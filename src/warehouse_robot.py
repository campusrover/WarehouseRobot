#!/usr/bin/env python3

import math
import rospy
import cv2
import cv_bridge
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist, Point
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry

# Constants
MAX_SPEED = 0.1
MIN_SPEED = 0.02
TURN_SPEED = 0.2
APPROACH_SPEED = 0.05
LOWER_ORANGE = np.array([10, 150, 150])
UPPER_ORANGE = np.array([20, 255, 255])
KP = 0.005
KD = 0.001

class WarehouseRobot:
    def __init__(self, target_fiducial):
        rospy.init_node('warehouse_robot', anonymous=True)
        
        # Store target fiducial ID
        self.target_fiducial = target_fiducial
        rospy.loginfo(f"Targeting fiducial ID: {self.target_fiducial}")
        
        # Initialize state
        self.robot_stopped = False
        self.current_state = "LINE_FOLLOWING"  # States: LINE_FOLLOWING, APPROACHING, RETURN_TO_LINE, TURN_TO_HOME, RETURNING, FACE_FIDUCIALS
        
        # Publishers and Subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.image_sub = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.image_cb)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        self.bridge = cv_bridge.CvBridge()
        self.twist = Twist()
        self.line_detected = False
        self.control_signal = 0.0
        self.prev_time = rospy.Time.now().to_sec()
        self.prev_err = 0.0
        
        # TF setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        
        # Position tracking
        self.pin_position = None
        self.fiducial_found = False
        self.distance_to_pin = float('inf')
        self.angle_to_pin = 0.0
        self.current_yaw = 0.0
        self.stored_line_yaw = None  # Will store yaw when we first turn towards fiducial
        self.temp_line_position = None
        self.temp_line_yaw = None

        # Initial position and orientation
        self.initial_yaw = None
        self.initial_position = None
        self.line_return_yaw = None  # Store yaw when leaving the line
        self.turn_direction = None  # Will store 1 for clockwise, -1 for counterclockwise
        self.original_line_yaw = None  # Will store the orientation when on the line
        
        # Mission control
        self.mission_complete = False

        # Wait for initial odometry
        rospy.loginfo("Waiting for initial odometry...")
        while self.initial_yaw is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        rospy.loginfo(f"Initial yaw set to {math.degrees(self.initial_yaw):.2f} degrees")

    def odom_callback(self, msg):
        orientation = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([
            orientation.x, orientation.y, 
            orientation.z, orientation.w])
        self.current_yaw = yaw

        if self.initial_yaw is None:
            self.initial_yaw = yaw
            self.initial_position = msg.pose.pose.position
            rospy.loginfo(f"Initial yaw set to {math.degrees(self.initial_yaw):.2f} degrees")
            rospy.loginfo(f"Initial position set to ({self.initial_position.x:.2f}, {self.initial_position.y:.2f})")
    
    def image_cb(self, msg):
        if self.robot_stopped and self.current_state == "LINE_FOLLOWING":
            return
            
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
            h, w, _ = image.shape
            mask[:int(3 * h / 5), :] = 0
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                self.line_detected = True
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    err = cx - w // 2
                    self.pid_control(err)
                    cv2.drawContours(image, [largest_contour], -1, (0, 255, 255), 3)
                    cv2.circle(image, (cx, h-30), 5, (0, 0, 255), -1)
            else:
                self.line_detected = False
            
            self.draw_status_info(image)
            cv2.imshow("Line Following", image)
            cv2.waitKey(1)
            
        except cv_bridge.CvBridgeError as e:
            rospy.logerr(f"Image processing error: {e}")

    def draw_status_info(self, image):
        cv2.putText(image, f"State: {self.current_state}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if self.pin_position:
            cv2.putText(image, f"Distance to pin: {self.distance_to_pin:.2f}m", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image, f"Angle to pin: {math.degrees(self.angle_to_pin):.1f} deg", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def pid_control(self, err):
        current_time = rospy.Time.now().to_sec()
        delta_time = current_time - self.prev_time
        
        if delta_time > 0:
            derivative = (err - self.prev_err) / delta_time
            self.control_signal = (KP * err) + (KD * derivative)
            
        self.prev_time = current_time
        self.prev_err = err

    def update_pin_position(self):
        try:
            transform = self.tf_buffer.lookup_transform('base_link', f'pin_{self.target_fiducial}', rospy.Time(0))
            self.pin_position = transform.transform.translation
            
            self.distance_to_pin = math.sqrt(
                self.pin_position.x**2 + self.pin_position.y**2
            )
            self.angle_to_pin = math.atan2(self.pin_position.y, self.pin_position.x)
            
            self.fiducial_found = True
            return True
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            if not self.fiducial_found:
                rospy.logwarn_throttle(1, f"Cannot see pin {self.target_fiducial}: {e}")
            self.fiducial_found = False
            return False

    def is_parallel_to_pin(self):
        if not self.pin_position:
            return False
        
        pin_x = self.pin_position.x
        pin_y = self.pin_position.y
        
        in_line_with_pin = abs(pin_x) < 0.10
        correct_distance = 0.15 < abs(pin_y) < 0.8
        
        if in_line_with_pin and correct_distance:
            return True
        return False

    def calculate_movement(self):
        self.update_pin_position()
        
        # In the calculate_movement method, modify the LINE_FOLLOWING state:
        if self.current_state == "LINE_FOLLOWING":
            if not self.line_detected:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.3
                rospy.loginfo_throttle(1.0, "Searching for line...")
                return
                        
            if self.fiducial_found and self.is_parallel_to_pin():
                # Stop and store current position and orientation
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
                pos, _ = self.get_current_position()
                if pos:
                    self.temp_line_position = (pos.x, pos.y)
                    self.original_line_yaw = self.current_yaw
                    
                    # Determine which way we need to turn initially
                    # If pin is on the right, we'll turn clockwise (negative angular.z)
                    # If pin is on the left, we'll turn counterclockwise (positive angular.z)
                    if self.pin_position.y > 0:  # Pin is on the left
                        self.turn_direction = 1  # Counterclockwise
                    else:  # Pin is on the right
                        self.turn_direction = -1  # Clockwise
                        
                    rospy.loginfo(f"Stored line position at ({pos.x:.2f}, {pos.y:.2f})")
                    rospy.loginfo(f"Original orientation: {math.degrees(self.original_line_yaw):.2f}")
                    rospy.loginfo(f"Turn direction: {'counterclockwise' if self.turn_direction > 0 else 'clockwise'}")
                    
                rospy.sleep(0.5)
                self.current_state = "APPROACHING"
                return
            
            # Normal line following
            self.twist.linear.x = MAX_SPEED
            self.twist.angular.z = -self.control_signal
                
        elif self.current_state == "APPROACHING":
            if not self.fiducial_found:
                rospy.logwarn("Lost sight of fiducial during approach")
                return
                
            # Calculate smooth approach speed
            approach_speed = min(APPROACH_SPEED, max(MIN_SPEED, self.distance_to_pin * 0.3))
            angle_correction = 0.3 * self.angle_to_pin
            
            # If very close to fiducial, prepare to stop
            if self.distance_to_pin < 0.20:  # 20cm from fiducial
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
                self.robot_stopped = True
                
                response = input("At fiducial. Type 'continue' to return home, or 'shutdown': ").lower()
                if response == 'continue':
                    self.robot_stopped = False
                    self.current_state = "RETURN_TO_LINE"
                    rospy.loginfo("Starting return journey")
                elif response == 'shutdown':
                    self.shutdown_requested = True
                return
                
            # Smooth approach
            self.twist.linear.x = approach_speed
            self.twist.angular.z = angle_correction

        elif self.current_state == "RETURN_TO_LINE":
            if not self.temp_line_position or self.original_line_yaw is None:
                rospy.logwarn("Missing line position or original orientation!")
                return
                    
            # Get current position
            pos, _ = self.get_current_position()
            if pos:
                dx = self.temp_line_position[0] - pos.x
                dy = self.temp_line_position[1] - pos.y
                dist_to_temp = math.sqrt(dx*dx + dy*dy)
                angle_to_temp = math.atan2(dy, dx)
                
                rospy.loginfo_throttle(1.0, f"Distance to line: {dist_to_temp:.2f}m")
                
                if dist_to_temp > 0.1:  # Not at line position yet
                    # First turn to face line position
                    angle_diff = self.normalize_angle(angle_to_temp - self.current_yaw)
                    
                    if abs(angle_diff) > 0.05:  # Need to turn more
                        self.twist.linear.x = 0.0
                        self.twist.angular.z = TURN_SPEED if angle_diff > 0 else -TURN_SPEED
                        return
                        
                    # Move towards line position
                    self.twist.linear.x = min(MAX_SPEED, dist_to_temp * 0.5)
                    self.twist.angular.z = 0.3 * angle_diff
                    return
                        
                else:  # At line position, turn opposite of original turn
                    self.twist.linear.x = 0.0
                    # Use the opposite of the original turn direction
                    self.twist.angular.z = TURN_SPEED * (-self.turn_direction)
                    
                    # Check if we've turned back to approximately the opposite of original orientation
                    target_yaw = self.normalize_angle(self.original_line_yaw + math.pi)
                    yaw_diff = self.normalize_angle(self.current_yaw - target_yaw)
                    
                    rospy.loginfo_throttle(1.0, 
                        f"Turning to face home. Current: {math.degrees(self.current_yaw):.1f}, " +
                        f"Target: {math.degrees(target_yaw):.1f}, " +
                        f"Diff: {math.degrees(yaw_diff):.1f}")
                    
                    if abs(yaw_diff) < 0.1:  # Successfully turned to face home
                        self.twist.linear.x = 0.0
                        self.twist.angular.z = 0.0
                        rospy.sleep(0.5)
                        self.current_state = "RETURNING"
                        rospy.loginfo("Oriented towards home, starting return journey")
                    return

        elif self.current_state == "TURN_TO_HOME":
            # Calculate vector from current position to home
            pos, _ = self.get_current_position()
            if pos and self.initial_position:
                dx = self.initial_position.x - pos.x
                dy = self.initial_position.y - pos.y
                target_yaw = math.atan2(dy, dx)  # Calculate angle to home
                yaw_diff = self.normalize_angle(target_yaw - self.current_yaw)
                
                rospy.loginfo_throttle(1.0, 
                    f"Turn to home - Current: {math.degrees(self.current_yaw):.1f}, " +
                    f"Target: {math.degrees(target_yaw):.1f}, " +
                    f"Diff: {math.degrees(yaw_diff):.1f}")
                
                if abs(yaw_diff) > 0.1:
                    turn_speed = min(TURN_SPEED, max(0.1, abs(yaw_diff) * 0.5))
                    self.twist.linear.x = 0.0
                    self.twist.angular.z = turn_speed if yaw_diff > 0 else -turn_speed
                    return
                
                # Done turning, start returning
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
                rospy.sleep(0.5)
                self.current_state = "RETURNING"
                rospy.loginfo("Aligned with home, starting return journey")
                return

        elif self.current_state == "RETURNING":
            # Follow the line back to home
            if not self.line_detected:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.3
                rospy.loginfo("Searching for line on return...")
                return

            # Follow the line
            self.twist.linear.x = MAX_SPEED
            self.twist.angular.z = -self.control_signal
            rospy.loginfo_throttle(1.0, "Returning to home...")

            # Check if back to initial orientation and position
            if self.is_at_home():
                # Stop completely first
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
                self.cmd_vel_pub.publish(self.twist)
                rospy.sleep(1.0)  # Pause to ensure complete stop
                
                # Start the 180-degree turn
                self.current_state = "TURN_180"
                self.turn_start_yaw = self.current_yaw  # Store starting orientation
                rospy.loginfo("At home - starting 180 degree turn")
                return

        elif self.current_state == "TURN_180":
            # Calculate how far we've turned
            angle_turned = self.normalize_angle(self.current_yaw - self.turn_start_yaw)
            angle_remaining = math.pi - abs(angle_turned)  # How far to go to reach 180°
            
            rospy.loginfo_throttle(1.0, f"Turning 180 - Turned: {math.degrees(abs(angle_turned)):.1f}°")
            
            if angle_remaining > 0.1:  # Still need to turn
                # Simple constant turn
                self.twist.linear.x = 0.0
                self.twist.angular.z = TURN_SPEED
                return
            
            # Turn complete - stop and get next fiducial
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.0
            self.cmd_vel_pub.publish(self.twist)
            rospy.sleep(0.5)  # Brief pause
            
            # Get next fiducial
            rospy.loginfo("Turn complete - getting next fiducial")
            new_id = self.get_next_fiducial()
            if new_id:
                self.target_fiducial = new_id
                rospy.loginfo(f"New target fiducial set to {self.target_fiducial}")
                self.current_state = "LINE_FOLLOWING"
            else:
                self.mission_complete = True
            return

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def get_current_position(self):
        """Get current position from odometry"""
        try:
            odom = rospy.wait_for_message('/odom', Odometry, timeout=1.0)
            return odom.pose.pose.position, odom.pose.pose.orientation
        except rospy.ROSException:
            rospy.logwarn("Failed to get current position")
            return None, None
        
    def is_at_home(self):
        """Determine if the robot has returned to home"""
        if not self.initial_position:
            return False
        
        # Get current position from odometry
        try:
            odom = rospy.wait_for_message('/odom', Odometry, timeout=1.0)
            current_pos = odom.pose.pose.position
            dx = current_pos.x - self.initial_position.x
            dy = current_pos.y - self.initial_position.y
            distance = math.sqrt(dx**2 + dy**2)
            rospy.loginfo(f"Returning - Distance from home: {distance:.3f}m")
            return distance < 0.2  # Threshold to consider as home
        except rospy.ROSException:
            return False

    def run(self):
        self.shutdown_requested = False
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown() and not self.shutdown_requested:
            rospy.loginfo(f"\nStarting mission to fiducial {self.target_fiducial}")
            
            # Reset state for new mission
            self.mission_complete = False
            self.robot_stopped = False
            self.current_state = "LINE_FOLLOWING"
            self.temp_line_position = None
            self.temp_line_yaw = None
            self.turn_direction = None
            
            # Main control loop
            while not rospy.is_shutdown() and not self.mission_complete:
                self.calculate_movement()
                if not self.robot_stopped:
                    self.cmd_vel_pub.publish(self.twist)
                rate.sleep()
            
            # After mission completion, ensure robot is fully stopped
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.0
            self.cmd_vel_pub.publish(self.twist)
            
            # Get next fiducial or shutdown
            response = input("\nMission complete. Enter 'next' for new fiducial or 'shutdown': ").lower()
            if response == 'next':
                new_id = self.get_next_fiducial()
                if new_id:
                    self.target_fiducial = new_id
                    rospy.loginfo(f"New target fiducial set to {self.target_fiducial}")
                else:
                    self.shutdown_requested = True
            else:
                self.shutdown_requested = True
        
        rospy.loginfo("Robot shutting down.")

    def is_at_home(self):
        """Determine if the robot has returned to home"""
        if not self.initial_position:
            return False

        # Get current position from odometry
        try:
            odom = rospy.wait_for_message('/odom', Odometry, timeout=1.0)
            current_pos = odom.pose.pose.position
            dx = current_pos.x - self.initial_position.x
            dy = current_pos.y - self.initial_position.y
            distance = math.sqrt(dx**2 + dy**2)
            rospy.loginfo_throttle(1.0, f"Distance from home: {distance:.3f}m")
            return distance < 0.2  # Threshold to consider as home
        except rospy.ROSException:
            return False

    def get_next_fiducial(self):
        """Get next fiducial ID from user"""
        while True:
            try:
                print("\nAvailable fiducial IDs: 106, 100, 108")
                fiducial_id = int(input("Enter the target fiducial ID: "))
                if fiducial_id in [106, 100, 108]:
                    return fiducial_id
                print("Invalid fiducial ID. Please choose from 106, 100, or 108.")
            except ValueError:
                print("Please enter a valid number.")

def get_valid_fiducial_id():
    """Get valid fiducial ID from user input"""
    while True:
        try:
            print("\nAvailable fiducial IDs: 106, 100, 108")
            fiducial_id = int(input("Enter the target fiducial ID: "))
            if fiducial_id in [106, 100, 108]:
                return fiducial_id
            print("Invalid fiducial ID. Please choose from 106, 100, or 108.")
        except ValueError:
            print("Please enter a valid number.")

if __name__ == '__main__':
    try:
        # Get target fiducial ID from user
        target_id = get_valid_fiducial_id()
        
        # Create and run robot with target fiducial
        robot = WarehouseRobot(target_id)
        robot.run()
    except rospy.ROSInterruptException:
        pass

