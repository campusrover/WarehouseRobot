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

        # Initial position and orientation
        self.initial_yaw = None
        self.initial_position = None
        self.line_return_yaw = None  # Store yaw when leaving the line
        
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
        
        if self.current_state == "LINE_FOLLOWING":
            if not self.line_detected:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.3
                return
                
            if self.fiducial_found and self.is_parallel_to_pin():
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
                self.line_return_yaw = self.current_yaw
                rospy.loginfo(f"Storing line return yaw: {math.degrees(self.line_return_yaw):.2f}")
                rospy.sleep(0.5)
                self.current_state = "APPROACHING"
                return
            
            self.twist.linear.x = MAX_SPEED
            self.twist.angular.z = -self.control_signal
                
        elif self.current_state == "APPROACHING":
            if not self.fiducial_found:
                return
                    
            # Smooth approach to fiducial
            if self.distance_to_pin < 0.20:  # Close enough to fiducial
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
                # Store exact position where we'll need to return
                pos, _ = self.get_current_position()
                if pos:
                    self.return_position = (pos.x, pos.y)
                    self.return_yaw = self.current_yaw
                    rospy.loginfo(f"Stored return position: ({pos.x:.2f}, {pos.y:.2f})")
                
                self.robot_stopped = True
                response = input("At fiducial. Type 'continue' to return home, or 'shutdown': ").lower()
                if response == 'continue':
                    self.robot_stopped = False
                    self.current_state = "RETURN_TO_LINE"
                elif response == 'shutdown':
                    self.mission_complete = True
                    self.shutdown_requested = True
                return
                    
            # Smoother approach speed
            approach_speed = min(APPROACH_SPEED, max(MIN_SPEED, self.distance_to_pin * 0.2))
            self.twist.linear.x = approach_speed
            self.twist.angular.z = 0.3 * self.angle_to_pin

        elif self.current_state == "RETURN_TO_LINE":
            if not hasattr(self, 'return_position') or not self.return_position:
                rospy.logwarn("No return position stored!")
                return
                
            # Calculate distance to return point
            pos, _ = self.get_current_position()
            if pos:
                dx = self.return_position[0] - pos.x
                dy = self.return_position[1] - pos.y
                dist_to_return = math.sqrt(dx*dx + dy*dy)
                angle_to_return = math.atan2(dy, dx)
                
                if dist_to_return > 0.1:  # Not at return point yet
                    # Turn towards return point
                    angle_diff = self.normalize_angle(angle_to_return - self.current_yaw)
                    if abs(angle_diff) > 0.1:
                        self.twist.linear.x = 0.0
                        self.twist.angular.z = TURN_SPEED if angle_diff > 0 else -TURN_SPEED
                    else:
                        # Move towards return point
                        self.twist.linear.x = min(MAX_SPEED, dist_to_return * 0.5)
                        self.twist.angular.z = 0.3 * angle_diff
                else:
                    # At return point, turn towards home
                    self.current_state = "TURN_TO_HOME"
                    rospy.loginfo("At return point, turning towards home")

        elif self.current_state == "TURN_TO_HOME":
            yaw_diff = self.normalize_angle(self.current_yaw - self.initial_yaw)
            
            if abs(yaw_diff) > 0.1:
                self.twist.linear.x = 0.0
                self.twist.angular.z = TURN_SPEED if yaw_diff < 0 else -TURN_SPEED
                return
            else:
                rospy.loginfo("Aligned with home direction")
                self.current_state = "RETURNING"
                return

        elif self.current_state == "RETURNING":
            if not self.line_detected:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.3
                return
            
            self.twist.linear.x = MAX_SPEED
            self.twist.angular.z = -self.control_signal
            
            if self.is_at_home():
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
                rospy.loginfo("Reached home position")
                self.current_state = "FACE_FIDUCIALS"
                return

        elif self.current_state == "FACE_FIDUCIALS":
            target_yaw = self.normalize_angle(self.initial_yaw + math.pi)
            yaw_diff = abs(self.normalize_angle(self.current_yaw - target_yaw))
            
            if yaw_diff > 0.1:
                self.twist.linear.x = 0.0
                self.twist.angular.z = TURN_SPEED
                return
            else:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
                self.robot_stopped = True
                self.mission_complete = True
                rospy.loginfo("Ready for next mission")
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
        if not self.initial_position:
            return False
        
        try:
            odom = rospy.wait_for_message('/odom', Odometry, timeout=1.0)
            current_pos = odom.pose.pose.position
            dx = current_pos.x - self.initial_position.x
            dy = current_pos.y - self.initial_position.y
            distance = math.sqrt(dx**2 + dy**2)
            
            return distance < 0.2
        except rospy.ROSException:
            return False

    def run(self):
        self.shutdown_requested = False
        
        while not rospy.is_shutdown() and not self.shutdown_requested:
            rospy.loginfo(f"Starting mission to fiducial {self.target_fiducial}")
            rate = rospy.Rate(10)
            
            # Reset state for new mission
            self.mission_complete = False
            self.robot_stopped = False
            self.current_state = "LINE_FOLLOWING"
            self.return_position = None
            
            # Main control loop
            while not rospy.is_shutdown() and not self.mission_complete:
                self.calculate_movement()
                if not self.robot_stopped:
                    self.cmd_vel_pub.publish(self.twist)
                rate.sleep()
            
            # If mission complete and not shutdown requested, get next fiducial
            if not self.shutdown_requested and self.mission_complete:
                response = input("Mission complete. 'next' for new fiducial or 'shutdown': ").lower()
                if response == 'next':
                    self.target_fiducial = self.get_next_fiducial()
                else:
                    self.shutdown_requested = True
            
            # Stop robot
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.0
            self.cmd_vel_pub.publish(self.twist)
        
        rospy.loginfo("Robot shutting down.")

def get_valid_fiducial_id():
    """Get valid fiducial ID from user input"""
    while True:
        try:
            print("\nAvailable fiducial IDs: 104, 100, 108")
            fiducial_id = int(input("Enter the target fiducial ID: "))
            if fiducial_id in [104, 100, 108]:
                return fiducial_id
            print("Invalid fiducial ID. Please choose from 104, 100, or 108.")
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
