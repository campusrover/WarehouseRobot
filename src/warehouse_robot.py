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
        self.current_state = "LINE_FOLLOWING"  # States: LINE_FOLLOWING, APPROACHING, TURNING_AROUND, RETURNING
        
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

        # Initial position and orientation
        self.initial_yaw = None
        self.initial_position = None

        # Wait for initial odometry to set initial yaw and position
        rospy.loginfo("Waiting for initial odometry...")
        while self.initial_yaw is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        self.current_state = "LINE_FOLLOWING"  # Initial state
        self.mission_complete = False  # Flag for mission completion

    def odom_callback(self, msg):
        # Get orientation quaternion
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
            
            # Create mask for orange line
            mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
            h, w, _ = image.shape
            mask[:int(3 * h / 5), :] = 0  # Focus on lower part of the image
            
            # Find line contours
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
            
            # Draw status information
            self.draw_status_info(image)
            
            cv2.imshow("Line Following", image)
            cv2.waitKey(1)
            
        except cv_bridge.CvBridgeError as e:
            rospy.logerr(f"Image processing error: {e}")

    def draw_status_info(self, image):
        """Draw status information on the image"""
        # Draw target fiducial and current state
        target_text = f"Target Fiducial: {self.target_fiducial}"
        cv2.putText(image, target_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        state_text = f"State: {self.current_state}"
        cv2.putText(image, state_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if self.pin_position:
            pin_info = f"Pin: ({self.pin_position.x:.2f}, {self.pin_position.y:.2f})"
            cv2.putText(image, pin_info, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            dist_info = f"Distance to pin: {self.distance_to_pin:.2f}m"
            cv2.putText(image, dist_info, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            angle_info = f"Angle to pin: {math.degrees(self.angle_to_pin):.1f} deg"
            cv2.putText(image, angle_info, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def pid_control(self, err):
        current_time = rospy.Time.now().to_sec()
        delta_time = current_time - self.prev_time
        
        if delta_time > 0:
            derivative = (err - self.prev_err) / delta_time
            self.control_signal = (KP * err) + (KD * derivative)
            
        self.prev_time = current_time
        self.prev_err = err

    def update_pin_position(self):
        """Get the position of the target fiducial pin"""
        try:
            transform = self.tf_buffer.lookup_transform('base_link', f'pin_{self.target_fiducial}', rospy.Time(0))
            self.pin_position = transform.transform.translation
            
            # Calculate distance and angle to pin
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
        """Check if we're at the right position relative to pin"""
        if not self.pin_position:
            return False
        
        # Get pin coordinates
        pin_x = self.pin_position.x  # Forward/backward position
        pin_y = self.pin_position.y  # Lateral position
        
        # Add more detailed logging
        rospy.loginfo(f"Pin {self.target_fiducial} detailed position:")
        rospy.loginfo(f"  X (forward/back): {pin_x:.3f}")
        rospy.loginfo(f"  Y (lateral): {pin_y:.3f}")
        rospy.loginfo(f"  Distance: {self.distance_to_pin:.3f}")
        rospy.loginfo(f"  Angle: {math.degrees(self.angle_to_pin):.1f} degrees")
        
        # More flexible thresholds
        in_line_with_pin = abs(pin_x) < 0.15  # Increased from 0.2
        correct_distance = 0.15 < abs(pin_y) < 0.6  # Widened range
        
        # Log the conditions
        rospy.loginfo(f"  In line check: {in_line_with_pin} (need < 0.25)")
        rospy.loginfo(f"  Distance check: {correct_distance} (need between 0.25 and 0.7)")
        
        if in_line_with_pin and correct_distance:
            rospy.loginfo(f"In position to turn towards pin {self.target_fiducial}")
            return True
            
        return False

    def calculate_movement(self):
        """Calculate robot movement based on current state"""
        # Update pin position
        self.update_pin_position()
        
        # State machine
        if self.current_state == "LINE_FOLLOWING":
            # First priority: Stay on the line
            if not self.line_detected:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.3
                rospy.loginfo("Searching for line...")
                return
                
            # Check if we're at the right position relative to pin
            if self.fiducial_found and self.is_parallel_to_pin():
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
                rospy.sleep(0.5)  # Brief pause before approaching
                self.current_state = "APPROACHING"
                rospy.loginfo(f"At correct position - starting approach to pin {self.target_fiducial}")
                return
            
            # Normal line following with debug info
            self.twist.linear.x = MAX_SPEED
            self.twist.angular.z = -self.control_signal
            rospy.loginfo_throttle(1.0, "Following line...")
                
        elif self.current_state == "APPROACHING":
            if not self.fiducial_found:
                rospy.logwarn(f"Lost sight of pin {self.target_fiducial} during approach")
                return
                
            # Move towards pin with detailed feedback
            rospy.loginfo_throttle(0.5, f"Approaching - Distance: {self.distance_to_pin:.3f}m")
            
            if self.distance_to_pin < 0.10:  # Close enough to pin
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
                rospy.sleep(0.5)  # Brief pause before turning
                self.current_state = "TURNING_AROUND"
                rospy.loginfo(f"Reached pin {self.target_fiducial}, starting turn around")
                return
                
            # Approach while maintaining alignment
            self.twist.linear.x = APPROACH_SPEED
            self.twist.angular.z = 0.5 * self.angle_to_pin

        elif self.current_state == "TURNING_AROUND":
            # Define target yaw as initial_yaw + 180 degrees
            target_yaw = self.normalize_angle(self.initial_yaw + math.pi)
            yaw_diff = self.normalize_angle(target_yaw - self.current_yaw)
            
            rospy.loginfo(f"Turning around - Target yaw: {math.degrees(target_yaw):.1f}, Current yaw: {math.degrees(self.current_yaw):.1f}, Diff: {math.degrees(yaw_diff):.1f}")
            
            if abs(yaw_diff) < 0.05:  # Within ~3 degrees
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
                rospy.sleep(0.5)  # Brief pause before returning
                self.current_state = "RETURNING"
                rospy.loginfo("Turned around - starting to return")
                return
            
            # Turn towards target yaw
            turn_direction = 1.0 if yaw_diff > 0 else -1.0
            self.twist.linear.x = 0.0
            self.twist.angular.z = turn_direction * TURN_SPEED
            rospy.loginfo_throttle(0.5, f"Turning around - Angular Z: {self.twist.angular.z:.2f}")
            
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
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
                self.robot_stopped = True
                self.mission_complete = True
                rospy.loginfo("Mission complete - returned to home")
                
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
        
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
        rospy.loginfo(f"Starting warehouse robot - targeting pin {self.target_fiducial}...")
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown() and not self.mission_complete:
            self.calculate_movement()
            
            if not self.robot_stopped:
                self.cmd_vel_pub.publish(self.twist)
                
            rate.sleep()
        
        # Stop the robot when mission is complete
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)
        rospy.loginfo("Robot stopped.")

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
