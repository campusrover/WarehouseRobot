#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

class LineFollower:
    def __init__(self):
        self.bridge = CvBridge()
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        
        # Tuning parameters
        self.linear_speed = 0.1
        self.angular_gain = 0.005

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {}".format(e))
            return
        
        height, width, _ = cv_image.shape
        # Focus on a region of interest near bottom of image
        roi = cv_image[int(0.7*height):height, 0:width]
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        M = cv2.moments(thresh)
        cmd = Twist()

        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            # Compute error from center
            error = cx - width/2
            cmd.linear.x = self.linear_speed
            cmd.angular.z = -error * self.angular_gain
        else:
            # If no line detected, stop or turn slowly to find line
            cmd.linear.x = 0.0
            cmd.angular.z = 0.1  # turn to try and relocate line

        self.cmd_pub.publish(cmd)

if __name__ == '__main__':
    rospy.init_node('line_follower')
    lf = LineFollower()
    rospy.spin()
