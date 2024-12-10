#!/usr/bin/env python3

import rospy
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion

class MyOdom:
    def __init__(self):
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odom_cb)
        self.my_odom_pub = rospy.Publisher('my_odom', Point, queue_size=1)
        self.old_pose = None
        self.dist = 0.0
        self.yaw = 0.0

    def odom_cb(self, msg):
        """Callback function for `odom_sub`."""
        cur_pose = msg.pose.pose
        self.update_dist(cur_pose)
        self.update_yaw(cur_pose.orientation)
        self.publish_data()

    def update_dist(self, cur_pose):
        """Update distance between current pose and previous pose."""
        if self.old_pose is not None:
            x_diff = cur_pose.position.x - self.old_pose.position.x
            y_diff = cur_pose.position.y - self.old_pose.position.y
            self.dist = math.sqrt(x_diff ** 2 + y_diff ** 2)
        self.old_pose = cur_pose

    def update_yaw(self, cur_orientation):
        """Update yaw based on orientation quaternion."""
        orientations = [
            cur_orientation.x,
            cur_orientation.y,
            cur_orientation.z,
            cur_orientation.w
        ]
        (roll, pitch, yaw) = euler_from_quaternion(orientations)
        self.yaw = yaw if yaw >= 0 else 2 * math.pi + yaw

    def publish_data(self):
        """Publish distance and yaw as a Point message."""
        data = Point()
        data.x = self.dist
        data.y = self.yaw
        self.my_odom_pub.publish(data)

if __name__ == '__main__':
    rospy.init_node('my_odom')
    MyOdom()
    rospy.spin()

