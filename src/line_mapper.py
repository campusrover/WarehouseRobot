#! /usr/bin/python3
import rospy
from geometry_msgs.msg import Point
from tf.transformations import quaternion_from_euler
import tf2_ros

class LineMapper:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.pin_dict = {}
        self.line_fiducials = []  # To store positions of gate fiducials

    def set_pins(self, fid_ids):
        """Initialize fiducial pins for mapping."""
        for id in fid_ids:
            tfs = tf2_ros.TransformStamped()
            tfs.header.frame_id = 'odom'
            tfs.child_frame_id = f'pin_{id}'
            self.pin_dict[id] = {'tfs': tfs, 'mapped': False}

    def update_line_positions(self):
        """Update positions of fiducials defining the gate."""
        self.line_fiducials = []
        for fid_id, pin in self.pin_dict.items():
            if pin['mapped']:
                pos = pin['tfs'].transform.translation
                self.line_fiducials.append((fid_id, Point(pos.x, pos.y, pos.z)))

    def is_intruder_near_gate(self, intruder_pos):
        """Check if the intruder is near the gate."""
        if len(self.line_fiducials) < 2:
            rospy.logwarn("Gate fiducials are not fully mapped!")
            return False

        _, fid1 = self.line_fiducials[0]
        _, fid2 = self.line_fiducials[1]

        # Check if the intruder is between fid1 and fid2
        line_vec = [fid2.x - fid1.x, fid2.y - fid1.y]
        intruder_vec = [intruder_pos[0] - fid1.x, intruder_pos[1] - fid1.y]

        # Check if the intruder is near the line segment
        cross_product = line_vec[0] * intruder_vec[1] - line_vec[1] * intruder_vec[0]
        return abs(cross_product) < 0.5  # Adjust threshold for sensitivity

if __name__ == '__main__':
    rospy.init_node('line_mapper_test')
    mapper = LineMapper()
    fid_ids = [107, 105]
    mapper.set_pins(fid_ids)
    rospy.loginfo("Testing LineMapper with fiducials [107, 105]")
    rospy.spin()
