#!/usr/bin/env python3
"""ROS integration regression: timestamps, RCS, partition, ground and temporal filtering."""
import math
import threading
import time
import unittest

import rospy
import rostest
from geometry_msgs.msg import TwistWithCovarianceStamped
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        self.messages, self.lock, self.subscribers = {}, threading.Lock(), []
        for mode in (2, 3):
            for name in ('inlier', 'outlier', 'flat', 'radar_vel', 'ego_vel'):
                key = (mode, name)
                kind = TwistWithCovarianceStamped if name.endswith('vel') else PointCloud2
                self.subscribers.append(rospy.Subscriber('/preprocess_test_%d/%s' % key,
                    kind, lambda msg, key=key: self.receive(key, msg), queue_size=10))
        self.publisher = rospy.Publisher('/preprocess_test_input', PointCloud2, queue_size=10)
        self.wait(lambda: self.publisher.get_num_connections() == 2 and
                  all(sub.get_num_connections() > 0 for sub in self.subscribers))

    def receive(self, key, msg):
        with self.lock:
            self.messages[key] = msg

    def wait(self, predicate):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and not rospy.is_shutdown():
            if predicate():
                return
            time.sleep(.02)
        self.fail('Timed out waiting for preprocessing outputs')

    def frame(self, stamp, points, velocity=True):
        names = ('x y z intensity range azimuth elevation velocity snr rcs detectionconfidence '
                 'mse_range mse_azimuth mse_elevation mse_velocity std_dev_range '
                 'std_dev_azimuthAngle std_dev_elevationAngle std_dev_velocity std_dev_rcs').split()
        fields = [PointField(name, i * 4, PointField.FLOAT32, 1) for i, name in enumerate(names)]
        rows = []
        for x, y, z, doppler, rcs in points:
            radius = math.sqrt(x*x + y*y + z*z)
            rows.append([x, y, z, 20, radius, math.atan2(y, x), math.atan2(z, math.hypot(x, y)),
                         doppler, 20, rcs, 1] + [0] * (len(names) - 11))
        msg = point_cloud2.create_cloud(Header(stamp=rospy.Time.from_sec(stamp), frame_id='raw_radar'), fields, rows)
        self.publisher.publish(msg)
        expected = [(mode, key) for mode in (2, 3) for key in
                    (('inlier', 'outlier', 'flat', 'radar_vel', 'ego_vel') if velocity else ('inlier', 'outlier', 'flat'))]
        def ready():
            with self.lock:
                return all(k in self.messages and self.messages[k].header.stamp == msg.header.stamp for k in expected)
        self.wait(ready)
        with self.lock:
            return {key: self.messages[key] for key in expected}

    def test_preprocessing_contract(self):
        static = [(10 + i*.4, (i % 7 - 3)*.3, (i % 5)*.1, 0, -8 + i*.5) for i in range(30)]
        # Trailing outliers expose the old inlier_idx[j] read past the end.
        dynamic = [(15, 2, .1, 4, 20), (16, -2, .2, 5, 21)]
        below_ground = [(12, 1, -3, 0, 10)]
        first = self.frame(10., static + dynamic + below_ground)
        for mode in (2, 3):
            inside = first[mode, 'inlier']
            self.assertEqual(inside.width * inside.height, 30)
            self.assertEqual(first[mode, 'outlier'].width * first[mode, 'outlier'].height, 2)
            self.assertEqual(inside.header.frame_id, 'radar')
            values = list(point_cloud2.read_points(inside, field_names=('intensity', 'rcs')))
            self.assertEqual(values[0], (-8., -8.))  # Signed RCS is not clamped by preprocessing.
            self.assertTrue(all(p[0] == 0 for p in point_cloud2.read_points(first[mode, 'flat'], field_names=('z',))))
            self.assertTrue(all(math.isfinite(x) for x in first[mode, 'ego_vel'].twist.covariance))

        isolated = [(100, 100, 1, 0, 30)]
        for stamp, expected_count in ((10.1, 30), (10.2, 31)):
            result = self.frame(stamp, static + isolated + dynamic)
            for mode in (2, 3):
                self.assertEqual(result[mode, 'inlier'].width * result[mode, 'inlier'].height, expected_count)
        empty = self.frame(10.3, [], velocity=False)
        self.assertTrue(all(msg.width * msg.height == 0 for msg in empty.values()))
        result = self.frame(10.4, static + isolated + dynamic)
        for mode in (2, 3):
            self.assertEqual(result[mode, 'inlier'].width * result[mode, 'inlier'].height, 31)


if __name__ == '__main__':
    rospy.init_node('test_radar_preprocessing')
    rostest.rosrun('dyna_rc', 'radar_preprocessing', PreprocessingTest)
