from unittest import TestCase
import numpy as np
import tensorflow as tf

from tensorflowlib.map2ptslayer import HeatmapToPointsLayer


class TestHeatmapToPointsLayer(TestCase):

    def test_call(self):
        map = np.zeros((2, 128, 128, 2), dtype='float32')
        map[0, 5, 77, 0] = 1
        map[0, 4, 1, 0] = 0.5
        map[0, 6, 4, 0] = -1

        map[0, 1, 66, 1] = 1

        map[1, 121, 2, 0] = 0.1

        map[1, 3, 45, 1] = 12

        map_tensor = tf.convert_to_tensor(map)

        pts = HeatmapToPointsLayer()(map_tensor)
        pts = pts.numpy()

        expected_pts = np.array([[[5, 1], [77, 66]], [[121, 3], [2, 45]]], dtype='float32')
        np.testing.assert_array_equal(pts, expected_pts)
