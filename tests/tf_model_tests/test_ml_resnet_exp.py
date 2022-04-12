from unittest import TestCase
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../../2_tensorflow')
import ml_resnet_exp


class TestMLResnet(TestCase):
    def test_mask(self):
        pts = np.random.randint(1, 92, (2, 2, 4), dtype=int)

        pts[0, 0, 3] = 0
        pts[1, 1, 0] = 0

        pts[0, 0, 0] = 0
        pts[0, 1, 0] = 0

        pts[1, 1, 3] = 0
        pts[1, 0, 3] = 0

        in_pts = tf.convert_to_tensor(pts, dtype=tf.float32)

        model = ml_resnet_exp.MLResnet()

        mask = model.generate_mask(in_pts)
        mask = mask.numpy()

        expected_mask = np.full((2, 1, 4), 1)
        expected_mask[0, 0, 0] = 0
        expected_mask[1, 0, 3] = 0

        np.testing.assert_array_equal(mask, expected_mask)

