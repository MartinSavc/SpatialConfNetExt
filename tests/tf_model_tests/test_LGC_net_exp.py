from unittest import TestCase
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../../2_tensorflow')

class TestKeyPointSpatialConfNet(TestCase):
    def test_mask_gen(self):
        image = np.zeros((2, 64, 64, 3))
        image[0, ..., 0] = 1
        image[1, ..., 2] = 0.1
        image = tf.convert_to_tensor(image, dtype=tf.float32)

        mask = tf.math.reduce_max(image, axis=(1, 2), keepdims=True)
        mask = tf.math.greater(mask, 0)
        mask = tf.cast(mask, tf.float32)
        actual = mask.numpy()

        expected = np.zeros((2, 1, 1, 3), dtype='float32')
        expected[0, ..., 0] = 1
        expected[1, ..., 2] = 1
        np.testing.assert_array_equal(actual, expected)
