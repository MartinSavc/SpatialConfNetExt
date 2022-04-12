from unittest import TestCase
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../../2_tensorflow')
import shn_net_exp


class TestKeyPointSpatialConfNet(TestCase):
    def test_split_image(self):
        tf.enable_eager_execution()

        model = shn_net_exp.KeyPointSpatialConfNet()

        image = np.zeros((2, 64, 64, 1))
        image[0, :32, :32] = 50
        image[0, 32:, :32] = 150
        image[0, :32, 32:] = 200

        image[1, :32, :32] = 175
        image[1, 32:, :32] = 125
        image[1, :32, 32:] = 75
        image = tf.convert_to_tensor(image, dtype=tf.float32)

        shape = tf.shape(image)

        splitted = []
        tile_length = 2
        for x in range(tile_length):
            for y in range(tile_length):
                slice_size = (shape[1] // tile_length)
                splitted.append(image[:, y * slice_size:(y+1) * slice_size,
                                x * slice_size:(x+1) * slice_size])

        unsplitted = model.unsplit_feat_map(splitted, tf.shape(image))
        np.testing.assert_array_equal(image, unsplitted)
