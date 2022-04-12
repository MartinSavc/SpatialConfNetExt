import numpy as np
import tensorflow as tf
import sys
import os
import unittest

sys.path.append('../../2_tensorflow')
import ta_graph_exp

# Invalid index exceptions are only available when running on the CPU.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class TestTAGraphNet(unittest.TestCase):

    def test_perpective_transform2(self):
        model = ta_graph_exp.TAGraphNet()

        coords = np.zeros((2, 2, 3), dtype='float32')

        coords[0, :, 0] = [4, 4]
        coords[0, :, 1] = [7, 1]
        coords[0, :, 2] = [7, 1]

        coords[1, :, 0] = [54, 543]
        coords[1, :, 1] = [4, 4]
        coords[1, :, 2] = [7, 1]

        coords = tf.convert_to_tensor(coords, dtype=tf.float32)

        M = np.array([
            [2, 0.5, -100],
            [0, 2, 0],
            [0, 0.005, 1]
        ])
        M = M[np.newaxis, ...]
        M = np.repeat(M, 2, axis=0)

        M = tf.convert_to_tensor(M, dtype=tf.float32)

        pred_vals = model.perspective_transform(coords, M)
        expected_vals = np.array([
            [
                [-88.23529412, -85.07462687, -85.07462687],
                [7.84313725, 1.99004975, 1.99004975]
            ],
            [
                [75.23553163, -88.23529412, -85.07462687],
                [292.32839838, 7.84313725, 1.99004975]
            ]
        ], dtype='float32')
        np.testing.assert_array_almost_equal(pred_vals, expected_vals, decimal=3)

    def test_shape_feat_extraction(self):
        model = ta_graph_exp.TAGraphNet()

        coords = np.zeros((2, 2, 2), dtype='float32')

        coords[0, :, 0] = [5, 4]
        coords[0, :, 1] = [1.75, 1.1]

        coords[1, :, 0] = [3.3, 7.5]
        coords[1, :, 1] = [5.5, 1.5]

        coords = tf.convert_to_tensor(coords, dtype=tf.float32)

        shape_feats = model.extract_shape_feats(coords)

        pred_vals = np.array([x.numpy() for x in shape_feats])
        expected_vals = np.array([
            [
                [[-3.25], [3.25]],
                [[2.2], [-2.2]]
            ],
            [
                [[-2.9], [2.9]],
                [[-6], [6]]
            ]
        ], dtype='float32')

        np.testing.assert_array_almost_equal(pred_vals, expected_vals)

    def test_visual_feat_extraction(self):
        model = ta_graph_exp.TAGraphNet()
        model.scale = 1

        feat_map = np.zeros((2, 64, 64, 2))

        x, y = np.mgrid[:64, :64]
        feat_map[0, ..., 0] = x + y
        feat_map[0, ..., 1] = x * y
        feat_map[1, ..., 0] = (x + y) + 5.7
        feat_map[1, ..., 1] = (x * y) - 1.53

        feat_map = tf.convert_to_tensor(feat_map, dtype=tf.float32)

        coords = np.zeros((2, 2, 2), dtype='float32')

        coords[0, :, 0] = [1.75, 1.75]
        coords[0, :, 1] = [1.75, 1.75]

        coords[1, :, 0] = [0.5, 0.5]
        coords[1, :, 1] = [0.5, 0.5]

        coords = tf.convert_to_tensor(coords, dtype=tf.float32)

        visual_feats = model.extract_visual_feats(feat_map, coords)

        pred_vals = visual_feats.numpy()
        expected_vals = np.array([
            [[3.5, 3.0625],
             [3.5, 3.0625]],
            [[6.7, -1.28],
             [6.7, -1.28]]
        ])

        np.testing.assert_array_almost_equal(pred_vals, expected_vals)