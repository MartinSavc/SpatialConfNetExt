import unittest
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras

from tensorflowlib.spatialsoftmax import PartitionsToFeatures

class PartitionsToFeaturesTest(unittest.TestCase):


    def test_1_general_data(self):
        data = np.zeros((3, 4, 6, 2))
        data[0, :, :, 0] = [
                [ 0,  1,  2,  3,  4,  5],
                [ 6,  7,  8,  9, 10, 11],
                [12, 13, 14, 15, 16, 17],
                [18, 19, 20, 21, 22, 23],
                ]
        data[0, :, :, 1] = [
                [1, 1, 2, 2, 3, 3],
                [1, 1, 2, 2, 3, 3],
                [4, 4, 5, 5, 6, 6],
                [4, 4, 5, 5, 6, 6],
                ]
        data[1, ...] = data[0, ...]*-1
        data[2, ...] = data[0, ...]*0.5

        feat_win_2_ref = np.zeros((3, 15, 8))
        feat_win_2_sum_ref = np.zeros((3, 15, 2))
        feat_win_3_ref = np.zeros((3, 6, 18))

        # win_size=2
        feat_win_2_ref[0, :, :] = [
                [0, 1, 1, 1, 6, 1, 7, 1],
                [2, 2, 3, 2, 8, 2, 9, 2],
                [4, 3, 5, 3, 10, 3, 11, 3],
                [12, 4, 13, 4, 18, 4, 19, 4],
                [14, 5, 15, 5, 20, 5, 21, 5],
                [16, 6, 17, 6, 22, 6, 23, 6],
                [1, 1, 2, 2, 7, 1, 8, 2],
                [3, 2, 4, 3, 9, 2, 10, 3],
                [13, 4, 14, 5, 19, 4, 20, 5],
                [15, 5, 16, 6, 21, 5, 22, 6],
                [6, 1, 7, 1, 12, 4, 13, 4],
                [8, 2, 9, 2, 14, 5, 15, 5],
                [10, 3, 11, 3, 16, 6, 17, 6],
                [7, 1, 8, 2, 13, 4, 14, 5],
                [9, 2, 10, 3, 15, 5, 16, 6],
                ]

        feat_win_2_sum_ref[0, :, :] = [
                [14, 4],
                [22, 8],
                [30, 12],
                [62, 16],
                [70, 20],
                [78, 24],
                [18, 6],
                [26, 10],
                [66, 18],
                [74, 22],
                [38, 10],
                [46, 14],
                [54, 18],
                [42, 12],
                [50, 16],
                ]

        feat_win_3_ref[0, :, :] = [
                [0, 1, 1, 1, 2, 2, 6, 1, 7, 1, 8, 2, 12, 4, 13, 4, 14, 5],
                [3, 2, 4, 3, 5, 3, 9, 2, 10, 3, 11, 3, 15, 5, 16, 6, 17, 6],
                [1, 1, 2, 2, 3, 2, 7, 1, 8, 2, 9, 2, 13, 4, 14, 5, 15, 5],
                [6, 1, 7, 1, 8, 2, 12, 4, 13, 4, 14, 5, 18, 4, 19, 4, 20, 5],
                [9, 2, 10, 3, 11, 3, 15, 5, 16, 6, 17, 6, 21, 5, 22, 6, 23, 6],
                [7, 1, 8, 2, 9, 2, 13, 4, 14, 5, 15, 5, 19, 4, 20, 5, 21, 5],
                ]

        feat_win_2_ref[1, ...] = feat_win_2_ref[0, ...]*-1
        feat_win_2_ref[2, ...] = feat_win_2_ref[0, ...]*0.5
        feat_win_2_sum_ref[1, ...] = feat_win_2_sum_ref[0, ...]*-1
        feat_win_2_sum_ref[2, ...] = feat_win_2_sum_ref[0, ...]*0.5
        feat_win_3_ref[1, ...] = feat_win_3_ref[0, ...]*-1
        feat_win_3_ref[2, ...] = feat_win_3_ref[0, ...]*0.5

        x_input = keras.Input(data.shape[1:])
        feat_win_2_tensor = PartitionsToFeatures(win_size=2)(x_input)
        feat_win_2_sum_tensor = PartitionsToFeatures(win_size=2, sum_over_window=True)(x_input)
        feat_win_3_tensor = PartitionsToFeatures(win_size=3)(x_input)
        test_model = keras.Model(inputs=[x_input], outputs=[feat_win_2_tensor, feat_win_2_sum_tensor, feat_win_3_tensor])
        feat_win_2, feat_win_2_sum, feat_win_3 = test_model.predict(data)
        B = data.shape[0]
        for b in range(B):
                np.testing.assert_equal(
                        feat_win_2[b, ...],
                        feat_win_2_ref[b, ...],
                        err_msg=f'extracted feature not equal to expected, batch {b}, win_size 2')

        for b in range(B):
                np.testing.assert_equal(
                        feat_win_2_sum[b, ...],
                        feat_win_2_sum_ref[b, ...],
                        err_msg=f'extracted feature not equal to expected, batch {b}, win_size 2, sum over window')

        for b in range(B):
                np.testing.assert_equal(
                        feat_win_3[b, ...],
                        feat_win_3_ref[b, ...],
                        err_msg=f'extracted feature not equal to expected, batch {b}, win_size 3')

    def test_2_compute_output_shape(self):
        part_gm_layer = PartitionsToFeatures(win_size=10)
        output_shape = part_gm_layer.compute_output_shape((12, 20, 30, 4))
        self.assertEqual(output_shape, (12, 15, 10*10*4))

        part_gm_layer = PartitionsToFeatures(win_size=8)
        output_shape = part_gm_layer.compute_output_shape((12, 20, 30, 4))
        self.assertEqual(output_shape, (12, 24, 8*8*4))

        part_gm_layer = PartitionsToFeatures(win_size=10, sum_over_window=True)
        output_shape = part_gm_layer.compute_output_shape((12, 20, 30, 4))
        self.assertEqual(output_shape, (12, 15, 4))

        part_gm_layer = PartitionsToFeatures(win_size=8, sum_over_window=True)
        output_shape = part_gm_layer.compute_output_shape((12, 20, 30, 4))
        self.assertEqual(output_shape, (12, 24, 4))
