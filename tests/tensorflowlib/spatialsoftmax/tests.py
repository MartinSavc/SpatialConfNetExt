import unittest
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras

from tensorflowlib.spatialsoftmax import (
        GeometricMean,
        SpatialMse,
        SpatialRmse,
        SpatialAdaptiveMse,
        )

def get_data(normalize=False):
    data = np.zeros((2, 4, 6, 3))
    data[0, :, :, 0] = [
            [0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            ]
    data[0, :, :, 1] = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
            ]
    data[0, :, :, 2] = [
            [0, 0, 0, 0, 0, 0],
            [0, 3, 0, 3, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            ]
    data[1, :, :, 0] = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 2],
            ]
    data[1, :, :, 1] = [
            [1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1],
            ]
    data[1, :, :, 2] = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            ]
    if normalize:
        data[0, ..., 0] /= 5
        data[0, ..., 1] /= 4
        data[0, ..., 2] /= 8
        data[1, ..., 0] /= 4
        data[1, ..., 1] /= 4
        data[1, ..., 2] /= 1

    geom_mean_ref = np.zeros((2, 2, 3))
    geom_mean_ref[0, :, 0] = 1., 2.
    geom_mean_ref[0, :, 1] = 1.5, 3.5
    geom_mean_ref[0, :, 2] = 1.25, 2.
    geom_mean_ref[1, :, 0] = 2.75, 4.75 
    geom_mean_ref[1, :, 1] = 1.5, 2.5
    geom_mean_ref[1, :, 2] = 3, 2

    return data, geom_mean_ref

class GeometricMeanTest(unittest.TestCase):
    def test_1_general_data(self):

        data, geom_mean_ref = get_data(normalize=False)

        x_input = keras.Input(data.shape[1:])

        geom_mean_layer = GeometricMean(normalize=True)(x_input)
        test_model = keras.Model(inputs=[x_input], outputs=[geom_mean_layer])
        geom_mean = test_model.predict(data)

        np.testing.assert_almost_equal(geom_mean, geom_mean_ref, 
                err_msg='calculated means do not equal reference means')

    def test_2_normalized_data(self):
        data, geom_mean_ref = get_data(normalize=True)

        x_input = keras.Input(data.shape[1:])

        geom_mean_layer = GeometricMean(normalize=False)(x_input)
        test_model = keras.Model(inputs=[x_input], outputs=[geom_mean_layer])
        geom_mean = test_model.predict(data)

        np.testing.assert_almost_equal(geom_mean, geom_mean_ref, 
                err_msg='calculated means do not equal reference means')


    def test_3_compute_output_shape(self):

        geom_mean_layer = GeometricMean(normalize=False)
        output_shape = geom_mean_layer.compute_output_shape((12, 40, 52, 102))
        self.assertEqual(output_shape, (12, 2, 102))

def get_data_2():
        y_1 = np.array([
            [[0, 1, 1, 0],
             [1, 1, 0, 2],],
            [[0.5, -0.5, 0, 2],
             [1.5,  1.5, 2, 0],],
            [[0, 0, 0, 0],
             [0, 0, 0, 0],],
            ])
        y_2 = np.array([
            [[1, 1, 0, 0],
             [0, 0, 1, 1],],
            [[1.5, 0.5, 1, 3],
             [2.5, 0.5, 1, -1],],
            [[0, 3, 0, 1],
             [2, 0, 4, 1],],
            ])
        mse_err_ref = np.array([
            [2, 1.0, 2, 1.0],
            [2, 2.0, 2, 2.0],
            [4, 9.0,16, 2.0],
            ])
        rmse_err_ref = mse_err_ref**0.5

        return y_1, y_2, mse_err_ref, rmse_err_ref

class SpatialRmseTest(unittest.TestCase):
    def test_1_spatial_mse(self):
        y_true = keras.Input((2, 4))
        y_pred = keras.Input((2, 4))

        mse_err = SpatialMse()(y_true, y_pred)
        test_model = keras.Model(inputs=[y_true, y_pred], outputs=[mse_err])

        y_1, y_2, mse_err_ref, _ = get_data_2()

        mse_err_res = test_model.predict((y_1, y_2))
        np.testing.assert_almost_equal(mse_err_res, mse_err_ref, decimal=5,
                err_msg='calculated mean squared error does not match reference')

    def test_2_spatial_rmse(self):
        y_true = keras.Input((2, 4))
        y_pred = keras.Input((2, 4))

        rmse_err = SpatialRmse()(y_true, y_pred)
        test_model = keras.Model(inputs=[y_true, y_pred], outputs=[rmse_err])

        y_1, y_2, _, rmse_err_ref = get_data_2()

        rmse_err_res = test_model.predict((y_1, y_2))
        np.testing.assert_almost_equal(rmse_err_res, rmse_err_ref, decimal=5,
                err_msg='calculated root mean squared error does not match reference')

    def test_3_spatial_adaptive_mse(self):
        y_true = keras.Input((2, 4))
        y_pred = keras.Input((2, 4))

        y_1 = np.array([
            [[0.0,  1.0, 1.0, 0.0],
             [1.0,  1.0, 0.0, 2.0],],
            [[0.5, -0.5, 0.0, 2.0],
             [1.5,  1.5, 2.0, 0.0],],
            [[0.0,  0.0, 0.0, 0.0],
             [0.0,  0.0, 0.0, 0.0],],
            ])
        y_2 = np.array([
            [[1, 1, 0, 0],
             [0, 0, 1, 1],],
            [[1.5, 0.5, 1, 3],
             [2.5, 0.5, 1, -1],],
            [[0, 3, 0, 1],
             [2, 0, 4, 1],],
            ])

        amse_err_ref = np.array([
           [9.25280978e-01, 4.30475761e-01, 9.25280978e-01, 5.97436816e-03],
           [1.76781774e-01, 1.76781774e-01, 1.19487363e-02, 1.19487363e-02],
           [2.00000000e+00, 4.50000000e+00, 8.00000000e+00, 1.00000000e+00],
           ])

        amse_err = SpatialAdaptiveMse(scale=1.2, pow_fact=10)(y_true, y_pred)
        test_model = keras.Model(inputs=[y_true, y_pred], outputs=[amse_err])

        amse_err_res = test_model.predict((y_1, y_2))
        np.testing.assert_almost_equal(amse_err_res, amse_err_ref, decimal=5,
                err_msg='calculated mean squared error does not match reference')

if __name__ == '__main__':
    unittest.main()
