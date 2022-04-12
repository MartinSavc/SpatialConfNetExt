import unittest
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflowlib.heatmaplayer import GaussianHeatmap


class GaussianHeatmapTest(unittest.TestCase):
    def test_1_generated_heatmap(self):
        gamma_ref = 0.5
        sigma_ref = np.array([2.0, 1.5, 1.0])
        pts_ref = np.array([[
            [2, 1.5, 1],
            [3, 5.5, 4],
            ],
            [
            [1, 2, 1],
            [5, 3, 1],
            ]])
        #
        # code to generate reference heatmaps:
        # B, _, N = pts_ref.shape
        # H, W = 5, 7
        # X, Y = np.meshgrid(np.arange(W), np.arange(H))
        # coords = np.dstack((Y, X)).reshape(1, H*W, 2, 1)
        # coords_zm = coords-pts_ref.reshape(B, 1, 2, N)
        # gauss_heatmap_ref = np.exp(-1*(coords_zm**2).sum(2)/(2*sigma_ref.reshape(1, 1, N)**2))
        # gauss_heatmap_ref *= gamma_ref/((2*np.pi)*sigma_ref.reshape(1, 1, N)**2)
        # gauss_heatmap_ref.shape = B, H, W, N
        #
        height = 5
        width = 7
        gauss_heatmap_ref = np.zeros((2, height, width, 3))
        gauss_heatmap_ref[0, :, :, 0] = [
                [0.0039, 0.0073, 0.0106, 0.0121, 0.0106, 0.0073, 0.0039],
                [0.0057, 0.0106, 0.0155, 0.0176, 0.0155, 0.0106, 0.0057],
                [0.0065, 0.0121, 0.0176, 0.0199, 0.0176, 0.0121, 0.0065],
                [0.0057, 0.0106, 0.0155, 0.0176, 0.0155, 0.0106, 0.0057],
                [0.0039, 0.0073, 0.0106, 0.0121, 0.0106, 0.0073, 0.0039],
            ]
        gauss_heatmap_ref[0, :, :, 1] = [
                [0., 0.0002, 0.0014, 0.0053, 0.013 , 0.0203, 0.0203],
                [0., 0.0004, 0.0022, 0.0083, 0.0203, 0.0316, 0.0316],
                [0., 0.0004, 0.0022, 0.0083, 0.0203, 0.0316, 0.0316],
                [0., 0.0002, 0.0014, 0.0053, 0.013 , 0.0203, 0.0203],
                [0., 0.0001, 0.0006, 0.0022, 0.0053, 0.0083, 0.0083],
            ]
        gauss_heatmap_ref[0, :, :, 2] = [
                [0., 0.0005, 0.0065, 0.0293, 0.0483, 0.0293, 0.0065],
                [0., 0.0009, 0.0108, 0.0483, 0.0796, 0.0483, 0.0108],
                [0., 0.0005, 0.0065, 0.0293, 0.0483, 0.0293, 0.0065],
                [0., 0.0001, 0.0015, 0.0065, 0.0108, 0.0065, 0.0015],
                [0., 0.    , 0.0001, 0.0005, 0.0009, 0.0005, 0.0001],
            ]




        gauss_heatmap_ref[1, :, :, 0] = [
                [0.0008, 0.0024, 0.0057, 0.0106, 0.0155, 0.0176, 0.0155],
                [0.0009, 0.0027, 0.0065, 0.0121, 0.0176, 0.0199, 0.0176],
                [0.0008, 0.0024, 0.0057, 0.0106, 0.0155, 0.0176, 0.0155],
                [0.0005, 0.0016, 0.0039, 0.0073, 0.0106, 0.0121, 0.0106],
                [0.0003, 0.0009, 0.0021, 0.0039, 0.0057, 0.0065, 0.0057],
            ]
        gauss_heatmap_ref[1, :, :, 1] = [
                [0.002 , 0.006 , 0.0116, 0.0145, 0.0116, 0.006 , 0.002 ],
                [0.0038, 0.0116, 0.0227, 0.0283, 0.0227, 0.0116, 0.0038],
                [0.0048, 0.0145, 0.0283, 0.0354, 0.0283, 0.0145, 0.0048],
                [0.0038, 0.0116, 0.0227, 0.0283, 0.0227, 0.0116, 0.0038],
                [0.002 , 0.006 , 0.0116, 0.0145, 0.0116, 0.006 , 0.002 ],
            ]
        gauss_heatmap_ref[1, :, :, 2] = [
                [0.0293, 0.0483, 0.0293, 0.0065, 0.0005, 0., 0.],
                [0.0483, 0.0796, 0.0483, 0.0108, 0.0009, 0., 0.],
                [0.0293, 0.0483, 0.0293, 0.0065, 0.0005, 0., 0.],
                [0.0065, 0.0108, 0.0065, 0.0015, 0.0001, 0., 0.],
                [0.0005, 0.0009, 0.0005, 0.0001, 0.    , 0., 0.],
            ]

        x_input = keras.Input(pts_ref.shape[1:])
        heatmap_output = GaussianHeatmap(gamma_ref, sigma_ref, 1.0, height, width)(x_input)
        test_model = keras.Model(inputs=[x_input], outputs=[heatmap_output])

        gauss_heatmap = test_model.predict(pts_ref)

        #import matplotlib.pyplot as pyplot
        #import pdb; pdb.set_trace()

        np.testing.assert_almost_equal(gauss_heatmap, gauss_heatmap_ref, decimal=4,
                err_msg='generated heatmaps don\'t equal reference heatmaps')

    def test_2_compute_output_shape(self):
        H, W = 7, 10
        heatmap_layer = GaussianHeatmap(0, 0, 1.0, height=H, width=W)
        output_shape = heatmap_layer.compute_output_shape((4, 2, 9))

        self.assertEqual(output_shape, (4, 7, 10, 9))


