import unittest
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflowlib.mulmasklayer import MultiplyMask

class MultiplyMaskTest(unittest.TestCase):
    def test_1_multiply_image(self):
        img_ref = np.array([[
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            ],
            [
            [1, 2, 10, 10., 1],
            [1, 2, 10, 5, 0.5],
            [1, 2, 10, 1/0.3, 1/3],
            [1, 2, 10, 1/0.4, 0.25],
            ]])
        y_ref = np.array([[
            [1, 1, 0.3, 0.4, 5],
            [5, 2, 0.3, 0.4, 2],
            [1, 0.5, 0.1, 0.3, 3],
            [1, 0.5, 0.1, 0.4, 4],
            ],
            [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            ]])
        mask = np.array([[
            [1, 0.5, 0.1, 0.1, 1],
            [1, 0.5, 0.1, 0.2, 2],
            [1, 0.5, 0.1, 0.3, 3],
            [1, 0.5, 0.1, 0.4, 4],
            ]])
        x_input = keras.Input(img_ref.shape[1:])
        mult_layer = MultiplyMask() 
        y_output = mult_layer(x_input)
        test_model = keras.Model(inputs=[x_input], outputs=[y_output])

        mult_layer.set_weights((mask,))

        y_res = test_model.predict(img_ref)
        np.testing.assert_almost_equal(y_res, y_ref, decimal=4,
                err_msg='generated output doesn\'t equal reference')


    def test_2_multiply_train(self):
        img_ref = np.array([[
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            ],
            [
            [1, 2, 10, 10., 1],
            [1, 2, 10, 5, 0.5],
            [1, 2, 10, 1/0.3, 1/3],
            [1, 2, 10, 1/0.4, 0.25],
            ]])
        y_ref = np.array([[
            [1, 1, 0.3, 0.4, 5],
            [5, 2, 0.3, 0.4, 2],
            [1, 0.5, 0.1, 0.3, 3],
            [1, 0.5, 0.1, 0.4, 4],
            ],
            [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            ]])
        mask_ref = np.array([[
            [1, 0.5, 0.2, 0.2, 1],
            [1, 0.5, 0.2, 0.2, 2],
            [1, 0.5, 0.2, 0.3, 3],
            [1, 0.5, 0.2, 0.4, 4],
            ]])

        x_input = keras.Input(img_ref.shape[1:])
        mult_layer = MultiplyMask(vmin=0.2, vmax=5)
        y_output = mult_layer(x_input)
        test_model = keras.Model(inputs=[x_input], outputs=[y_output])

        test_model.compile(loss='mse')
        test_model.fit(x=img_ref, y=y_ref, epochs=3000, verbose=0)

        mask_res = mult_layer.get_weights()[0]

        np.testing.assert_almost_equal(mask_res, mask_ref, decimal=1,
                err_msg='estimated mask doesn\'t match reference')
