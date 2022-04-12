
import unittest
import numpy as np
import tensorflow.keras as keras

from .poolinglayers import ArgMaxPool2D, ArgUpsample2D

class ArgMaxPool2DTest(unittest.TestCase):

    def test_1_output_sizes(self):
        input_shape = (3, 4, 4, 2)
        pooled_shape = (3, 2, 2, 2)
        output_shape = (3, 4, 4, 2)

        arg_max_pool_layer = ArgMaxPool2D()
        arg_upsample_layer = ArgUpsample2D()

        arg_max_pool_output_shape = arg_max_pool_layer.compute_output_shape(input_shape)
        arg_upsample_output_shape = arg_upsample_layer.compute_output_shape(arg_max_pool_output_shape)

        self.assertEqual(len(arg_max_pool_output_shape), 2,
                msg=f'expected 2 outputs from ArgMaxPool2D')
        self.assertEqual(arg_max_pool_output_shape[0], pooled_shape,
                msg=f'ArgMaxPool2D expected shape {pooled_shape}, returned shape {arg_max_pool_output_shape[0]}')
        self.assertEqual(arg_upsample_output_shape, output_shape,
                msg=f'ArgUpsample2D expected shape {output_shape}, returned shape {arg_upsample_output_shape}')

    def test_2_output_data(self):
        input_shape = (3, 4, 4, 2)
        pooled_shape = (3, 2, 2, 2)
        output_shape = (3, 4, 4, 2)

        a = np.zeros(input_shape, dtype=np.float32)
        a[0, :, :, 0] = [[0, 1, 2, 3],
                         [7, 6, 5, 4],
                         [2, 1, 1, 2],
                         [1, 1, 4, 1]]

        a[1, :, :, 0] = [[1, 1, 1, 2],
                         [1, 2, 1, 1],
                         [0, 2, 2, 1],
                         [1, 0, 1, 1]]

        a[2, :, :, 0] = [[9, 7, 6, 2],
                         [7, 2, 9, 1],
                         [2, 3, 7, 1],
                         [1, 0, 1, 1]]

        a[0, :, :, 1] = [[0, 3, 1, 2],
                         [3, 4, 4, 1],
                         [0, 4, 4, 3],
                         [1, 0, 1, 2]]

        a[1, :, :, 1] = [[2, 3, 5, 6],
                         [2, 1, 4, 7],
                         [7, 8, 1, 2],
                         [5, 4, 4, 3]]

        a[2, :, :, 1] = [[0, 1, 2, 3],
                         [7, 6, 5, 4],
                         [8, 9, 8, 7],
                         [3, 4, 5, 6]]

        b = np.zeros(pooled_shape, dtype=np.float32)
        b[0, :, :, 0] = [[7, 5], [2, 4]]
        b[1, :, :, 0] = [[2, 2], [2, 2]]
        b[2, :, :, 0] = [[9, 9], [3, 7]]
        b[0, :, :, 1] = [[4, 4], [4, 4]]
        b[1, :, :, 1] = [[3, 7], [8, 4]]
        b[2, :, :, 1] = [[7, 5], [9, 8]]

        c = np.zeros(output_shape, dtype=np.float32)
        c[0, :, :, 0] = [[0, 0, 0, 0],
                         [7, 0, 5, 0],
                         [2, 0, 0, 0],
                         [0, 0, 4, 0]]

        c[1, :, :, 0] = [[0, 0, 0, 2],
                         [0, 2, 0, 0],
                         [0, 2, 2, 0],
                         [0, 0, 0, 0]]

        c[2, :, :, 0] = [[9, 0, 0, 0],
                         [0, 0, 9, 0],
                         [0, 3, 7, 0],
                         [0, 0, 0, 0]]

        c[0, :, :, 1] = [[0, 0, 0, 0],
                         [0, 4, 4, 0],
                         [0, 4, 4, 0],
                         [0, 0, 0, 0]]

        c[1, :, :, 1] = [[0, 3, 0, 0],
                         [0, 0, 0, 7],
                         [0, 8, 0, 0],
                         [0, 0, 4, 0]]

        c[2, :, :, 1] = [[0, 0, 0, 0],
                         [7, 0, 5, 0],
                         [0, 9, 8, 0],
                         [0, 0, 0, 0]]

        v_in = keras.Input(batch_shape=a.shape)
        arg_max_pool_layer = ArgMaxPool2D()
        arg_upsample_layer = ArgUpsample2D()

        v_pool, v_pool_ind = arg_max_pool_layer(v_in)
        v_upsample = arg_upsample_layer((v_pool, v_pool_ind))

        test_model = keras.Model(v_in, outputs=[v_pool, v_upsample])
        b_res, c_res = test_model.predict(a)

        np.testing.assert_array_equal(b_res, b)
        np.testing.assert_array_equal(c_res, c)

    def test_3_ArgMaxPool2D_get_config(self):
        amp2d_orig = ArgMaxPool2D(pool_size=(3,3), strides=1, padding='valid', data_format='channels_last')

        amp2d_new = ArgMaxPool2D.from_config(amp2d_orig.get_config())


        self.assertEqual(amp2d_orig.pool_size, amp2d_new.pool_size, 
                f'pool_size differs, original: {amp2d_orig.pool_size} config: {amp2d_new.pool_size}')
        self.assertEqual(amp2d_orig.strides, amp2d_new.strides, 
                f'strides differ, original: {amp2d_orig.strides} config: {amp2d_new.strides}')
        self.assertEqual(amp2d_orig.padding, amp2d_new.padding, 
                f'padding differs, original: {amp2d_orig.padding} config: {amp2d_new.padding}')
        self.assertEqual(amp2d_orig.data_format, amp2d_new.data_format, 
                f'data_format differs, original: {amp2d_orig.data_format} config: {amp2d_new.data_format}')

    def test_4_ArgUpsample2D_get_config(self):
        aup2d_orig = ArgUpsample2D(pool_size=2, strides=3, padding='same', data_format='channels_last')

        aup2d_new = ArgUpsample2D.from_config(aup2d_orig.get_config())


        self.assertEqual(aup2d_orig.pool_size, aup2d_new.pool_size, 
                f'pool_size differs, original: {aup2d_orig.pool_size} config: {aup2d_new.pool_size}')
        self.assertEqual(aup2d_orig.strides, aup2d_new.strides, 
                f'strides differ, original: {aup2d_orig.strides} config: {aup2d_new.strides}')
        self.assertEqual(aup2d_orig.padding, aup2d_new.padding, 
                f'padding differs, original: {aup2d_orig.padding} config: {aup2d_new.padding}')
        self.assertEqual(aup2d_orig.data_format, aup2d_new.data_format, 
                f'data_format differs, original: {aup2d_orig.data_format} config: {aup2d_new.data_format}')


if __name__ == '__main__':
    unittest.main()
