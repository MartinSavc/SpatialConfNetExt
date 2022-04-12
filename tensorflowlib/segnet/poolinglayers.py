import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

class ArgMaxPool2D(keras.layers.Layer):
    '''
    Similar to MaxPooling2D layer but also returns indices of the selected values.
    This is used together with ArgUpsample2D to provide an improved upscaling
    for max pooling.

    Arguments are compatible with MaxPooling2D. 
    This layer outputs two tensors - pooled tensor and indice tensor. The
    pooled tesor is processed further, the indice is used with ArgUpsample2D.

    The first input tensor of ArgUpsample2D must have the same shape as x_pool.

    x_input
    x_pool, x_ind = arg_max_pool_layer = ArgMaxPool2D()(x_input)

    x_pool_conv = Layer()(x_pool)

    x_output = arg_upsample_layer = ArgUpsample2D()((x_pool_conv, x_ind))
    
    Example

    the 2d image:
    [1 2 3 4]
    [4 3 1 2]
    [2 1 4 3]
    [3 4 2 1]

    after MaxPooling2D is reduced to:
    [4 4]
    [4 4]
    
    and after ArgUpsample2D:
    [0 0 0 4]
    [4 0 0 0]
    [0 0 4 0]
    [0 4 0 0]

    The size of inputs must be known in advanced - including batch size.
    '''
    def __init__(self, pool_size=(2, 2), strides=None,  padding='valid', data_format=None, **kwargs):
        super().__init__(**kwargs)

        if isinstance(pool_size, int):
            pool_size = (1, pool_size, pool_size, 1)
        elif isinstance(pool_size, (tuple, list)) and len(pool_size) == 2:
            pool_size = (1, *pool_size, 1)

        if strides is None:
            strides = pool_size
        elif isinstance(strides, int):
            strides = (1, strides, strides, 1)
        elif isinstance(strides, (tuple, list)) and len(strides) == 2:
            strides = (1, *strides, 1)

        if padding in ('same', 'SAME'):
            padding = 'SAME'
        elif padding in ('valid', 'VALID'):
            padding = 'VALID'

        if data_format is None:
            data_format = 'channels_last'

        if data_format == 'channels_last':
            data_format = 'NHWC'
        elif data_format == 'channels_first':
            data_format = 'NCHW'

        self.strides = strides
        self.pool_size = pool_size
        self.padding = padding
        self.data_format = data_format

    def get_config(self):
        conf_dict = super().get_config()
        conf_dict.update({
            'pool_size':  self.pool_size,
            'strides' : self.strides,
            'padding' : self.padding,
            'data_format' : self.data_format,
            })
        return conf_dict

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        _, dh, dw, _ = self.strides
        n, h, w, c = input_shape
        return (n, h//dh, w//dw, c), (n, h//dh, w//dw, c)

    def call(self, x_in):
        x_out, x_ind_out = tf.nn.max_pool_with_argmax(
                x_in,
                self.pool_size,
                self.strides,
                self.padding,
                self.data_format,
                include_batch_in_index=True,
                name=None)

        return x_out, x_ind_out

class ArgUpsample2D(keras.layers.Layer):
    '''
    '''
    def __init__(self, pool_size=(2, 2), strides=None,  padding='valid', data_format=None, **kwargs):
        super().__init__(**kwargs)

        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        elif isinstance(pool_size, (tuple, list)) and len(pool_size) == 2:
            self.pool_size = (1, *pool_size, 1)

        if strides is None:
            strides = pool_size
        elif isinstance(strides, int):
            strides = (strides, strides)
        elif isinstance(strides, (tuple, list)) and len(strides) == 2:
            self.strides = (1, *strides, 1)

        self.strides = strides
        self.pool_size = pool_size
        self.padding = padding
        self.data_format = data_format

    def get_config(self):
        conf_dict = super().get_config()
        conf_dict.update({
            'pool_size':  self.pool_size,
            'strides' : self.strides,
            'padding' : self.padding,
            'data_format' : self.data_format,
            })
        return conf_dict

    def build(self, input_shape):
        data_shape, ind_shape = input_shape

        self.output_reshape = self.compute_output_shape(input_shape)
        self.ind_reshape = (*ind_shape, 1)
        self.lin_reshape = (np.prod(self.output_reshape),)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        dh, dw = self.strides
        (n, h, w, c), _ = input_shape
        return n, h*dh, w*dw, c

    def call(self, inputs):
        x_in, x_ind_in = inputs

        indices = tf.reshape(x_ind_in, self.ind_reshape)
        updates = x_in

        x_out_lin = tf.scatter_nd(indices, updates, self.lin_reshape, name=None)
        x_out = tf.reshape(x_out_lin, self.output_reshape)
        return x_out

