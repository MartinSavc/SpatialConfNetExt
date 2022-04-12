import tensorflow as tf
import tensorflow.keras as keras

class SpatialSoftmax(keras.layers.Layer):
    '''
    The layers calculates softmax over the second and third dimension 
    (usually width and height).

    Receives a tensor of shape:
    BxHxWxN

    returns a tensor of shape:
    BxHxWxN

    H, W and N must be known ahead of time (cannot work with variable width
    and height inputs).
    '''
    def __init__(self, **kwargs):
        self.ref_input_shape = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        _, H, W, chns = input_shape
        if H is None or W is None or chns is None:
            raise Exception(f'Cannot build SpatialSoftmax with shape {input_shape}')
        self.ref_input_shape = H, W, chns
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        H, W, chns = self.ref_input_shape
        inputs_lin = tf.reshape(inputs, (-1, H*W, chns))
        softmax_lin = tf.nn.softmax(inputs_lin, 1)
        softmax = tf.reshape(softmax_lin, (-1, H, W, chns))
        return softmax
