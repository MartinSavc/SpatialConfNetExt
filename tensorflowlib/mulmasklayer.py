import tensorflow.keras as keras

class MultiplyMask(keras.layers.Layer):
    '''
    Train a multiplication mask for the input. Can be used for as a classification
    prior.

    vmin - double
    vmax - double
        minimum and maximum value in the mask.
    '''
    def __init__(self, vmin=1e-5, vmax=1.0, axis=0, **kwargs):
        super().__init__(**kwargs)
        self.vmin = vmin
        self.vmax = vmax
        self.axis = axis

    def get_config(self):
        conf = super().get_config()
        conf.update({
            'vmin' : self.vmin,
            'vmax' : self.vmax,
            'axis' : self.axis,
            })
        return conf

    def build(self, input_shape):
        self.map = self.add_weight(
                name='map',
                shape=(1, )+input_shape[1:],
                initializer=keras.initializers.Ones(),
                constraint=keras.constraints.MinMaxNorm(self.vmin, self.vmax, axis=0),
                trainable=True)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        return inputs*self.map
