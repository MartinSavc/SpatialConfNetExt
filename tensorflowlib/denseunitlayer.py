import tensorflow.keras as keras

class DenseUnitLayer(keras.layers.Layer):
    '''
    A series of convolutions connected together into a DenseNet unit.

    The input is convolved multiple times using convolutions with a linear activation function.
    The given activation function is applied to the last layer.

    The output is the concatenation of the last convolution output and original input.

    conv_count - int
        Number of convolutions to apply.
    filters - int
        Number of additional output channels 
    kernel_size - int
        size of each convolution kernel
    activation - 
        activation function or its name, passed on to keras.Conv2D
    kernel_regularizer - 
        regularizer function or name, passed on to keras.Conv2D
    '''
    def __init__(self, conv_count, filters, kernel_size=3, activation=None, kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.conv_count = conv_count
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.kernel_regularizer = kernel_regularizer

        self.conv_layers = []
        self.concat_layer = None

    def get_config(self):
        conf_dict = super().get_config()
        conf_dict.update({
            'conv_count' : self.conv_count,
            'filters' : self.filters,
            'kernel_size' : self.kernel_size,
            'activation' : self.activation,
            'kernel_regularizer' : self.kernel_regularizer,
            })
        return conf_dict

    def build(self, input_shape):
        for n in range(self.conv_count-1):
            conv_layer = keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                padding='same',
                kernel_regularizer=self.kernel_regularizer,
                )

            self.conv_layers.append(conv_layer)

        conv_layer = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            activation=self.activation,
            kernel_regularizer=self.kernel_regularizer,
            )
        self.conv_layers.append(conv_layer)

        self.concat_layer = keras.layers.Concatenate()


    def compute_output_shape(self, input_shape):
        output_shape = [*input_shape,]
        output_shape[3] += self.filters
        return output_shape

    def call(self, inputs):
        in_tensor = inputs
        
        cur_tensor = in_tensor
        for n in range(self.conv_count):
            cur_tensor = self.conv_layers[n](cur_tensor)

        out_tensor = self.concat_layer([in_tensor, cur_tensor])
        return out_tensor
