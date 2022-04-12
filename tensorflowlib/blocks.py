import tensorflow.keras as keras


def simple_conv_block(in_tensor,
                      num_filters,
                      kernel_size,
                      stride=1,
                      padding='same',
                      activation='relu',
                      batch_norm=False,
                      kernel_initializer=None,
                      bias_initializer=None):

    # We want to use default initializers in case the ones passed to this function are None.
    kwargs = {
        'filters': num_filters,
        'kernel_size': kernel_size,
        'strides': stride,
        'padding': padding,
        'activation': activation,
        'kernel_initializer': kernel_initializer,
        'bias_initializer': bias_initializer
    }
    out = keras.layers.Conv2D(**{k: v for k, v in kwargs.items() if v is not None})(in_tensor)

    if batch_norm:
        out.use_bias = False
        out = keras.layers.BatchNormalization()(out)

    if activation:
        out = keras.layers.Activation(activation)(out)

    return out
