import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from .segnet import segnet_layer_gen, ArgMaxPool2D, ArgUpsample2D

def segnet_model_gen(
        input_shape,
        chn_out,
        n_layers=4,
        n_convs=1,
        features=64,
        filter_size=7,
        pool_size=2,
        weight_decay=0,
        softmax_output=False,
        activation='relu',
        batchnorm=True,
        dropout=False,
        dropout_rate=0.25,
    ):
    '''
    Creates a generic segnet model for image segmentation. With optional
    softmax over channels output.


    input_shape - tuple of ints (N, H, W, C)
        Shape of input data. Must be fully defined, including batch size.

    chn_out - int
        Number of output channels - target classes.

    n_layers - int
        Number of segnet layers

    n_convs - int
        Numer of convolutions per each layer.

    features - int
        Number of features/filters per each convolution.

    filter_size - int
        Size of convolution filters.

    pool_size - int
        Size of pooling region.
    '''
    x_in = keras.Input(batch_shape=input_shape)

    x_out = segnet_layer_gen(
            x_in,
            n_layers,
            n_convs,
            features,
            filter_size,
            pool_size,
            weight_decay,
            activation,
            batchnorm,
            dropout,
            dropout_rate,
            )

    if softmax_output:
        out_activation = 'softmax'
    else:
        out_activation = 'linear'

    x_out = layers.Conv2D(
            filters=chn_out,
            kernel_size=1,
            padding='same',
            activation=out_activation,
            use_bias=True,
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            )(x_out)


    model = keras.Model(inputs=x_in, outputs=x_out)
    return model

