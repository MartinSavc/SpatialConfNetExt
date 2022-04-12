
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from .poolinglayers import ArgMaxPool2D, ArgUpsample2D

def segnet_layer_gen(
        in_tensor,
        n_layers=4,
        n_convs=1,
        features=64,
        filter_size=7,
        pool_size=2,
        weight_decay=0,
        activation='relu',
        batchnorm=True,
        dropout=False,
        dropout_rate=0.25,
        ):
    '''
    Construct a series of layers following the segnet pattern. 

    in_tensor - tensorflow.tensor
        Input tensor, starting point. Must have all of its dimensions specified,
        including batch, width and height. It's width and height must be divisible
        by pool_size**n_layers

    n_layers - int
        Number of segnet layers.

    n_convs - int
        Number of convolutions done on each layer, once before downsampling, once
        before upsampling. Must be >1.

    features - int
        Number of features/kernel used for each convolution.

    filter_size - int or tuple of (int, int)
        Sizes of filter used. Can be a tuple of 2 ints.

    pool_size - int
        Pooling size parameter. Passed to pooling layer,  only tested with 2.
        Must be integer. 

    weight_decay - float
        Weight of kernel l2 regularizer added to convolution layers.

    activation - string,...
        Activation layer to use. Is passed on to keras.Activation as first parameter.

    batchnorm - bool
        If true, batchnormalization are added after convolutions and before activation.

    dropout - bool
        If true, dropout layers are added after convolutions and before activation.
        If both batchnorm and dropout are set, batchonorm is used first and 
        dropout after. This is however not recommended.

    dropout_rate - float
        Dropout rate parameter.

    Returns:
        out_tensor - tensorflow.tensor
            The output tensor, after applying all the segnet layers.
    '''

    out_tensor = in_tensor
    ind_list = []
    for l in range(n_layers):
        for k in range(n_convs):
            out_tensor = layers.Conv2D(
                    filters=features,
                    kernel_size=filter_size,
                    padding='same',
                    activation='linear',
                    use_bias=True,
                    kernel_regularizer=keras.regularizers.l2(weight_decay),
                    )(out_tensor)

            if batchnorm:
                out_tensor = layers.BatchNormalization()(out_tensor)
            if dropout:
                out_tensor = layers.Dropout(rate=dropout_rate)(out_tensor)
            out_tensor = layers.Activation(activation)(out_tensor)

        out_tensor, ind = ArgMaxPool2D(
                pool_size=(pool_size, pool_size),
                padding='valid',
                )(out_tensor)
        ind_list.append(ind)

    for l in range(n_layers):
        ind = ind_list.pop()
        out_tensor = ArgUpsample2D(
                pool_size=(pool_size, pool_size),
                padding='valid',
                )((out_tensor, ind))

        for k in range(n_convs):
            out_tensor = layers.Conv2D(
                    filters=features,
                    kernel_size=filter_size,
                    padding='same',
                    activation='linear',
                    use_bias=True,
                    kernel_regularizer=keras.regularizers.l2(weight_decay),
                    )(out_tensor)

            if batchnorm:
                out_tensor = layers.BatchNormalization()(out_tensor)
            if dropout:
                out_tensor = layers.Dropout(rate=dropout_rate)(out_tensor)
            out_tensor = layers.Activation(activation)(out_tensor)

    return out_tensor

def calc_segnet_cropping_size(
        size,
        n_layers,
        n_convs,
        filter_size,
        pool_size,
        ):
    '''
    Calculates the cropping size for segnet architecture with the given parameters.

    size - int
    Single number representing the dimension. If the input image has an un-even
    size, call the function twice once with rows and once with columns to get
    row and column cropping separately. 

    Returns:
    start, end - int, int
    Two integers representing the number of values cropped from start and end 
    of the input of the given size. Both are positive, to comply with the use
    of keras.layers.Cropping2D
    '''
    size_cropped = size

    # down sampling
    for l in range(n_layers):
        size_cropped -= (filter_size-1)*n_convs
        size_cropped //= pool_size

    # up sampling
    for l in range(n_layers):
        size_cropped *= pool_size
        size_cropped -= (filter_size-1)*n_convs

    start = (size-size_cropped)//2
    end = size-(start+size_cropped)

    return start, end





