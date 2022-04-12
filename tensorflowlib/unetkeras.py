'''
Based on Unet implementation https://github.com/jakeret/tf_unet/pulls
the implementation follows the original network described in:
O. Ronneberger etal, U-Net: Convolutional Networks for Biomedical Image Segmentation

'''
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

def unet_model_gen(
        chn_in,
        chn_out,
        n_layers=3,
        features_root=16,
        filter_size=3,
        pool_size=2,
        dropout_rate=0.25,
        dropout_layer=True,
        batchnorm_layer=False
    ):
    '''
    Create a unet keras Model, with a simple softmax function 
    on top.
    '''
    if isinstance(chn_in, int):
        x_in = keras.Input((None, None, chn_in))
    else:
        x_in = keras.Input(chn_in)

    x_out = unet_layers_gen(
        x_in,
        n_layers,
        features_root,
        filter_size,
        pool_size,
        dropout_rate,
        dropout_layer,
        batchnorm_layer,
        )

    x_out = layers.Conv2D(
        filters=chn_out,
        kernel_size=1,
        activation='softmax',
        use_bias=True,
        kernel_initializer=keras.initializers.he_normal(),
        )(x_out)

    model = keras.Model(inputs=x_in, outputs=x_out)
    return model

def unet_mlp_model_gen(
        chn_in,
        chn_hidden,
        chn_out,
        n_layers=3,
        features_root=16,
        filter_size=3,
        pool_size=2,
        dropout_rate=0.25,
        dropout_layer=True,
        batchnorm_layer=False
    ):
    if isinstance(chn_in, int):
        x_in = keras.Input((None, None, chn_in))
    else:
        x_in = keras.Input(chn_in)

    x_out = unet_layers_gen(
        x_in,
        n_layers,
        features_root,
        filter_size,
        pool_size,
        dropout_rate,
        dropout_layer,
        batchnorm_layer,
        )

    x_out = layers.Conv2D(
        filters=chn_hidden,
        kernel_size=1,
        activation='relu',
        use_bias=True,
        kernel_initializer=keras.initializers.he_normal(),
        )(x_out)
    x_out = layers.Conv2D(
        filters=chn_out,
        kernel_size=1,
        activation='softmax',
        use_bias=True,
        kernel_initializer=keras.initializers.he_normal(),
        )(x_out)

    model = keras.Model(inputs=x_in, outputs=x_out)
    return model

def unet_layers_gen(
        x_in,
        n_layers=3,
        features=16,
        filter_size=3,
        pool_size=2,
        dropout_rate=0.25,
        dropout_layer=True,
        batchnorm_layer=False
        ):
    '''
    Based on the input tensor create and connect additional
    unet layers. Returns the unets' output tensor, the
    same shape as the input tensor.
    '''
    x_out = x_in
    # down sampling
    x_down_list = []
    for l in range(n_layers-1):
        feat_count = 2**l*features

        for i in range(2):
            x_out = layers.Conv2D(
                filters=feat_count,
                kernel_size=filter_size,
                padding='same',
                activation='relu',
                use_bias=True,
                kernel_initializer=keras.initializers.he_normal(),
                )(x_out)
            if dropout_layer:
                x_out = layers.Dropout(
                    rate=dropout_rate,
                    )(x_out)
            if batchnorm_layer:
                x_out = layers.BatchNormalization()(x_out)


        x_down_list.append(x_out)

        x_out = layers.MaxPool2D(
            pool_size=pool_size,
            strides=pool_size,
            )(x_out)

    # bottle neck
    feat_count = 2**(n_layers-1)*features
    for i in range(2):
        x_out = layers.Conv2D(
            filters=feat_count,
            kernel_size=filter_size,
            padding='same',
            activation='relu',
            use_bias=True,
            kernel_initializer=keras.initializers.he_normal(),
            )(x_out)
        if dropout_layer:
            x_out = layers.Dropout(
                rate=dropout_rate,
                )(x_out)
        if batchnorm_layer:
            x_out = layers.BatchNormalization()(x_out)

    # up sampling
    for l in range(n_layers-2, -1, -1):
        feat_count = 2**l*features
        x_out = layers.Conv2DTranspose(
            filters=feat_count,
            kernel_size=pool_size,
            strides=pool_size,
            padding='valid',
            output_padding=0,
            activation='relu',
            use_bias=True,
            )(x_out)

        x_out_1 = x_out
        x_out_2 = x_down_list[l]
        x_out = layers.Concatenate(
            )([x_out_1, x_out_2])

        for i in range(2):
            x_out = layers.Conv2D(
                filters=feat_count,
                kernel_size=filter_size,
                padding='same',
                activation='relu',
                use_bias=True,
                kernel_initializer=keras.initializers.he_normal(),
                )(x_out)
            if batchnorm_layer:
                x_out = layers.BatchNormalization()(x_out)

    return x_out

def unet_cropped_loss_fun_gen(
        in_shape,
        loss_fun=keras.losses.categorical_crossentropy,
        n_layers=3,
        filter_size=3,
        pool_size=2,
        ):
    '''
    in_shape - 
        size/shape of the input image
    loss_fun -
        generic loss function for keras models

    n_layers - int
    filter_size - int
    pool_size - int
        unet parameters that influence cropping size

    returns:
        loss function where the ground truth is cropped 
        to the valid output region. The output of the network
        (prediction) is already expected to be cropped.
    '''

    #crop_size = 2*(filter_size-1)
    #for l in range(n_layers-1):
        #crop_size = crop_size*pool_size
        #crop_size += 4*(filter_size-1)

    #tl_crop = crop_size//2
    tl, br = calc_unet_cropping_size(in_shape, n_layers, filter_size, pool_size)

    def cropped_loss(x_gt, x_pred):
        cropp_fun = keras.layers.Cropping2D((tl, br))

        x_gt_crop = cropp_fun(x_gt)
        x_pred_crop = cropp_fun(x_pred)

        return loss_fun(x_gt_crop, x_pred_crop)
    return cropped_loss

def calc_unet_cropping_size(
        size,
        n_layers,
        filter_size,
        pool_size,
        ):
    size_cropped = size 
    # down
    for l in range(n_layers-1):
        size_cropped -= 2*(filter_size-1)
        size_cropped //= 2
    # middle
    size_cropped -= 2*(filter_size-1)
    # up
    for l in range(n_layers-1):
        size_cropped *= 2
        size_cropped -= 2*(filter_size-1)

    start = (size-size_cropped)//2
    end = size-(start+size_cropped)

    return start, end

def calc_unet_padding_size(
        size,
        n_layers,
        filter_size,
        pool_size,
        ):

    size_padded = size

    # up
    for l in range(n_layers-1):
        size_padded += 2*(filter_size-1)
        if not size_padded%pool_size:
            size_padded //= pool_size
            size_padded += 1
        else:
            size_padded //= pool_size
    # middle
    size_padded += 2*(filter_size-1)

    # down
    for l in range(n_layers-1):
        size_padded *= 2
        size_padded += 2*(filter_size-1)

    # adjust the size, so that the decimation leaves no remainder
    decimation_factor = pool_size**(n_layers-1)
    decimation_fraction = size_padded//decimation_factor
    decimation_remainder = size_padded%decimation_factor
    if decimation_remainder != 0:
        size_padded = (decimation_fraction+1)*decimation_factor

    start = (size_padded-size)//2
    end = size_padded-(start+size)
    return start, end

def init_conv2Dtranspose_unit(unit_val=1, rand_init=None):
    '''
    Initialize the Conv2Transpose kernels to unit kernels,
    that will upscale the inputs using repetition. Optionally
    adds random noise using the additional initializer.
    This initialization attempts to prevent channel cross talk
    when upscaling.

    if input were an image:

    [[1, 2],
     [3, 4]]

     Upscaling using a 2x2 filter and this initialization will produce:
     [[1, 1, 2, 2],
      [1, 1, 2, 2],
      [3, 3, 4, 4],
      [3, 3, 4, 4]]

    the kernel will be initialized to:
    [[1, 1],
     [1, 1]]

    unit_val - int, float
        The multiplication factor for input values. 

    rand_init - initializer function
        And additional initializer whos initialization values
        will be added to this one.
    '''
    zeros_init = keras.initializers.Zeros()
    unit_init = keras.initializers.Constant(unit_val)

    def init_fun(shape, dtype=None, partition_info=None):
        H, W, C_in, C_out = shape

        np_array = np.zeros(shape)

        for c in range(max(C_in, C_out)):
            np_array[:, :, c%C_in, c%C_out] = unit_val

        np_array /= np_array.sum(axis=2, keepdims=True)

        tensor = tf.convert_to_tensor(np_array, dtype=dtype)

        if rand_init is not None:
            tensor = tensor+rand_init(shape, dtype, partition_info)
        return tensor
    return init_fun

