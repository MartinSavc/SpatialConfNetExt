import tensorflow.keras as keras
from .cephdataseq import CephDataSequence

def convnet_build(
        input_shape,
        layer_reps_list,
        filters_list,
        kernel_sizes_list,
        output_chns=1,
        ):
    '''
    Creates a Convolutional NN, with multiple layers.
    Layer configuration can be specified using number of repetitions, filter sizes and kernel sizes.
    The ouput is kept the same size as input, the number of
    output channels can be specified.


    input_shape - tuple of ints
        shape of input, unknown dimensions (such as image width and height) can be None

    layer_reps_list - list or tuple of integers
        number of repetitions of each layer in groups
    filters_list - list or tuple of integers
        number of filters in each layer group
        each item is passed onto keras.layers.Conv2D a filters
    kernel_sizes_list - list or tuple of integers
        kernel sizes for each layer in group
        each item is passed onto keras.layers.Conv2D a kernel_size

    output_chns - integer
        number of channels of the output image

    '''

    input_tens = keras.Input(shape=input_shape, dtype='float32')

    output_tens = input_tens
    for reps, filt, ksize in zip(layer_reps_list,
                                 filters_list,
                                 kernel_sizes_list):
        for _ in range(reps):
            conv_layer = keras.layers.Conv2D(
                filters=filt,
                kernel_size=ksize,
                padding='same',
                activation='relu',
                )
            output_tens = conv_layer(output_tens)

            bn_layer = keras.layers.BatchNormalization()
            output_tens = bn_layer(output_tens)

    output_layer = keras.layers.Conv2D(
        filters=output_chns,
        kernel_size=(1, 1),
        activation='sigmoid',
        )
    output_tens = output_layer(output_tens)

    model = keras.Model(inputs=input_tens, outputs=output_tens)

    return model

if __name__ == '__main__':
    import os
    import time
    import sys

    #sys.path  += [os.path.abspath('../')]

    #try:
    #    import KPnet_config
    #except:
    #    print()
    #    print('missing commonlib and KPnet_config modules')
    #    print('try adding the root directory to PYTHONPATH environment variable')

    #sample_size = (128, 128)

    train_data_config = {
            'image_list_file': '../1_data/train_data.list',
            #'image_shape': sample_size,
            'resample': (128, 128),
            'batch_size' : 16,
            'map_sigma' : 2.,
            'intensity' : {'min':0.8, 'max':1.2}, 
            'black_level' : {'min':0.0, 'max':0.2}, 
            'gamma' : {'min':0.8, 'max':1.2}, 
            'rotate' : {'min' : -10, 'max':10}, 
            'scale' : {'min' : 0.9, 'max':1.1}, 
            }
    test_data_config = {
            'image_list_file': '../1_data/test_data.list',
            #'image_shape': sample_size,
            'resample': (128, 128),
            'map_sigma': 2.,
            }
    train_data_seq = CephDataSequence(train_data_config)
    test_data_seq = CephDataSequence(test_data_config)


    # prepare folders for this network
    time_stamp = time.strftime('%Y_%m_%d-%H_%M_%S')

    target_dir = './convnet_train/'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    target_dir = os.path.join(target_dir, time_stamp)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    else:
        raise Exception(f'folder {target_dir} already exists')
    


    # prepare model
    convnet_model = convnet_build(
        input_shape=(None, None, 1),
        output_chns=72,
        layer_reps_list =   ( 2,  2, 1),
        filters_list =      (12, 24, 128),
        kernel_sizes_list = ( 9,  5, 3),
        )
    convnet_model.compile(
        optimizer='rmsprop',#'rmsprop',
        loss='mse', # Mean Square Error
        #loss=keras.losses.categorical_crossentropy,
        metrics=['mae']
        )

    # tensorboard callback init
    tboard_cb = keras.callbacks.TensorBoard(
        log_dir=os.path.join(target_dir, './Graph'),
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        )

    try:
        # call fitting function
        convnet_model.fit(
            train_data_seq,
            epochs=100,
            verbose=1,
            callbacks=[tboard_cb],
            validation_data=test_data_seq,
            use_multiprocessing=False,
            )
    finally:
        convnet_model.save(os.path.join(target_dir, 'model.h5'))

