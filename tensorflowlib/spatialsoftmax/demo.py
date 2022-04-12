'''
Demonstrate the usage of SpatialSoftmax and GeometricMean layers.
'''

import matplotlib.pyplot as pyplot
import numpy as np
from . import GeometricMean, SpatialSoftmax

import tensorflow.keras as keras

def generate_sample(win_size, marker_size, noise_level=1.00):
    H, W = win_size
    image = np.ones(win_size)/2.

    y = np.random.randint(0, H)
    x = np.random.randint(0, W)

    size = marker_size//2
    image[y, x-size:x+size+1] = 0.0
    image[y-size:y+size+1, x] = 0.0

    image += np.random.randn(*image.shape)*noise_level

    return image, np.array((y, x))

def sample_generator(win_size, marker_size, noise_level):
    while True:
        img, pt = generate_sample(win_size, marker_size, noise_level)
        img = img.reshape(1, win_size[0], win_size[1], 1)
        pt = pt.reshape(1, 2, 1)
        yield img, pt

if __name__ == '__main__':

    win_size = 128, 128
    marker_size = 12
    noise_level = 0.2

    n_samples = 2000
    data = np.zeros((n_samples, win_size[0], win_size[1], 1))
    pts = np.zeros((n_samples, 2, 1))
    for n in range(n_samples):
        d, p = generate_sample(win_size, marker_size, noise_level)
        data[n, ..., 0] = d
        pts[n, ..., 0] = p


    x_input_tens = keras.Input((win_size+(1,)))
    conv_1_tens = keras.layers.Conv2D(8, 5, padding='same', activation='relu')(x_input_tens)
    conv_2_tens = keras.layers.Conv2D(4, 5, padding='same', activation='relu')(conv_1_tens)
    conv_3_tens = keras.layers.Conv2D(1, 1, padding='same')(conv_2_tens)
    softmax_tens = SpatialSoftmax()(conv_3_tens)
    geom_tens = GeometricMean(normalize=False)(softmax_tens)
    model = keras.Model(inputs=[x_input_tens], outputs=[geom_tens])
    model.compile(optimizer='adam', loss = 'mean_squared_error')
    model.fit(
            data,
            pts,
            epochs=10,
            validation_split=0.25,
            use_multiprocessing=False,
            verbose=2)

    model_test = keras.Model(
            inputs=[x_input_tens],
            outputs=[conv_3_tens, softmax_tens, geom_tens],
            )

    d, pt_gt = generate_sample(win_size, marker_size, noise_level)

    logit_output, sm_output, pt_pred = model_test.predict(d.reshape(1, win_size[0], win_size[1], 1))

    fig, ax = pyplot.subplots(1, 3, True, True)

    ax[0].imshow(d)
    ax[1].imshow(logit_output[0,...,0])
    ax[2].imshow(sm_output[0,...,0])

    for n in range(3):
        ax[n].plot(pt_gt[1], pt_gt[0], 'og')
        ax[n].plot(pt_pred[0, 1, 0], pt_pred[0, 0, 0], 'xr')

    pyplot.show()

