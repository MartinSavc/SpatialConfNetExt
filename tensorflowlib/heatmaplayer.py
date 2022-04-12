import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K


class GaussianHeatmap(keras.layers.Layer):
    '''
    Layer constructs a gaussian heatmap with trainable sigma (standard deviation)
    for a given set of points. Each point gets its own sigma parameter.

    gamma - float
        scaling parameter for the resulting heatmap
    sigma - float
        initial sigma parameters, passed as parameter to keras.initializers.Constant
    alpha - float
        l2 regularization weight for the sigma parameters
    height - int
    width - int
        size (dimensions) of the resulting map

    Other keyword parameters are passed on to keras.layers.Layer.

    The layer receives a tensor of 2D points with shape:
    B x 2 x N
    and returns/outputs a tensor of heatmaps with shape:
    B x H x W x N
    
    where B is batch size, H and W are height and width, and N is the number of 
    points.

    Learned sigma parameters can be read from member tensor
    GaussianHeatmap.sigma_var with shape 1 x 1 x 1 x N
    '''
    def __init__(self, gamma, sigma, alpha, height, width, trainable=True, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.sigma = sigma
        self.alpha = alpha
        self.spatial_shape = height, width
        self.trainable = trainable
        self.n_pts = None
        self.sigma_var = None
        self.y_coord = None
        self.x_coord = None


    def get_config(self):
        conf_dict = super().get_config()
        conf_dict.update({
            'gamma' : self.gamma,
            'sigma' : self.sigma,
            'alpha' : self.alpha,
            'height' : self.spatial_shape[0],
            'width' : self.spatial_shape[1],
            })
        return conf_dict

    def build(self, input_shape):
        B, D, N = input_shape
        H, W = self.spatial_shape
        self.sigma_var = self.add_weight(
                'sigma',
                (1, 1, 1, N), 
                dtype='float32',
                initializer=keras.initializers.Constant(self.sigma),
                regularizer=keras.regularizers.l2(self.alpha),
                trainable=self.trainable)
        self.y_coord = K.reshape(K.constant(np.arange(H)), (1, H, 1, 1))
        self.x_coord = K.reshape(K.constant(np.arange(W)), (1, 1, W, 1))
        self.n_pts = N

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        B, D, N = input_shape
        H, W = self.spatial_shape
        return B, H, W, N

    def call(self, inputs):
        in_y_coord = K.reshape(inputs[:, 0, :], (-1, 1, 1, self.n_pts))
        in_x_coord = K.reshape(inputs[:, 1, :], (-1, 1, 1, self.n_pts))

        y_coord_zm = self.y_coord-in_y_coord
        x_coord_zm = self.x_coord-in_x_coord

        gauss_y_map = K.exp(-0.5*(y_coord_zm/(self.sigma_var))**2)
        gauss_x_map = K.exp(-0.5*(x_coord_zm/(self.sigma_var))**2)
        gauss_heatmap = gauss_y_map*gauss_x_map
        gauss_heatmap = gauss_heatmap*(self.gamma/((2*np.pi)*self.sigma_var**2))

        return gauss_heatmap

class HeatmapErr(keras.layers.Layer):
    '''
    Layers calculates mean squared error between two heatmaps - 4D tensors
    of compatible shapes.

    The layer receives two tensors with shapes:
    B x H x W x N

    and returns a tensor with shape:
    B x 1
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_build = None


    def build(self, input_shape):
        self.input_shape_build = input_shape
        print(f'input shape: {input_shape}')
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1


    def call(self, inputs):
        gauss_hm_est, gauss_hm_ref = inputs

        err = gauss_hm_est - gauss_hm_ref
        sq_err = keras.backend.square(err)

        _, height, width, n_pts = self.input_shape_build[0]

        sq_err = keras.backend.reshape(sq_err, (-1, height*width*n_pts))
        sum_sq_err = keras.backend.sum(sq_err, 1, keepdims=True)

        return sum_sq_err

