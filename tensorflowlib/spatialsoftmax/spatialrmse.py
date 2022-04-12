import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K

class SpatialMse():
    '''
    Calculate mean square error between two sets of points.
    Used as a loss.
    '''
    def __init__(self, linearize=False):
        self.__name__ = 'spatial_mse'
        self.linearize = linearize

    def __call__(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        y_mse = K.sum(math_ops.squared_difference(y_pred, y_true), axis=-2)

        if self.linearize:
            return tf.reshape(y_mse, (-1, 1))
        else:
            return y_mse


class SpatialRmse():
    '''
    Calculate root mean square error between two sets of points.
    Used as a metric.
    '''
    def __init__(self, linearize=False):
        self.linearize = linearize
        self.__name__ = 'spatial_rmse'

    def __call__(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        y_rmse = math_ops.sqrt(K.sum(math_ops.squared_difference(y_pred, y_true), axis=-2)) 

        if self.linearize:
            return tf.reshape(y_rmse, (-1, 1))
        else:
            return y_rmse

class SpatialAdaptiveMse():
    '''
    Calculate an adaptive mean square error between two sets of points.
    Used as a loss.

    When the target vector is shorter than a given scale, the 
    error acts as the mean-squared error. When the target vector is longer
    the error is atenuated. This means that when the target is further away
    than the scale it's prediction becomes less important.

    parameters:
    scale - float,
    The distance threshold at which prediction is still important. After
    this, the error is atenuated rapidly.

    pow_fact - int,
    Scale of atenuation. An even power factor, larger then 2.

    '''
    def __init__(self, scale=10, pow_fact=10):
        self.__name__ = 'spatial_adaptive_mse'
        self.norm_pow = pow_fact
        self.norm_scale = scale
        self.norm_offset = 1

    def __call__(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        adapt_norm_fact = math_ops.scalar_mul(1/self.norm_scale, y_true)
        adapt_norm_fact = math_ops.pow(adapt_norm_fact, self.norm_pow)
        adapt_norm_fact = K.mean(adapt_norm_fact, axis=-2)
        adapt_norm_fact = math_ops.add(adapt_norm_fact, self.norm_offset)

        sq_err = math_ops.squared_difference(y_pred, y_true)
        y_mse = K.mean(sq_err, axis=-2)
        norm_y_mse = math_ops.divide(y_mse, adapt_norm_fact)

        return norm_y_mse
