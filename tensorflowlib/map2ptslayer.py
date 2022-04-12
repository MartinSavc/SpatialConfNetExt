import tensorflow as tf


class HeatmapToPointsLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        map_shape = inputs.get_shape()

        max_ind = tf.reshape(inputs, (tf.shape(inputs)[0], -1, map_shape[3]))
        max_ind = tf.argmax(max_ind, 1)
        max_ind = tf.reshape(max_ind, [-1])
        max_ind = tf.cast(max_ind, tf.int32)
        max_pts = tf.unravel_index(max_ind, map_shape[1:3])
        reshape = tf.transpose(max_pts, (1, 0))
        reshape = tf.reshape(reshape, (-1, map_shape[-1], 2))
        reshape = tf.transpose(reshape, (0, 2, 1))
        return tf.cast(reshape, tf.float32)
