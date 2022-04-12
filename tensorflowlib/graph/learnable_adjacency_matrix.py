import tensorflow.keras as keras
import tensorflow as tf


# Trainable weighted adjacency matrix with optionally added self-loops.
# Weights are initialized to a value (1.0 / node_num).
# If add_self_loops parameter is true, diagonal elements are forced to always be 1.0.
class LearnableAdjacencyMatrix(keras.layers.Layer):

    def __init__(self, node_num, add_self_loops=False, **kwargs):
        super().__init__(**kwargs)
        self.node_num = node_num
        self.add_self_loops = add_self_loops
        self.adjacency_matrix = None
        self.mask = None
        self.identity = None

    def get_config(self):
        conf_dict = super().get_config()
        conf_dict.update({
            'node_num': self.node_num,
        })
        return conf_dict

    def build(self, input_shape):
        self.adjacency_matrix = self.add_weight(
            'adjacency_matrix',
            (self.node_num, self.node_num),
            dtype=tf.float32,
            initializer=keras.initializers.Constant(1 / self.node_num),
            trainable=True)

        self.identity = tf.eye(self.node_num)

        self.mask = tf.eye(self.node_num, dtype=tf.bool)
        self.mask = tf.logical_not(self.mask)
        self.mask = tf.cast(self.mask, tf.float32)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        res = self.adjacency_matrix
        if self.add_self_loops:
            res = self.identity + res * self.mask
        return res
