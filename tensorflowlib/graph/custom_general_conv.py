import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
# import spektral


# # Custom layer that builds on the Spektral library GeneralConv layer.
# # Adds support for trainable weighted adjacency matrix, which the library lacks of.
# # Aggregation function is 'sum' and can't be changed by the supplied argument!
# class CustomGeneralConv(spektral.layers.GeneralConv):
# 
#     def __init__(self, channels=256, batch_norm=False, aggregate='sum', **kwargs):
#         if aggregate != 'sum':
#             raise NotImplementedError
# 
#         super().__init__(channels, batch_norm, aggregate=aggregate, **kwargs)
# 
# 
#     def propagate(self, x, a, e=None, **kwargs):
#         x = tf.transpose(x, (0, 2, 1))
#         out = tf.matmul(x, a)
#         out = tf.transpose(out, (0, 2, 1))
#         return out
# 
#     @staticmethod
#     def get_inputs(inputs):
#         if len(inputs) == 3:
#             x, a, e = inputs
#             assert K.ndim(e) == 2, 'E must have rank 2'
#         elif len(inputs) == 2:
#             x, a = inputs
#             e = None
#         else:
#             raise ValueError('Expected 2 or 3 inputs tensors (X, A, E), got {}.'
#                              .format(len(inputs)))
#         assert K.ndim(x) in (2, 3), 'X must have rank 2 or 3.'
#         assert K.ndim(a) in (2, 3), 'A must have rank 2 or 3.'
# 
#         return x, a, e
# 
#     # We don't use these methods. Make sure that they're not called from somewhere unknowingly.
#     def get_j(self, x):
#         raise NotImplementedError
# 
#     def get_i(self, x):
#         raise NotImplementedError

'''***'''
class CustomGeneralConv2(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        conf_dict = super().get_config()
        return conf_dict

    def build(self, input_shape):
        graph_feat_shape, adj_mat_shape = input_shape

        batch, node_count, feat_len = graph_feat_shape
        # node_count, node_count = adj_mat_shape

        self.W1 = self.add_weight(
                'W1',
                (feat_len, feat_len),
                dtype=tf.float32,
                initializer='glorot_uniform',
                regularizer=keras.regularizers.l2(1e-4),
                trainable=True,
                )
        self.W2 = self.add_weight(
                'W2',
                (feat_len, feat_len),
                dtype=tf.float32,
                initializer='glorot_uniform',
                regularizer=keras.regularizers.l2(1e-4),
                trainable=True,
                )

        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        # F.shape -> B, N, F
        # E.shape -> N, N
        # W.shape -> F, F
        F, E = inputs # graph features, adjacency_matrix
        F1 = tf.matmul(F, self.W1)
        F2 = tf.matmul(F, self.W2)

        F_new = tf.add(F1, tf.matmul(E, F2))
        return F_new
'''***'''

class CustomResnetGraphConv(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gconv_layer_1 = CustomGeneralConv2()
        self.gconv_layer_2 = CustomGeneralConv2()

    def get_config(self):
        conf_dict = super().get_config()
        return conf_dict

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        F, E = inputs
        R = self.gconv_layer_1([F, E])
        R = keras.activations.relu(R)
        R_new = self.gconv_layer_2([R, E])
        F_new = keras.activations.relu(R_new+F)
        return F_new
