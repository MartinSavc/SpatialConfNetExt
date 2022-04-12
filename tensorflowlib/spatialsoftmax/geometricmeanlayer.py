import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class GeometricMean(keras.layers.Layer):
    '''
    The layers transforms input heatmap into 2D points using geometric mean.

    normalize - bool
        If True, heatmaps are normalized over second and third dimension
        before calculating geometric means. This should only be set to false,
        if the input heatmaps are already normalized (when used with SpatialSoftmax).

    Receives a tensor of shape:
    BxHxWxN

    returns a tensor of shape:
    Bx2xN

    H, W and N must be known ahead of time (cannot work with variable width
    and height inputs).
    '''
    def __init__(self, normalize=True, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize
        self.pos_ind = None
        self.input_shape_build = None

    def get_config(self):
        conf_dict = super().get_config()
        conf_dict.update({
            'normalize':  self.normalize,
            })
        return conf_dict

    def build(self, input_shape):
        self.input_shape_build = input_shape
        _, height, width, _ = input_shape

        lin_shape = height*width

        y_ind, x_ind = tf.meshgrid(np.arange(int(height)), np.arange(int(width)), indexing='ij')
        y_ind_lin = tf.reshape(y_ind, (1, lin_shape, 1))
        x_ind_lin = tf.reshape(x_ind, (1, lin_shape, 1))
        pos_ind_lin = tf.stack((y_ind_lin, x_ind_lin), 2)

        pos_ind = tf.reshape(pos_ind_lin, (1, height, width, 2, 1))
        self.pos_ind = tf.cast(pos_ind, tf.float32)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        batch, _, _, chns = input_shape
        return batch, 2, chns

    def call(self, inputs):
        _, height, width, chns = self.input_shape_build
        inputs = tf.reshape(inputs, (-1, height, width, 1, chns))
        if self.normalize:
            inputs_sum = tf.reshape(tf.reduce_sum(inputs, (1, 2)), (-1, 1, 1, 1, chns))
            inputs_norm = inputs/inputs_sum
        else:
            inputs_norm = inputs

        geom_mean = tf.reduce_sum(inputs_norm*self.pos_ind, (1, 2))
        return geom_mean

class PartitionedGeometricMean(keras.layers.Layer):
    '''
    The layer transforms input heatmap into 2D points using geometric means.
    The heatmap is partitioned using a window, one point is predicted per each
    window position over the heatmap.

    Receives a tensor of shape:
    BxHxWxN

    Generates M partitions:
    M_1 = H//win_size * W//win_size
    M_2 = (H-win_size//2)//win_size * (W-win_size//2)//win_size
    M = M_1 + M_2

    Where M_1 are partitions using non overlaping windows starting at position
    (0, 0), while M_2 are partitions using non overlaping windows shifted half
    of a widow down and left. Value win_size is the window size.

    and returns a tensor of shape:
    Bx2xMxN

    The dimensions of the input tensor must be known ahead of time. It is
    prefered that the size of the input tensor is an integer multiple
    of window size.

    For example given an input of (20x30), 10 is a good window
    size, both 20 and 30 are divisible by 10. 15 is not, since 20 is not
    divisible by 15.

    In this example, a window size of 10 generates 2*3+1*2 = 8 partitions.
    '''

    def __init__(self, win_size, use_softmax=False, **kwargs):
        super().__init__(**kwargs)
        self.win_size = int(win_size)
        self.use_softmax = use_softmax
        self.chn_masks = None
        self.even_cells = None
        self.odd_cells = None
        self.even_cell_win = None
        self.odd_cell_win = None
        self.pos_ind = None
        self.cell_offsets = None

    def get_config(self):
        conf_dict = super().get_config()
        conf_dict.update({
            'win_size': self.win_size,
            'use_softmax': self.use_softmax,
            })
        return conf_dict

    def build(self, input_shapes):
        _, height, width, chn_masks = input_shapes
        height = int(height)
        width = int(width)
        chn_masks = int(chn_masks)

        self.chn_masks = chn_masks
        win_size = self.win_size

        lin_shape = win_size**2

        y_ind, x_ind = tf.meshgrid(np.arange(int(win_size)), np.arange(int(win_size)), indexing='ij')
        y_ind_lin = tf.reshape(y_ind, (1, lin_shape, 1))
        x_ind_lin = tf.reshape(x_ind, (1, lin_shape, 1))
        pos_ind_lin = tf.stack((y_ind_lin, x_ind_lin), 2)

        pos_ind = tf.reshape(pos_ind_lin, (1, 1, win_size, 1, win_size, 2, 1))
        pos_ind = tf.transpose(pos_ind, (0, 1, 3, 2, 4, 5, 6))
        pos_ind = tf.cast(pos_ind, tf.float32)
        self.pos_ind = pos_ind

        cells_q1 = height//win_size, width//win_size
        cells_q2 = height//win_size, (width-win_size//2)//win_size
        cells_q3 = (height-win_size//2)//win_size, width//win_size
        cells_q4 = (height-win_size//2)//win_size, (width-win_size//2)//win_size

        cell_win_q1 = 0, cells_q1[0]*win_size, 0, cells_q1[1]*win_size
        cell_win_q2 =  (0, cells_q2[0]*win_size,
                        win_size//2, win_size//2 + cells_q2[1]*win_size)
        cell_win_q3 =  (win_size//2, win_size//2 + cells_q3[0]*win_size,
                        0, cells_q3[1]*win_size)
        cell_win_q4 =  (win_size//2, win_size//2 + cells_q4[0]*win_size,
                        win_size//2, win_size//2 + cells_q4[1]*win_size)

        self.cells_q1 = cells_q1
        self.cells_q2 = cells_q2
        self.cells_q3 = cells_q3
        self.cells_q4 = cells_q4
        self.cell_win_q1 = cell_win_q1
        self.cell_win_q2 = cell_win_q2
        self.cell_win_q3 = cell_win_q3
        self.cell_win_q4 = cell_win_q4


        cell_offsets_list = []
        for cells, cell_win in (
                (cells_q1, cell_win_q1),
                (cells_q2, cell_win_q2),
                (cells_q3, cell_win_q3),
                (cells_q4, cell_win_q4),
                ):

            y_offsets = np.arange(cell_win[0], cell_win[1], win_size)
            x_offsets = np.arange(cell_win[2], cell_win[3], win_size)

            Y_offsets, X_offsets = tf.meshgrid(y_offsets, x_offsets, indexing='ij')
            Y_lin_offsets = tf.reshape(Y_offsets, (1, cells[0]*cells[1], 1))
            X_lin_offsets = tf.reshape(X_offsets, (1, cells[0]*cells[1], 1))
            cell_offsets = tf.stack((Y_lin_offsets, X_lin_offsets), 1)
            cell_offsets_list.append(cell_offsets)

        cell_offsets = tf.concat(cell_offsets_list, 2)
        self.cell_offsets = tf.cast(cell_offsets, tf.float32)

        super().build(input_shapes)

    def compute_output_shape(self, input_shapes):
        batch, height, width, c_mask = input_shapes
        c_feat = 2

        cell_cols = width//self.win_size
        cell_rows = height//self.win_size
        cell_cols_shift = (width-self.win_size//2)//self.win_size
        cell_rows_shift = (height-self.win_size//2)//self.win_size

        cells = cell_cols*cell_rows + cell_cols*cell_rows_shift + cell_cols_shift*cell_rows+ cell_cols_shift*cell_rows_shift

        return batch, c_feat, cells, c_mask

    def call(self, inputs):

        masks = inputs
        pos_ind = self.pos_ind

        win_size = self.win_size
        chn_masks = self.chn_masks

        gm_cells_list = []
        for cells, cell_win in (
                (self.cells_q1, self.cell_win_q1),
                (self.cells_q2, self.cell_win_q2),
                (self.cells_q3, self.cell_win_q3),
                (self.cells_q4, self.cell_win_q4),
                ):
            masks_cells = tf.reshape(
                masks[:, cell_win[0]:cell_win[1], cell_win[2]:cell_win[3], :],
                (-1, cells[0], win_size, cells[1], win_size, 1, chn_masks),
                )

            masks_cells_tr = tf.transpose(masks_cells, (0, 1, 3, 2, 4, 5, 6))
            masks_cells_lin_tr = tf.reshape(masks_cells_tr, (-1, win_size**2, chn_masks))+1e-12
            if self.use_softmax:
                masks_cells_lin_tr = tf.nn.softmax(masks_cells_lin_tr, 1)
            else:
                masks_cells_sum = tf.reduce_sum(masks_cells_lin_tr, 1, keepdims=True)
                masks_cells_lin_tr = masks_cells_lin_tr/masks_cells_sum

            masks_cells_tr = tf.reshape(masks_cells_lin_tr, (-1, cells[0], cells[1], win_size, win_size, 1, chn_masks))
            gm_cells = tf.reduce_sum(masks_cells_tr*pos_ind, (3, 4))
            gm_cells = tf.reshape(gm_cells, (-1, cells[0]*cells[1], 2, chn_masks))
            gm_cells = tf.transpose(gm_cells, (0, 2, 1, 3))

            gm_cells_list.append(gm_cells)

        gm_cells_all = tf.concat(gm_cells_list, 2)+self.cell_offsets

        return gm_cells_all
