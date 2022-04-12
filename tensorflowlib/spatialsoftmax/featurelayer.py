import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

class PartitionsToFeatures(keras.layers.Layer):
    '''
    Partition the BxHxWxF tensor alongside H and W dimensions using ovelaping windows.
    
    The partitions are first extracted for four offsets relative to top-left.
    The four offsets are (0, 0), (0, win_size//2), (win_size//2, 0) and (win_size//2, win_size//2)

    The extracted dat is first reshapen to:
    B x H' x S1 x W' x S2 x F
    where S1 and S2 are win size, and H' and W' are ~H//S1 and ~W//S2. The data is then transposed to:
    B x H' x W' x S1 x S2 x F
    and finaly reshapen to:
    B x M' x (S1*S2*F)

    This is done for each of the four offsets. Only the windows that are completely 
    withing the HxW of the original tensor are taken, no padding is done.

    The extracted features are concatenated, returning a tensor of shape:
    B x M x (S*S*F)
    
    M is the total number of partitions - extracted window samples
    S*S*F is the size of the linearized extracted window.

    win_size - int
        The size of the window. The overlap is half (rounded down) the window size.
    '''
    def __init__(self, win_size, sum_over_window=False, **kwargs):
        super().__init__(**kwargs)
        self.win_size = win_size
        self.sum_over_window = sum_over_window
        self.B = None
        self.C = None
        self.cells_q1 = None
        self.cells_q2 = None
        self.cells_q3 = None
        self.cells_q4 = None
        self.cell_win_q1 = None
        self.cell_win_q2 = None
        self.cell_win_q3 = None
        self.cell_win_q4 = None

    def get_config(self):
        conf_dict = super().get_config()
        conf_dict.update({
            'win_size': self.win_size,
            'sum_over_window': self.sum_over_window,
            })
        return conf_dict

    def build(self, input_shape):
        self.B, height, width, self.C = input_shape
        if self.B is None or self.B.value is None:
            self.B = -1
        win_size = self.win_size

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

    def compute_output_shape(self, input_shape):
        batch, height, width, feats_in = input_shape

        cell_cols = width//self.win_size
        cell_rows = height//self.win_size
        cell_cols_shift = (width-self.win_size//2)//self.win_size
        cell_rows_shift = (height-self.win_size//2)//self.win_size

        cells = cell_cols*cell_rows + cell_cols*cell_rows_shift + cell_cols_shift*cell_rows+ cell_cols_shift*cell_rows_shift

        if self.sum_over_window:
            feats_out = feats_in
        else:
            feats_out = feats_in*self.win_size**2

        return batch, cells, feats_out

    def call(self, inputs):
        partitions_list = []
        for cells, cell_win in (
                (self.cells_q1, self.cell_win_q1),
                (self.cells_q2, self.cell_win_q2),
                (self.cells_q3, self.cell_win_q3),
                (self.cells_q4, self.cell_win_q4),
                ):
            input_part = tf.reshape(
                inputs[:, cell_win[0]:cell_win[1], cell_win[2]:cell_win[3], :],
                (self.B, cells[0], self.win_size, cells[1], self.win_size, self.C),
                )
            M = cells[0]*cells[1]
            if self.sum_over_window:
                input_part_lin = tf.reduce_sum(input_part, (2, 4))
                input_part_lin = tf.reshape(input_part_lin, (self.B, M, self.C))
            else:
                input_part = tf.transpose(input_part, perm=(0, 1, 3, 2, 4, 5))
                input_part_lin = tf.reshape(input_part, (self.B, M, self.C*self.win_size*self.win_size))
            partitions_list.append(input_part_lin)
        return tf.concat(partitions_list, 1)


class PartitionedPointFeatures(keras.layers.Layer):
    '''

    '''
    def __init__(self, win_size, **kwargs):
        super().__init__(**kwargs)
        self.win_size = win_size
        self.chn_feats = None
        self.chn_masks = None
        self.even_cells = None
        self.odd_cells = None
        self.even_cell_win = None
        self.odd_cell_win = None

    def get_config(self):
        conf_dict = super().get_config()
        conf_dict.update({
            'win_size': self.win_size,
            })
        return conf_dict

    def build(self, input_shapes):
        mask_shape, features_shape = input_shapes
        _, _, _, chn_masks = mask_shape
        _, height, width, chn_feats = features_shape
        win_size = self.win_size

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

        self.chn_feats = chn_feats
        self.chn_masks = chn_masks

        super().build(input_shapes)


    def compute_output_shape(self, input_shapes):
        mask_shape, features_shape = input_shapes
        batch, height, width, c_mask = mask_shape
        _, _, _, c_feat = features_shape

        cell_cols = width//self.win_size
        cell_rows = height//self.win_size
        cell_cols_shift = (width-self.win_size//2)//self.win_size
        cell_rows_shift = (height-self.win_size//2)//self.win_size

        cells = cell_cols*cell_rows + cell_cols*cell_rows_shift + cell_cols_shift*cell_rows+ cell_cols_shift*cell_rows_shift

        return batch, c_feat, cells, c_mask

    def call(self, inputs):
        masks, features = inputs
        win_size = self.win_size
        chn_masks = self.chn_masks
        chn_feats = self.chn_feats

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
                )+1e-12

            features_cells = tf.reshape(
                features[:, cell_win[0]:cell_win[1], cell_win[2]:cell_win[3], :],
                (-1, cells[0], win_size, cells[1], win_size, chn_feats, 1),
                )

            masks_cells_sum = tf.reduce_sum(masks_cells, (2, 4))

            gm_cells = tf.reduce_sum(masks_cells*features_cells, (2, 4))
            gm_cells = gm_cells/masks_cells_sum
            gm_cells = tf.reshape(gm_cells, (-1, cells[0]*cells[1], chn_feats, chn_masks))
            gm_cells = tf.transpose(gm_cells, (0, 2, 1, 3))

            gm_cells_list.append(gm_cells)

        gm_cells_all = tf.concat(gm_cells_list, 2)

        return gm_cells_all
