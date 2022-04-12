import unittest
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras

from tensorflowlib.spatialsoftmax import PartitionedGeometricMean

def get_data():
    data = np.zeros((2, 4, 6, 3))
    data[0, :, :, 0] = [
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        ]
    data[0, :, :, 1] = [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        ]
    data[0, :, :, 2] = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        ]
    data[1, :, :, 0] = [
        [0, 1, 0, 0, 0, 0],
        [1, 2, 1, 1, 1, 1],
        [0, 1, 0, 1, 2, 1],
        [0, 1, 1, 0, 1, 0],
        ]
    data[1, :, :, 1] = [
        [1, 0, 1, 0, 1, 0],
        [1, 2, 1, 2, 1, 2],
        [2, 1, 2, 1, 2, 1],
        [0, 1, 0, 1, 0, 1],
        ]
    data[1, :, :, 2] = [
        [0, 1, 0, 0, 0, 0],
        [0, 3, 0, 3, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 3, 0, 0],
        ]

    # the geom. mean predictions are interlaced. They are predicted from four subsets of non-overlaping window. The first subset contains the windows that start with 0,0 offset from the top left corner (index 0,0). The second subset contains windows that start with 0, win_size/2 offset, the third with win_size/2, 0, and the fourth with offset win_size/2, win_size/2. The predictions are ordered by subsets (first, second, ....) and order from left to right, and top to bottom within subsets. This is in contrast to ordering all of them as they appear withing the image, fom left to right and top to bottom.
    part_gm_ref = np.zeros((2, 2, 15, 3))
    part_gm_ref[0, :, :, 0] = [
       [1,    0,    1,    3,    2,    3,    1,    0,    3,     2.5,  1,    2,     1,   1,    2    ],
       [1,    3,    5,    1,    3,    4,    1,    3,    1,     3.5,  1,    3,     5,   1,    3    ],
        ]
    part_gm_ref[0, :, :, 1] = [
       [0.5,  0.5,  0.5,  2.5,  2.5,  2.5,  0.5,  0.5,  2.5,   2.5,  1.5,  1.5,   1.5, 1.5,  1.5  ],
       [0.5,  2.5,  4.5,  0.5,  2.5,  4.5,  1.5,  3.5,  1.5,   3.5,  0.5,  2.5,   4.5, 1.5,  3.5  ],
        ]
    part_gm_ref[0, :, :, 2] = [
       [0.5,  0.5,  0.5,  2.5,  2.5,  2.5,  0.5,  0.5,  2.5,   2.5,  1.5,  1.5,   1.5, 1.5,  1.5  ],
       [0.5,  2.5,  4.5,  0.5,  2.5,  4.5,  1.5,  3.5,  1.5,   3.5,  0.5,  2.5,   4.5, 1.5,  3.5  ],
        ]
    part_gm_ref[1, :, :, 0] = [
       [0.75, 1.0,  1.0,  2.5,  2.5,  2.25, 0.75, 1.0,  2.6666667, 2.25, 1.25, 1.3333334, 1.6, 1.25, 1.6  ],
       [0.75, 2.5,  4.5,  1.0,  2.5,  4.25, 1.25, 3.5,  1.3333334, 3.75, 0.75, 2.6666667, 4.4, 1.25, 3.6  ],
        ]
    part_gm_ref[1, :, :, 1] = [
       [0.75, 0.75, 0.75, 2.25, 2.25, 2.25, 0.75, 0.75, 2.25,  2.25, 1.5,  1.5,   1.5, 1.5,  1.5  ],
       [0.50, 2.50, 4.50, 0.50, 2.50, 4.50, 1.50, 3.50, 1.50,  3.50, 0.5,  2.5,   4.5, 1.5,  3.5  ],
        ]
    part_gm_ref[1, :, :, 2] = [
       [0.75, 1.,   0.5,  2.5,  2.75, 2.,   0.75, 1.,   2.,    2.75, 1.,   1.25,  2.,  1.25, 1.25 ],
       [1.00, 3.,   4.5,  0.5,  2.75, 4.,   1.00, 3.,   2.,    3.25, 1.,   2.75,  4.,  1.25, 3.25 ],
        ]
    return data, part_gm_ref

class PartitionedGeometricMeanTest(unittest.TestCase):
    def test_1_general_data(self):
        data, part_gm_ref = get_data()

        x_input = keras.Input(data.shape[1:])
        part_gm_tensor = PartitionedGeometricMean(win_size=2)(x_input)
        test_model = keras.Model(inputs=[x_input], outputs=[part_gm_tensor])
        part_gm = test_model.predict(data)

        B = part_gm.shape[0]
        C = part_gm.shape[3]
        for b in range(B):
            for c in range(C):
                np.testing.assert_almost_equal(
                        part_gm[b, ..., c],
                        part_gm_ref[b, ..., c], 
                        err_msg=f'calculated means do not equal reference means, '+\
                                f'batch {b}, channel {c}')

    def test_3_compute_output_shape(self):
        part_gm_layer = PartitionedGeometricMean(win_size=10)
        output_shape = part_gm_layer.compute_output_shape((12, 20, 30, 4))
        self.assertEqual(output_shape, (12, 2, 15, 4))

        part_gm_layer = PartitionedGeometricMean(win_size=8)
        output_shape = part_gm_layer.compute_output_shape((12, 20, 30, 4))
        self.assertEqual(output_shape, (12, 2, 24, 4))

