import os
import sys
import copy
import functools
import tensorflow.keras as keras
import numpy as np

sys.path += [os.path.abspath('../')]
try:
    import KPnet_config
except:
    print()
    print('missing commonlib and KPnet_config modules')
    print('try adding the root directory to PYTHONPATH environment variable')

from tensorflowlib.kpnetkeras import KeyPointKerasNetwork
from tensorflowlib.heatmaplayer import GaussianHeatmap, HeatmapErr
import tensorflowlib.spatialsoftmax as spatialsoftmax

import tensorflowlib.cephdataseq as cdseq
import commonlib.cephdataloaders as dataload

from spatial_conf_net_exp import KeyPointSpatialConfNet, modify_data_samples, OutputIsLoss

class KeyPointSpatialConfNetV5(KeyPointSpatialConfNet):
    def __init__(self):
        super().__init__()

        self.name = 'spatial-config-net-v5'

        self.gamma_weight = 100
        self.l2_weight = 0.0005
        self.alpha = 20

        self.lr_base = 1e-4
        self.step_size = 50
        self.gamma = 0.5

        self.net_count = 5

        self.la_layers = 4
        self.la_filters = 32
        self.la_kernel_size = 3

        self.spc_layers = 3
        self.spc_subsample = 16
        self.spc_filters = 32
        self.spc_kernel_size = 11

        self.data_loader_conf = {
            #'rotate' : {'min':-15, 'max':15},
            #'scale' : {'min':0.6, 'max':1.4},
            'intensity' : {'min':0.75, 'max':1.25}, 
            'black_level' : {'min':-0.25, 'max':0.25},
            #'points_selection' : (61,),
            'output_fun': modify_data_samples,
            }

    def generate_params_dict(self):
        params_dict = super().generate_params_dict()
        params_dict['net_count'] = self.net_count
        return params_dict

    def add_cmd_params(self, arg_parser):
        super().add_cmd_params(arg_parser)
        arg_parser.add_argument(
            '--net_count', 
            default=self.net_count,
            type=int,
            help='number of subnets to produce',
            )

    def set_cmd_params(self, args_env):
        super().set_cmd_params(args_env)
        self.net_count = args_env.net_count

    def apply_params_dict(self, params_dict):
        super().apply_params_dict(params_dict)
        self.net_count = params_dict['net_count'] 

    def init_model(self):
        if 'resample' in self.data_loader_conf:
            H, W = self.data_loader_conf['resample']
        else:
            H, W = self.data_loader_conf['image_shape']

        if 'points_selection' in self.data_loader_conf:
            N = len(self.data_loader_conf['points_selection'])
        else:
            N = self.n_pts

        in_tensor = keras.Input((H, W, 1))
        in_pts = keras.Input((2, N))

        pts_feat_list = []
        la_submaps_list = []
        spc_submaps_list = []
        for n in range(self.net_count):
            la_submap = self.init_local_appearance_layers(in_tensor, self.la_filters)
            spc_submap = self.init_spatial_configuration_layers(la_submap, self.spc_filters, None)

            la_submaps_list.append(la_submap)
            spc_submaps_list.append(spc_submap)

        la_submap_tens = keras.layers.Concatenate()(la_submaps_list)
        spc_submap_tens = keras.layers.Concatenate()(spc_submaps_list)

        la_map = keras.layers.Conv2D(
            filters=N,
            kernel_size=(1, 1),
            padding='same',
            activation='linear',
            kernel_regularizer=keras.regularizers.l2(self.l2_weight),
            bias_initializer=keras.initializers.Constant(0),
            kernel_initializer='he_uniform',
            )(la_submap_tens)
        spc_map = keras.layers.Conv2D(
            filters=N,
            kernel_size=(1, 1),
            padding='same',
            activation='tanh',
            kernel_regularizer=keras.regularizers.l2(self.l2_weight),
            bias_initializer=keras.initializers.Constant(0),
            kernel_initializer='he_uniform',
            )(spc_submap_tens)

        output_map = keras.layers.Multiply()([la_map, spc_map])

        gauss_heatmap_ref = GaussianHeatmap(
            gamma=self.gamma_weight,
            sigma=1.0,
            alpha=self.alpha,
            height=H,
            width=W,
            )(in_pts)

        heatmap_err = HeatmapErr()([output_map, gauss_heatmap_ref])
        self.model = keras.Model(inputs=[in_tensor, in_pts], outputs=heatmap_err)

        self.model.compile(
            optimizer='adam',
            loss=OutputIsLoss,
            )


if __name__ == '__main__':
    KeyPointSpatialConfNetV5.main()
