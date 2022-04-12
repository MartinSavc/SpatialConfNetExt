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

class GlobalMulMapLayer(keras.layers.Layer):
    '''
    '''
    def get_config(self):
        conf = super().get_config()
        return conf

    def build(self, input_shape):
        _, H, W, C = input_shape

        self.map = self.add_weight(
                name='map',
                shape=(1, H, W, C),
                initializer=keras.initializers.Ones(),
                constraint=keras.constraints.MinMaxNorm(1e-5, 1.0, axis=[0, 1]),
                trainable=True)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        return inputs*self.map

class KeyPointSpatialConfNetV51(KeyPointSpatialConfNet):
    def __init__(self):
        super().__init__()

        self.name = 'spatial-config-net-v51'

        self.gamma_weight = 100
        self.l2_weight = 0.0005
        self.alpha = 20

        self.lr_base = 1e-4
        self.step_size = 50
        self.gamma = 0.5

        self.net_count = 5
        self.global_prior = False

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

    def get_custom_layers_dict(self):
        custom_layers_dict = super().get_custom_layers_dict()
        custom_layers_dict['GlobalMulMapLayer'] = GlobalMulMapLayer
        return custom_layers_dict

    def generate_params_dict(self):
        params_dict = super().generate_params_dict()
        params_dict['net_count'] = self.net_count
        params_dict['global_prior'] = self.global_prior
        return params_dict

    def add_cmd_params(self, arg_parser):
        super().add_cmd_params(arg_parser)
        arg_parser.add_argument(
            '--net_count', 
            default=self.net_count,
            type=int,
            help='number of subnets to produce',
            )
        arg_parser.add_argument(
            '--global_prior',
            action='store_true',
            default=self.global_prior,
            help='add a global multiplier prior layer for spatial configuration',
            )

    def set_cmd_params(self, args_env):
        super().set_cmd_params(args_env)
        self.net_count = args_env.net_count
        self.global_prior = args_env.global_prior

    def apply_params_dict(self, params_dict):
        super().apply_params_dict(params_dict)
        self.net_count = params_dict['net_count'] 
        self.global_prior = params_dict['global_prior'] 

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
            spc_submap = self.init_spatial_configuration_layers(la_submap, self.spc_filters, None, False)

            la_submaps_list.append(la_submap)
            spc_submaps_list.append(spc_submap)

        if self.net_count > 1:
            la_submap_tens = keras.layers.Concatenate()(la_submaps_list)
            spc_submap_tens = keras.layers.Concatenate()(spc_submaps_list)
        else:
            la_submap_tens = la_submaps_list[0]
            spc_submap_tens = spc_submaps_list[0]

        la_map = keras.layers.Conv2D(
            filters=N,
            kernel_size=(1, 1),
            padding='same',
            activation='linear',
            kernel_regularizer=keras.regularizers.l2(self.l2_weight),
            bias_initializer=keras.initializers.Constant(0),
            kernel_initializer='he_uniform',
            name='local_map',
            )(la_submap_tens)
        spc_map = keras.layers.Conv2D(
            filters=N,
            kernel_size=(1, 1),
            padding='same',
            activation='tanh',
            kernel_regularizer=keras.regularizers.l2(self.l2_weight),
            bias_initializer=keras.initializers.Constant(0),
            kernel_initializer='he_uniform',
            name='spatial_configuration_map',
            )(spc_submap_tens)

        if self.global_prior:
            spc_map = GlobalMulMapLayer(name='apriori_map')(spc_map)

        spc_map = keras.layers.UpSampling2D(size=self.spc_subsample, interpolation='bilinear')(spc_map)

        output_map = keras.layers.Multiply(name='pts_map')([la_map, spc_map])

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

    def preview(self, data):
        import matplotlib.pyplot as pyplot

        local_map_layer = self.model.get_layer('local_map')
        sp_conf_map_layer = self.model.get_layer('spatial_configuration_map')
        if self.global_prior:
            apriori_map_layer = self.model.get_layer('apriori_map')
        pts_map_layer = self.model.get_layer('pts_map')
        preview_model = keras.Model(
            inputs=self.model.inputs[0],
            outputs=[local_map_layer.output, sp_conf_map_layer.output, pts_map_layer.output],
            )

        global_map = apriori_map_layer.get_weights()[0]

        image, pts_target_list = data
        data_loader = dataload.CephDataWrapper((image,), pts_target_list)
        data_config = self.data_loader_conf.copy()
        data_config.pop('rotate', None)
        data_config.pop('scale', None)
        data_config.pop('translate', None)
        data_config.pop('intensity', None)
        data_config.pop('batch_size', None)
        data_config.pop('black_level', None)
        data_config.pop('gamma', None)
        data_config['image_loader'] = data_loader
        predict_data_seq = cdseq.CephDataSequence(data_config)

        for (image_patch, kp_ref), _target_loss in predict_data_seq:

            la_map, spc_map, output_map = preview_model.predict(image_patch)

            max_ind = output_map.reshape(-1, output_map.shape[3]).argmax(0)
            max_pts = np.unravel_index(max_ind, output_map.shape[1:3])

            fig, ax_table = pyplot.subplots(2, 3, True, True)
            ax_table = ax_table.ravel()
            ax_table[0].set_title('image')
            ax_table[0].imshow(image_patch[0, ..., 0])
            ax_table[0].plot(max_pts[1], max_pts[0], 'xr', label='predicted')
            ax_table[0].plot(
                kp_ref[0, 1, :],
                kp_ref[0, 0, :],
                'ob',
                label='reference',
                picker=5,
                )

            ax_table[0].plot(
                (kp_ref[0, 1, :], max_pts[1]),
                (kp_ref[0, 0, :], max_pts[0]),
                'g')
            ax_table[0].legend(loc='best')

            ax_table[1].set_title('la map')
            ax_table[1].imshow(la_map[0].max(2))
            ax_table[2].set_title('spc map')
            ax_table[2].imshow(spc_map[0].max(2))
            ax_table[3].set_title('global map')
            ax_table[3].imshow(global_map[0].max(2))
            ax_table[4].set_title('output map')
            ax_table[4].imshow(output_map[0].max(2))
            ax_table[5].cla()

            def on_pick_event(event, ax_table, la_map, spc_map, global_map, output_map):
                i = event.ind[0]
                ax_table[1].cla()
                ax_table[1].imshow(la_map[0, :, :, i])
                ax_table[2].cla()
                ax_table[2].imshow(spc_map[0, :, :, i])
                ax_table[3].cla()
                ax_table[3].imshow(global_map[0, :, :, i])
                ax_table[4].cla()
                ax_table[4].imshow(output_map[0, :, :, i])

            fig.canvas.mpl_connect('pick_event',
                    functools.partial(
                        on_pick_event,
                        ax_table=ax_table,
                        la_map=la_map,
                        spc_map=spc_map,
                        global_map=global_map,
                        output_map=output_map,
                        ))

    def get_exportable_model(self):
        pts_map_layer = self.model.get_layer('pts_map')
        predict_model = keras.Model(
            inputs=self.model.inputs[0],
            outputs=[pts_map_layer.output],
            )
        return predict_model

if __name__ == '__main__':
    KeyPointSpatialConfNetV51.main()
