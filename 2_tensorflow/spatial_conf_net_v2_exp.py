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

from spatial_conf_net_exp import KeyPointSpatialConfNet

class KeyPointSpatialConfNetV2(KeyPointSpatialConfNet):
    def __init__(self):
        super().__init__()

        self.name = 'spatial-config-net-v2'

        self.gamma_weight = 100
        self.l2_weight = 0.0005
        self.alpha = 20

        self.lr_base = 1e-6
        self.step_size = 50
        self.gamma = 0.5

        self.la_layers = 4
        self.la_filters = 128
        self.la_kernel_size = 3

        self.spc_layers = 3
        self.spc_subsample = 16
        self.spc_filters = 128
        self.spc_kernel_size = 11

        self.data_loader_conf = {
            #'rotate' : {'min':-15, 'max':15},
            #'scale' : {'min':0.6, 'max':1.4},
            'intensity' : {'min':0.75, 'max':1.25}, 
            'black_level' : {'min':-0.25, 'max':0.25},
            #'points_selection' : (61,),
            }

    def generate_params_dict(self):
        params_dict = super(KeyPointSpatialConfNet, self).generate_params_dict()
        params_dict['l2_weight'] = self.l2_weight
        params_dict['la_params'] = {
            'layers':self.la_layers,
            'filters':self.la_filters,
            'kernel_size':self.la_kernel_size,
            }
        params_dict['spc_params'] = {
            'layers':self.spc_layers,
            'subsample':self.spc_subsample,
            'filters':self.spc_filters,
            'kernel_size':self.spc_kernel_size,
            }
        return params_dict

    def set_cmd_params(self, args_env):
        super(KeyPointSpatialConfNet, self).set_cmd_params(args_env)
        self.l2_weight = args_env.l2_weight
        self.la_layers = args_env.la_layers
        self.la_filters = args_env.la_filters
        self.la_kernel_size = args_env.la_kernel_size
        self.spc_layers = args_env.spc_layers
        self.spc_subsample = args_env.spc_subsample
        self.spc_filters = args_env.spc_filters
        self.spc_kernel_size = args_env.spc_kernel_size

    def apply_params_dict(self, params_dict):
        super(KeyPointSpatialConfNet, self).apply_params_dict(params_dict)

        self.l2_weight = params_dict['l2_weight']

        self.la_layers = params_dict['la_params']['layers']
        self.la_filters = params_dict['la_params']['filters']
        self.la_kernel_size = params_dict['la_params']['kernel_size']

        self.spc_layers = params_dict['spc_params']['layers']
        self.spc_subsample = params_dict['spc_params']['subsample']
        self.spc_filters = params_dict['spc_params']['filters']
        self.spc_kernel_size = params_dict['spc_params']['kernel_size']


    def get_custom_layers_dict(self):
        custom_layers_dict = super(KeyPointSpatialConfNet, self).get_custom_layers_dict()
        custom_layers_dict['SpatialSoftmax'] = spatialsoftmax.SpatialSoftmax
        custom_layers_dict['GeometricMean'] = spatialsoftmax.GeometricMean
        custom_layers_dict['spatial_mse'] = spatialsoftmax.SpatialMse(False)
        custom_layers_dict['spatial_rmse'] = spatialsoftmax.SpatialRmse(False)
        return custom_layers_dict


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

        la_map = self.init_local_appearance_layers(in_tensor, N)
        spc_map = self.init_spatial_configuration_layers(la_map, N, 'tanh')

        detect_map = keras.layers.Multiply()([la_map, spc_map])

        sm_detect_map = spatialsoftmax.SpatialSoftmax()(detect_map)
        pts_target = spatialsoftmax.GeometricMean(normalize=False)(sm_detect_map)

        self.model = keras.Model(inputs=in_tensor, outputs=pts_target)

        self.model.compile(
            optimizer='adam',
            loss=spatialsoftmax.SpatialRmse(False),
            metrics=[spatialsoftmax.SpatialRmse(False)],
            )

    def preview(self, data):
        import matplotlib.pyplot as pyplot

        mult_layer = self.model.layers[-3]
        soft_max_layer = self.model.layers[-2]
        geom_mean_layer = self.model.layers[-1]

        preview_model = keras.Model(
            inputs=self.model.inputs[0],
            outputs=[
                *mult_layer.input,
                mult_layer.output,
                soft_max_layer.output,
                geom_mean_layer.output
                ],
            )

        image, pts_target_list = data
        data_loader = dataload.CephDataWrapper((image,), pts_target_list)
        data_config = self.data_loader_conf.copy()
        data_config.pop('rotate', None)
        data_config.pop('scale', None)
        data_config.pop('pw_affine_warp', None)
        data_config.pop('translate', None)
        data_config.pop('intensity', None)
        data_config.pop('batch_size', None)
        data_config.pop('black_level', None)
        data_config.pop('gamma', None)
        data_config.pop('shuffle', None)
        data_config.pop('cache', None)
        data_config['image_loader'] = data_loader
        predict_data_seq = cdseq.CephDataSequence(data_config)

        #image_patch, kp_ref = predict_data_seq[0]
        for image_patch, kp_ref in predict_data_seq:
            la_map, spc_map, mult_map, sm_map, pts_output = preview_model.predict(image_patch)

            fig, ax_table = pyplot.subplots(2, 2, True, True)
            ax_table = ax_table.ravel()

            ax_table[0].set_title('image')
            ax_table[0].imshow(image_patch[0, ..., 0])
            ax_table[0].plot(pts_output[0, 1, :], pts_output[0, 0, :], 'xr', label='predicted')
            ax_table[0].plot(
                kp_ref[0, 1, :],
                kp_ref[0, 0, :],
                'ob',
                label='reference',
                picker=5,
                )

            ax_table[0].plot(
                (kp_ref[0, 1, :], pts_output[0, 1, :]),
                (kp_ref[0, 0, :], pts_output[0, 0, :]),
                'g',
                )
            ax_table[0].legend(loc='best')

            ax_table[1].set_title('la map')
            ax_table[1].imshow(la_map[0, :, :, 0])
            ax_table[2].set_title('spc map')
            ax_table[2].imshow(spc_map[0, :, :, 0])
            ax_table[3].set_title('output map')
            ax_table[3].imshow(sm_map[0, :, :, 0]**0.1)

            def on_pick_event(event, ax_table, la_map, spc_map, sm_map):
                i = event.ind[0]
                ax_table[1].cla()
                ax_table[1].imshow(la_map[0, :, :, i])
                ax_table[2].cla()
                ax_table[2].imshow(spc_map[0, :, :, i])
                ax_table[3].cla()
                ax_table[3].imshow(sm_map[0, :, :, i]**0.1)

            fig.canvas.mpl_connect('pick_event',
                    functools.partial(
                        on_pick_event,
                        ax_table=ax_table,
                        la_map=la_map,
                        spc_map=spc_map,
                        sm_map=sm_map,
                        )
                    )

    def get_exportable_model(self):
        soft_max_layer = self.model.layers[-2]
        geom_mean_layer = self.model.layers[-1]
        predict_model = keras.Model(
            inputs=self.model.inputs[0],
            outputs=[
                soft_max_layer.output,
                geom_mean_layer.output
                ],
            )
        return predict_model

    def predict(self, image, pts_target=None):
        predict_model = self.get_exportable_model()

        data_loader = dataload.CephDataWrapper((image,), pts_target)
        data_config = self.data_loader_conf.copy()
        data_config.pop('rotate', None)
        data_config.pop('scale', None)
        data_config.pop('intensity', None)
        data_config.pop('batch_size', None)
        data_config.pop('black_level', None)
        data_config.pop('gamma', None)
        data_config['image_loader'] = data_loader
        predict_data_seq = cdseq.CephDataSequence(data_config)

        pts_list = data_config.get('points_selection', None)
        if not pts_list:
            pts_list = [*range(self.n_pts)]

        pts_result = np.array([[(0, 0, -1) for _ in pts_target[0]]], dtype=float)
        pts_target = np.array([[(y, x, 0) for y, x in pts_target[0]]], dtype=float)

        if 'subselect' in data_config:
            kp_maps = []
            for i, pt_ind in enumerate(pts_list):
                image_patch, _ = predict_data_seq[i]
                output_map, pts_est = predict_model.predict(image_patch)

                pts_est = [[(*pts_est[0, :, n],) for n in range(pts_est.shape[2])]]

                pts_est_inv = predict_data_seq.invert_pts_modification(i, pts_est)
                pt = pts_est_inv[0][pt_ind]
                pts_result[0, pt_ind] = pt[0], pt[1], 0

                kp_maps.append(output_map[..., i]/output_map[..., i].max())

            kp_maps = np.array(kp_maps).transpose(1, 2, 3, 0)
        else:
            image_patch, _ = predict_data_seq[0]
            output_map, pts_est = predict_model.predict(image_patch)

            pts_est = [[(*pts_est[0, :, n],) for n in range(pts_est.shape[2])]]
            pts_est_inv = predict_data_seq.invert_pts_modification(0, pts_est)

            for i, pt_ind in enumerate(pts_list):
                pt = pts_est_inv[0][pt_ind]
                pts_result[0, pt_ind] = pt[0], pt[1], 0
            kp_maps = output_map

        return kp_maps, pts_result, pts_target


if __name__ == '__main__':
    KeyPointSpatialConfNetV2.main()
