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

def OutputIsLoss(_, loss):
    return loss


def modify_data_samples(sample):
    image, pts = sample
    B = image.shape[0]
    return (image, pts), np.zeros((B, 1))

class KeyPointSpatialConfNet(KeyPointKerasNetwork):
    def __init__(self):
        super().__init__()

        self.name = 'spatial-config-net'

        self.gamma_weight = 100
        self.l2_weight = 0.0005
        self.alpha = 20

        self.lr_base = 1e-4#0.01
        # momentum - 0.99
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
            #'image_shape' : (1024, 1024),
            #'rotate' : {'min':-15, 'max':15},
            #'scale' : {'min':0.6, 'max':1.4},
            'intensity' : {'min':0.75, 'max':1.25}, 
            'black_level' : {'min':-0.25, 'max':0.25},
            #'points_selection' : (61,),
            'output_fun': modify_data_samples,
            }

    def generate_params_dict(self):
        del self.data_loader_conf['output_fun']
        params_dict = copy.deepcopy(super().generate_params_dict())
        self.data_loader_conf['output_fun'] = modify_data_samples
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

    def apply_params_dict(self, params_dict):
        super().apply_params_dict(params_dict)
        self.data_loader_conf['output_fun'] = modify_data_samples

        self.l2_weight = params_dict['l2_weight']

        self.la_layers = params_dict['la_params']['layers']
        self.la_filters = params_dict['la_params']['filters']
        self.la_kernel_size = params_dict['la_params']['kernel_size']

        self.spc_layers = params_dict['spc_params']['layers']
        self.spc_subsample = params_dict['spc_params']['subsample']
        self.spc_filters = params_dict['spc_params']['filters']
        self.spc_kernel_size = params_dict['spc_params']['kernel_size']

    def add_cmd_params(self, arg_parser):
        super().add_cmd_params(arg_parser)

        arg_parser.add_argument(
            '--l2_weight', 
            default=self.l2_weight,
            type=float,
            help='weight for the l2 regularization parameter',
            )
        arg_parser.add_argument(
            '--la_layers', 
            default=self.la_layers,
            type=int,
            help='number of layers of the Local Appearance network',
            )
        arg_parser.add_argument(
            '--la_filters', 
            default=self.la_filters,
            type=int,
            help='number of filters for each leayer of the Local Appearance network',
            )
        arg_parser.add_argument(
            '--la_kernel_size', 
            default=self.la_kernel_size,
            type=int,
            help='size of kernels of the Local Appearance network',
            )
        arg_parser.add_argument(
            '--spc_layers', 
            default=self.spc_layers,
            type=int,
            help='number of layers of the Spatial Configuration network',
            )
        arg_parser.add_argument(
            '--spc_subsample', 
            default=self.spc_subsample,
            type=int,
            help='submsampling factor of the Spatial Configuration network',
            )
        arg_parser.add_argument(
            '--spc_filters', 
            default=self.spc_filters,
            type=int,
            help='number of filters for each layer of the Spatial Configuration network',
            )
        arg_parser.add_argument(
            '--spc_kernel_size', 
            default=self.spc_kernel_size,
            type=int,
            help='size of kernel of the Spatial Configuration network',
            )

    def set_cmd_params(self, args_env):
        super().set_cmd_params(args_env)
        self.data_loader_conf['output_fun'] = modify_data_samples

        self.l2_weight = args_env.l2_weight
        self.la_layers = args_env.la_layers
        self.la_filters = args_env.la_filters
        self.la_kernel_size = args_env.la_kernel_size
        self.spc_layers = args_env.spc_layers
        self.spc_subsample = args_env.spc_subsample
        self.spc_filters = args_env.spc_filters
        self.spc_kernel_size = args_env.spc_kernel_size

    def get_custom_layers_dict(self):
        custom_layers_dict = super().get_custom_layers_dict()
        custom_layers_dict['HeatmapErr'] = HeatmapErr
        custom_layers_dict['GaussianHeatmap'] = GaussianHeatmap
        custom_layers_dict['OutputIsLoss'] = OutputIsLoss
        return custom_layers_dict

    def init_local_appearance_layers(self, in_tensor, out_ndim):
        in_down = in_tensor
        out_layers = []
        for l in range(self.la_layers):
            conv_1 = keras.layers.Conv2D(
                filters=self.la_filters,
                kernel_size=self.la_kernel_size,
                padding='same',
                kernel_regularizer=keras.regularizers.l2(self.l2_weight),
                bias_initializer=keras.initializers.Constant(0),
                kernel_initializer='he_uniform',
                )(in_down)
            conv_1 = keras.layers.LeakyReLU(alpha=0.1)(conv_1)
            dropout = keras.layers.Dropout(0.5)(conv_1)
            conv_2 = keras.layers.Conv2D(
                filters=self.la_filters,
                kernel_size=self.la_kernel_size,
                padding='same',
                kernel_regularizer=keras.regularizers.l2(self.l2_weight),
                bias_initializer=keras.initializers.Constant(0),
                kernel_initializer='he_uniform',
                )(dropout)
            conv_2 = keras.layers.LeakyReLU(alpha=0.1)(conv_2)
            conv_3 = keras.layers.Conv2D(
                filters=self.la_filters,
                kernel_size=self.la_kernel_size,
                padding='same',
                kernel_regularizer=keras.regularizers.l2(self.l2_weight),
                bias_initializer=keras.initializers.Constant(0),
                kernel_initializer='he_uniform',
                )(conv_2)
            conv_3 = keras.layers.LeakyReLU(alpha=0.1)(conv_3)
            in_down = keras.layers.AvgPool2D(pool_size=2, padding='same')(conv_2)
            out_layers.append(conv_3)

        out_layer = out_layers.pop()
        while out_layers:
            out_layer_2 = out_layers.pop()
            out_layer_upsample = keras.layers.UpSampling2D(
                size=2,
                interpolation='bilinear'
                )(out_layer)
            out_layer = keras.layers.Add()([out_layer_2, out_layer_upsample])

        la_map = keras.layers.Conv2D(
            filters=out_ndim,
            kernel_size=self.la_kernel_size,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(self.l2_weight),
            bias_initializer=keras.initializers.Constant(0),
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
            )(out_layer)
        return la_map

    def init_spatial_configuration_layers(
        self, 
        in_tensor,
        out_ndim,
        output_activation='tanh',
        upsample=True,
        ):
        # spatial configuration
        spc_conf_conv = keras.layers.AvgPool2D(
            pool_size=self.spc_subsample,
            padding='same'
            )(in_tensor)

        for l in range(self.spc_layers):
            spc_conf_conv = keras.layers.Conv2D(
                filters=self.spc_filters,
                kernel_size=self.spc_kernel_size,
                padding='same',
                kernel_regularizer=keras.regularizers.l2(self.l2_weight),
                bias_initializer=keras.initializers.Constant(0),
                kernel_initializer='he_uniform',
                )(spc_conf_conv)
            spc_conf_conv = keras.layers.LeakyReLU(alpha=0.1)(spc_conf_conv)

        spc_map_lr = keras.layers.Conv2D(
            filters=out_ndim,
            kernel_size=self.spc_kernel_size,
            padding='same',
            activation=output_activation,
            kernel_regularizer=keras.regularizers.l2(self.l2_weight),
            bias_initializer=keras.initializers.Constant(0),
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
            )(spc_conf_conv)

        if upsample:
            spc_map = keras.layers.UpSampling2D(size=self.spc_subsample, interpolation='bilinear')(spc_map_lr)
        else:
            spc_map = spc_map_lr

        return spc_map

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

        la_map = self.init_local_appearance_layers(in_tensor, N)
        spc_map = self.init_spatial_configuration_layers(la_map, N)

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

    def preview(self, data):
        import matplotlib.pyplot as pyplot

        mult_layer = self.model.layers[-3]
        preview_model = keras.Model(
            inputs=self.model.inputs[0],
            outputs=[*mult_layer.input, mult_layer.output],
            )

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

            fig, ax_table = pyplot.subplots(2, 2, True, True)
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
            ax_table[3].set_title('output map')
            ax_table[3].imshow(output_map[0].max(2))

            def on_pick_event(event, ax_table):
                i = event.ind[0]
                ax_table[1].cla()
                ax_table[1].imshow(la_map[0, :, :, i])
                ax_table[2].cla()
                ax_table[2].imshow(spc_map[0, :, :, i])
                ax_table[3].cla()
                ax_table[3].imshow(output_map[0, :, :, i])

            fig.canvas.mpl_connect('pick_event',
                    functools.partial(on_pick_event, ax_table=ax_table))



    def get_exportable_model(self):
        mult_layer = self.model.layers[-3]
        predict_model = keras.Model(
            inputs=self.model.inputs[0],
            outputs=[mult_layer.output],
            )
        return predict_model 

    def predict(self, image, pts_target=None):
        predict_model = self.get_exportable_model()

        data_loader = dataload.CephDataWrapper((image,), pts_target)
        data_config = self.data_loader_conf.copy()
        data_config.pop('rotate', None)
        data_config.pop('scale', None)
        data_config.pop('pw_affine_warp', None)
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
            # multiple prediction
            kp_maps = []
            for i, pt_ind in enumerate(pts_list):
                (image_patch, kp_ref), _target_loss = predict_data_seq[i]
                output_map = predict_model.predict(image_patch)

                max_ind = output_map.reshape(-1, output_map.shape[3]).argmax(0)
                max_val = output_map.reshape(-1, output_map.shape[3]).max(0)
                max_pts = np.unravel_index(max_ind, output_map.shape[1:3])

                pts_est = [[*zip(*max_pts)]]

                pts_est_inv = predict_data_seq.invert_pts_modification(i, pts_est)
                pt = pts_est_inv[0][pt_ind]
                pts_result[0, pt_ind] = pt[0], pt[1], max_val[i]

                kp_maps.append(output_map[..., i])
            kp_maps = np.array(kp_maps).transpose(1, 2, 3, 0)
        else:
            # single prediction
            (image_patch, kp_ref), _target_loss = predict_data_seq[0]
            output_map = predict_model.predict(image_patch)

            max_ind = output_map.reshape(-1, output_map.shape[3]).argmax(0)
            max_val = output_map.reshape(-1, output_map.shape[3]).max(0)
            max_pts = np.unravel_index(max_ind, output_map.shape[1:3])

            pts_est = [[*zip(*max_pts)]]
            pts_est_inv = predict_data_seq.invert_pts_modification(0, pts_est)

            for i, pt_ind in enumerate(pts_list):
                pt = pts_est_inv[0][pt_ind]
                pts_result[0, pt_ind] = pt[0], pt[1], max_val[i]

            kp_maps = output_map

        return kp_maps, pts_result, pts_target

if __name__ == '__main__':
    KeyPointSpatialConfNet.main()
