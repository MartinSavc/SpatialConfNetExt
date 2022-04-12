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
import tensorflowlib.spatialsoftmax as spatialsoftmax

import tensorflowlib.cephdataseq as cdseq
import commonlib.cephdataloaders as dataload

class GlobalAddMapLayer(keras.layers.Layer):
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
                initializer=keras.initializers.Zeros(),
                regularizer=keras.regularizers.l2(1e-5),
                trainable=True)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        return inputs+self.map

class KeyPointSpatialConfNetV7(KeyPointKerasNetwork):
    def __init__(self):
        super().__init__()

        self.name = 'spatial-config-net-v7'

        self.l2_weight = 0.0005
        self.alpha = 20
        self.dropout_rate = 0.5

        self.lr_base = 1e-4#0.01
        # momentum - 0.99
        self.step_size = 30
        self.gamma = 0.5

        self.net_count = 5
        self.global_prior = False
        self.partition_win_size = 32

        self.la_layers = 4
        self.la_filters = 64
        self.la_kernel_size = 5

        self.spc_layers = 3
        self.spc_subsample = 16
        self.spc_filters = 64
        self.spc_kernel_size = 11

        self.data_loader_conf = {
            'image_shape' : (512, 512),
            'rotate' : {'min':-5, 'max':5},
            'scale' : {'min':0.6, 'max':1.2},
            'intensity' : {'min':0.75, 'max':1.25}, 
            'black_level' : {'min':-0.25, 'max':0.25},
            #'points_selection' : (61,),
            }

    def generate_params_dict(self):
        params_dict = super().generate_params_dict()
        params_dict['l2_weight'] = self.l2_weight
        params_dict['dropout_rate'] = self.dropout_rate
        params_dict['partition_win_size'] = self.partition_win_size
        params_dict['net_count'] = self.net_count
        params_dict['global_prior'] = self.global_prior
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

        self.l2_weight = params_dict['l2_weight']
        self.dropout_rate = params_dict['dropout_rate']
        self.partition_win_size = params_dict['partition_win_size']
        self.net_count = params_dict['net_count'] 
        self.global_prior = params_dict['global_prior'] 

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
            '--dropout_rate', 
            default=self.dropout_rate,
            type=float,
            help='dropout rate for all dropout layers',
            )
        arg_parser.add_argument(
            '--win_size',
            default=self.partition_win_size,
            type=int,
            help='window size for point extraction',
            )
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

        self.l2_weight = args_env.l2_weight
        self.dropout_rate = args_env.dropout_rate
        self.partition_win_size = args_env.win_size
        self.net_count = args_env.net_count
        self.global_prior = args_env.global_prior
        self.la_layers = args_env.la_layers
        self.la_filters = args_env.la_filters
        self.la_kernel_size = args_env.la_kernel_size
        self.spc_layers = args_env.spc_layers
        self.spc_subsample = args_env.spc_subsample
        self.spc_filters = args_env.spc_filters
        self.spc_kernel_size = args_env.spc_kernel_size

    def get_custom_layers_dict(self):
        custom_layers_dict = super().get_custom_layers_dict()
        custom_layers_dict['PartitionedGeometricMean'] = spatialsoftmax.PartitionedGeometricMean
        custom_layers_dict['PartitionsToFeatures'] = spatialsoftmax.PartitionsToFeatures
        custom_layers_dict['spatial_mse'] = spatialsoftmax.SpatialMse(False)
        custom_layers_dict['spatial_rmse'] = spatialsoftmax.SpatialRmse(False)
        custom_layers_dict['GlobalAddMapLayer'] = GlobalAddMapLayer
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
            dropout = keras.layers.Dropout(self.dropout_rate)(conv_1)
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
        W_part = self.partition_win_size

        if W_part%self.spc_subsample:
            raise Exception(f'window size {W_part} is not divisible by subsample rate {self.coarse_subsample}')
        W_part_subsample = W_part//self.spc_subsample

        pts_feat_list = []
        la_submaps_list = []
        spc_submaps_list = []
        for n in range(self.net_count):
            la_submap = self.init_local_appearance_layers(in_tensor, N)
            spc_submap = self.init_spatial_configuration_layers(la_submap, N, output_activation='linear', upsample=False)

            la_submaps_list.append(la_submap)
            spc_submaps_list.append(spc_submap)

        if self.net_count > 1:
            la_map = keras.layers.Add()(la_submaps_list)
            spc_map = keras.layers.Add()(spc_submaps_list)
        else:
            la_map = la_submaps_list[0]
            spc_map = spc_submaps_list[0]

        if self.global_prior:
            spc_map = GlobalAddMapLayer()(spc_map)

        # B x 2 x M x N
        # M point estimates from M partitions for N points
        part_gm_layer = spatialsoftmax.PartitionedGeometricMean(W_part, name='pts_targets', use_softmax=True)
        pts_targets = part_gm_layer(la_map)

        # B x M x N
        # classification for each of the M partitions
        logits_targets = spatialsoftmax.PartitionsToFeatures(W_part_subsample, sum_over_window=True, name='logits_targets')(spc_map)
        class_targets = keras.layers.Softmax(axis=1, name='class_targets')(logits_targets)

        _, _, M, _ = part_gm_layer.compute_output_shape((-1, H, W, N))
        class_targets = keras.layers.Reshape((1, M, N))(class_targets)

        pts_weighted = keras.layers.Multiply()([pts_targets, class_targets])
        pts_output = keras.layers.Lambda(lambda x: keras.backend.sum(x, 2))(pts_weighted)

        self.model = keras.Model(inputs=in_tensor, outputs=pts_output)

        self.model.compile(
            optimizer='adam',
            loss=spatialsoftmax.SpatialRmse(False),
            metrics=[spatialsoftmax.SpatialRmse(False)],
            )

    def preview(self, data):
        import matplotlib.pyplot as pyplot

        pts_targets_layer = self.model.get_layer('pts_targets')
        if self.global_prior:
            logits_map_layer = self.model.get_layer('global_add_map_layer')
            global_logits_maps = logits_map_layer.get_weights()[0]
        else:
            logits_map_layer = self.model.get_layers('logits_targets')
            global_logits_maps = None
        class_targets_layer = self.model.get_layer('class_targets')

        class_targets_max = keras.layers.GlobalMaxPool1D()(class_targets_layer.output)
        class_targets_max = keras.layers.Reshape((1, class_targets_max.shape[1]), name='exp_reshape')(class_targets_max)

        new_model_output = keras.layers.Concatenate(axis=1)([self.model.output, class_targets_max])

        preview_model = keras.Model(
            inputs=self.model.inputs[0],
            outputs=[
                logits_map_layer.input,
                pts_targets_layer.input,
                pts_targets_layer.output,
                class_targets_layer.output,
                new_model_output
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

        for image_patch, kp_ref in predict_data_seq:
            logits_maps, pts_maps, pts_targets, class_targets, kp_est = preview_model.predict(image_patch)

            print(kp_est[0, 2, :])

            fig, ax_table = pyplot.subplots(2, 2)
            ax_table = ax_table.ravel()
            ax_table[0].imshow(image_patch[0, :, :, 0])
            ax_table[0].plot(kp_est[0, 1, :], kp_est[0, 0, :], 'xr', label='predicted')
            ax_table[0].plot(kp_ref[0, 1, :], kp_ref[0, 0, :], 'ob', label='reference', picker=5)
            ax_table[0].plot(
                (kp_ref[0, 1, :], kp_est[0, 1, :]),
                (kp_ref[0, 0, :], kp_est[0, 0, :]),
                'g',
                )
            ax_table[0].legend(loc='best')

            ax_table[1].set_title('pts maps')
            ax_table[1].imshow(pts_maps[0, :, :, 0])
            ax_table[2].set_title('logits maps')
            ax_table[2].imshow(np.exp(logits_maps[0, :, :, 0]))
            ax_table[3].set_title('global maps')
            if global_logits_maps is not None:
                ax_table[3].imshow(np.exp(global_logits_maps[0, :, :, 0]))

            def on_pick_event(event, ax_table, image_patch, kp_est, kp_ref, pts_maps, logits_maps, global_logits_maps, pts_targets, class_targets):
                i = event.ind[0]
                [ax.cla() for ax in ax_table]

                ax_table[0].set_title('image')
                ax_table[0].imshow(image_patch[0, :, :, 0])
                ax_table[0].plot(kp_est[0, 1, :], kp_est[0, 0, :], 'xr', label='predicted')
                ax_table[0].plot(kp_ref[0, 1, :], kp_ref[0, 0, :], 'ob', label='reference', picker=5)
                ax_table[0].plot(
                    (kp_ref[0, 1, :], kp_est[0, 1, :]),
                    (kp_ref[0, 0, :], kp_est[0, 0, :]),
                    'g',
                    )

                ax_table[1].set_title('pts maps')
                ax_table[1].imshow(pts_maps[0, :, :, i])
                ax_table[2].set_title('logits maps')
                ax_table[2].imshow(np.exp(logits_maps[0, :, :, i]))
                ax_table[3].set_title('global maps')
                if global_logits_maps is not None:
                    ax_table[3].imshow(np.exp(global_logits_maps[0, :, :, i]))

                ax_table[1].plot(kp_est[0, 1, i], kp_est[0, 0, i], 'xr', label='predicted')
                ax_table[1].plot(kp_ref[0, 1, i], kp_ref[0, 0, i], 'ob', label='reference')
                ax_table[1].scatter(pts_targets[0, 1, :, i], pts_targets[0, 0, :, i],
                        s=100*class_targets[0, :, i]+0.1,
                        c=['g' if v>1e-3 else 'r' for v in class_targets[0, :, i]])

                ax_table[0].legend(loc='best')

            fig.canvas.mpl_connect('pick_event',
                    functools.partial(on_pick_event, ax_table=ax_table, image_patch=image_patch, kp_est=kp_est, kp_ref=kp_ref, pts_maps=pts_maps, logits_maps=logits_maps, global_logits_maps=global_logits_maps, pts_targets=pts_targets, class_targets=class_targets))

    def get_exportable_model(self):
        class_targets_layer = self.model.get_layer('class_targets')
        class_targets_max = keras.layers.GlobalMaxPool1D()(class_targets_layer.output)
        class_targets_max = keras.layers.Reshape((1, class_targets_max.shape[1]), name='exp_reshape')(class_targets_max)
        new_model_output = keras.layers.Concatenate(axis=1)([self.model.output, class_targets_max])
        export_model = keras.Model(
            inputs=self.model.inputs[0],
            outputs=[
                new_model_output
                ],
            )

        return export_model


    def predict(self, image, pts_target=None):
        predict_model = self.get_exportable_model()

        data_loader = dataload.CephDataWrapper((image,), pts_target)
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

        pts_list = data_config.get('points_selection', None)
        if not pts_list:
            pts_list = [*range(self.n_pts)]

        pts_result = np.array([[(0, 0, -1) for _ in pts_target[0]]], dtype=float)
        pts_target = np.array([[(y, x, 0) for y, x in pts_target[0]]], dtype=float)

        if 'subselect' in data_config:
            for i, pt_ind in enumerate(pts_list):
                image_patch, _ = predict_data_seq[i]
                pts_est = predict_model.predict(image_patch)

                pts_est_list = [[(*pts_est[0, :, n],) for n in range(pts_est.shape[2])]]

                pts_est_inv = predict_data_seq.invert_pts_modification(i, pts_est_list)
                pt = pts_est_inv[0][0]
                pts_result[0, pt_ind] = pt[0], pt[1], pts_est[0, 2, 0]


        else:
            image_patch, _ = predict_data_seq[0]
            pts_est = predict_model.predict(image_patch)

            pts_est_list = [[(*pts_est[0, :2, n],) for n in range(pts_est.shape[2])]]
            pts_est_inv = predict_data_seq.invert_pts_modification(0, pts_est_list)

            for i, pt_ind in enumerate(pts_list):
                pt = pts_est_inv[0][i]
                pts_result[0, pt_ind] = pt[0], pt[1], pts_est[0, 2, i]

        return None, pts_result, pts_target

if __name__ == '__main__':
    KeyPointSpatialConfNetV7.main()
