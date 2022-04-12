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

class KeyPointSpatialConfNetV6(KeyPointKerasNetwork):
    def __init__(self):
        super().__init__()
        self.name = 'spatial-config-net-v6'

        self.lr_base = 1e-4
        self.step_size = 30
        self.gamma = 0.5

        self.l2_weight = 0.0005
        self.activation = 'relu'
        self.dropout_rate = 0.5
        self.partition_win_size = 32

        self.coarse_subnets = 3
        self.coarse_levels = 4
        self.coarse_convolutions = [1, 2, 3]
        self.coarse_kernel_size = [3, 5, 7]
        self.coarse_filter_mult = [1, 2, 3]
        self.coarse_init_filters = [4, 6, 8]

        self.local_subnets = 2
        self.local_levels = [2, 3]
        self.local_kernel_size = [5, 3]
        self.local_filters = [64, 32]

        self.data_loader_conf = {
            'image_shape' : (1024, 1024),
            'intensity' : {'min':0.75, 'max':1.25}, 
            'black_level' : {'min':-0.25, 'max':0.25},
            }

    def generate_params_dict(self):
        params_dict = super().generate_params_dict()
        params_dict.update({
            'l2_weight' : self.l2_weight,
            'activation' : self.activation,
            'dropout_rate' : self.dropout_rate,
            'partition_win_size' : self.partition_win_size,
            'coarse_net_params' : {
                'subnets' : self.coarse_subnets,
                'levels' : self.coarse_levels,
                'convolutions' : self.coarse_convolutions,
                'kernel_size' : self.coarse_kernel_size,
                'filter_mult' : self.coarse_filter_mult,
                'init_filters' : self.coarse_init_filters,
                },
            'local_net_params' : {
                'subnets' : self.local_subnets,
                'levels' : self.local_levels,
                'kernel_size' : self.local_kernel_size,
                'filters' : self.local_filters,
                },
            })
        return params_dict

    def apply_params_dict(self, params_dict):
        super().apply_params_dict(params_dict)
        self.l2_weight = params_dict['l2_weight']
        self.activation = params_dict['activation']
        self.dropout_rate = params_dict['dropout_rate']
        self.partition_win_size = params_dict['partition_win_size']

        self.coarse_subnets = params_dict['coarse_net_params']['subnets']
        self.coarse_levels = params_dict['coarse_net_params']['levels']
        self.coarse_convolutions = params_dict['coarse_net_params']['convolutions']
        self.coarse_kernel_size = params_dict['coarse_net_params']['kernel_size']
        self.coarse_filter_mult = params_dict['coarse_net_params']['filter_mult']
        self.coarse_init_filters = params_dict['coarse_net_params']['init_filters']

        self.local_subnets = params_dict['local_net_params']['subnets']
        self.local_levels = params_dict['local_net_params']['levels']
        self.local_kernel_size = params_dict['local_net_params']['kernel_size']
        self.local_filters = params_dict['local_net_params']['filters']

    def add_cmd_params(self, arg_parser):
        super().add_cmd_params(arg_parser)
        arg_parser.add_argument(
            '--win_size',
            default=self.partition_win_size,
            type=int,
            help='window size for point extraction',
            )
        arg_parser.add_argument(
            '--l2_weight',
            default=self.l2_weight,
            type=float,
            help='l2 weight for all network parameters',
            )
        arg_parser.add_argument(
            '--activation',
            default=self.activation,
            type=str,
            help='activation function for all convolutions (except specific layers)',
            )
        arg_parser.add_argument(
            '--dropout_rate',
            default=self.dropout_rate,
            type=float,
            help='dropout rate for dropout layers',
            )


        arg_parser.add_argument(
            '--coarse_subnets',
            default=self.coarse_subnets,
            type=int,
            help='number of coarse sub networks',
            )
        arg_parser.add_argument(
            '--coarse_levels',
            default=self.coarse_levels,
            type=int,
            help='number of levels for all coarse subnetworks',
            )
        arg_parser.add_argument(
            '--coarse_convolutions', '--coarse_conv',
            default=self.coarse_convolutions,
            type=int,
            nargs='*',
            help='number of convolutions per level of each coarse subnetwork',
            )
        arg_parser.add_argument(
            '--coarse_kernel_size',
            default=self.coarse_kernel_size,
            type=int,
            nargs='*',
            help='size of convolution kernels of each coarse subnetwork',
            )
        arg_parser.add_argument(
            '--coarse_filter_mult',
            default=self.coarse_filter_mult,
            type=float,
            nargs='*',
            help='filter multiplier of each coarse subnetwork',
            )
        arg_parser.add_argument(
            '--coarse_init_filters', '--coarse_filters',
            default=self.coarse_init_filters,
            type=int,
            nargs='*',
            help='number of initial filters of each coarse subnetwork',
            )

        arg_parser.add_argument(
            '--local_subnets',
            default=self.local_subnets,
            type=int,
            help='number of local sub networks',
            )
        arg_parser.add_argument(
            '--local_levels',
            default=self.local_levels,
            type=int,
            nargs='*',
            help='number of levels for each local subnetwork',
            )
        arg_parser.add_argument(
            '--local_kernel_size',
            default=self.local_kernel_size,
            type=int,
            nargs='*',
            help='size of convolution kernels of each local subnetwork',
            )
        arg_parser.add_argument(
            '--local_filters',
            default=self.local_filters,
            type=int,
            nargs='*',
            help='number of filters of convolutions in each local subnetwork',
            )



    def set_cmd_params(self, args_env):
        super().set_cmd_params(args_env)

        self.l2_weight = args_env.l2_weight
        self.activation = args_env.activation
        self.dropout_rate = args_env.dropout_rate
        self.partition_win_size = args_env.win_size

        self.coarse_subnets = args_env.coarse_subnets
        self.coarse_levels = args_env.coarse_levels
        self.coarse_convolutions = args_env.coarse_convolutions
        self.coarse_kernel_size = args_env.coarse_kernel_size
        self.coarse_filter_mult = args_env.coarse_filter_mult
        self.coarse_init_filters = args_env.coarse_init_filters

        self.local_subnets = args_env.local_subnets
        self.local_levels = args_env.local_levels
        self.local_kernel_size = args_env.local_kernel_size
        self.local_filters = args_env.local_filters

    def get_custom_layers_dict(self):
        custom_layers_dict = super().get_custom_layers_dict()
        custom_layers_dict['PartitionedGeometricMean'] = spatialsoftmax.PartitionedGeometricMean
        custom_layers_dict['PartitionsToFeatures'] = spatialsoftmax.PartitionsToFeatures
        custom_layers_dict['spatial_mse'] = spatialsoftmax.SpatialMse(False)
        custom_layers_dict['spatial_rmse'] = spatialsoftmax.SpatialRmse(False)
        custom_layers_dict['GlobalAddMapLayer'] = GlobalAddMapLayer
        return custom_layers_dict

    def init_coarse_map(self, in_tensor, levels, convolutions, kernel_size, filter_mult, init_filters, l2_weight, activation, dropout_rate):
        conv_tensor = in_tensor
        output_tensors_list = []
        for l in range(levels):
            filters = init_filters*int(filter_mult**l)

            for c in range(convolutions):
                conv_tensor = keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding='same',
                    activation=activation,
                    kernel_regularizer=keras.regularizers.l2(l2_weight),
                    )(conv_tensor)
            conv_tensor = keras.layers.AvgPool2D(pool_size=2, padding='same')(conv_tensor)
            conv_tensor = keras.layers.SpatialDropout2D(rate=dropout_rate)(conv_tensor)
            output_tensors_list.append(conv_tensor)

        return output_tensors_list

    def init_local_map(self, in_tensor, levels, kernel_size, filters, activation, dropout_rate, l2_weight):
        out_layers = []
        for l in range(levels):
            conv_1 = keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                activation=activation,
                kernel_regularizer=keras.regularizers.l2(l2_weight),
                )(in_tensor)
            dropout_1 = keras.layers.SpatialDropout2D(rate=dropout_rate)(conv_1)

            conv_2 = keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                activation=activation,
                kernel_regularizer=keras.regularizers.l2(l2_weight),
                )(dropout_1)

            conv_3 = keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                activation=activation,
                kernel_regularizer=keras.regularizers.l2(l2_weight),
                )(conv_2)

            if l < levels-1:
                in_tensor = keras.layers.AvgPool2D(pool_size=2, padding='same')(conv_2)

            out_layers.append(conv_3)

        out_layer = out_layers.pop()
        while out_layers:
            out_layer_2 = out_layers.pop()
            out_layer_upsample = keras.layers.UpSampling2D(
                size=2,
                interpolation='bilinear'
                )(out_layer)
            out_layer = keras.layers.Add()([out_layer_2, out_layer_upsample])

        return out_layer


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

        subsample = (2**self.coarse_levels)

        if W_part%subsample:
            raise Exception(f'window size {W_part} is not divisible by subsample rate {self.coarse_subsample}')
        W_part_subsample = W_part//subsample

        # coarse map B X H_ss X W_ss X F
        coarse_submaps = []
        for n in range(self.coarse_subnets):
            coarse_layers = self.init_coarse_map(
                in_tensor,
                self.coarse_levels,
                self.coarse_convolutions[n],
                self.coarse_kernel_size[n],
                self.coarse_filter_mult[n],
                self.coarse_init_filters[n],
                self.l2_weight,
                self.activation,
                self.dropout_rate,
                )
            coarse_submap = coarse_layers[-1]
            coarse_submaps.append(coarse_submap)
        if len(coarse_submaps) > 1:
            coarse_map = keras.layers.Concatenate()(coarse_submaps)
        else:
            coarse_map = coarse_submaps[0]

        # reduce F features to N point classes
        logits_map = keras.layers.Conv2D(
            filters=N,
            kernel_size=1,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(self.l2_weight),
            )(coarse_map)

        # global map  B X H_ss X W_ss X N
        logits_map = GlobalAddMapLayer(name='logits_map')(logits_map)

        # local map B X H X W X F
        local_submaps = []
        for n in range(self.local_subnets):
            local_submap = self.init_local_map(
                in_tensor,
                self.local_levels[n],
                self.local_kernel_size[n],
                self.local_filters[n],
                self.activation,
                self.dropout_rate,
                self.l2_weight,
                )
            local_submaps.append(local_submap)
        if len(local_submaps) > 1:
            local_map = keras.layers.Concatenate()(local_submaps)
        else:
            local_map = local_submaps[0]

        # reduce F features to N point classes
        local_map = keras.layers.Conv2D(
            filters=N,
            kernel_size=1,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(self.l2_weight),
            )(local_map)

        # B x 2 x M x N
        # M point estimates from M partitions for N points
        part_gm_layer = spatialsoftmax.PartitionedGeometricMean(W_part, name='pts_targets', use_softmax=True)
        pts_targets = part_gm_layer(local_map)

        # B x M x N
        # classification for each of the M partitions
        logits_targets = spatialsoftmax.PartitionsToFeatures(W_part_subsample, sum_over_window=True, name='logits_targets')(logits_map)

        class_targets = keras.layers.Softmax(axis=1, name='class_targets')(logits_targets)

        _, _, M, _ = part_gm_layer.compute_output_shape((-1, H, W, N))
        class_targets = keras.layers.Reshape((1, M, N))(class_targets)

        pts_weighted = keras.layers.Multiply()([pts_targets, class_targets])
        # B x 2 x N
        pts_output = keras.backend.sum(pts_weighted, 2)

        self.model = keras.Model(inputs=in_tensor, outputs=pts_output)

        self.model.compile(
            optimizer='adam',
            loss=spatialsoftmax.SpatialMse(False),
            metrics=[spatialsoftmax.SpatialRmse(False)],
            )

    def preview(self, data):
        import matplotlib.pyplot as pyplot

        pts_targets_layer = self.model.get_layer('pts_targets')
        logits_map_layer = self.model.get_layer('logits_map')
        class_targets_layer = self.model.get_layer('class_targets')

        preview_model = keras.Model(
            inputs=self.model.inputs[0],
            outputs=[
                logits_map_layer.input,
                pts_targets_layer.input,
                pts_targets_layer.output,
                class_targets_layer.output,
                self.model.output,
                ],
            )

        global_logits_maps = logits_map_layer.get_weights()[0]

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
        data_config.pop('shuffle', None)
        data_config.pop('cache', None)
        data_config['image_loader'] = data_loader
        predict_data_seq = cdseq.CephDataSequence(data_config)

        for image_patch, kp_ref in predict_data_seq:
            logits_maps, pts_maps, pts_targets, class_targets, kp_est = preview_model.predict(image_patch)

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

            def on_pick_event(event, ax_table, image_patch, kp_est, kp_ref, pts_maps, logits_maps, global_logits_maps, pts_targets, class_targets):
                i = event.ind[0]
                [ax.cla() for ax in ax_table]

                ax_table[0].imshow(image_patch[0, :, :, 0])
                ax_table[0].plot(kp_est[0, 1, :], kp_est[0, 0, :], 'xr', label='predicted')
                ax_table[0].plot(kp_ref[0, 1, :], kp_ref[0, 0, :], 'ob', label='reference', picker=5)
                ax_table[0].plot(
                    (kp_ref[0, 1, :], kp_est[0, 1, :]),
                    (kp_ref[0, 0, :], kp_est[0, 0, :]),
                    'g',
                    )

                ax_table[1].imshow(pts_maps[0, :, :, i])
                ax_table[2].imshow(np.exp(logits_maps[0, :, :, i]))
                ax_table[3].imshow(np.exp(global_logits_maps[0, :, :, i]))

                ax_table[1].plot(kp_est[0, 1, i], kp_est[0, 0, i], 'xr', label='predicted')
                ax_table[1].plot(kp_ref[0, 1, i], kp_ref[0, 0, i], 'ob', label='reference')
                ax_table[1].scatter(pts_targets[0, 1, :, i], pts_targets[0, 0, :, i],
                        s=100*class_targets[0, :, i]+0.1,
                        c=['g' if v>1e-3 else 'r' for v in class_targets[0, :, i]])

                ax_table[0].legend(loc='best')

            fig.canvas.mpl_connect('pick_event',
                    functools.partial(on_pick_event, ax_table=ax_table, image_patch=image_patch, kp_est=kp_est, kp_ref=kp_ref, pts_maps=pts_maps, logits_maps=logits_maps, global_logits_maps=global_logits_maps, pts_targets=pts_targets, class_targets=class_targets))

    def predict(self, image, pts_target=None):

        class_targets_layer = self.model.get_layer('class_targets')
        predict_model = keras.Model(
            inputs=self.model.inputs[0],
            outputs=[
                class_targets_layer.output,
                self.model.output,
                ],
            )

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
            for i, pt_ind in enumerate(pts_list):
                image_patch, _ = predict_data_seq[i]
                pts_class, pts_est = predict_model.predict(image_patch)

                pts_est = [[(*pts_est[0, :, n],) for n in range(pts_est.shape[2])]]

                pts_est_inv = predict_data_seq.invert_pts_modification(i, pts_est)
                pt = pts_est_inv[0][pt_ind]
                pts_result[0, pt_ind] = pt[0], pt[1], pts_class[0, :, i].max()


        else:
            image_patch, _ = predict_data_seq[0]
            pts_class, pts_est = predict_model.predict(image_patch)

            pts_est = [[(*pts_est[0, :, n],) for n in range(pts_est.shape[2])]]
            pts_est_inv = predict_data_seq.invert_pts_modification(0, pts_est)

            for i, pt_ind in enumerate(pts_list):
                pt = pts_est_inv[0][pt_ind]
                pts_result[0, pt_ind] = pt[0], pt[1], pts_class[0, :, i].max()

        return None, pts_result, pts_target

if __name__ == '__main__':
    KeyPointSpatialConfNetV6.main()
