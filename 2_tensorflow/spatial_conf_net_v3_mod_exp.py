import os
import sys
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

import tensorflowlib.spatialsoftmax as spatialsoftmax

import tensorflowlib.cephdataseq as cdseq
import commonlib.cephdataloaders as dataload

from spatial_conf_net_exp import KeyPointSpatialConfNet


class KeyPointSpatialConfNetV3(KeyPointSpatialConfNet):
    def __init__(self):
        super().__init__()

        self.name = 'spatial-config-net-v3-mod'

        self.partition_win_size = 32
        self.pt_chns = 72
        self.feat_chns = 96

        self.gamma_weight = 100
        self.l2_weight = 0.0005
        self.alpha = 20

        self.lr_base = 1e-4
        self.step_size = 20
        self.gamma = 0.5

        self.la_layers = 8
        self.la_filters = 8
        self.la_kernel_size = 3
        self.la_chn_base = 1.44
        self.la_pool_size = 2
        self.la_downsample_interval = 2
        self.la_activation_string = "relu"
        self.part_gm_layers = [0, 2, 4]
        self.dropout_rate = 0

        self.la_activation = keras.activations.get(self.la_activation_string)
        self.la_chn_getter = lambda i: int(self.la_chn_base ** i) * self.la_filters

        self.data_loader_conf = {
            'image_shape': (1024, 1024),
            # 'rotate' : {'min':-15, 'max':15},
            # 'scale' : {'min':0.6, 'max':1.4},
            'intensity': {'min': 0.75, 'max': 1.25},
            'black_level': {'min': -0.25, 'max': 0.25},
            # 'points_selection' : (61,),
        }

    def add_cmd_params(self, arg_parser):
        super().add_cmd_params(arg_parser)

        arg_parser.add_argument(
            '--activation',
            default=self.la_activation_string,
            type=str,
            help='activation function',
        )

    def generate_params_dict(self):
        params_dict = super(KeyPointSpatialConfNet, self).generate_params_dict()
        params_dict['win_size'] = self.partition_win_size
        params_dict['pt_chns'] = self.pt_chns
        params_dict['feat_chns'] = self.feat_chns
        params_dict['dropout_rate'] = self.dropout_rate

        params_dict['la_params'] = {
            'layers': self.la_layers,
            'filters': self.la_filters,
            'kernel_size': self.la_kernel_size,
            'chn_base': self.la_chn_base,
            'pool_size': self.la_pool_size,
            'downsample_interval': self.la_downsample_interval,
            'activation': self.la_activation_string,
            'part_gm_layers': self.part_gm_layers,
        }
        return params_dict

    def set_cmd_params(self, args_env):
        super(KeyPointSpatialConfNet, self).set_cmd_params(args_env)
        self.la_activation_string = args_env.activation

    def apply_params_dict(self, params_dict):
        super(KeyPointSpatialConfNet, self).apply_params_dict(params_dict)
        self.partition_win_size = params_dict['win_size']
        self.pt_chns = params_dict['pt_chns']
        self.feat_chns = params_dict['feat_chns']
        self.dropout_rate = params_dict['dropout_rate']

        self.la_layers = params_dict['la_params']['layers']
        self.la_filters = params_dict['la_params']['filters']
        self.la_kernel_size = params_dict['la_params']['kernel_size']
        self.la_downsample_interval = params_dict['la_params']['downsample_interval']
        self.la_pool_size = params_dict['la_params']['pool_size']
        self.la_chn_base = params_dict['la_params']['chn_base']
        self.la_activation_string = params_dict['la_params']['activation']
        self.part_gm_layers = params_dict['la_params']['part_gm_layers']


    def get_custom_layers_dict(self):
        custom_layers_dict = super(KeyPointSpatialConfNet, self).get_custom_layers_dict()
        custom_layers_dict['PartitionedGeometricMean'] = spatialsoftmax.PartitionedGeometricMean
        custom_layers_dict['PartitionsToFeatures'] = spatialsoftmax.PartitionsToFeatures
        custom_layers_dict['spatial_mse'] = spatialsoftmax.SpatialMse(False)
        custom_layers_dict['spatial_rmse'] = spatialsoftmax.SpatialRmse(False)
        return custom_layers_dict

    def init_local_appearance_layers(self, in_tensor, out_ndim, out_feat):
        in_down = in_tensor
        out_layers = []
        for l in range(self.la_layers):
            conv_1 = keras.layers.Conv2D(
                filters=self.la_chn_getter(l),
                kernel_size=self.la_kernel_size,
                padding='same',
                activation=self.la_activation_string,
                kernel_regularizer=keras.regularizers.l2(self.l2_weight),
                bias_initializer=keras.initializers.Constant(0),
                kernel_initializer='he_uniform',
            )(in_down)

            if not (l + 1) % self.la_downsample_interval:
                pooling = keras.layers.MaxPool2D(pool_size=self.la_pool_size, padding='same')(conv_1)
            else:
                pooling = conv_1

            dropout = keras.layers.Dropout(self.dropout_rate)(pooling)
            conv_2 = keras.layers.Conv2D(
                filters=self.la_chn_getter(l),
                kernel_size=self.la_kernel_size,
                padding='same',
                activation=self.la_activation_string,
                kernel_regularizer=keras.regularizers.l2(self.l2_weight),
                bias_initializer=keras.initializers.Constant(0),
                kernel_initializer='he_uniform',
            )(dropout)

            conv_3 = keras.layers.Conv2D(
                filters=self.la_chn_getter(l),
                kernel_size=self.la_kernel_size,
                padding='same',
                activation=self.la_activation_string,
                kernel_regularizer=keras.regularizers.l2(self.l2_weight),
                bias_initializer=keras.initializers.Constant(0),
                kernel_initializer='he_uniform',
            )(conv_2)
            out_layers.append(conv_3)
            in_down = conv_2

        part_gm = out_layers[self.part_gm_layers[-1]]
        for l in range(len(self.part_gm_layers) - 2, -1, -1):
            combine_layer = out_layers[self.part_gm_layers[l]]
            upsampled_layer = keras.layers.UpSampling2D((combine_layer.get_shape()[1] // part_gm.get_shape()[1],
                                                        combine_layer.get_shape()[2] // part_gm.get_shape()[2]),
                                                        interpolation='bilinear')(part_gm)
            part_gm = keras.layers.Concatenate()([combine_layer, upsampled_layer])

        part_gm = keras.layers.Conv2D(
            filters=out_ndim,
            kernel_size=self.la_kernel_size,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(self.l2_weight),
            bias_initializer=keras.initializers.Constant(0),
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
        )(part_gm)

        part_feat = out_layers[-1]

        part_feat = keras.layers.Conv2D(
            filters=out_feat,
            kernel_size=self.la_kernel_size,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(self.l2_weight),
            bias_initializer=keras.initializers.Constant(0),
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
        )(part_feat)

        return part_gm, part_feat

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
        F = self.feat_chns

        # B x H x W x N
        part_gm_map, part_feat_map = self.init_local_appearance_layers(in_tensor, N, F)

        subsample = 2 ** (self.la_layers // self.la_downsample_interval)

        if W_part % subsample:
            raise Exception(f'window size {W_part} is not divisible by subsample rate {subsample}')

        W_part_subsample = W_part // subsample

        # B x 2 x M x N
        # M point estimates from M partitions for N points
        part_gm_layer = spatialsoftmax.PartitionedGeometricMean(W_part, name='pts_targets', use_softmax=True)
        pts_targets = part_gm_layer(part_gm_map)

        # B x M x F
        # feature vector for each of the M partitions
        feats_targets = spatialsoftmax.PartitionsToFeatures(W_part_subsample, name='feats_targets')(part_feat_map)

        # B x M x N
        # a classification for each of the M partitions to hold one of the N points
        # each point can only appear in one partition
        logits_targets = keras.layers.Conv1D(
            filters=N,
            kernel_size=1,
            kernel_regularizer=keras.regularizers.l2(self.l2_weight),
        )(feats_targets)
        class_targets = keras.layers.Softmax(axis=1, name='class_targets')(logits_targets)

        # B x 1 x M x N
        _, _, M, _ = part_gm_layer.compute_output_shape((-1, H, W, N))
        class_targets = keras.layers.Reshape((1, M, N))(class_targets)

        # multiply the point estimates with the classification probability/weight
        # and sum them together to get one point estimate from all partitions
        # B x 2 x M x N
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
        feats_targets_layer = self.model.get_layer('feats_targets')
        class_targets_layer = self.model.get_layer('class_targets')
        # permute_1 layer - part_points, Nx2
        # permute_2 layer - part_feaures, NxF

        preview_model = keras.Model(
            inputs=self.model.inputs[0],
            outputs=[
                pts_targets_layer.input,
                pts_targets_layer.output,
                feats_targets_layer.input,
                class_targets_layer.output,
                self.model.output,
            ],
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
        data_config.pop('shuffle', None)
        data_config.pop('cache', None)
        data_config['image_loader'] = data_loader
        predict_data_seq = cdseq.CephDataSequence(data_config)

        # image_patch, kp_ref = predict_data_seq[0]
        for image_patch, kp_ref in predict_data_seq:
            pt_maps, pts_array, feats_maps, class_array, kp_est = preview_model.predict(image_patch)
            fig = pyplot.figure()
            ax = pyplot.subplot(121)
            ax_map = pyplot.subplot(122)
            ax.imshow(image_patch[0, :, :, 0])
            ax.plot(kp_est[0, 1, :], kp_est[0, 0, :], 'xr', label='predicted')
            ax.plot(kp_ref[0, 1, :], kp_ref[0, 0, :], 'ob', label='reference', picker=5)
            ax.plot(
                (kp_ref[0, 1, :], kp_est[0, 1, :]),
                (kp_ref[0, 0, :], kp_est[0, 0, :]),
                'g',
            )
            ax.legend(loc='best')

            def on_pick_event(event, ax):
                i = event.ind[0]
                ax.cla()
                ax_map.cla()
                ax.imshow(image_patch[0, :, :, 0])
                ax.plot(kp_est[0, 1, :], kp_est[0, 0, :], 'xr', label='predicted')
                ax.plot(kp_ref[0, 1, :], kp_ref[0, 0, :], 'ob', label='reference', picker=5)
                ax.plot(
                    (kp_ref[0, 1, :], kp_est[0, 1, :]),
                    (kp_ref[0, 0, :], kp_est[0, 0, :]),
                    'g',
                )

                ax_map.imshow(pt_maps[0, :, :, i] ** 0.1)
                ax_map.plot(kp_est[0, 1, i], kp_est[0, 0, i], 'xr', label='predicted')
                ax_map.scatter(pts_array[0, 1, :, i], pts_array[0, 0, :, i],
                               s=10 * class_array[0, :, i] + 0.1,
                               c=['g' if v > 1e-3 else 'r' for v in class_array[0, :, i]])

                ax.legend(loc='best')

            fig.canvas.mpl_connect('pick_event',
                                   functools.partial(on_pick_event, ax=ax))

    def predict(self, image, pts_target=None):
        part_gm_layer = self.model.layers[-7]
        predict_model = keras.Model(
            inputs=self.model.inputs[0],
            outputs=[
                part_gm_layer.input,
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
            kp_maps = []
            for i, pt_ind in enumerate(pts_list):
                image_patch, _ = predict_data_seq[i]
                output_map, pts_est = predict_model.predict(image_patch)

                pts_est = [[(*pts_est[0, :, n],) for n in range(pts_est.shape[2])]]

                pts_est_inv = predict_data_seq.invert_pts_modification(i, pts_est)
                pt = pts_est_inv[0][pt_ind]
                pts_result[0, pt_ind] = pt[0], pt[1], 0

                kp_maps.append(output_map[..., i] / output_map[..., i].max())

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
    KeyPointSpatialConfNetV3.main()
