import os
import sys
import copy
import functools

import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

sys.path += [os.path.abspath('../')]
try:
    import KPnet_config
except:
    print()
    print('missing commonlib and KPnet_config modules')
    print('try adding the root directory to PYTHONPATH environment variable')

from tensorflowlib.kpnetkeras import KeyPointKerasNetwork

import tensorflowlib.cephdataseq as cdseq
import commonlib.cephdataloaders as dataload
'''***'''
def modify_data_samples(sample):
    R=40
    image, pts = sample
    image = np.repeat(image, 3, axis=3)

    B, H, W, _ = image.shape
    _, _, N = pts.shape

    y_coord = np.arange(0, H)
    x_coord = np.arange(0, W)
    y_coord, x_coord = np.meshgrid(y_coord, x_coord, indexing='ij')

    x_coord.shape = (1,)+x_coord.shape+(1,)
    y_coord.shape = (1,)+y_coord.shape+(1,)

    y_pts = pts[:, 0, :].reshape(B, 1, 1, N)
    x_pts = pts[:, 1, :].reshape(B, 1, 1, N)

    y_offset = (y_pts-y_coord)/R
    x_offset = (x_pts-x_coord)/R

    heatmap = (y_offset**2+x_offset**2)<=1

    heatmaps_offsets = np.concatenate((heatmap, y_offset, x_offset), axis=-1)
    return image, heatmaps_offsets
'''***'''

class AttentiveFeature(tf.keras.layers.Layer):
    def __init__(self, out_chns, **kwargs):
        super(AttentiveFeature, self).__init__(**kwargs)
        self.out_chns = out_chns

    def get_config(self):
        conf_dict = super().get_config()
        conf_dict.update({
            'out_chns':self.out_chns
            })
        return conf_dict

    def build(self, inputs_shape):
        _, N, F = inputs_shape
        self.w2 = self.add_weight(
                shape=(1, self.out_chns, self.out_chns, N, 1),
                initializer='glorot_uniform',
                name='W1',
                )
        self.w1 = self.add_weight(
                shape=(1, self.out_chns, self.out_chns, 1),
                initializer='glorot_uniform',
                name='W2',
                )

    def call(self, inputs):
        feats = inputs
        B, N, F = feats.shape
        feats = tf.reshape(feats, (-1, 1, 1, N, F))
        feats = tf.multiply(feats, self.w2) # B x C x N x N x F
        feats = tf.reduce_sum(feats, axis=3) # B x C x N x F
        feats = tf.tanh(feats)

        feats = tf.multiply(feats, self.w1) # B x C x N x F
        feats = tf.reduce_sum(feats, axis=2) # B x C x F
        return tf.nn.softmax(feats, axis=2)

class HeatmapOffsetLoss(tf.keras.losses.Loss):
    def __init__(self, n_pts, **kwargs):
        super(HeatmapOffsetLoss, self).__init__(**kwargs)
        self.n_pts = n_pts

    def get_config(self):
        conf_dict = super().get_config()
        conf_dict.update({
            'n_pts':self.n_pts
            })
        return conf_dict

    def call(self, y_target, y_est):
        _, H, W, C_3 = y_est.shape
        C = self.n_pts

        heatmap_est = y_est[..., :C]
        y_off_est = y_est[..., C:2*C]
        x_off_est = y_est[..., 2*C:]
       
        heatmap_target = y_target[..., :C]
        y_off_target = y_target[..., C:2*C]
        x_off_target = y_target[..., 2*C:]

        loss_heatmap = tf.losses.binary_crossentropy(heatmap_target, heatmap_est, from_logits=True)
        loss_heatmap = tf.reduce_mean(loss_heatmap)

        loss_offset = tf.abs(x_off_est-x_off_target)+tf.abs(y_off_est-y_off_target)
        loss_offset = tf.reduce_sum(loss_offset*heatmap_target)/tf.reduce_sum(heatmap_target)

        return loss_heatmap*2/3 + loss_offset*1/3

class HeatmapLoss(tf.keras.losses.Loss):
    def __init__(self, n_pts, **kwargs):
        super(HeatmapLoss, self).__init__(**kwargs)
        self.n_pts = n_pts

    def get_config(self):
        conf_dict = super().get_config()
        conf_dict.update({
            'n_pts':self.n_pts
            })
        return conf_dict

    def call(self, y_target, y_est):
        _, H, W, C_3 = y_est.shape
        C = self.n_pts

        heatmap_est = y_est[..., :C]
        heatmap_target = y_target[..., :C]
        loss_heatmap = tf.losses.binary_crossentropy(heatmap_target, heatmap_est, from_logits=True)
        loss_heatmap = tf.reduce_mean(loss_heatmap)

        return loss_heatmap

class OffsetLoss(tf.keras.losses.Loss):
    def __init__(self, n_pts, **kwargs):
        super(OffsetLoss, self).__init__(**kwargs)
        self.n_pts = n_pts

    def get_config(self):
        conf_dict = super().get_config()
        conf_dict.update({
            'n_pts':self.n_pts
            })
        return conf_dict

    def call(self, y_target, y_est):
        _, H, W, C_3 = y_est.shape
        C = self.n_pts

        y_off_est = y_est[..., C:2*C]
        x_off_est = y_est[..., 2*C:]

        heatmap_target = y_target[..., :C]
        y_off_target = y_target[..., C:2*C]
        x_off_target = y_target[..., 2*C:]

        loss_offset = tf.abs(y_off_est-y_off_target)+tf.abs(x_off_est-x_off_target)
        loss_offset = tf.reduce_sum(loss_offset*heatmap_target)/tf.reduce_sum(heatmap_target)

        return tf.reduce_mean(loss_offset)


class ConvIdentityInit(tf.keras.initializers.Initializer):

    def __init__(self):
        pass

    def __call__(self, shape, dtype=None, **kwargs):
        H, W, C_in, C_out = shape
        assert C_in==C_out, f'{C_in} {C_out} are not identical'
        v = np.random.randn(H, W, C_in, C_out)/(C_in*10)
        #v = np.zeros((H, W, C_in, C_out))

        v[H//2, W//2, np.arange(C_in), np.arange(C_out)] = 1
        return tf.Variable(v, dtype=dtype)

    def get_config(self):  # To support serialization
        return {}

class AFPFNet(KeyPointKerasNetwork):
    '''
    '''

    def __init__(self):
        super().__init__()
        self.name='afpf'

        self.data_loader_conf = {
            'image_shape': (800, 640),
            'batch_size': 1,
            #'rotate': {'min': -30, 'max': 30},
            #'scale': {'min': 0.7, 'max': 1.3},
            #'map_sigma': 'bilinear',
            # 'intensity': {'min': 0.75, 'max': 1.25},
            # 'black_level': {'min': -0.25, 'max': 0.25},
            # 'points_selection' : (61,),
            'output_fun': modify_data_samples,
        }

    def generate_params_dict(self):
        del self.data_loader_conf['output_fun']
        params_dict = copy.deepcopy(super().generate_params_dict())
        self.data_loader_conf['output_fun'] = modify_data_samples
        return params_dict

    def apply_params_dict(self, params_dict):
        super().apply_params_dict(params_dict)
        self.data_loader_conf['output_fun'] = modify_data_samples

    def add_cmd_params(self, arg_parser):
        super().add_cmd_params(arg_parser)

    def set_cmd_params(self, args_env):
        super().set_cmd_params(args_env)
        self.data_loader_conf['output_fun'] = modify_data_samples

    def get_custom_layers_dict(self):
        custom_layers_dict = super().get_custom_layers_dict()
        custom_layers_dict['HeatmapOffsetLoss'] = HeatmapOffsetLoss
        custom_layers_dict['HeatmapLoss'] = HeatmapLoss
        custom_layers_dict['OffsetLoss'] = OffsetLoss
        custom_layers_dict['AttentiveFeature'] = AttentiveFeature
        custom_layers_dict['ConvIdentityInit'] = ConvIdentityInit
        return custom_layers_dict

    def generate_keras_callbacks(self):
        keras_callbacks = [c for c in super().generate_keras_callbacks() if not isinstance(c, tf.keras.callbacks.LearningRateScheduler)]
        return keras_callbacks

    def init_model(self):
        n_pts = self.n_pts

        # backbone 
        in_tens = tf.keras.Input((800, 640, 3))
        vgg19_backbone = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_tensor=in_tens)
        #vgg19_backbone.trainable=False
        feat_l1 = vgg19_backbone.get_layer('block2_pool').output
        feat_l2 = vgg19_backbone.get_layer('block3_pool').output
        feat_l3 = vgg19_backbone.get_layer('block4_pool').output
        feat_l4 = vgg19_backbone.get_layer('block5_pool').output

        feat_l1 = tf.keras.layers.Conv2D(filters=64, kernel_size=1)(feat_l1)
        feat_l2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1)(feat_l2)
        feat_l3 = tf.keras.layers.Conv2D(filters=64, kernel_size=1)(feat_l3)
        feat_l4 = tf.keras.layers.Conv2D(filters=64, kernel_size=1)(feat_l4)

        feat_l2 = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(feat_l2)
        feat_l3 = tf.keras.layers.UpSampling2D(size=4, interpolation='bilinear')(feat_l3)
        feat_l4 = tf.keras.layers.UpSampling2D(size=8, interpolation='bilinear')(feat_l4)

        feat_map = tf.keras.layers.Concatenate()((feat_l1, feat_l2, feat_l3, feat_l4))

        # dilated convolutional block
        # folowing F. Yu, V. Koltun, MULTI-SCALE CONTEXT AGGREGATION BY DILATED CONVOLUTIONS
        conv_block_map = feat_map
        feat_chns=feat_map.shape[3]
        for dilation, chn_mult in zip([1, 1, 2, 4, 8, 16, 1], [1, 1, 1, 1, 1, 1, 1]):
            ## TODO should be initialized to identity, whatever that means
            conv_block_map = tf.keras.layers.Conv2D(
                    filters=feat_chns*chn_mult,
                    kernel_size=3,
                    dilation_rate=dilation,
                    activation='relu',
                    padding='same',
                    kernel_initializer=ConvIdentityInit(),
                    )(conv_block_map)
        conv_block_map = tf.keras.layers.Conv2D(filters=feat_chns, kernel_size=1, dilation_rate=dilation)(conv_block_map)

        # attention maps
        feat_vects = conv_block_map
        feat_vects = tf.keras.layers.AveragePooling2D(pool_size=8)(feat_vects)
        feat_vects = tf.keras.layers.Reshape((25*20, feat_chns))(feat_vects)

        heatmap_af = AttentiveFeature(n_pts)(feat_vects)
        y_offset_af = AttentiveFeature(n_pts)(feat_vects)
        x_offset_af = AttentiveFeature(n_pts)(feat_vects)

        heatmap_af = tf.keras.layers.Reshape((1, 1, n_pts, feat_chns))(heatmap_af)
        y_offset_af = tf.keras.layers.Reshape((1, 1, n_pts, feat_chns))(y_offset_af)
        x_offset_af = tf.keras.layers.Reshape((1, 1, n_pts, feat_chns))(x_offset_af)

        # apply attention maps to feature images
        _, H, W, _ = conv_block_map.shape
        conv_block_map_ext = tf.keras.layers.Reshape((H, W, 1, feat_chns))(conv_block_map)

        heatmap_feat = tf.keras.layers.Multiply()([heatmap_af, conv_block_map_ext])*feat_chns
        y_offset_feat = tf.keras.layers.Multiply()([y_offset_af, conv_block_map_ext])*feat_chns
        x_offset_feat = tf.keras.layers.Multiply()([x_offset_af, conv_block_map_ext])*feat_chns

        heatmap = tf.keras.layers.Conv3D(filters=1, kernel_size=1)(heatmap_feat)
        y_offset = tf.keras.layers.Conv3D(filters=1, kernel_size=1)(y_offset_feat)
        x_offset = tf.keras.layers.Conv3D(filters=1, kernel_size=1)(x_offset_feat)

        heatmap = tf.keras.layers.Reshape((H, W, n_pts))(heatmap)
        y_offset = tf.keras.layers.Reshape((H, W, n_pts))(y_offset)
        x_offset = tf.keras.layers.Reshape((H, W, n_pts))(x_offset)
        output_tens = tf.keras.layers.Concatenate()((heatmap, y_offset, x_offset))

        # output_tens = tf.keras.layers.Conv2D(filters=n_pts*3*10, kernel_size=1, activation='relu')(conv_block_map)
        # output_tens = tf.keras.layers.Conv2D(filters=n_pts*3, kernel_size=1)(output_tens)

        output_tens_int = tf.keras.layers.UpSampling2D(size=4, interpolation='bilinear')(output_tens)
        self.model = tf.keras.Model(inputs=in_tens, outputs=output_tens_int)
        self.model.summary()
        self.model.compile(
                optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.9, epsilon=1e-6),
                loss=HeatmapOffsetLoss(n_pts),
                metrics=(HeatmapLoss(n_pts, name='heatmap_loss'),
                    OffsetLoss(n_pts, name='offset_loss')),
                )

    def preview(self, data):
        import matplotlib.pyplot as pyplot
        pyplot.ion()

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

        for image, target in predict_data_seq:
            prediction = self.model.predict(image)

            heatmap_pred = prediction[0, :, :, :self.n_pts]
            y_off_pred = prediction[0, :, :, self.n_pts:2*self.n_pts]
            x_off_pred = prediction[0, :, :, 2*self.n_pts:]

            vote_map = self.regression_voting_prediction(heatmap_pred, y_off_pred, x_off_pred)

            heatmap_pred[:] = 1/(1+np.exp(-heatmap_pred))

            max_ind = vote_map.reshape(-1, vote_map.shape[2]).argmax(0)
            max_pts = np.unravel_index(max_ind, vote_map.shape[:2])

            kp_ref = np.zeros((1, 2, self.n_pts))
            for n in range(self.n_pts):
                y_pts, x_pts = np.where(target[0, :, :, n])
                kp_ref[0, :, n] = np.mean(y_pts), np.mean(x_pts)

            fig, ax_table = pyplot.subplots(2, 3, sharex=True, sharey=True)
            ax_table = ax_table.ravel()
            ax_table[0].set_title('image')
            ax_table[0].imshow(image[0, ..., 0])
            ax_table[0].plot(max_pts[1], max_pts[0], 'xr', label='predicted')
            ax_table[0].plot(
                kp_ref[0, 1, :],
                kp_ref[0, 0, :],
                'ob',
                label='reference',
                picker=5,
                #pickradius=5.0,
                )

            ax_table[0].plot(
                (kp_ref[0, 1, :], max_pts[1]),
                (kp_ref[0, 0, :], max_pts[0]),
                'g')
            ax_table[0].legend(loc='best')

            ax_table[1].set_title('reference')
            ax_table[1].imshow(target[0, :, :, :self.n_pts].mean(2))
            ax_table[2].set_title('heatmap')
            ax_table[2].imshow(heatmap_pred.mean(2))
            ax_table[3].set_title('vote map')
            ax_table[3].imshow(vote_map.mean(2))

            def on_pick_event(event, ax_table, reference, heatmap, vote_map, y_offset, x_offset):
                i = event.ind[0]
                ax_table[1].cla()
                ax_table[1].set_title('reference')
                ax_table[1].imshow(reference[:, :, i])
                ax_table[2].cla()
                ax_table[2].set_title('heatmap')
                ax_table[2].imshow(heatmap[:, :, i])
                ax_table[3].cla()
                ax_table[3].set_title('vote map')
                ax_table[3].imshow(vote_map[:, :, i])

                ax_table[4].cla()
                ax_table[4].set_title('y offset')
                ax_table[4].imshow(y_offset[:, :, i])
                ax_table[5].cla()
                ax_table[5].set_title('vote map')
                ax_table[5].imshow(x_offset[:, :, i])

            fig.canvas.mpl_connect('pick_event',
                    functools.partial(
                        on_pick_event,
                        ax_table=ax_table,
                        reference=target[0, :, :, :],
                        heatmap=heatmap_pred,
                        vote_map=vote_map,
                        y_offset=y_off_pred,
                        x_offset=x_off_pred,
                        ))

            

    def regression_voting_prediction(self, heatmap, y_offset, x_offset, R=40):
        vote_map = np.zeros(heatmap.shape, dtype=np.int32)
        H, W, N = heatmap.shape
        P_max = int(np.pi*R**2+0.5)
        ind_lin = np.argsort(heatmap.reshape(H*W, N), 0)[-P_max:, :]
        y_pts, x_pts = np.unravel_index(ind_lin, (H, W))
        ind_pt = np.arange(N)

        y_pts_adj = y_pts + np.int32(y_offset[y_pts, x_pts, ind_pt]*R)
        x_pts_adj = x_pts + np.int32(x_offset[y_pts, x_pts, ind_pt]*R)

        ###
        mask = np.ones(heatmap.shape, dtype=bool)
        mask[y_pts, x_pts, ind_pt] = False
        y_offset[mask] = 0
        x_offset[mask] = 0
        ###

        outlier_mask = (y_pts_adj<0) + (x_pts_adj<0) + (y_pts_adj>=H) + (x_pts_adj>=W)

        y_pts_adj[outlier_mask] = -1
        x_pts_adj[outlier_mask] = -1


        for p in range(P_max):
            vote_map[y_pts_adj[p, :], x_pts_adj[p, :], ind_pt] += 1
        vote_map[-1, -1, :] = 0
        return vote_map
        

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

        image_patch, _ = predict_data_seq[0]
        prediction = predict_model.predict(image_patch)

        heatmap_pred = prediction[0, :, :, :self.n_pts]
        y_off_pred = prediction[0, :, :, self.n_pts:2*self.n_pts]
        x_off_pred = prediction[0, :, :, 2*self.n_pts:]

        vote_map = self.regression_voting_prediction(heatmap_pred, y_off_pred, x_off_pred)

        max_ind = vote_map.reshape(-1, vote_map.shape[2]).argmax(0)
        max_val = vote_map.reshape(-1, vote_map.shape[2]).max(0)
        max_pts = np.unravel_index(max_ind, vote_map.shape[:2])

        pts_est = [[*zip(*max_pts)]]
        pts_est_inv = predict_data_seq.invert_pts_modification(0, pts_est)

        for i, pt_ind in enumerate(pts_list):
            pt = pts_est_inv[0][pt_ind]
            pts_result[0, pt_ind] = pt[0], pt[1], max_val[i]

        kp_maps = vote_map.reshape(heatmap_pred.shape)

        return kp_maps, pts_result, pts_target


    def get_exportable_model(self):
        return self.model


if __name__ == '__main__':
    AFPFNet.main()
