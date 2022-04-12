'''
Modifiers for cephalometric datasets for data augmentation.
'''

import numpy as np
import copy
from .resize_and_labels_fun import rotate_and_scale_im_points, translate_im_points
import skimage.transform as sktransform

class RandFunGenerator():
    '''
    '''
    def __init__(self, rand_var_conf):
        conf_keys = set(rand_var_conf)
        if conf_keys == {'constant'}:
            self.cval = rand_var_conf['constant']
            self._call_proxy = self._constant
        elif conf_keys == {'min', 'max'}: 
            self.min_val = rand_var_conf['min']
            self.max_val = rand_var_conf['max']
            self._call_proxy = self._uniform
        elif conf_keys == {'mean', 'std'}: 
            self.mean_val = rand_var_conf['mean']
            self.std_val = rand_var_conf['std']
            self._call_proxy = self._normal
        else:
            raise Exception(f'random variable configuration {rand_var_conf} is not valid')

    def __call__(self, N):
        return self._call_proxy(N)

    def _constant(self, N):
        return np.ones(N)*self.cval

    def _uniform(self, N):
        rand_val = np.random.random((N,))
        rand_val *= (self.max_val-self.min_val)
        rand_val += self.min_val
        return rand_val

    def _normal(self, N):
        rand_val = np.random.randn(N)
        rand_val *= self.std_val
        rand_val += self.mean_val
        return rand_val

class CephDataModifier:
    '''
    '''
    def __init__(
            self,
            data_loader,
            rand_var_fun,
        ):
        self.data_loader = data_loader
        self.rand_var_fun = rand_var_fun
        self.rand_val = None
        self.generate_rand_values()

    def __len__(self):
        return len(self.data_loader)

    def __getitem__(self, ind):
        return self.data_loader[ind]

    def generate_rand_values(self):
        N = len(self.data_loader)
        self.rand_val = self.rand_var_fun(N)

    def invert_pts_modification(self, ind, pts):
        return self.data_loader.invert_pts_modification(ind, pts)

class NormalizeIntensityModifier(CephDataModifier):
    def __init__(
            self,
            data_loader,
            scale,
            offset
        ):
        self.data_loader = data_loader
        self.scale = scale
        self.offset = offset

    def __getitem__(self, ind):
        ceph_image, kp_map = self.data_loader[ind]

        ceph_image = ceph_image*self.scale+self.offset

        return ceph_image, kp_map



class CephDataBlackLevelMod(CephDataModifier):
    '''
    '''
    def __getitem__(self, ind):
        ceph_image, kp_map = self.data_loader[ind]
        black_level = self.rand_val[ind]

        ceph_image += black_level

        return ceph_image, kp_map

class CephDataIntensityMod(CephDataModifier):
    '''
    '''
    def __getitem__(self, ind):
        ceph_image, kp_map = self.data_loader[ind]
        int_mul = max(1e-2, self.rand_val[ind])
        ceph_image *= int_mul

        return ceph_image, kp_map

class CephDataGammaMod(CephDataModifier):
    '''
    '''
    def __getitem__(self, ind):
        ceph_image, kp_map = self.data_loader[ind]
        gamma = max(1e-2, self.rand_val[ind])
        ceph_image **= (1/gamma)

        return ceph_image, kp_map

class CephDataRotateScaleMod(CephDataModifier):
    '''
    '''
    def __init__(
            self,
            data_loader,
            scale_rand_var_fun,
            rotate_rand_var_fun,
        ):
        self.data_loader = data_loader
        self.rotate_rand_var_fun = rotate_rand_var_fun
        self.scale_rand_var_fun = scale_rand_var_fun
        self.rand_val = None
        self.generate_rand_values()
        self.remove_invalid = False

    def __getitem__(self, ind):
        ceph_image, kp_list = self.data_loader[ind]
        theta = self.rotate_rand_val[ind]
        scale_fact = self.scale_rand_val[ind]

        _, H, W, _ = ceph_image.shape

        ceph_image_tr, kp_array_tr = rotate_and_scale_im_points(
                ceph_image[0, :, :, 0],
                np.array(kp_list[0]).T,
                theta,
                scale_fact)

        ceph_image_tr = ceph_image_tr.reshape((1, H, W, 1))
        kp_list_tr = [[]]
        for n in range(len(kp_list[0])):
            y, x = kp_array_tr[:, n]
            if self.remove_invalid and (y<0 or y>=H or x<0 or x>=W):
                kp_list_tr[0].append((np.nan, np.nan))
            else:
                kp_list_tr[0].append((y, x))

        return ceph_image_tr, kp_list_tr

    def generate_rand_values(self):
        N = len(self.data_loader)
        self.rotate_rand_val = self.rotate_rand_var_fun(N)
        self.scale_rand_val = self.scale_rand_var_fun(N)

    def invert_pts_modification(self, ind, pts):
        ceph_image, _ = self.data_loader[ind]
        theta = self.rotate_rand_val[ind]
        scale_fact = self.scale_rand_val[ind]

        ceph_image_tr, kp_array_tr = rotate_and_scale_im_points(
                ceph_image[0, :, :, 0],
                np.array(pts[0]).T,
                -theta,
                1/scale_fact)
        pts = [[(y, x) for y, x in kp_array_tr.T]]

        return self.data_loader.invert_pts_modification(ind, pts)

class CephDataTranslateMod(CephDataModifier):
    '''
    '''
    def __init__(
            self,
            data_loader,
            translate_rand_var_fun,
        ):
        self.data_loader = data_loader
        self.translate_rand_var_fun = translate_rand_var_fun
        self.trans_rand_vec = None
        self.generate_rand_values()
        self.remove_invalid = False

    def __getitem__(self, ind):
        ceph_image, kp_list = self.data_loader[ind]
        trans_vec = self.trans_rand_vec[ind]

        _, H, W, _ = ceph_image.shape

        ceph_image_tr, kp_array_tr = translate_im_points(
                ceph_image[0, :, :, 0],
                np.array(kp_list[0]).T,
                trans_vec)

        ceph_image_tr = ceph_image_tr.reshape((1, H, W, 1))
        kp_list_tr = [[]]
        for n in range(len(kp_list[0])):
            y, x = kp_array_tr[:, n]
            if self.remove_invalid and (y<0 or y>=H or x<0 or x>=W):
                kp_list_tr[0].append((np.nan, np.nan))
            else:
                kp_list_tr[0].append((y, x))

        return ceph_image_tr, kp_list_tr

    def generate_rand_values(self):
        N = len(self.data_loader)
        self.trans_rand_vec = self.translate_rand_var_fun(N*2).reshape(N, 2)

    def invert_pts_modification(self, ind, pts):
        dy, dx = self.trans_rand_vec[ind]

        pts = [[(y-dy, x-dx) for y, x in pts]]

        return self.data_loader.invert_pts_modification(ind, pts)

class PiecewiseAffineWarpModifier(CephDataModifier):
    def __init__(
            self,
            data_loader,
            rand_var_fun,
            vert_split=8,
            horz_split=8,
        ):
        self.vert_split = vert_split
        self.horz_split = horz_split
        self.rand_tr_vec = None
        super().__init__(data_loader, rand_var_fun)

    def __getitem__(self, ind):
        ceph_image, kp_list = self.data_loader[ind]
        trans_vecs = self.rand_tr_vec[ind]

        _, H, W, _ = ceph_image.shape

        y_src = np.linspace(0, H, self.vert_split+1)
        x_src = np.linspace(0, W, self.horz_split+1)
        y_src, x_src = np.meshgrid(y_src, x_src)

        pts_src = np.array([y_src, x_src]).transpose(1, 2, 0)
        pts_dst = pts_src.copy()
        pts_dst[1:-1, 1:-1, :] += trans_vecs.transpose(1, 0, 2)

        pts_src = pts_src.reshape(-1, 2)
        pts_dst = pts_dst.reshape(-1, 2)

        tform = sktransform.PiecewiseAffineTransform()
        tform.estimate(pts_src, pts_dst)

        ceph_image_tr = sktransform.warp(ceph_image[0, :, :, 0], tform)
        ceph_image_tr = ceph_image_tr.reshape(1, H, W, 1)

        kp_list = np.array(kp_list)
        kp_list_tr = tform.inverse(kp_list[0, :, ::-1])

        kp_list_tr = [[(y, x) for x, y in kp_list_tr]]

        return ceph_image_tr, kp_list_tr

    def generate_rand_values(self):
        N = len(self.data_loader)
        M_vert = self.vert_split-1
        M_horz = self.horz_split-1
        self.rand_tr_vec = self.rand_var_fun(N*M_vert*M_horz*2).reshape(N, M_vert, M_horz, 2)

    def invert_pts_modification(self, ind, pts):
        #raise Exception(f'inverse for PiecewiseAffineWarpModifier is broken')
        trans_vecs = self.rand_tr_vec[ind]
        ceph_image, _ = self.data_loader[ind]

        _, H, W, _ = ceph_image.shape

        y_src = np.linspace(0, H, self.vert_split+1)
        x_src = np.linspace(0, W, self.horz_split+1)
        y_src, x_src = np.meshgrid(y_src, x_src)

        pts_src = np.array([y_src, x_src]).transpose(1, 2, 0)
        pts_dst = pts_src.copy()
        pts_dst[1:-1, 1:-1, :] += trans_vecs.transpose(1, 0, 2)

        pts_src = pts_src.reshape(-1, 2)
        pts_dst = pts_dst.reshape(-1, 2)

        tform = sktransform.PiecewiseAffineTransform()
        tform.estimate(pts_dst, pts_src)

        pts_ar = np.array(pts)
        pts_tr_ar = tform.inverse(pts_ar[0, :, ::-1])

        pts_tr = [[(y, x) for x, y in pts_tr_ar]]

        return self.data_loader.invert_pts_modification(ind, pts_tr)


class PointResampleDataModifier(CephDataModifier):
    '''
    '''
    def __init__(
            self,
            data_loader,
            sample_size,
            translate_rand_var_fun=None,
            ):
        _, pts_list = data_loader[0]
        self.data_loader = data_loader
        self.n_pts = len(pts_list[0])
        self.sample_size = sample_size

        if translate_rand_var_fun is None:
            translate_rand_var_fun = RandFunGenerator({'constant': 0})
        self.translate_rand_var_fun = translate_rand_var_fun
        self.trans_rand_vec = None

        self.cached_data_ind = None
        self.cached_data = None
        self.remove_invalid = False

        self.generate_rand_values()

    def __len__(self):
        return len(self.data_loader)*self.n_pts

    def get_sample_offsets(self, data_ind, pt_ind):
        if self.cached_data_ind == data_ind:
            ceph_image, kp_list = self.cached_data
        else:
            ceph_image, kp_list = self.data_loader[data_ind]
            self.cached_data_ind = data_ind
            self.cached_data = ceph_image, kp_list

        _, H, W, _ = ceph_image.shape
        if kp_list[0][pt_ind] == (np.nan, np.nan):
            y, x = H/2, W/2
        else:
            y, x = kp_list[0][pt_ind]
        s_height, s_width = self.sample_size

        # add random offset
        dy, dx = self.trans_rand_vec[data_ind, pt_ind, :]
        y += dy
        x += dx
        # keep the offset inside the image
        y = min(max(y, 0), H-1)
        x = min(max(x, 0), W-1)

        t, l = 0, 0 #top, left
        b, r = self.sample_size # bottom, right

        y0, x0 = int(y)-(s_height-1)//2, int(x)-(s_width-1)//2
        y1, x1 = y0+s_height, x0+s_height

        if y0 < 0:
            t += -y0
            y0 = 0
        if x0 < 0:
            l += -x0
            x0 = 0
        if y1 >= H:
            b -= y1-H
            y1 = H
        if x1 >= W:
            r -= x1-W
            x1 = W

        return t, l, b, r, y0, x0, y1, x1

    def __getitem__(self, ind):
        data_ind = ind//self.n_pts
        pt_ind = ind%self.n_pts

        t, l, b, r, y0, x0, y1, x1 = self.get_sample_offsets(data_ind, pt_ind)
        ceph_image, kp_list = self.cached_data

        ceph_sample = np.zeros((1,)+self.sample_size+(1,), np.float32)
        ceph_sample[0, t:b, l:r, 0] = ceph_image[0, y0:y1, x0:x1, 0]
        kp_sample_list = [[]]
        for y, x in kp_list[0]:
            y, x = y-y0+t, x-x0+l
            # find the ones outside the image
            if(self.remove_invalid and 
            (y<0 or y>=self.sample_size[0] or 
             x<0 or x>=self.sample_size[1])):
                kp_sample_list[0].append((np.nan, np.nan))
            else:
                kp_sample_list[0].append((y, x))


        return ceph_sample, kp_sample_list

    def invert_pts_modification(self, ind, pts):
        data_ind = ind//self.n_pts
        pt_ind = ind%self.n_pts

        t, l, _, _, y0, x0, _, _ = self.get_sample_offsets(data_ind, pt_ind)

        pts = [[(y-t+y0, x-l+x0) for y, x in pts[0]]]

        return self.data_loader.invert_pts_modification(data_ind, pts)

    def generate_rand_values(self):
        N = len(self.data_loader)
        P = self.n_pts
        self.trans_rand_vec = self.translate_rand_var_fun(N*P*2)
        self.trans_rand_vec = self.trans_rand_vec.reshape(N, P, 2)

class SubselectDataModifier(CephDataModifier):
    '''
    '''
    def __init__(self, data_loader, select_fraction):
        self.data_loader = data_loader
        self.n_orig = len(data_loader)
        self.n_sel = int(self.n_orig*select_fraction)
        self.sel_inds = None
        self.generate_rand_values()

    def __len__(self):
        return self.n_sel

    def __getitem__(self, ind):
        return self.data_loader[self.sel_inds[ind]]

    def invert_pts_modification(self, ind, pts):
        return self.data_loader.invert_pts_modification(ind, self.sel_inds[pts])

    def generate_rand_values(self):
        sel_inds = np.random.permutation(self.n_orig)
        sel_inds = sel_inds[:self.n_sel]
        self.sel_inds = np.sort(sel_inds)

class CacheDataModifier(CephDataModifier):
    def __init__(self, data_loader):
        self.cache = {}
        super().__init__(data_loader, None)

    def __getitem__(self, ind):
        if ind in self.cache:
            return copy.deepcopy(self.cache[ind])
        else:
            data = self.data_loader[ind]
            self.cache[ind] = data
            return copy.deepcopy(data)

    def generate_rand_values(self):
        pass


class ShuffleDataModifier(CephDataModifier):
    def __init__(self, data_loader):
        super().__init__(data_loader, None)

    def __getitem__(self, ind):
        return self.data_loader[self.ind_shuffle[ind]]

    def generate_rand_values(self):
        self.ind_shuffle = np.random.permutation(len(self.data_loader))

class CenterPointsDataModifier(CephDataModifier):
    def __init__(self, data_loader):
        super().__init__(data_loader, None)

    def __getitem__(self, ind):
        ceph_image, kp_list = self.data_loader[ind]

        _, H, W, _ = ceph_image.shape

        kp_list_tr = [[(y-(H-1)/2, x-(W-1)/2) for y, x in kp_list[0]]]

        return ceph_image, kp_list_tr

    def invert_pts_modification(self, ind, pts):
        ceph_image, _ = self.data_loader[ind]

        _, H, W, _ = ceph_image.shape

        pts = [[(y+(H-1)/2, x+(W-1)/2) for y, x in pts[0]]]

        return self.data_loader.invert_pts_modification(ind, pts)

    def generate_rand_values(self):
        pass

class AlignBBoxModifier(CephDataModifier):
    def __init__(
            self,
            data_loader,
            border_frac,
        ):
        super().__init__(data_loader, None)
        self.grow_fact = 1+2*border_frac

    def __getitem__(self, ind):
        ceph_image, kp_list = self.data_loader[ind]
        kp_list_array = np.array(kp_list)
        B, H, W, C = ceph_image.shape

        bb_b, bb_r = kp_list_array.max(1)[0]
        bb_t, bb_l = kp_list_array.min(1)[0]
        bb_h = int(bb_b-bb_t)
        bb_w = int(bb_r-bb_l)
        bb_center = int(bb_t+bb_h//2), int(bb_l+bb_w//2)

        bb_diff = int((max(bb_h, bb_w)*self.grow_fact)/2)
        ceph_image_bbox = np.zeros((B, bb_diff*2+1, bb_diff*2+1, C))

        y0, y1, x0, x1 = (
                bb_center[0]-bb_diff, # top
                bb_center[0]+bb_diff+1, # bottom
                bb_center[1]-bb_diff, # left 
                bb_center[1]+bb_diff+1, # right
                )
        t, b, l, r = 0, bb_diff*2+1, 0, bb_diff*2+1

        if y0 < 0:
            t += -y0
            y0 = 0
        if x0 < 0:
            l += -x0
            x0 = 0
        if y1 >= H:
            b -= y1-H
            y1 = H
        if x1 >= W:
            r -= x1-W
            x1 = W

        ceph_image_bbox[:, t:b, l:r, :] = ceph_image[:, y0:y1, x0:x1, :]

        kp_list_bbox = [[(y-y0+t, x-x0+l) for y, x in kp_list[0]]]

        return ceph_image_bbox, kp_list_bbox

    def generate_rand_values(self):
        pass

    def invert_pts_modification(self, ind, pts):
        ceph_image, kp_list = self.data_loader[ind]
        kp_list_array = np.array(kp_list)
        B, H, W, C = ceph_image.shape

        bb_b, bb_r = kp_list_array.max(1)[0]
        bb_t, bb_l = kp_list_array.min(1)[0]
        bb_h = int(bb_b-bb_t)
        bb_w = int(bb_r-bb_l)
        bb_center = int(bb_t+bb_h//2), int(bb_l+bb_w//2)

        bb_diff = int((max(bb_h, bb_w)*self.grow_fact)/2)

        y0, y1, x0, x1 = (
                bb_center[0]-bb_diff, # top
                bb_center[0]+bb_diff+1, # bottom
                bb_center[1]-bb_diff, # left 
                bb_center[1]+bb_diff+1, # right
                )
        t, b, l, r = 0, bb_diff*2+1, 0, bb_diff*2+1

        if y0 < 0:
            t += -y0
            y0 = 0
        if x0 < 0:
            l += -x0
            x0 = 0
        if y1 >= H:
            b -= y1-H
            y1 = H
        if x1 >= W:
            r -= x1-W
            x1 = W

        pts = [[(y+y0-t, x+x0-l) for y, x in pts[0]]]
        return self.data_loader.invert_pts_modification(ind, pts)


def main_test():
    '''
    test this module
    '''
    plot_samples = True

    if plot_samples:
        import matplotlib.pyplot as pyplot
    import sys
    import time

    from .cephdataloaders import (
        CephDataLoader,
        DataResizeLoader,
        KeypointMapDataLoader,
        DataBatchLoader,
        )

    data_loader = CephDataLoader.fromDirs(sys.argv[1])
    data_loader_orig = data_loader

    data_loader = DataResizeLoader(data_loader, (512, 512))

    def gen_rand_val(m, M, N):
        #class rand_val():
        #    def __init__(self, m, M, N):
        #        self.arr = np.random.uniform(m, M, N)
        #    def __getitem__(self, i):
        #        v = self.arr[i]
        #        #print(v)
        #        return v
        #return rand_val(m, M, N) 
        return np.random.uniform(m, M, N)

    piecewise_mod = PiecewiseAffineWarpModifier(data_loader, lambda N: gen_rand_val(-5, 5, N))
    data_loader = piecewise_mod

    rotscale_mod = CephDataRotateScaleMod(data_loader,
        lambda N: gen_rand_val(0.9, 1.1, N),
        lambda N: gen_rand_val(-10., 10., N))
    data_loader = rotscale_mod

    #pt_resample_mod = PointResampleDataModifier(data_loader,
    #        sample_size=(128, 128))
    #data_loader = pt_resample_mod

    #translate_mod = CephDataTranslateMod(data_loader,
    #    lambda N: gen_rand_val(-32, 32, N))
    #data_loader = translate_mod

    #bl_mod = CephDataBlackLevelMod(data_loader,
    #    lambda N: gen_rand_val(0, 0.5, N))
    #data_loader = bl_mod
    #int_mod = CephDataIntensityMod(data_loader,
    #    lambda N: gen_rand_val(0.5, 1.5, N))
    #data_loader = int_mod
    #gamma_mod = CephDataGammaMod(data_loader,
    #    lambda N: gen_rand_val(0.5, 1.5, N))
    #data_loader = gamma_mod

    modifiers_list = [
        #bl_mod,
        #int_mod,
        #gamma_mod,
        rotscale_mod,
        #translate_mod,
        piecewise_mod,
        ]

    #B = 2 
    #data_loader = DataBatchLoader(data_loader, B)
    #data_loader = KeypointMapDataLoader(data_loader, 2.0)


    t0 = time.time()
    while True:
        print('click image to go to next')
        print('press button to restart')
        print('press Ctrl+C to stop')
        if plot_samples:
            pyplot.close('all')
            fig = pyplot.figure()

        for modifier in modifiers_list:
            modifier.generate_rand_values()

        #for img_batch, kp_map_batch in data_loader:
        for ind in range(len(data_loader)):
            #img_batch, kp_map_batch = data_loader[ind]
            img_batch, kp_batch = data_loader[ind]
            img_orig, kp_orig = data_loader_orig[ind]

            kp_inv = data_loader.invert_pts_modification(ind, kp_batch)

            kp_batch = np.array(kp_batch)
            kp_orig = np.array(kp_orig)
            kp_inv = np.array(kp_inv)

            dt = time.time()-t0
            t0 = time.time()
            print(f'image: {img_batch.shape},time: {dt:.3f}s')
            if plot_samples:
                pyplot.clf()
                _, ax = pyplot.subplots(1, 2, num=fig.number)

                ax[0].imshow(img_batch[0, :, :, 0])
                ax[0].plot(kp_batch[0, :, 1], kp_batch[0, :, 0], 'x')
                ax[0].set_title(f'im tr {ind}')

                ax[1].imshow(img_orig[0, :, :, 0])
                ax[1].plot(kp_orig[0, :, 1], kp_orig[0, :, 0], 'x')
                ax[1].plot(kp_inv[0, :, 1], kp_inv[0, :, 0], 'o')
                ax[1].set_title(f'im orig {ind}')

                #for b in range(img_batch.shape[0]):
                #    ax[b, 0].imshow(img_batch[b, :, :, 0]+kp_map_batch[b, :, :].max(2))
                #    ax[b, 0].set_title(f'im {b}')
                #    ax[b, 1].imshow(kp_map_batch[b, :, :].max(2))
                #    ax[b, 1].set_title(f'map {b}')

                pyplot.draw()
                if pyplot.waitforbuttonpress():
                    break

if __name__ == '__main__':
    main_test()
