'''
Loaders and Generators for cephalometric datasets.
'''
import os
import re
import numpy as np
import scipy.ndimage as ndimage
import PIL.Image

from .resize_and_labels_fun import (
        getPointsFileFromImg,
        readPointsInTxt,
        resizeAndPadImageArray,
        resizePoints,
        calcResizeRatio
        )

class CephDataLoader:
    '''
    The base CephDataLoader that reads the list of images
    in the source directory and than loads each individual image
    and it's coreponding keypoints.

    source_dir - str
        path of directory containing the images. The labels
        are automatically located relative to this directory.

    '''
    def __init__(
            self,
            img_files_list,
            pts_files_list=None,
        ):

        if pts_files_list is None:
            pts_files_list = [getPointsFileFromImg(f) for f in img_files_list]

        self.img_files_list = img_files_list
        self.pts_files_list = pts_files_list
        self.files_count = len(img_files_list)

    def __len__(self):
        return self.files_count

    @classmethod
    def fromDirs(
            cls,
            images_dir,
            pts_dir=None,
            img_file_pattern='.*png',
            pts_file_pattern='.*points.txt',
        ):
        '''
        '''
        img_files_list = [
            f
            for f in os.listdir(images_dir)
            if re.fullmatch(img_file_pattern, f, re.IGNORECASE)
            ]
        img_files_list.sort()
        img_files_list = [
            os.path.join(images_dir, f)
            for f in img_files_list
            ]

        if pts_dir is not None:
            pts_files_list = [
                f
                for f in os.listdir(pts_dir)
                if re.fullmatch(pts_file_pattern, f, re.IGNORECASE)
                ]
            pts_files_list.sort()
            pts_files_list = [
                os.path.join(pts_dir, f)
                for f in pts_files_list
                ]
        else:
            pts_files_list = None

        return cls(img_files_list, pts_files_list)

    @classmethod
    def fromFiles(
            cls,
            images_list_file,
            pts_list_file=None,
        ):
        '''
        '''
        img_files_list = []
        pts_files_list = []

        if isinstance(images_list_file, str):
            images_list_file = [images_list_file]
        if isinstance(pts_list_file, str):
            pts_list_file = [pts_list_file]



        for f in images_list_file:
            images_dir = os.path.dirname(f)
            with open(f, 'r') as in_file:
                img_files_list_part = in_file.readlines()
            img_files_list_part = [f.rstrip() for f in img_files_list_part]
            img_files_list_part = [
                os.path.join(images_dir, f)
                for f in img_files_list_part
                ]
            img_files_list += img_files_list_part

        if pts_list_file is not None:
            for f in pts_list_file:
                pts_dir = os.path.dirname(f)
                with open(f, 'r') as in_file:
                    pts_files_list_part = in_file.readlines()
                pts_files_list_part = [f.rstrip() for f in pts_files_list_part]
                pts_files_list_part = [
                    os.path.join(pts_dir, f)
                    for f in pts_files_list_part
                    ]
                pts_files_list += pts_files_list_part
        else:
            pts_files_list = None

        return cls(img_files_list, pts_files_list)


    def __getitem__(self, idx):
        '''
        Returns
        ceph_image, point_map
        ceph_image - np.ndarray <BxHxW>, float32
            Source image, XRay of face.

        points_list - list [[(float, float)],]
            List of lists of 2D pairs of cephalometric points in the image.
            The outter list contains batches of images. The simple loader always 
            returns 1 image per batch.
            The inner list contains points in a single image.
            Missing points may be marked as None.

        '''

        img_file_path = self.img_files_list[idx]
        ceph_image = np.array(PIL.Image.open(img_file_path))

        if ceph_image.dtype == np.uint8:
            ceph_image = ceph_image/255
        if ceph_image.ndim == 3:
            ceph_image = ceph_image.mean(2)


        ceph_image = ceph_image.reshape((1,)+ceph_image.shape+(1,))

        pts_file_path = self.pts_files_list[idx]
        points_array = readPointsInTxt(pts_file_path, ',', pointCount=None)

        points_list = [[(x, y) for x, y in points_array.T]]

        return np.float32(ceph_image), points_list

    def get_file_name(self, idx):
        '''
        idx - int
            index of item

        returns:
        str, str - path to image, path to points file
        '''
        return self.img_files_list[idx], self.pts_files_list[idx]

    def random_train_split(self, train_frac):
        '''
        Split this data loader into even partitions and use the specified partition.

        val_frac - array of two integers
            First value specifies number of partitions to generate. Second value specifies
            which partition to use.

        Returns:
            CephDataLoader with a specified partition of a training dataset.
        '''

        files_ind = np.random.RandomState(seed=123).permutation(self.files_count)
        train_files_ind = np.array_split(files_ind, train_frac[0])[train_frac[1]]

        train_img_list = [self.img_files_list[i] for i in train_files_ind]
        train_pts_list = [self.pts_files_list[i] for i in train_files_ind]

        train_data_loader = CephDataLoader(train_img_list, train_pts_list)

        return train_data_loader

    def random_validation_split(self, val_frac=0.25, rand_seed=None):
        '''
        Split this data loader into two non-overlaping sets.

        val_frac - float, in the range (0, 1)
            The fraction of images to put in the validation set. The actual 
            number of images split is rounded.
        rand_seed - int or None
            seed for np.random.RandomState(seed).permutation()
            if None, np.random.permutation() will be used

        Returns:
            train_data, valid_data
            Two new CephDataLoader objects.
        '''
        valid_count = int(self.files_count*val_frac+0.5)
        if rand_seed is None:
            files_ind = np.random.permutation(self.files_count)
        else:
            files_ind = np.random.RandomState(seed=rand_seed).permutation(self.files_count)
        valid_files_ind = files_ind[:valid_count]
        train_files_ind = files_ind[valid_count:]

        valid_files_ind.sort()
        train_files_ind.sort()

        train_img_list = [self.img_files_list[i] for i in train_files_ind]
        train_pts_list = [self.pts_files_list[i] for i in train_files_ind]
        valid_img_list = [self.img_files_list[i] for i in valid_files_ind]
        valid_pts_list = [self.pts_files_list[i] for i in valid_files_ind]

        train_data_loader = CephDataLoader(train_img_list, train_pts_list)
        valid_data_loader = CephDataLoader(valid_img_list, valid_pts_list)

        return train_data_loader, valid_data_loader

    def invert_pts_modification(self, ind, pts):
        return pts

class CephDataWrapper:
    '''
    A simple wrapper around preloaded data, that is compatible with CephDataLoader.
    '''

    def __init__(
            self,
            images_list,
            pts_list,
        ):
        self.images_list = images_list
        self.pts_list = pts_list

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        return self.images_list[idx], [self.pts_list[idx]]


    def get_file_name(self, idx):
        return f'image_{idx+1}'

    def random_validation_split(self, val_frac=0.25):
        return None, None

    def invert_pts_modification(self, ind, pts):
        return pts

class DataResizeLoader:
    '''
    target_size - Tuple[int, int]
        Target size for all images.
    kpoint_generator -
        Generator or iterable that contains/returns:
            image, keypoint list

        Both image and keypoint list resized to the requested
        target_size.
    '''
    def __init__(self, data_loader, target_size):
        self.data_loader = data_loader
        self.target_size = target_size

    def __len__(self):
        return len(self.data_loader)

    def __getitem__(self, ind):
        ceph_image, points_list = self.data_loader[ind]
        ceph_image_res = resizeAndPadImageArray(
            ceph_image[0, ..., 0],
            self.target_size[0],
            self.target_size[1],
            )
        points_list_res = resizePoints(
            ceph_image[0, ..., 0],
            points_list[0],
            self.target_size[0],
            self.target_size[1],
            )

        ceph_image_res = ceph_image_res.reshape(
            (1,)+ceph_image_res.shape+(1,),
            )

        return np.float32(ceph_image_res), [points_list_res]

    def invert_pts_modification(self, ind, pts):
        ceph_image, _ = self.data_loader[ind]
        h_src, w_src = ceph_image.shape[1:3]
        h_tar, w_tar = self.target_size

        scale_fact = calcResizeRatio((h_src, w_src), (h_tar, w_tar))
        pts = [[(y/scale_fact, x/scale_fact) for y, x in pts[0]]]
        return self.data_loader.invert_pts_modification(ind, pts)


class KeypointMapDataLoader:
    '''
    gaussian_sigma - float or str
        If float, then sigma is used to generate keypoint maps.
        Each point will be represented by a gaussian kernel.
        If string, then if:
            - 'single' - each point is represented by a 1 at the rounded location.
            - 'bilinear' - each point is represented with up to 4 neighbouring
                           pixels that sum up to 1.

    softmax_target - bool
        Add an aditional channel and adapt it for a softmax target.

    kpoint_generator -
        Generator or iterable that contains/returns:
            image, keypoint_list

        image - numpy.ndarray, <HxW>
            Image from dataset.

        keypoint_list - List[Tuple[float, float]]
            Coresponding keypoints for the image.
            Each keypoint is represented by two floating point coordinates,
            or None if keypoint is missing in the image.


    generates/yields:
        image, kp_map

        image - numpy.ndarray <HxW>
            The same image retrieved from the iterable.

        kp_map - numpy.ndarray <HxWxN>
            Map of keypoints, where N is the number of keypoints
            returned from the iterable.
    '''
    def __init__(self, data_loader, gaussian_sigma=None, softmax_target=False):
        self.data_loader = data_loader
        self.softmax_target = softmax_target
        self.gaussian_sigma = gaussian_sigma

        assert type(gaussian_sigma) is float or \
               type(gaussian_sigma) is str and gaussian_sigma in ('single', 'bilinear'), \
               'map_sigma -> gaussian_sigma argument is invalid!'

        if type(gaussian_sigma) is float:
            patch_size = int(gaussian_sigma*8)
            kp_patch = np.zeros((patch_size*2+1, patch_size*2+1))
            kp_patch[patch_size, patch_size] = 1.
            kp_patch = ndimage.gaussian_filter(
                kp_patch,
                gaussian_sigma,
                mode='constant',
                cval=0,
                truncate=8,
                )
            kp_patch /= kp_patch.max()
            self.kp_patch = kp_patch
        else:
            self.kp_patch = None

    def __len__(self):
        return len(self.data_loader)

    def __getitem__(self, ind):
        ceph_image, points_list = self.data_loader[ind]

        _, H, W, _ = ceph_image.shape
        B = len(points_list)
        N = len(points_list[0])
        if self.softmax_target:
            kp_map = np.zeros((B, H, W, N+1), dtype=np.float32)
        else:
            kp_map = np.zeros((B, H, W, N), dtype=np.float32)

        if self.kp_patch is not None:
            kp_map[:] = self.kp_patch.min()

        for n_batch in range(B):
            for n in range(N):
                if points_list[n_batch][n] != (np.nan, np.nan):
                    y, x = points_list[n_batch][n]
                    if self.gaussian_sigma == 'single':
                        # Some points are completely out of image bounds.
                        if not(H > y > 0 and W > x > 0):
                            continue

                        kp_map[n_batch, int(y), int(x), n] = 1.
                    elif self.gaussian_sigma == 'bilinear':
                        # Some points are completely out of image bounds.
                        if not(H > y > 0 and W > x > 0):
                            continue

                        x_frac, y_frac = np.modf(x)[0], np.modf(y)[0]
                        x_int, y_int = int(x), int(y)

                        f_x = np.asarray([1 - x_frac, x_frac]).reshape((2, 1))
                        f_y = np.asarray([1 - y_frac, y_frac]).reshape((1, 2))
                        kp_patch = f_x @ f_y

                        y_max, x_max = min(y_int + 2, H), min(x_int + 2, W)
                        kp_map[n_batch, y_int:y_max, x_int:x_max, n] = kp_patch[0:y_max - y_int, 0:x_max - x_int]
                    else:
                        t, l = 0, 0 #top, left
                        b, r = self.kp_patch.shape # bottom, right

                        patch_size = b

                        y0 = int(y)-(patch_size-1)//2
                        y1 = y0+patch_size
                        x0 = int(x)-(patch_size-1)//2
                        x1 = x0+patch_size

                        if y0 < 0:
                            t += -y0
                            y0 = 0
                        if x0 < 0:
                            l += -x0
                            x0 = 0
                        if y1 > H:
                            b -= y1-H
                            y1 = H
                        if x1 > W:
                            r -= x1-W
                            x1 = W

                        # patch ouside of bounds?
                        if y1<y0 or x1<x0:
                            continue

                        kp_map[n_batch, y0:y1, x0:x1, n] = self.kp_patch[t:b, l:r]

        if self.softmax_target:
            kp_map[..., -1] = (1-kp_map).prod(3)
            kp_map /= kp_map.sum(3, keepdims=True)+1e-5

        return ceph_image, kp_map

    def invert_pts_modification(self, ind, pts):
        return self.data_loader.invert_pts_modification(ind, pts)

class KeypointWeightsDataLoader:
    '''
    Takes a CephDataLoader that loads images with lists of keypoints. It
    packages the keypoints into an array, with the shape Bx2xC, where B is the 
    size of the batch and C is the number of points. B is at least 1.
    
    It also generates an array of weights with the shape Bx2xC. If any
    point is missing (it's coordinates are np.nan), then it's weight is set to 0.

    Accesing an element returns: image, keypoint_array, weight_array
    '''
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __len__(self):
        return len(self.data_loader)

    def __getitem__(self, ind):
        ceph_image, points_list = self.data_loader[ind]

        points_array = np.array(points_list)

        # switch number of points, and point coordinate dimensions
        points_array = points_array.transpose(0, 2, 1).copy()
        weights_array = np.ones(points_array.shape)
        nan_mask = np.isnan(points_array)
        weights_array[nan_mask] = 0
        points_array[nan_mask] = 0

        return ceph_image, points_array

    def invert_pts_modification(self, ind, pts):
        return self.data_loader.invert_pts_modification(ind, pts)


class DataBatchLoader:
    '''
    batch_size - size of batch, maximal size
    '''
    def __init__(self, data_loader, batch_size):
        batch_count = len(data_loader)//batch_size

        self.data_loader = data_loader
        self.batch_size = batch_size
        self.batch_count = batch_count

    def __len__(self):
        return self.batch_count

    def __getitem__(self, ind):
        b_start = self.batch_size*ind
        b_end = min(self.batch_size*(ind+1), len(self.data_loader))

        ceph_image, points_list = self.data_loader[b_start]

        ceph_image_batch = np.zeros(
            (self.batch_size, )+ceph_image.shape[1:],
            dtype=np.float32,
            )
        points_list_batch = points_list

        ceph_image_batch[0] = ceph_image[0]
        b = 1
        for b_ind in range(b_start+1, b_end):
            ceph_image, points_list = self.data_loader[b_ind]
            ceph_image_batch[b] = ceph_image[0]
            points_list_batch += points_list
            b += 1

        return ceph_image_batch[:b], points_list_batch

    def invert_pts_modification(self, ind, pts):
        b_start = self.batch_size*ind
        b_end = min(self.batch_size*(ind+1), len(self.data_loader))

        for p_ind, b_ind in enumerate(range(b_start, b_end)):
            pts[p_ind] = self.data_loader.invert_pts_modification(b_ind, [pts[p_ind]])[0]
        return pts

class PointSelectionLoader:
    '''
    pts_sel_inds - list of point indices
    '''
    def __init__(
            self,
            data_loader,
            pts_sel_inds,
        ):
        self.data_loader = data_loader
        if isinstance(pts_sel_inds, int):
            pts_sel_inds = (pts_sel_inds,)
        self.pts_sel_inds = pts_sel_inds

        _, points_list = data_loader[0]
        self.n_pts = len(points_list[0])


    def __len__(self):
        return len(self.data_loader)

    def __getitem__(self, ind):
        ceph_image, points_list = self.data_loader[ind]

        points_list_sel = [[pts_batch[i] for i in self.pts_sel_inds] for pts_batch in points_list]

        return ceph_image, points_list_sel

    def invert_pts_modification(self, ind, pts):
        b_size = len(pts)
        pts_new = [[(0, 0) for _ in range(self.n_pts)] for _ in range(b_size)]
        for b in range(b_size):
            for n, i in enumerate(self.pts_sel_inds):
                pts_new[b][i] = pts[b][n]

        return self.data_loader.invert_pts_modification(ind, pts_new)



def main_test():
    '''
    test this module
    '''
    plot_samples = True

    if plot_samples:
        import matplotlib.pyplot as pyplot
    import sys

    data_loader = CephDataLoader.fromDirs(sys.argv[1])
    print(f'number of images: {len(data_loader)}')
    data_loader = PointSelectionLoader(data_loader, 4)
    data_loader = DataResizeLoader(data_loader, (512, 512))
    print(f'number of resized images: {len(data_loader)}')
    data_loader = DataBatchLoader(data_loader, 4)
    print(f'number of batches: {len(data_loader)}')
    data_loader = KeypointMapDataLoader(data_loader, 8)
    print(f'number of maps: {len(data_loader)}')


    while True:
        print('click image to go to next')
        print('press button to restart')
        print('press Ctrl+C to stop')
        if plot_samples:
            pyplot.close('all')
            fig, ax = pyplot.subplots(4, 2)

        for img_batch, kp_map_batch in data_loader:
            print(f'image: {img_batch.shape}, kp map: {kp_map_batch.shape}')
            if plot_samples:
                for a in ax.ravel():
                    a.cla()
                for b in range(img_batch.shape[0]):
                    ax[b, 0].imshow(img_batch[b, :, :, 0]+kp_map_batch[b, :, :].max(2))
                    ax[b, 1].imshow(kp_map_batch[b, :, :].max(2))
                pyplot.draw()
                if pyplot.waitforbuttonpress():
                    break



if __name__ == '__main__':
    main_test()

