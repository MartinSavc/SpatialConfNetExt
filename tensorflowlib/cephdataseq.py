import sys
import os
import time
import argparse
import itertools
import numpy as np
from tensorflow.keras.utils import Sequence

sys.path += [os.path.abspath('../')]

import commonlib.cephdataloaders as dataload
import commonlib.cephdatamodifiers as datamod

supported_kw = [
        'image_list_file',
        'label_list_file',
        'image_dir',
        'label_dir',
        'points_selection',
        'bbox_align',
        'image_shape',
        'normalize_intensity',
        'shuffle',
        'cache',
        'center_targets',
        'pw_affine_warp',
        'rotate',
        'scale',
        'translate',
        'gamma',
        'intensity',
        'black_level',
        'resample',
        'subselect',
        'map_sigma',
        'batch_size',
        'softmax_targets',
        'output_fun',
        ]


class CephDataSequence(Sequence):
    '''
    config_dict may contain keys (with appropriate values):

    To select the source of data, at least one of the following information
    must be given:

    'image_dir' : str (valid path to folder)
    'label_dir' : str (valid path to folder)
        Load all image and label files from the image_dir directory.
        If label_dir is not given, the names for label data are automatically
        generated from individual image file names.

    'image_list_file' : str (valid file name)
    'label_list_file' : str (valid file name)
        Load all image and label files from a list in a file.
        If label_list_file is not given, the names for label data are
        automatically generated from individual image file names.

    'image_loader' : commonlib.cephdataloader.CephDataLoader
        Load images and labels from this existing data loader.

    Fixed configuration modification can be made using the following flags.
    All are optional, but some depend on each other ('batch_size' requires
    'image_shape')

    'points_selection' : (int,...)
        Tuple of integers - indices of points to use. This reduces the number of 
        output channels to the selected points.

    'bbox_align' : float,
        Images and data are aligned by centering on the bounding box of key points.
        The bounding box is a square. The floating point parameter is a factor of
        growth that increases the  bounding box size giving it an additional border.
        Given parameter 0, a tight bounding box is used. Given the parameter 0.2,
        a border the size of 0.2*box width/height is added. This increases the 
        size of the bounding box by 1.4.
        
    'image_shape' : (int, int)
        Tuple of two labels : height and width.
        All images and labels are resized to size H - height and W - width.

    'normalize_intensity' : (float, float)
        Tuple of two floats : scale and offset.
        All images have their intensity normalized/modified by first multiplying by
        scale and then adding offset.

    'cache': bool
        When set, resized images and data points are cached before agumentation.
        The cacheing method is simple and naive - all data points in the
        sequence are remembered in a dictionary, with their index as id.

    'shuffle': bool
        When set, data samples are reshufled on each epoch.
        This shufling also mixed data samples between batches (which a regular
        shuffle parameter in the model cannot do).


    'batch_size' : int
        Number of images and labels gathered into a single batch. If not given,
        each image is returned on its own. All images must be the same size,
        so passing image_shape is also expected.

    'map_sigma' : float or str
        If float, sigma is used to generate gaussian points in keypoint maps.
        If string, then if:
            - 'single' - each point is represented by a 1 at the rounded location.
            - 'bilinear' - each point is represented with up to 4 neighbouring pixels
                           that sum up to 1.
        If not given, no map is generated.

    'center_targets' : bool
        When set, the positions returned are relative to the image center. By
        default the positions are relative to upper left corner.

    'output_fun' : callable
        A final function to modify the data as needed before it is returned.
        Data is passed in as the arguments, and it's return values are 
        returned as the item.

    Randomized modification (data augmentation can be made using the following
    flags. These are configured using random variable specifications,
    described below.

    The modifications are applied in the order described here.

    'resample' : (int, int)
        Resample the images at given keypoints with the specified window size.
        For each keypoint a new sample is created using the specified
        window placed over the keypoint position. If a keypoint is missing 
        the center of the image is sampled.

    'subselect' : float
        Randomly select a subset of samples. The parameter is the
        fraction of samples to select, from 0 to 1. The samples are
        not shuffled. The selection is randomized on each epoch.

    'pw_affine_warp' : int int int
        Place a grid of points over the image. Randomly move the points in the grid
        and use these movements to define an piecewise affine warping of the image.

        The grid size is specified by the first two parameters, the third parameter specifies
        the maximum movement of grid points.


    'rotate' :  <random variable spec>
        Rotate the image by a angle. The random value is interpreted
        as degrees. 

    'scale' : <random varible spec>
        Scale the image by a factor. When the value is >1, the image
        iz zoomed in, when value is <1 the image iz zoomed out.

    'translate' : <random varible spec>
        Translate - shift the image. This is usefull, when resampling
        aligns points with the center of the image and this alignment might
        influence training and evaluation.

        Translation is applied after resampling.

    'gamma' : <random variable specs>
        Changed the gamma correction of the image. Values are limited to
        [0, inf). The image values are transformed as v**(1/g), where v is
        the image value and g is the random variable. Values g<1 emphasize
        bright parts, values g>1 emphasize the dark parts.
    'intensity' : <random variable specs>
        Multiply the image with the random value. Values are limited to
        [0, inf). Values <1 make the image darker, values >1 make image
        brighter. Image values above 1 are saturated after the operation
        (set to 1).
    'black_level' : <random variable specs>
        Add a random value to the image. Values are limited to [0, 1].
        Image values above 1 are saturated after the operation (set to 1).

    'softmax_targets' : bool
        When set, the outputs maps are modified for softmax output or other classification. A null class channel is added, its value beeing the 1 - sum of other channels.



    Random variable specification
    Random variables are specified using additional dictionaries to determine
    the type of random distribution used. These number are generated on each
    epoch of the dataset (if needed).

    For a uniformly distributed real random variable between min_val and
    max_val:
    {'min':min_val, 'max':max_val}

    For a normaly distributed real random variable with mean mean_val and
    standard deviation std_val:
    {'mean':mean_val, 'std':std_val}

    For a constant value const_val (this will only be initialized once):
    {'constant': const_val}
    '''
    def __init__(self, config_dict):
        self.data_loader = None

        # convert any lists to tuples
        for k, v in config_dict.items():
            if isinstance(v, list):
                config_dict[k] = tuple(v)

        if config_dict.get('image_list_file'):
            self.data_loader = dataload.CephDataLoader.fromFiles(config_dict['image_list_file'])

        if config_dict.get('image_dir'):
            self.data_loader = dataload.CephDataLoader.fromDirs(config_dict['image_dir'])

        if config_dict.get('image_loader'):
            self.data_loader = config_dict['image_loader']

        if self.data_loader is None:
            raise Exception('must pass image path')

        if config_dict.get('points_selection'):
            self.data_loader = dataload.PointSelectionLoader(
                self.data_loader,
                config_dict['points_selection'],
                )

        if config_dict.get('bbox_align') is not None:
            self.data_loader = datamod.AlignBBoxModifier(
                    self.data_loader, 
                    config_dict['bbox_align'],
                    )


        if config_dict.get('image_shape'):
            self.data_loader = dataload.DataResizeLoader(
                self.data_loader,
                config_dict['image_shape'],
                )

        if config_dict.get('normalize_intensity'):
            scale, offset = config_dict['normalize_intensity']
            self.data_loader = datamod.NormalizeIntensityModifier(self.data_loader, scale, offset)

        if config_dict.get('cache'):
            self.data_loader = datamod.CacheDataModifier(self.data_loader)

        self.data_modifiers = []

        if config_dict.get('shuffle'):
            shuffle_mod = datamod.ShuffleDataModifier(self.data_loader)
            self.data_modifiers += [shuffle_mod]
            self.data_loader = shuffle_mod

        if config_dict.get('pw_affine_warp'):
            vert_split, horz_split, warp_max = config_dict['pw_affine_warp']
            pw_affine_mod = datamod.PiecewiseAffineWarpModifier(
                    self.data_loader,
                    datamod.RandFunGenerator({'min':-warp_max, 'max':warp_max}),
                    vert_split,
                    horz_split,
                    )
            self.data_modifiers += [pw_affine_mod]
            self.data_loader = pw_affine_mod

        if config_dict.get('rotate') or config_dict.get('scale'):
            if config_dict.get('rotate'):
                rot_rv_conf = config_dict['rotate']
                rot_rv_fun = datamod.RandFunGenerator(rot_rv_conf)
            else:
                rot_rv_fun = datamod.RandFunGenerator({'constant':0})
            if config_dict.get('scale'):
                scale_rv_conf = config_dict['scale']
                scale_rv_fun = datamod.RandFunGenerator(scale_rv_conf)
            else:
                rot_rv_fun = datamod.RandFunGenerator({'constant':1})

            rotscale_mod = datamod.CephDataRotateScaleMod(self.data_loader,
                scale_rv_fun,
                rot_rv_fun,
                )
            self.data_modifiers += [rotscale_mod]
            self.data_loader = rotscale_mod

        for mod_name, mod_class in [
                ('gamma', datamod.CephDataGammaMod),
                ('intensity', datamod.CephDataIntensityMod),
                ('black_level', datamod.CephDataBlackLevelMod),
                ]:
            if config_dict.get(mod_name):
                rv_conf = config_dict[mod_name]
                rv_fun = datamod.RandFunGenerator(rv_conf)
                data_mod = mod_class(
                    self.data_loader,
                    rv_fun,
                    )

                if 'constant' not in rv_conf:
                    self.data_modifiers += [data_mod]
                self.data_loader = data_mod

        if config_dict.get('resample'):
            if config_dict.get('translate'):
                translate_rv_conf = config_dict['translate']
                translate_rv_fun = datamod.RandFunGenerator(translate_rv_conf)

                data_mod = datamod.PointResampleDataModifier(
                    self.data_loader,
                    config_dict['resample'],
                    translate_rv_fun,
                    )

                if 'constant' not in translate_rv_conf:
                    self.data_modifiers += [data_mod]

            else:
                data_mod = datamod.PointResampleDataModifier(
                    self.data_loader,
                    config_dict['resample'],
                    )
            self.data_loader = data_mod

        if config_dict.get('subselect'):
            data_mod = datamod.SubselectDataModifier(
                self.data_loader,
                config_dict['subselect'],
                )
            self.data_modifiers += [data_mod]
            self.data_loader = data_mod

        if config_dict.get('softmax_targets'):
            self.softmax_targets = config_dict['softmax_targets']
        else:
            self.softmax_targets = False

        if config_dict.get('center_targets'):
            self.data_loader = datamod.CenterPointsDataModifier(self.data_loader)

        if config_dict.get('batch_size'):
            self.data_loader = dataload.DataBatchLoader(
                self.data_loader,
                config_dict['batch_size'],
                )

        if config_dict.get('map_sigma'):
            self.data_loader = dataload.KeypointMapDataLoader(
                self.data_loader,
                config_dict['map_sigma'],
                softmax_target=self.softmax_targets,
                )
        else:
            self.data_loader = dataload.KeypointWeightsDataLoader(self.data_loader)

        if config_dict.get('output_fun'):
            self.output_fun = config_dict['output_fun']
        else:
            self.output_fun = lambda data: data



    def __len__(self):
        return len(self.data_loader)

    def __getitem__(self, ind):
        return self.output_fun(self.data_loader[ind])

    def on_epoch_end(self):
        for data_modifier in self.data_modifiers:
            data_modifier.generate_rand_values()

    def invert_pts_modification(self, ind, pts):
        return self.data_loader.invert_pts_modification(ind, pts)


class RandValArgType():
    def __init__(self):
        self.rand_dist_name = None
        self.rand_dist_params = []
        pass

    def __call__(self, arg_str):
        if self.rand_dist_name is None:
            if arg_str in ['uniform', 'normal', 'constant']:
                self.rand_dist_name = arg_str
                return arg_str
            else:
                msg = f'"{arg_str}" is not a valid random variable distribution'
                raise argparse.ArgumentTypeError(msg)
        else:
            arg_val = float(arg_str)
            return arg_val


def add_cmd_params_for_config(arg_parser, defaults={}):
    arg_parser.add_argument(
        '--image_dir',
        default=defaults['image_dir'] if 'image_dir' in defaults else None,
        type=str,
        help='directory of training/validation images',
        )
    arg_parser.add_argument(
        '--label_dir',
        default=defaults['label_dir'] if 'label_dir' in defaults else None,
        type=str,
        help='directory of training/validation label files',
        )
    arg_parser.add_argument(
        '--image_list_file',
        default=defaults['image_list_file'] if 'image_list_file' in defaults else None,
        type=str,
        help='file with list of training/validation images',
        )
    arg_parser.add_argument(
        '--label_list_file',
        default=defaults['label_list_file'] if 'label_list_file' in defaults else None,
        type=str,
        help='file with list of training/validation label files',
        )
    arg_parser.add_argument(
        '--points_selection',
        default=defaults['points_selection'] if 'points_selection' in defaults else None,
        nargs='*',
        type=int,
        help='a selection of point indices to train with',
        )
    arg_parser.add_argument(
        '--bbox_align',
        default=defaults['bbox_align'] if 'bbox_align' in defaults else None,
        type=float,
        help='align the images using a bbox around the points, increasing the size of the box by the given relative border size',
        )
    arg_parser.add_argument(
        '--image_shape',
        default=defaults['image_shape'] if 'image_shape' in defaults else None,
        nargs=2,
        type=int,
        help='fixed size target to rescale and pad all images to',
        )
    arg_parser.add_argument(
        '--normalize_intensity',
        default=defaults['normalize_intensity'] if 'normalize_intensity' in defaults else None,
        nargs=2,
        type=float,
        help='scale and offset for image intensity normalization (img*scale + offset)',
        )
    arg_parser.add_argument(
        '--cache',
        action='store_true',
        help='cache all data samples in memory on first epoch',
        )
    arg_parser.add_argument(
        '--shuffle',
        action='store_true',
        help='shuffle the data samples on each epoch',
        )
    arg_parser.add_argument(
        '--center_targets',
        action='store_true',
        help='center the target points, so that the coordinates are relative to the image center',
        )
    arg_parser.add_argument(
        '--batch_size',
        default=defaults['batch_size'] if 'batch_size' in defaults else None,
        type=int,
        help='batch size for training',
        )
    arg_parser.add_argument(
        '--map_sigma',
        default=defaults['map_sigma'] if 'map_sigma' in defaults else None,
        help='Sigma used to generate gaussian kernels as target points. When set to 0, only the specified point position is set to 1.',
        )
    arg_parser.add_argument(
        '--resample',
        default=defaults['resample'] if 'resample' in defaults else None,
        nargs=2,
        type=int,
        help='Size of resampling window. When set, images are resampled,\n'+\
        'for each point in an image a sample is created by centerting a\n'+\
        'a window of the specified size on the point.',
        )
    arg_parser.add_argument(
        '--subselect',
        default=defaults['subselect'] if 'subselect' in defaults else None,
        type=float,
        help='Fraction of samples to keep. The samples are selected'+\
        ' at random each epoch.'
        )

    arg_parser.add_argument(
        '--pw_affine_warp',
        default=defaults['pw_affine_warp'] if 'pw_affine_warp' in defaults else None,
        nargs=3,
        type=int,
        help='Random piecewise affine warping of an image. A grid of points is placed'+\
             'over the image. The points are randomly moved, and the image is warped'+\
             'acording to their movement. Specify 3 integer values: rows cols max'+\
             'Where rows and cols defines the size of the grid, and max defines the maximum'+\
             'displacement of the points for warping.',
        )
    if 'rotate' in defaults:
        default_val = [str(v) for v in itertools.chain(*defaults['rotate'].items())]
    else:
        default_val = None
    arg_parser.add_argument(
        '--rotate',
        default=default_val,
        nargs='*',
        type=str,
        help='Random rotation of images. Specify a random variable either\n'+\
             'uniform: min <min> max <max>\n'+\
             'normal: mean <mean> std <std>\n'+\
             'constant: constant <value>\n',
        )
    if 'scale' in defaults:
        default_val = [str(v) for v in itertools.chain(*defaults['scale'].items())]
    else:
        default_val = None
    arg_parser.add_argument(
        '--scale',
        default=default_val,
        nargs='*',
        type=str,
        help='Random scaling of images. Specify a random variable either\n'+\
             'uniform: min <min> max <max>\n'+\
             'normal: mean <mean> std <std>\n'+\
             'constant: constant <value>\n',
        )
    if 'translate' in defaults:
        default_val = [str(v) for v in itertools.chain(*defaults['translate'].items())]
    else:
        default_val = None
    arg_parser.add_argument(
        '--translate',
        default=default_val,
        nargs='*',
        type=str,
        help='Random translation of images. Specify a random variable either\n'+\
             'uniform: min <min> max <max>\n'+\
             'normal: mean <mean> std <std>\n'+\
             'constant: constant <value>\n',
        )
    if 'gamma' in defaults:
        default_val = [str(v) for v in itertools.chain(*defaults['gamma'].items())]
    else:
        default_val = None
    arg_parser.add_argument(
        '--gamma',
        default=default_val,
        nargs='*',
        type=str,
        help='Random gamma intensity transformations. Specify a random variable either\n'+\
             'uniform: min <min> max <max>\n'+\
             'normal: mean <mean> std <std>\n'+\
             'constant: constant <value>\n',
        )
    if 'intensity' in defaults:
        default_val = [str(v) for v in itertools.chain(*defaults['intensity'].items())]
    else:
        default_val = None
    arg_parser.add_argument(
        '--intensity',
        default=default_val,
        nargs='*',
        type=str,
        help='Random intensity transformations. Specify a random variable either\n'+\
             'uniform: min <min> max <max>\n'+\
             'normal: mean <mean> std <std>\n'+\
             'constant: constant <value>\n',
        )
    if 'black_level' in defaults:
        default_val = [str(v) for v in itertools.chain(*defaults['black_level'].items())]
    else:
        default_val = None
    arg_parser.add_argument(
        '--black_level',
        default=default_val,
        nargs='*',
        type=str,
        help='Random black level transformations. Specify a random variable either\n'+\
             'uniform: min <min> max <max>\n'+\
             'normal: mean <mean> std <std>\n'+\
             'constant: constant <value>\n',
        )
    arg_parser.add_argument(
        '--softmax_targets',
        action='store_true',
        help='Add an aditional channel to the outputs to\n'+\
             'represent the no-point class.',
        )

    return arg_parser

def config_from_cmd_params(arg_env):
    config = {k:v for k, v in vars(arg_env).items() if v and k in supported_kw}

    for k in config:
        if isinstance(config[k], list):
            config[k] = tuple(config[k])
    
    for rv_conf_key in ['gamma', 'rotate', 'scale', 'translate', 'intensity', 'black_level']:
        if rv_conf_key in config:
            rv_params = config[rv_conf_key]
            config[rv_conf_key] = {
                    k:float(v) 
                    for k, v in zip(rv_params[::2], rv_params[1::2])
                    }
    return config

if __name__ == '__main__':
    data_loader = dataload.CephDataLoader.fromDirs('../1_data/1_images/')
    config = {
        #'image_list_file':'../1_data/test_data.list',
        'image_loader' : data_loader,
        'map_sigma': 8.0,
        #'points_selection': (4, 12, 64),
        'image_shape': (512, 512),
        'normalize_intensity' : (2.0, -1.0),
        #'resample': (128, 128),
        #'subselect': 0.1,
        #'batch_size' : 4,
        #'black_level' : {'min':-1., 'max':0.},
        #'intensity' : {'min': 0.8, 'max':1.2},
        #'gamma' : {'constant' : 1.},
        #'pw_affine_warp' : [32, 32, 5],
        'rotate' : {'min' : -10, 'max':10}, 
        'scale' : {'min' : 0.9, 'max':1.1}, 
        'translate' : {'min' : -24, 'max' : 24},
        #'softmax_targets' : False,
        }
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--plot',
        action='store_true',
        help='plot the images and keypoint maps',
        )

    add_cmd_params_for_config(parser, config)
    arg_env = parser.parse_args()
    plot_data  = arg_env.plot
    config = config_from_cmd_params(arg_env)

    config['image_loader'] = data_loader

    if plot_data:
        import matplotlib.pyplot as pyplot

    print(config)

    ceph_data_seq = CephDataSequence(config)

    print(f'data length: {len(ceph_data_seq)}')
    t0 =time.time()

    while True:
        #for n, (img, kp_map) in enumerate(ceph_data_seq):
        for n, (img, kp_array) in enumerate(ceph_data_seq):
            t1 = time.time()
            dt = t1-t0
            t0 = t1

            #print(f'image {n}: {img.shape}, map: {kp_map.shape} '+\
            print(f'image {n}: {img.shape}, keypoints: {kp_array.shape} '+\
                  f'load time {dt:.3} s')

            img_max = img.max(axis=(1, 2, 3))
            img_min = img.min(axis=(1, 2, 3))
            print('image values: [min, max]')
            for vmin, vmax in zip(img_min, img_max):
                print(f'[{vmin}, {vmax}]')

            B = img.shape[0]

            R = B
            C = 2

            if plot_data:
                pyplot.clf()

                _, ax = pyplot.subplots(R, C, num=0)
                ax = ax.reshape(B, 2)
                #ax = ax.reshape(B, 1)

                for b in range(img.shape[0]):
                    im = ax[b, 0].imshow(img[b, :, :, 0])
                    pyplot.colorbar(im, ax=ax[b, 0])
                    ax[b, 0].set_title(f'im {b}')

                    if 'map_sigma' in config:
                        ax[b, 1].set_title(f'map_sigma combined')
                        im = ax[b, 1].imshow(kp_array[b, ...].max(2))
                    else:
                        ax[b, 0].plot(
                            kp_array[b, 1, :],
                            kp_array[b, 0, :],
                            'ro')
                        # ax[b, 0].imshow(img[b, :, :, 0]+kp_map[b, :, :].max(2))
                        # ax[b, 1].imshow(kp_map[b, :, :].max(2))
                        # ax[b, 1].set_title(f'map {b}')

                pyplot.gcf().canvas.mpl_connect('close_event', lambda ev: exit())

                pyplot.draw()
                if pyplot.waitforbuttonpress():
                    break
        ceph_data_seq.on_epoch_end()


