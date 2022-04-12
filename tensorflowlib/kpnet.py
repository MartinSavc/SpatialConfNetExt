from abc import ABC, abstractmethod
import os
import sys
import re
import time
import argparse
import numpy as np
import h5py
from matplotlib.pyplot import imsave

import tensorflow as tf
import tensorflow.keras as keras

sys.path += [os.path.abspath('../')]
try:
    import KPnet_config as config
    from commonlib.cephdataloaders import CephDataLoader

except ModuleNotFoundError as e:
    configPathMissing = os.path.abspath(sys.path[-1])
    print('\nERROR: KPnet_config.py missing in: ' + configPathMissing + '\n\n')
    print(e)
    raise e


class KeyPointNetwork(ABC):
    '''
    Base class for training, testing and previewing key point detection networks.


    Defines and assumes the folowing members:
    self.name - str
        Name of the model. This is used to define the output folder,
        when saving a new model. The output folder is:
        f'{self.name}_models/model_{timestamp}/'

    self.model_path - str
        Model path, where a new model is stored or a saved model is loaded from.

    '''

    def __init__(self):
        self.name = 'NONE'
        self.n_pts = None

    @property
    def model_path(self):
        '''
        Save/load model path. Either specified with cmd argument or generated
        based on model name and current timestamp.
        '''
        return self._model_path

    @model_path.setter
    def model_path(self, path):
        self._model_path = path

    @property
    def eval_path(self):
        '''
        Get evaluation path. This is usually equal to or derived from model_path.
        '''
        return self.model_path

    @abstractmethod
    def train(self, train_loader, validation_loader, epochs):
        '''
        train_loader - a commonlib.cephdataloaders.CephDataLoader for training data
        validation_loader - a commonlib.cephdataloaders.CephDataLoader for validation data
        epochs - numer of epochs to train for

        Called for command train.
        '''
        pass

    @abstractmethod
    def preview(self, data):
        '''
        data - tuple of image and list of groud truth points

        Called for command preview.
        Should process the given image with the current model and
        display it's results alongside the ground truth points.
        '''
        pass

    @abstractmethod
    def export(self, export_path):
        '''
        export_path - path to export to

        Called for command export.
        Should prep the model and save it as pb and pbascii files.
        '''
        pass

    @abstractmethod
    def predict(self, image, pts_target=None):
        '''
        image - np.ndarray, (1, H, W, 1)
            image to process
        pts_target - np.ndarray, (1, N, 2)
            reference/groundtruth points

        Returns:
        pts_predict - np.ndarray, (1, N, 3)
        pts_target - np.ndarray, (1, N, 3)
        image_map - np.ndarray,  (1, H, W, N)

        If pts_target is supplied, the method is expected to return them.
        Return point values have 3 dimensions - y, x, v
        where v is the value/probability of the point (y, x), based on models
        output.

        Called for command evaluate.
        Called for each image in the dataset.
        
        '''
        pass

    @abstractmethod
    def init_model(self):
        '''
        Called to initialize/setup the model. Called when a new
        model is created. Isn't called during loading of an existing model.
        '''
        pass

    @abstractmethod
    def preload_params(self):
        '''
        Called to load parameters from a previously saved model. Value model_path
        will be set. This is called before add_cmd_params, so that default
        parameter values can be read from file.
        '''
        pass

    @abstractmethod
    def load_model(self):
        '''
        Called to load a save model. Value model_path will be set.
        '''
        pass

    @abstractmethod
    def save_model(self):
        '''
        Save the model to file. Value model_path will be set.
        '''
        pass

    def add_cmd_params(self, arg_parser):
        '''
        arg_parser - argparse.ArgumentParser

        Add your own parameters to arg_parser, these will be available as cmd
        arguments, if the main method is run.
        '''
        pass

    def set_cmd_params(self, args_env):
        '''
        args_env - environment returned by argparse.ArgumentParser.parse_args()

        If you added any cmd arguments through the add_cmd_params,
        you can read them from the args_env here.
        '''
        pass

    def load_pretrained_model(self):
        '''
            Loads a pretrained model. After the model is loaded,
            it can be accessed by pretrained_model instance variable.
        '''
        pass

    @classmethod
    def main(cls):
        '''
        Main method to call, when training, testing or previewing a model.
        '''
        # allocate GPU memory dynamically 
        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_sess = tf.compat.v1.Session(config=tf_config)
        tf.compat.v1.keras.backend.set_session(tf_sess)

        arg_parser_part = argparse.ArgumentParser(add_help=False)
        arg_parser_part.add_argument(
            '-M', '-m', '--model',
            metavar='file', type=str,
            dest='model_path',
        )

        args_env_part, _ = arg_parser_part.parse_known_args()


        arg_parser = argparse.ArgumentParser(
            'evaluate the trained model on train and test images, ' + \
            'producing result data',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        kp_net = cls()
        if args_env_part.model_path:
            kp_net.model_path = args_env_part.model_path
            kp_net.preload_params()

        kp_net.add_cmd_params(arg_parser)

        arg_parser.add_argument(
            'command',
            choices=['train', 'evaluate', 'preview', 'export'],
        )
        arg_parser.add_argument(
            '-M', '-m', '--model',
            metavar='file', type=str,
            help='path to trained model (folder)',
            dest='model_path',
        )
        arg_parser.add_argument(
            '--export_path',
            metavar='file', type=str,
            help='path to export model to (folder)',
            dest='export_path',
        )
        arg_parser.add_argument(
            '--save_maps',
            action='store_true',
            help='also save the generated maps as images',
        )
        arg_parser.add_argument(
            '--save_maps_hdf5',
            action='store_true',
            help='also save the generated maps in hdf5 file, fails if outputs already exist',
        )
        arg_parser.add_argument(
            '-e', '--epochs',
            type=int,
            default=100,
            help='number of epochs to train',
            dest='epochs',
        )
        arg_parser.add_argument(
            '--vf', '--validation-frac',
            type=float,
            default=0.25,
            help='the fraction of training images to use in validation',
            dest='valid_frac',
            )
        arg_parser.add_argument(
            '--valid_rand_seed',
            default=None,
            help='int, seed for random in random_validation_split(). Default=None (seed will be random)',
            )
        arg_parser.add_argument(
            '-to', '--test_list_only',
            action='store_true',
            help='only uses testImagesFile list for evaluate command (disables train images)',
            )
        arg_parser.add_argument(
            '--tp', '--train-part',
            default=None,
            nargs='*',
            type=int,
            dest='train_part',
            help='split training dataset into multiple partitions. Partitions are split evenly and do not overlap.'
                 'Example: argument "2 0" would split the dataset into two partitions and use the '
                 'first one to train current stage model. After the first model is trained, we '
                 'can train the next-stage model using the second partition by specifying "2 1".',
        )
        arg_parser.add_argument(
            '-p', '--pretrained_model',
            metavar='file', type=str,
            help='path to trained model (folder), whose output can be used for further training.'
                 'Shoudln\'t be used together with the --model flag',
            dest='pretrained_model',
        )


        args_env = arg_parser.parse_args()
        command = args_env.command
        save_maps = args_env.save_maps
        save_maps_hdf5 = args_env.save_maps_hdf5
        epochs = args_env.epochs
        valid_frac = args_env.valid_frac
        test_list_only = args_env.test_list_only
        train_part = args_env.train_part
        valid_rand_seed = args_env.valid_rand_seed
        if str(valid_rand_seed).upper() == 'NONE':
            valid_rand_seed = None
        else:
            valid_rand_seed = int(valid_rand_seed)

        if hasattr(args_env, 'testImagesFile'):
            testImagesFile = args_env.testImagesFile
        else:
            testImagesFile = config.testImagesFile

        if hasattr(args_env, 'trainImagesFile'):
            trainImagesFile = args_env.trainImagesFile
        else:
            trainImagesFile = config.trainImagesFile

        if hasattr(args_env, 'pointsCnt'):
            pointsCnt = args_env.pointCnt
        else:
            pointsCnt = config.pointCnt
        if hasattr(args_env, 'resLabelsDir'):
            resLabelsDir = args_env.resLabelsDir
        else:
            resLabelsDir = config.resLabelsDir
        if hasattr(args_env, 'analysisImagesFile'):
            analysisImagesFile = args_env.analysisImagesFile
        else:
            analysisImagesFile = config.analysisImagesFile
        if hasattr(args_env, 'resMatrixFile'):
            resMatrixFile = args_env.resMatrixFile
        else:
            resMatrixFile = config.resMatrixFile
        if hasattr(args_env, 'GTmatrixFile'):
            GTmatrixFile = args_env.GTmatrixFile
        else:
            GTmatrixFile = config.GTmatrixFile
        if hasattr(args_env, 'diffMatrixFile'):
            diffMatrixFile = args_env.diffMatrixFile
        else:
            diffMatrixFile = config.diffMatrixFile

        if args_env.model_path:
            kp_net.model_path = args_env.model_path
            kp_net.n_pts = pointsCnt
            kp_net.set_cmd_params(args_env)
            kp_net.load_model()
        else:
            if args_env.pretrained_model:
                kp_net.model_path = args_env.pretrained_model
                kp_net.load_pretrained_model()

            time_stamp = time.strftime('%Y_%m_%d-%H_%M_%S')
            model_path = os.path.join(f'{kp_net.name}_models', f'model_{time_stamp}')

            if os.path.exists(model_path):
                raise Exception(f'output model path {model_path} ' \
                                f'already exists, quiting')
            os.makedirs(model_path)
            kp_net.model_path = model_path
            kp_net.n_pts = pointsCnt
            kp_net.set_cmd_params(args_env)
            kp_net.init_model()

        print(f'model path:{kp_net.model_path}')
        print(f'cmd arguments:\n{args_env}')

        if command == 'train':
            kp_net.save_model()
            try:
                kp_net.main_train(
                    trainImagesFile,
                    epochs,
                    valid_frac,
                    train_part,
                    valid_rand_seed
                )
            finally:
                print(f'model path:{kp_net.model_path}')
                kp_net.save_model()
        elif command == 'preview':
            kp_net.main_preview(
                trainImagesFile,
                testImagesFile,
            )

        elif command == 'evaluate':
            imagesFiles = []
            if test_list_only:
                imagesFiles = [testImagesFile]
            else:
                imagesFiles = [trainImagesFile, testImagesFile]
            kp_net.main_evaluate(
                    imagesFiles,
                    resLabelsDir,
                    analysisImagesFile,
                    resMatrixFile,
                    GTmatrixFile,
                    diffMatrixFile,
                    save_maps,
                    save_maps_hdf5,
                    )
        elif command == 'export':
            if args_env.export_path:
                export_path = args_env.export_path
            else:
                export_path = os.path.join(kp_net.model_path, 'export')
            if os.path.exists(export_path):
                raise Exception(f'export path {export_path} ' \
                                f'already exists, quiting')
            os.makedirs(export_path)

            kp_net.main_export(export_path)
        else:
            raise Exception('no command given')

    def main_train(
            self,
            trainImagesFile,
            epochs,
            val_frac,
            train_part,
            valid_rand_seed
    ):
        '''
        Called when train command is set.
        '''
        images_loader = CephDataLoader.fromFiles(trainImagesFile)
        if train_part is not None:
            images_loader = images_loader.random_train_split(train_part)
        if val_frac > 0:
            train_images_loader, valid_images_loader = images_loader.random_validation_split(
                val_frac, rand_seed=valid_rand_seed)
        else:
            train_images_loader = images_loader
            valid_images_loader = None
        self.train(train_images_loader, valid_images_loader, epochs)

    def main_preview(
            self,
            trainImagesFile,
            testImagesFile,
    ):
        '''
        Called when preview command is set.
        '''
        import cmd
        import matplotlib.pyplot as pyplot

        pyplot.ion()

        import matplotlib.pyplot as pyplot

        class cmd_interface(cmd.Cmd):
            def __init__(self, kpnet, trainImagesFile, testImagesFile, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self.kpnet = kpnet
                self.train_images_loader = CephDataLoader.fromFiles(trainImagesFile)
                self.test_images_loader = CephDataLoader.fromFiles(testImagesFile)

                self.ind_train = 0
                self.ind_test = 0

            def do_info(self, arg):
                '''
                print the number of testing and training images and
                the last previewed images
                '''

                print(f'training images {len(self.train_images_loader)}')
                print(f'last training image {self.ind_train}')
                print(f'testing images {len(self.test_images_loader)}')
                print(f'last testing image {self.ind_test}')

            def do_train(self, arg):
                '''
                preview a training image
                train
                    preview the next training image
                train 10
                    preview the 10th training image
                '''
                if arg:
                    i = int(arg)
                    if i >= len(self.train_images_loader) or i < 0:
                        print('not a valid training image index')

                    self.ind_train = i
                else:
                    self.ind_train += 1
                    self.ind_train = min(self.ind_train, len(self.train_images_loader) - 1)

                file_name = self.train_images_loader.get_file_name(self.ind_train)
                print(f'training file: {os.path.basename(file_name[0])}')

                train_data = self.train_images_loader[self.ind_train]
                self.kpnet.preview(train_data)

            def do_test(self, arg):
                '''
                preview a testing image
                test 
                    preview the next image
                test 10 
                    preview the 10the image
                '''
                if arg:
                    i = int(arg)
                    if i >= len(self.test_images_loader) or i < 0:
                        print('not a valid testing image index')

                    self.ind_test = i
                else:
                    self.ind_test += 1
                    self.ind_test = min(self.ind_test, len(self.test_images_loader) - 1)

                file_name = self.test_images_loader.get_file_name(self.ind_test)
                print(f'test file: {os.path.basename(file_name[0])}')

                test_data = self.test_images_loader[self.ind_test]
                self.kpnet.preview(test_data)

            def do_exit(self, arg):
                return True

        cmd_interface(self, trainImagesFile, testImagesFile).cmdloop()

    def main_export(
            self,
            export_path,
    ):
        '''
        Export the model to a tensorflow pb and readable pbascii format.
        '''
        self.export(export_path)

    def main_evaluate(
            self,
            imagesFilesList,
            resLabelsDir,
            analysisImagesFile,
            resMatrixFile,
            GTmatrixFile,
            diffMatrixFile,
            save_maps=False,
            save_maps_hdf5=False,
    ):
        '''
        Called when evaluate command is set.
        '''

        # sedej: added - can put extra text behind "date"
        model_pattern = f'{self.name}_models/model_[0-9\-_]+.*(/.*)?'
        path_mod = re.fullmatch(model_pattern, self.eval_path)
        if path_mod is None:
            raise Exception(f'pattern {model_pattern} does not match model path: {self.eval_path}')
        path_mod = path_mod[0]

        analysisImagesFile = config.getModifiedPath(analysisImagesFile, path_mod, True)
        resMatrixFile = config.getModifiedPath(resMatrixFile, path_mod, True)
        GTmatrixFile = config.getModifiedPath(GTmatrixFile, path_mod, True)
        diffMatrixFile = config.getModifiedPath(diffMatrixFile, path_mod, True)

        data_loader = CephDataLoader.fromFiles(imagesFilesList)
        n_imgs = len(data_loader)

        img_files_list = []
        pts_gt_matrix = np.zeros((n_imgs, self.n_pts, 3))
        pts_res_matrix = np.zeros((n_imgs, self.n_pts, 3))

        t0 = time.time()
        for ind, (img, kp_map_gt) in enumerate(data_loader):
            img_orig, pts_orig = data_loader[ind]
            img_file_path, _ = data_loader.get_file_name(ind)

            img_files_list += [img_file_path]

            kp_map, pts_pred, pts_orig = self.predict(img_orig, pts_orig)

            for n in range(self.n_pts):
                pts_gt_matrix[ind, n, :] = pts_orig[0][n]
                pts_res_matrix[ind, n, :] = pts_pred[0][n]

            if save_maps_hdf5:
                im_name = os.path.basename(img_file_path).rsplit(os.path.extsep, 1)[0]
                h5_file_path = os.path.join(config.resLabelsDir, f'{im_name}.h5')
                h5_file_path = config.getModifiedPath(h5_file_path, path_mod, True)
                with h5py.File(h5_file_path, 'w-') as h5_file:
                    h5_data = h5_file.create_dataset(
                        'heatmaps',
                        kp_map.shape,
                        dtype=np.float32,
                        compression='gzip',
                    )
                    h5_data[:] = np.float32(kp_map)
            if save_maps:
                n_maps = kp_map.shape[3]
                label_image_paths = generateLabelImageNames(img_file_path, n_maps)
                for l, label_path in enumerate(label_image_paths):
                    label_path = config.getModifiedPath(label_path, path_mod, True)

                    imsave(label_path, kp_map[0, :, :, l],
                           vmin=0, vmax=1, cmap='gray')

            if time.time() - t0 > 10:
                t0 = time.time()
                print(f'processed {ind}/{n_imgs} images')

        pts_diff_matrix = calcDiffMatrix(
            pts_gt_matrix,
            pts_res_matrix,
            pts_gt_matrix.shape[0],
            pts_gt_matrix.shape[1],
        )
        with open(analysisImagesFile, 'w') as f:
            target_dir = os.path.dirname(analysisImagesFile)
            for fn in img_files_list:
                fn_rel = os.path.relpath(fn, target_dir)
                f.write(f'{fn_rel}\n')
        np.save(resMatrixFile, pts_res_matrix)
        print("saved resMatrix\n" + resMatrixFile)
        np.save(GTmatrixFile, pts_gt_matrix)
        print("saved GTMatrix\n" + GTmatrixFile)
        np.save(diffMatrixFile, pts_diff_matrix)
        print("saved diffMatrix\n" + diffMatrixFile)


def generateLabelImageNames(image_name, n_pts=None):
    if n_pts is None:
        n_pts = config.pointCnt

    image_name = os.path.basename(image_name).rsplit(os.path.extsep, 1)[0]
    pts_paths_list = []
    for pt_n in range(n_pts):
        pts_path = os.path.join(config.resLabelsDir, f'{image_name}-lab-{pt_n:02}.jpg')
        pts_paths_list += [pts_path]

    return pts_paths_list


def calcDiffMatrix(GTmatrixOrig, resMatrix, imCnt=150, pointCnt=19):
    # GTmatrix plus probability form res
    GTmatrix = np.zeros((imCnt, pointCnt, 3))
    GTmatrix[:, :, 0:2] = GTmatrixOrig[:, :, 0:2]
    diffMatrix = np.zeros(resMatrix.shape)

    for i in range(0, imCnt):  # 150
        for p in range(0, pointCnt):  # 19
            # if point (mandmark) is missing -> set as np.nan
            if GTmatrix[i, p, 0] == 0.:
                diffMatrix[i, p, :] = np.nan
            else:
                diffMatrix[i, p, :] = resMatrix[i, p, :] - GTmatrix[i, p]
    return diffMatrix
