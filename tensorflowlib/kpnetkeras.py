import os
import json
import tensorflow.keras as keras
import numpy as np
import tempfile
import shutil

from . import cephdataseq as cdseq
from . import keras2tf
from .kpnet import KeyPointNetwork


class KeyPointKerasNetwork(KeyPointNetwork):
    '''
    Base class for Keras network models, taking care of some
    basic configuration and setup with sensible defaults.

    Defines and assumes the folowing members:
    self.name - str
        Name of the model. This is used to define the output folder,
        when saving a new model. The output folder is:
        f'{self.name}_models/model_{timestamp}/'

    self.model - keras.models.Model
        The model that is trained, tested, ...

    self.data_loader_conf - dictionary
        Configuration for tensorflowlib.cephdataseq.CephDataSequence
        Adds all required cmd parameters for CephDataSequence.

    self.lr_base - float (0.001 default)
        Base learning rate.
    self.step_size - int (50 default)
        Base step size.
    self.gamma - float (0.5 default)
        Base learning rate reduction.

        Parameters for adaptation of learning rate.
        Learning rate is decreased each step_size epochs, by
        multiplying it by gamma.

    '''

    def __init__(self):
        super().__init__()
        self.name = None
        self.data_loader_conf = {}
        self.model = None
        self.pretrained_model = None
        self.load_weights = None
        self.lr_base = 0.001
        self.step_size = 50
        self.gamma = 0.5
        self.early_stop = None
        self.weights_only = None

        self.load_best = False
        self.load_last = False

    @KeyPointNetwork.eval_path.getter
    def eval_path(self):
        if self.load_best:
            return os.path.join(self.model_path, 'model_best')
        if self.load_last:
            return os.path.join(self.model_path, 'model_last')
        return self.model_path

    def add_cmd_params(self, arg_parser):
        '''
        arg_parser - argparse.ArgumentParser

        Add your own parameters to arg_parser, these will be available as cmd
        arguments, if the main method is run.
        '''
        super().add_cmd_params(arg_parser)

        arg_parser.add_argument(
            '-s', '--early_stop',
            help='stop training after N epochs without improvement',
            nargs='?',
            const=5,
            type=int,
            default=self.early_stop,
            dest='early_stop',
        )
        arg_parser.add_argument(
            '-w', '--weights_only',
            action='store_true',
            help='Save and load model from a file that contains weights only (.pb).'
                 'On loading, initialized model architecture must be identical to the one described by weights.',
        )
        arg_parser.add_argument(
            '--best',
            action='store_true',
        )
        arg_parser.add_argument(
            '--last',
            action='store_true',
        )

        arg_parser.add_argument(
            '--lr_base',
            default=self.lr_base,
            type=float,
            help='base learning rate, learning rate is calculated as base*gamma^int(epoch/step)',
        )
        arg_parser.add_argument(
            '--lr_step',
            default=self.step_size,
            type=int,
            help='learning rate step size, learning rate is calculated as base*gamma^int(epoch/step)',
        )
        arg_parser.add_argument(
            '--lr_gamma',
            default=self.gamma,
            type=float,
            help='learning rate gamma, learning rate is calculated as base*gamma^int(epoch/step)',
        )

        cdseq.add_cmd_params_for_config(arg_parser, self.data_loader_conf)

    def set_cmd_params(self, args_env):
        '''
        args_env - environment returned by argparse.ArgumentParser.parse_args()

        If you added any cmd arguments through the add_cmd_params,
        you can read them from the args_env here.
        '''
        super().set_cmd_params(args_env)

        self.load_best = args_env.best
        self.load_last = args_env.last
        self.early_stop = args_env.early_stop
        self.weights_only = args_env.weights_only

        self.lr_base = args_env.lr_base
        self.step_size = args_env.lr_step
        self.gamma = args_env.lr_gamma

        self.data_loader_conf.update(cdseq.config_from_cmd_params(args_env))

    def generate_params_dict(self):
        '''
        Generate a dictionary of parameters, that will be stored in a json file.
        The dictionary can only contain primitive types, lists and other
        dictionaries of primitive types.

        Return the dictionary.

        When subclassing call super().generate_params_dict(), and add your own
        parameters to the returned dictionary.
        '''
        return {'data_loader_conf': self.data_loader_conf,
                'learning_rate':{
                    'base': self.lr_base,
                    'step_size': self.step_size,
                    'gamma': self.gamma,
                    },
                'early_stop': self.early_stop,
                }

    def apply_params_dict(self, params_dict):
        '''
        Set the parameters of params_dict to init/load parameters. Usually
        the same set, that was returned from generate_params_dict.

        Don't forget to call super().apply_params_dict(params_dict)
        '''
        self.data_loader_conf = params_dict['data_loader_conf']
        if 'learning_rate' in params_dict:
            self.lr_base = params_dict['learning_rate']['base']
            self.step_size = params_dict['learning_rate']['step_size']
            self.gamma = params_dict['learning_rate']['gamma']
        if 'early_stop' in params_dict:
            self.early_stop = params_dict['early_stop']

    def get_custom_layers_dict(self):
        '''
        Function is called during model loading. Should return a dictionary
        that will be passes to keras.models.load_model as the custom_object
        parameter.

        This dicionary is mainly used for custom keras layers or loss functions,
        during the loading of the model. The names/dicionary keys are usualy
        the same as the name of the class of function.
        '''
        return {}

    def load_pretrained_model(self):
        '''
        Loads the pretrained model from model.h5 file from the selected model path.
        '''
        super().load_pretrained_model()

        if self.load_best:
            model_name = 'model_best.h5'
        elif self.load_last:
            model_name = 'model_last.h5'
        else:
            model_name = 'model.h5'

        model_file_path = os.path.join(self.model_path, model_name)
        self.pretrained_model = keras.models.load_model(
            model_file_path,
            custom_objects=self.get_custom_layers_dict()
        )


    def preload_params(self):
        '''
        Loads the model parameters from params.json before setting
        '''
        super().preload_params()
        params_file_path = os.path.join(self.model_path, 'params.json')
        with open(params_file_path, 'r') as fd:
            params_dict = json.load(fd)
            self.apply_params_dict(params_dict)


    def load_model(self):
        '''
        Loads the model from model.h5 file from the selected model path.
        '''
        super().load_model()
        params_file_path = os.path.join(self.model_path, 'params.json')
        with open(params_file_path, 'r') as fd:
            params_dict = json.load(fd)
            self.apply_params_dict(params_dict)

        if self.load_best:
            model_name = 'model_best.h5'
        elif self.load_last:
            model_name = f'model_last.h5'
        else:
            model_name = f'model.h5'

        model_file_path = os.path.join(self.model_path, model_name)
        if self.weights_only:
            self.init_model()
            self.model.load_weights(model_file_path)
        else:
            self.model = keras.models.load_model(
                model_file_path,
                custom_objects=self.get_custom_layers_dict(),
            )

    def save_model(self):
        '''
        Saves the model the model.h5 file from the selected model path.
        '''
        if self.model is None:
            raise Exception('model not created yet')
        model_file_path = os.path.join(self.model_path, 'model.h5')
        if self.weights_only:
            self.model.save_weights(model_file_path)
        else:
            self.model.save(model_file_path)

        params_file_path = os.path.join(self.model_path, 'params.json')
        with open(params_file_path, 'w') as fd:
            json.dump(
                self.generate_params_dict(),
                fd,
                indent=4,
            )

    def generate_keras_callbacks(self):
        '''
        Return a list of callbacks that will be passed onto
        keras.models.Model.fit as list of callbacks.

        By default returns the following callbacks:
        [
            keras.callbacks.TensorBoard,
            keras.callbacks.ModelCheckpoint,
            keras.callbacks.LearningRateScheduler,
            keras.callbacks.TerminateOnNaN,
        ]

        These can be added to or completely replaced.
        '''
        callbacks = []

        tboard_cb = keras.callbacks.TensorBoard(
            log_dir=os.path.join(self.model_path, './Graph'),
            update_freq='batch',
            histogram_freq=0,
            write_graph=True,
            write_images=False,
        )
        callbacks.append(tboard_cb)

        checkpoint_best_cb = keras.callbacks.ModelCheckpoint(
            os.path.join(self.model_path, 'model_best.h5'),
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=self.weights_only,
            mode='auto',
            period=1,
        )
        callbacks.append(checkpoint_best_cb)

        checkpoint_last_cb = keras.callbacks.ModelCheckpoint(
            os.path.join(self.model_path, 'model_last.h5'),
            monitor='val_loss',
            verbose=0,
            save_best_only=False,
            save_weights_only=self.weights_only,
            period=1,
        )
        callbacks.append(checkpoint_last_cb)

        reducelr_cb = keras.callbacks.LearningRateScheduler(
            lambda e: self.lr_base*self.gamma**(int(e/self.step_size)),
            1,
        )
        callbacks.append(reducelr_cb)

        termnan_cb = keras.callbacks.TerminateOnNaN()
        callbacks.append(termnan_cb)

        if self.early_stop:
            earlystop_cb = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                mode='auto',
                min_delta=0,
                patience=self.early_stop,
                verbose=1
            )
            callbacks.append(earlystop_cb)

        return callbacks

    def train(self, train_loader, validation_loader, epochs):
        '''
        Train the model.

        train_loader - a commonlib.cephdataloaders.CephDataLoader for training data
        validation_loader - a commonlib.cephdataloaders.CephDataLoader for validation data
        epochs - numer of epochs to train for
        '''
        data_config = self.data_loader_conf.copy()
        data_config['image_loader'] = train_loader
        train_data_seq = cdseq.CephDataSequence(data_config)

        if validation_loader is not None:
            data_config['image_loader'] = validation_loader
            # remove some of the data augmentation modifiers
            # since these are not really usefull for validation or testing
            data_config.pop('pw_affine_warp', None)
            data_config.pop('rotate', None)
            data_config.pop('scale', None)
            data_config.pop('gamma', None)
            data_config.pop('intensity', None)
            data_config.pop('black_level', None)
            valid_data_seq = cdseq.CephDataSequence(data_config)
        else:
            valid_data_seq = None

        keras_callbacks = self.generate_keras_callbacks()
        keras_callbacks += [keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: [train_data_seq.on_epoch_end()], #valid_data_seq.on_epoch_end()],
        )]
        print('current configuration (before fit() )')
        print(self.generate_params_dict())

        self.model.fit(
            train_data_seq,
            epochs=epochs,
            verbose=1,
            callbacks=keras_callbacks,
            validation_data=valid_data_seq,
            use_multiprocessing=False,
            shuffle=False,
        )

    def get_exportable_model(self):
        '''
        Convert the model to an exportable form. This should remove any layers
        used specifically for training but are removed or ignored for inference.
        '''
        return self.model

    def export(self, export_path):
        '''
        Export the model.

        Convert the keras model to a tensorflow graph and export it.
        '''

        model = self.get_exportable_model()

        if self.load_best:
            keras2tf.convert_keras_to_const_graph(model, export_path, 'model_best')
        elif self.load_last:
            keras2tf.convert_keras_to_const_graph(model, export_path, 'model_last')
        else:
            keras2tf.convert_keras_to_const_graph(model, export_path)
