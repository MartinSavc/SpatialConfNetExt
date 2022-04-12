from abc import ABC, abstractmethod
import tensorflow.keras as keras
from tensorflowlib.blocks import simple_conv_block


class ResNetBlock(ABC):

    @staticmethod
    @abstractmethod
    def build(in_tensor, num_filters, stride, downsample, batch_norm,
              kernel_initializer, bias_initializer):
        """
            Parameter 'downsample' should point to a partial function that implements downsampling operation on a tensor
            'in_tensor' that is passed to it on the invocation. Processed tensor should be returned to the caller.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_expansion():
        pass


class Bottleneck(ResNetBlock):

    @staticmethod
    def get_expansion():
        return 4

    @staticmethod
    def build(in_tensor, num_filters, stride=1, downsample=None, batch_norm=False,
              kernel_initializer=None, bias_initializer=None):
        res = in_tensor

        out = simple_conv_block(in_tensor, num_filters, 1, batch_norm=batch_norm)

        out = simple_conv_block(out, num_filters, 3, stride=stride, batch_norm=batch_norm)

        out = simple_conv_block(out, num_filters * Bottleneck.get_expansion(), 1, activation=None,
                                batch_norm=batch_norm)

        if downsample is not None:
            res = downsample(in_tensor=res)

        out = out + res
        out = keras.layers.ReLU()(out)

        return out


class BasicBlock(ResNetBlock):

    @staticmethod
    def get_expansion():
        return 1

    @staticmethod
    def build(in_tensor, num_filters, stride=1, downsample=None, batch_norm=False,
              kernel_initializer=None, bias_initializer=None):
        res = in_tensor

        out = simple_conv_block(in_tensor, num_filters, 3, stride=stride, batch_norm=batch_norm)

        out = simple_conv_block(out, num_filters, 3, activation=None, batch_norm=batch_norm)

        if downsample is not None:
            res = downsample(in_tensor=res)

        out = out + res
        out = keras.layers.ReLU()(out)

        return out
