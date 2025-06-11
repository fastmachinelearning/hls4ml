import inspect
import typing
from collections.abc import Sequence
from math import prod

import numpy as np

from ._base import KerasV3LayerHandler, register

if typing.TYPE_CHECKING:
    import keras
    from keras import KerasTensor


@register
class DenseHandler(KerasV3LayerHandler):
    handles = ('keras.src.layers.core.dense.Dense',)

    def handle(
        self,
        layer: 'keras.layers.Dense',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        kernel = self.load_weight(layer, 'kernel')
        bias = self.load_weight(layer, 'bias') if layer.use_bias else None
        n_in, n_out = kernel.shape  # type: ignore

        config = {
            'data_format': 'channels_last',
            'weight_data': kernel,
            'bias_data': bias,
            'n_out': n_out,
            'n_in': n_in,
        }
        return config


@register
class InputHandler(KerasV3LayerHandler):
    handles = ('keras.src.layers.core.input_layer.InputLayer',)

    def handle(
        self,
        layer: 'keras.layers.InputLayer',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        config = {'input_shape': list(layer._batch_shape[1:])}
        return config


@register
class ActivationHandler(KerasV3LayerHandler):
    handles = ('keras.src.layers.activations.activation.Activation',)

    def handle(
        self,
        layer: 'keras.layers.Activation',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        import keras

        config = {}
        config.update(self.default_config)

        activation = getattr(layer, 'activation', keras.activations.linear)
        match activation:
            case keras.activations.softmax:
                class_name = 'Softmax'
                config['axis'] = -1
            case keras.activations.hard_sigmoid:
                class_name = 'HardActivation'
            case keras.activations.leaky_relu:
                class_name = 'LeakyReLU'
                signature = inspect.signature(keras.activations.leaky_relu)
                config['activ_param'] = signature.parameters['negative_slope'].default
            case keras.activations.elu:
                class_name = 'ELU'
                signature = inspect.signature(keras.activations.elu)
                config['activ_param'] = signature.parameters['alpha'].default
            case _:
                class_name = 'Activation'

        config['activation'] = activation.__name__
        config['class_name'] = class_name
        config['n_in'] = prod(in_tensors[0].shape[1:])  # type: ignore
        return (config,)


@register
class ReLUHandler(KerasV3LayerHandler):
    handles = (
        'keras.src.layers.activations.leaky_relu.LeakyReLU',
        'keras.src.layers.activations.prelu.PReLU',
        'keras.src.layers.activations.relu.ReLU',
    )

    def handle(
        self,
        layer: 'keras.layers.ReLU',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        config = {}
        config.update(self.default_config)

        if layer.__class__.__name__ == 'ReLU':
            config['class_name'] = 'Activation'
            config['activation'] = 'relu'
            return config

        if layer.__class__.__name__ == 'PReLU':
            config['class_name'] = 'PReLU'
            config['param_data'] = np.array(layer.alpha)
            config['activation'] = 'prelu'
        else:
            config['class_name'] = 'LeakyReLU'
            config['activ_param'] = float(layer.negative_slope)
            config['activation'] = 'leaky_relu'

        return (config,)


@register
class SoftmaxHandler(KerasV3LayerHandler):
    handles = ('keras.src.layers.activations.softmax.Softmax',)

    def handle(
        self,
        layer: 'keras.layers.Softmax',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        ax = layer.axis
        ax = ax if ax >= 0 else len(in_tensors[0].shape) + ax
        # io_stream asserts axis=-1, convert to -1 when it is
        n_outer: int = prod(in_tensors[0].shape[1:ax])  # type: ignore
        n_inner: int = prod(in_tensors[0].shape[ax + 1 :])  # type: ignore
        ax = -1 if ax == len(in_tensors[0].shape) - 1 else ax
        config = {}
        config.update(self.default_config)
        if len(in_tensors) == 2:
            raise NotImplementedError('Masked softmax not supported yet')
            config['class_name'] = 'MaskedSoftmax'
        elif len(in_tensors) == 1:
            config['class_name'] = 'Softmax'
        else:
            raise ValueError(f'Too many inputs for softmax layer {layer.name}: expected 1 or 2, got {len(in_tensors)}')
        config['axis'] = layer.axis
        config['activation'] = 'softmax'
        config['n_outer'] = n_outer
        config['n_inner'] = n_inner

        return (config,)


@register
class EluHandler(KerasV3LayerHandler):
    handles = ('keras.src.layers.activations.elu.ELU',)

    def handle(
        self,
        layer: 'keras.layers.ELU',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        config = {}
        config.update(self.default_config)

        config['class_name'] = 'ELU'
        config['activ_param'] = float(layer.alpha)
        config['activation'] = 'elu'
        config['n_in'] = prod(in_tensors[0].shape[1:])  # type: ignore

        return (config,)


@register
class ReshapeHandler(KerasV3LayerHandler):
    handles = ('keras.src.layers.reshaping.reshape.Reshape', 'keras.src.layers.reshaping.flatten.Flatten')

    def handle(
        self,
        layer: 'keras.layers.Reshape',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        return {
            'class_name': 'Reshape',
            'target_shape': list(out_tensors[0].shape[1:]),
        }


@register
class PermuteHandler(KerasV3LayerHandler):
    handles = ('keras.src.layers.reshaping.permute.Permute',)

    def handle(
        self,
        layer: 'keras.layers.Permute',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        config = {'class_name': 'Transpose', 'perm': [dim - 1 for dim in layer.dims]}  # rm batch dim
        return config
