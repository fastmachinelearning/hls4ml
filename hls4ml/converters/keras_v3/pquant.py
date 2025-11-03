import typing
from collections.abc import Sequence

import numpy as np

from hls4ml.model.types import FixedPrecisionType

from ._base import KerasV3LayerHandler, register
from .conv import gen_conv_config

if typing.TYPE_CHECKING:
    import pquant
    from keras import KerasTensor


@register
class PQuantReLUHandler(KerasV3LayerHandler):
    handles = ('pquant.core.activations_quantizer.QuantizedReLU',)

    def handle(
        self,
        layer: 'pquant.core.activations_quantizer.QuantizedReLU',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        config = {}
        config.update(self.default_config)
        config['quantization_parameters'] = layer.quantization_parameters

        if (
            not config['quantization_parameters']['use_high_granularity_quantization']
            and layer.config['quantization_parameters']['use_relu_multiplier']
        ):
            config['class_name'] = 'MultiplierReLU'
            config['param_data'] = np.array(layer.multiplier)
            config['activation'] = 'multiplier_relu'

        else:
            config['class_name'] = 'QActivation'
            config['activation'] = 'relu'

        return (config,)


@register
class PQuantTanhHandler(KerasV3LayerHandler):
    handles = ('pquant.core.activations_quantizer.QuantizedTanh',)

    def handle(
        self,
        layer: 'pquant.core.activations_quantizer.QuantizedTanh',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        config = {}
        config.update(self.default_config)
        config['quantization_parameters'] = layer.quantization_parameters

        if not layer.config['quantization_parameters']['use_real_tanh']:
            config['class_name'] = 'HardActivation'
            config['slope'] = 0.5  # the default values in QKeras
            config['shift'] = 0.5
            # Quartus seems to have trouble if the width is 1.
            config['slope_prec'] = FixedPrecisionType(width=2, integer=0, signed=False)
            config['shift_prec'] = FixedPrecisionType(width=2, integer=0, signed=False)
            config['activation'] = 'hard_tanh'

        else:
            config['class_name'] = 'QActivation'
            config['activation'] = 'tanh'

        return (config,)


@register
class PQuantPoolingHandler(KerasV3LayerHandler):
    handles = ('pquant.core.tf_impl.compressed_layers_tf.QuantizedPooling',)

    def handle(
        self,
        layer: 'pquant.core.tf_impl.compressed_layers_tf.QuantizedPooling',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        assert len(in_tensors) == 1, f'Layer {layer.name} has more than one input'
        assert len(out_tensors) == 1, f'Layer {layer.name} has more than one output'

        in_shape: tuple[int, ...] = in_tensors[0].shape[1:]  # type: ignore
        out_shape: tuple[int, ...] = out_tensors[0].shape[1:]  # type: ignore
        assert all(isinstance(x, int) for x in in_shape), f'Layer {layer.name} has non-fixed size input: {in_shape}'
        assert all(isinstance(x, int) for x in out_shape), f'Layer {layer.name} has non-fixed size output: {out_shape}'

        data_format = layer.data_format

        if data_format == 'channels_last':
            *px_in_shape, _ = in_shape
        else:
            _, *px_in_shape = in_shape

        pool_size: tuple[int, ...] = layer.pool_size

        strides = layer.strides
        padding = layer.padding
        pooling_config = gen_conv_config(
            in_shape=in_shape,
            out_shape=out_shape,
            ker_px_shape=pool_size,
            strides=strides,
            data_format=data_format,
            padding=padding,
            name=layer.name,
        )

        pooling_config['pool_width'] = pooling_config.pop('filt_width')
        if 'filt_height' in pooling_config:
            pooling_config['pool_height'] = pooling_config.pop('filt_height')
        if len(px_in_shape) == 1:
            # inconsistent pooling1d config key name...
            pooling_config['n_in'] = pooling_config['in_width']
            pooling_config['n_out'] = pooling_config['out_width']

        config = {}
        config.update(self.default_config)
        config.update(pooling_config)
        config['class_name'] = f'AveragePooling{layer.dimensions}D'
        config['quantization_parameters'] = layer.quantization_parameters
        return (config,)
