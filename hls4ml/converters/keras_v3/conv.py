import typing
from collections.abc import Sequence
from math import ceil
from typing import Any

from ._base import KerasV3LayerHandler, register

if typing.TYPE_CHECKING:
    import keras
    from keras import KerasTensor


def gen_conv_config(
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    ker_px_shape: tuple[int, ...],
    strides: tuple[int, ...],
    padding: str,
    data_format: str,
    name: str,
) -> dict[str, Any]:
    if data_format == 'channels_last':
        *px_in_shape, ch_in = in_shape
        *px_out_shape, ch_out = out_shape
    else:
        ch_in, *px_in_shape = in_shape
        ch_out, *px_out_shape = out_shape
    if not px_out_shape:
        px_out_shape = [1] * len(px_in_shape)

    if padding == 'same':
        n_padding = [ceil(N / n) * n - N for N, n in zip(px_in_shape, ker_px_shape)]
        n_padding0 = [p // 2 for p in n_padding]
        n_padding1 = [p - p0 for p, p0 in zip(n_padding, n_padding0)]
    elif padding == 'valid':
        n_padding0 = [0] * len(px_in_shape)
        n_padding1 = [0] * len(px_in_shape)
    elif padding == 'causal':
        assert len(px_in_shape) == 1, f'Invalid padding mode {padding} for layer {name}: ndim > 1'
        n_padding0 = [ker_px_shape[0] - 1] + [0] * (len(px_in_shape) - 1)
        n_padding1 = [0] * len(px_in_shape)
    else:
        raise ValueError(f'Invalid padding mode {padding} for layer {name}')

    if len(ker_px_shape) == 1:
        config = {
            'filt_width': ker_px_shape[0],
            'stride_width': strides[0],
            'pad_left': n_padding0[0],
            'pad_right': n_padding1[0],
            'in_width': px_in_shape[0],
            'out_width': px_out_shape[0],
        }

    elif len(ker_px_shape) == 2:
        config = {
            'filt_height': ker_px_shape[0],
            'filt_width': ker_px_shape[1],
            'stride_height': strides[0],
            'stride_width': strides[1],
            'pad_top': n_padding0[0],
            'pad_bottom': n_padding1[0],
            'pad_left': n_padding0[1],
            'pad_right': n_padding1[1],
            'in_height': px_in_shape[0],
            'in_width': px_in_shape[1],  # type: ignore
            'out_height': px_out_shape[0],
            'out_width': px_out_shape[1],
        }
    else:
        raise ValueError(f'Only 1D and 2D layers are supported, got {len(ker_px_shape)}D')

    config['n_filt'] = ch_out
    config['n_chan'] = ch_in
    return config


@register
class ConvHandler(KerasV3LayerHandler):
    handles = (
        'keras.src.layers.convolutional.conv1d.Conv1D',
        'keras.src.layers.convolutional.conv2d.Conv2D',
        'keras.src.layers.convolutional.depthwise_conv1d.DepthwiseConv1D',
        'keras.src.layers.convolutional.depthwise_conv2d.DepthwiseConv2D',
        'keras.src.layers.convolutional.separable_conv1d.SeparableConv1D',
        'keras.src.layers.convolutional.separable_conv2d.SeparableConv2D',
    )

    def handle(
        self,
        layer: 'keras.layers.Conv1D|keras.layers.Conv2D|keras.layers.DepthwiseConv1D|keras.layers.DepthwiseConv2D',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        from keras.src.layers.convolutional.base_conv import BaseConv
        from keras.src.layers.convolutional.base_depthwise_conv import BaseDepthwiseConv
        from keras.src.layers.convolutional.base_separable_conv import BaseSeparableConv

        assert len(in_tensors) == 1, f'Layer {layer.name} has more than one input'
        assert len(out_tensors) == 1, f'Layer {layer.name} has more than one output'

        in_shape: tuple[int, ...] = in_tensors[0].shape[1:]  # type: ignore
        out_shape: tuple[int, ...] = out_tensors[0].shape[1:]  # type: ignore
        assert all(isinstance(x, int) for x in in_shape), f'Layer {layer.name} has non-fixed size input: {in_shape}'
        assert all(isinstance(x, int) for x in out_shape), f'Layer {layer.name} has non-fixed size output: {out_shape}'

        kernel = self.load_weight(layer, 'kernel')
        if layer.use_bias:
            bias = self.load_weight(layer, 'bias')
        else:
            bias = None

        ker_px_shape: tuple[int, ...] = layer.kernel_size
        data_format = layer.data_format

        config = gen_conv_config(
            in_shape=in_shape,
            out_shape=out_shape,
            ker_px_shape=ker_px_shape,
            strides=layer.strides,
            data_format=data_format,
            padding=layer.padding,
            name=layer.name,
        )

        config.update(
            {
                'bias_data': bias,
                'data_format': data_format,
                'weight_data': kernel,
            }
        )

        if isinstance(layer, BaseDepthwiseConv):
            config['depthwise_data'] = kernel
            config['depth_multiplier'] = layer.depth_multiplier
        elif isinstance(layer, BaseSeparableConv):
            config['depthwise_data'] = kernel
            config['pointwise_data'] = self.load_weight(layer, 'pointwise_kernel')
            config['depth_multiplier'] = layer.depth_multiplier
        elif isinstance(layer, BaseConv):
            config['weight_data'] = kernel

        return config
