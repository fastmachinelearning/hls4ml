import typing
from collections.abc import Sequence

from ._base import KerasV3LayerHandler, register
from .conv import gen_conv_config

if typing.TYPE_CHECKING:
    from keras import KerasTensor
    from keras.src.layers.pooling.base_global_pooling import BaseGlobalPooling
    from keras.src.layers.pooling.base_pooling import BasePooling


@register
class PoolingHandler(KerasV3LayerHandler):
    handles = (
        'keras.src.layers.pooling.max_pooling1d.MaxPooling1D',
        'keras.src.layers.pooling.max_pooling2d.MaxPooling2D',
        'keras.src.layers.pooling.max_pooling3d.MaxPooling3D',
        'keras.src.layers.pooling.average_pooling1d.AveragePooling1D',
        'keras.src.layers.pooling.average_pooling2d.AveragePooling2D',
        'keras.src.layers.pooling.average_pooling3d.AveragePooling3D',
        'keras.src.layers.pooling.global_average_pooling1d.GlobalAveragePooling1D',
        'keras.src.layers.pooling.global_average_pooling2d.GlobalAveragePooling2D',
        'keras.src.layers.pooling.global_average_pooling3d.GlobalAveragePooling3D',
        'keras.src.layers.pooling.global_max_pooling1d.GlobalMaxPooling1D',
        'keras.src.layers.pooling.global_max_pooling2d.GlobalMaxPooling2D',
        'keras.src.layers.pooling.global_max_pooling3d.GlobalMaxPooling3D',
    )

    def handle(
        self,
        layer: 'BasePooling | BaseGlobalPooling',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        from keras.src.layers.pooling.base_pooling import BasePooling

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

        pool_size: tuple[int, ...] = layer.pool_size if isinstance(layer, BasePooling) else tuple(px_in_shape)

        strides = layer.strides if isinstance(layer, BasePooling) else pool_size
        padding = layer.padding if isinstance(layer, BasePooling) else 'valid'
        config = gen_conv_config(
            in_shape=in_shape,
            out_shape=out_shape,
            ker_px_shape=pool_size,
            strides=strides,
            data_format=data_format,
            padding=padding,
            name=layer.name,
        )

        config['pool_width'] = config.pop('filt_width')
        if 'filt_height' in config:
            config['pool_height'] = config.pop('filt_height')
        if len(px_in_shape) == 1:
            # inconsistent pooling1d config key name...
            config['n_in'] = config['in_width']
            config['n_out'] = config['out_width']
        return config
