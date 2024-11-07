import typing
from typing import Any, Sequence

import numpy as np

from ._base import KerasV3LayerHandler, register

if typing.TYPE_CHECKING:
    import keras
    from keras.api import KerasTensor
    from keras.src.layers.merging.base_merge import Merge


@register
class KV3DenseHandler(KerasV3LayerHandler):
    handles = ('keras.src.layers.core.dense.Dense',)

    def handle(
        self,
        layer: 'keras.layers.Dense',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        kernel = np.array(layer.kernel)
        assert layer._build_shapes_dict is not None, f"Layer {layer.name} is not built"
        # inp_shape = layer._build_shapes_dict['input_shape'][1:]
        config = {
            'data_format': 'channels_last',
            'weight_data': kernel,
            'bias_data': np.array(layer.bias) if layer.use_bias else None,
            'n_out': kernel.shape[1],
            'n_in': kernel.shape[0],
        }
        return config


@register
class KV3InputHandler(KerasV3LayerHandler):
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
class KV3MergeHandler(KerasV3LayerHandler):
    handles = (
        'keras.src.layers.merging.add.Add',
        'keras.src.layers.merging.multiply.Multiply',
        'keras.src.layers.merging.average.Average',
        'keras.src.layers.merging.maximum.Maximum',
        'keras.src.layers.merging.minimum.Minimum',
        'keras.src.layers.merging.concatenate.Concatenate',
        'keras.src.layers.merging.subtract.Subtract',
        'keras.src.layers.merging.dot.Dot',
    )

    def handle(
        self,
        layer: 'Merge',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        assert len(out_tensors) == 1, f"Merge layer {layer.name} has more than one output"
        output_shape = list(out_tensors[0].shape[1:])

        config: dict[str, Any] = {
            'output_shape': output_shape,
            'op': layer.__class__.__name__.lower(),
        }

        match layer.__class__.__name__:
            case 'Concatenate':
                rank = len(output_shape)
                class_name = f'Concatenate{rank}d'
                config['axis'] = layer.axis
            case 'Dot':
                class_name = f'Dot{len(output_shape)}d'
                rank = len(output_shape)
                assert rank == 1, f"Dot product only supported for 1D tensors, got {rank}D on layer {layer.name}"
            case _:
                class_name = 'Merge'

        config['class_name'] = class_name
        return config
