import typing
from collections.abc import Sequence
from typing import Any

from ._base import KerasV3LayerHandler, register

if typing.TYPE_CHECKING:
    from keras import KerasTensor
    from keras.src.layers.merging.base_merge import Merge


@register
class MergeHandler(KerasV3LayerHandler):
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
        cls_name: str | None = None,
    ):
        assert len(out_tensors) == 1, f'Merge layer {layer.name} has more than one output'
        output_shape = list(out_tensors[0].shape[1:])

        cls_name = cls_name or layer.__class__.__name__
        config: dict[str, Any] = {'output_shape': output_shape}

        op = cls_name.lower()
        match cls_name:
            case 'Concatenate':
                rank = len(output_shape)
                class_name = f'Concatenate{rank}d'
                config['axis'] = layer.axis
            case 'Dot':
                msg = (
                    'Dot product only supported flatten tensors, got input shapes'
                    f'{in_tensors[0].shape} and {in_tensors[1].shape} for layer {layer.name}.'
                )
                assert all(len(t.shape) == 2 for t in in_tensors), msg
                assert in_tensors[0].shape[1] == in_tensors[1].shape[0], f'Input shape mismatch for layer {layer.name}.'
                class_name = 'Dot'
                op = 'dot1d'
                config['axes'] = layer.axes
            case _:
                class_name = 'Merge'

        config['class_name'] = class_name
        config['op'] = op
        return config
