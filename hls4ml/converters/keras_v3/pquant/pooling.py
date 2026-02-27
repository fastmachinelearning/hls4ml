from collections.abc import Sequence
from typing import TYPE_CHECKING

from hls4ml.converters.keras_v3._base import register
from hls4ml.converters.keras_v3.pooling import PoolingHandler

from ._base import PQLayerHandler

if TYPE_CHECKING:
    import pquant
    from keras import KerasTensor


@register
class PQAvgPoolHandler(PQLayerHandler, PoolingHandler):
    handles = (
        'pquant.core.keras.layers.PQAvgPool1d',
        'pquant.core.keras.layers.PQAvgPool2d',
    )

    def handle(
        self,
        layer: 'pquant.core.keras.layers.PQAvgPool1d | pquant.core.keras.layers.PQAvgPool2d',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        conf = super().handle(layer, in_tensors, out_tensors)
        conf['class_name'] = 'AveragePooling' + layer.__class__.__name__[-2] + 'D'

        return conf
