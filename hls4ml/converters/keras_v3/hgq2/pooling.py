from ..pooling import PoolingHandler
from ._base import QLayerHandler, register


@register
class QPoolingHandler(PoolingHandler, QLayerHandler):
    handles = (
        'hgq.layers.pooling.QMaxPooling1D',
        'hgq.layers.pooling.QMaxPooling2D',
        'hgq.layers.pooling.QMaxPooling3D',
        'hgq.layers.pooling.QAveragePooling1D',
        'hgq.layers.pooling.QAveragePooling2D',
        'hgq.layers.pooling.QAveragePooling3D',
        'hgq.layers.pooling.QGlobalAveragePooling1D',
        'hgq.layers.pooling.QGlobalAveragePooling2D',
        'hgq.layers.pooling.QGlobalAveragePooling3D',
        'hgq.layers.pooling.QGlobalMaxPooling1D',
        'hgq.layers.pooling.QGlobalMaxPooling2D',
        'hgq.layers.pooling.QGlobalMaxPooling3D',
    )
