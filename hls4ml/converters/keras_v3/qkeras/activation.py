from typing import Any

from hls4ml.converters.utils import IsolatedLayerReader

from ..core import KerasV3LayerHandler
from .utils import set_default_config


class QKerasQActivationHandler(KerasV3LayerHandler):
    handles = ('qkeras.qlayers.QActivation', 'QActivation')

    def handle(
        self,
        layer,
        in_tensors,
        out_tensors,
    ) -> tuple[dict[str, Any], ...]:

        config = layer.get_config()
        layer_dict = {'config': config, 'class_name': layer.__class__.__name__}

        reader = IsolatedLayerReader(layer)
        input_shapes = [list(t.shape) for t in in_tensors]
        input_names = [t.name for t in in_tensors]

        from hls4ml.converters.keras_v2_to_hls import layer_handlers as v2_layer_handlers

        v2_handler = v2_layer_handlers.get(layer.__class__.__name__)
        if v2_handler is None:
            raise ValueError(f'No v2 handler found for {layer.__class__.__name__}')

        hls_conf, _ = v2_handler(layer_dict, input_names, input_shapes, reader)
        hls_conf = set_default_config(hls_conf, self.default_config)

        return (hls_conf,)
