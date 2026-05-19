from typing import Any

from ..core import KerasV3LayerHandler
from .util import IsolatedLayerReader


class QKerasQActivationHandler(KerasV3LayerHandler):
    handles = ('qkeras.qlayers.QActivation', 'QActivation')

    def handle(
        self,
        layer,  # qkeras.qlayers.QActivation
        in_tensors,
        out_tensors,
    ) -> tuple[dict[str, Any], ...]:

        config = layer.get_config()
        layer_dict = {'config': config, 'class_name': layer.__class__.__name__}

        reader = IsolatedLayerReader(layer)
        input_shapes = [list(t.shape) for t in in_tensors]
        input_names = [t.name for t in in_tensors]
        output_names = [t.name for t in out_tensors]

        from hls4ml.converters.keras_v2_to_hls import layer_handlers as v2_layer_handlers

        v2_handler = v2_layer_handlers.get(layer.__class__.__name__)
        if v2_handler is None:
            raise ValueError(f'No v2 handler found for {layer.__class__.__name__}')

        hls_conf, _ = v2_handler(layer_dict, input_names, input_shapes, reader)

        hls_conf['input_keras_tensor_names'] = list(input_names)
        hls_conf['output_keras_tensor_names'] = list(output_names)
        hls_conf.setdefault('name', layer.name)
        hls_conf.setdefault('class_name', 'QActivation')

        return (hls_conf,)
