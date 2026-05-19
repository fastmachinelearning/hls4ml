import numpy as np

from ..core import KerasV3LayerHandler
from .utils import set_default_config

from hls4ml.converters.utils import IsolatedLayerReader

class QKerasQConv2DBatchnormHandler(KerasV3LayerHandler):
    handles = ('qkeras.qconv2d_batchnorm.QConv2DBatchnorm')

    def handle(self, layer, in_tensors, out_tensors):
        config = layer.get_config()
        layer_dict = {'config': config, 'class_name': layer.__class__.__name__}

        reader = IsolatedLayerReader(layer)
        input_shapes = [list(t.shape) for t in in_tensors]
        input_names = [t.name for t in in_tensors]

        from hls4ml.converters.keras_v2_to_hls import layer_handlers as v2_layer_handlers

        v2_handler = v2_layer_handlers.get(layer.__class__.__name__)
        if v2_handler is None:
            raise ValueError(f'No v2 handler found for {layer.__class__.__name__}')

        ret, _ = v2_handler(layer_dict, input_names, input_shapes, reader)
        ret = set_default_config(ret, self.default_config)

        activation = config.get('activation')
        if activation not in (None, 'linear'):
            from hls4ml.converters.keras.qkeras import get_activation_quantizer

            activation_config = get_activation_quantizer(layer_dict, input_names)
            intermediate_tensor_name = f'{out_tensors[0].name}_activation'
            ret['output_keras_tensor_names'] = [intermediate_tensor_name]
            activation_config.update(
                {
                    'name': f'{layer.name}_activation',
                    'input_keras_tensor_names': [intermediate_tensor_name],
                    'output_keras_tensor_names': [out_tensors[0].name],
                }
            )
            return ret, activation_config

        return ret
