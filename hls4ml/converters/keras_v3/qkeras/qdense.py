from collections.abc import Sequence
import numpy as np

from ..core import KerasV3LayerHandler


class QKerasQDenseHandler(KerasV3LayerHandler):
    handles = ("qkeras.qlayers.QDense", "QDense")

    def handle(self, layer, in_tensors: Sequence["KerasTensor"], out_tensors: Sequence["KerasTensor"]):
        config = layer.get_config()
        layer_dict = {"config": config, "class_name": layer.__class__.__name__}

        class IsolatedLayerReader:
            def get_weights_data(self, layer_name, var_name):
                assert layer_name == layer.name, f"Processing {layer.name}, but handler tried to read {layer_name}"
                for w in layer.weights:
                    if var_name in w.name:
                        return np.array(w)
                return None

        reader = IsolatedLayerReader()
        input_shapes = [list(t.shape) for t in in_tensors]
        input_names = [t.name for t in in_tensors]

        from hls4ml.converters.keras_v2_to_hls import layer_handlers as v2_layer_handlers

        v2_handler = v2_layer_handlers.get(layer.__class__.__name__)
        if v2_handler is None:
            raise ValueError(f"No v2 handler found for {layer.__class__.__name__}")

        ret, _ = v2_handler(layer_dict, input_names, input_shapes, reader)

        # override / normalize the names used by the v3 graph parser
        ret["name"] = layer.name
        ret["class_name"] = ret.get("class_name", "QDense")
        ret["module"] = layer.__module__
        ret["input_keras_tensor_names"] = [t.name for t in in_tensors]
        ret["input_shape"] = [list(t.shape[1:]) for t in in_tensors]
        ret["output_keras_tensor_names"] = [t.name for t in out_tensors]

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