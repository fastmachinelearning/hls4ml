from collections.abc import Sequence
from typing import Any

import numpy as np

from hls4ml.converters.keras_v3.core import KerasV3LayerHandler  # adjust import to your tree


class QKerasQActivationHandler(KerasV3LayerHandler):
    # IMPORTANT: match dispatcher key(s)
    handles = ("qkeras.qlayers.QActivation", "QActivation")

    def handle(
        self,
        layer,  # qkeras.qlayers.QActivation
        in_tensors: Sequence["KerasTensor"],
        out_tensors: Sequence["KerasTensor"],
    ) -> tuple[dict[str, Any], ...]:

        # --- v2 handler plumbing (same pattern as your dispatcher.v2_call) ---
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
        output_names = [t.name for t in out_tensors]

        from hls4ml.converters.keras_v2_to_hls import layer_handlers as v2_layer_handlers

        v2_handler = v2_layer_handlers.get(layer.__class__.__name__)
        if v2_handler is None:
            raise ValueError(f"No v2 handler found for {layer.__class__.__name__}")

        hls_conf, _ = v2_handler(layer_dict, input_names, input_shapes, reader)

        hls_conf["input_keras_tensor_names"] = list(input_names)
        hls_conf["output_keras_tensor_names"] = list(output_names)
        hls_conf.setdefault("name", layer.name)
        hls_conf.setdefault("class_name", "QActivation")

        return (hls_conf,)
