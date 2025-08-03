# Typing imports
from __future__ import annotations # makes all annotations into strings
from typing import List, Literal, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from hls4ml.model.graph import ModelGraph
    from hls4ml.model.layers import Layer

from hls4ml.model.optimizer import OptimizerPass

    
class MergeDenseRelu(OptimizerPass):
    """Merges a dense layer followed by a relu layer in one layer by
    applying the relu function immediately after each dot product. 
    """

    def match(self, node) -> bool:
        """We first match a dense layer and in the transform step we merge any following ReLU layers."""
        if node.class_name == 'Dense':
            return True
        return False

    def transform(self, model: ModelGraph, node: Layer) -> Literal[False]:        

        layers: list[Layer] = list(model.get_layers())
        for i, layer in enumerate(layers[:-1]):
            next_layer = layers[i + 1]
            if layer == node and next_layer.class_name == 'Activation':
                new_func_call = f'fc::dense_relu<{layer.get_attr("in_nb")}, {layer.get_attr("in_en")}, {layer.get_attr("in_bu")}, {next_layer.get_attr("out_nb")}, {next_layer.get_attr("out_en")}, {next_layer.get_attr("out_bu")}>'
                layer.set_attr('func_call', new_func_call)
                next_layer.set_attr('write_func', False)

        return False

