import numpy as np

from hls4ml.model.layers import BatchNormalization, Conv1D, Conv2D, Dense
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import UnspecifiedPrecisionType


class FuseBatchNormalization(OptimizerPass):
    def match(self, node):
        prev_node = node.get_input_node(node.inputs[0])
        basic_match = (
            isinstance(node, BatchNormalization)
            and isinstance(prev_node, (Dense, Conv1D, Conv2D))
            and isinstance(prev_node.get_output_variable().type.precision, UnspecifiedPrecisionType)
        )
        if basic_match:
            s0 = prev_node.weights['weight'].data_unquantized
            b0 = prev_node.weights['bias'].data_unquantized
            s1 = node.weights['scale'].data_unquantized
            b1 = node.weights['bias'].data_unquantized
            scale_compatible = (
                (prev_node.get_attr('weight_quantizer') is None and node.get_attr('scale_quantizer') is None)
                or (s0 == np.ones_like(s0)).all()
                or (s1 == np.ones_like(s1)).all()
            )
            bias_compatible = (
                (prev_node.get_attr('bias_quantizer') is None and node.get_attr('bias_quantizer') is None)
                or (b0 == np.zeros_like(b0)).all()
                or (b1 == np.zeros_like(b1)).all()
            )
            return scale_compatible and bias_compatible

        else:
            return False

    def transform(self, model, node):
        """Fuse weight and bias of Dense/Conv1D/Conv2D layer with BN values."""
        parent_node = node.get_input_node()
        parent_map = parent_node.get_output_use_map()
        if len(parent_map[parent_node.outputs[0]]) > 1:
            return False

        # # Not sure why this part is needed
        # node_map = node.get_output_use_map()
        # if len(node_map[node.outputs[0]]) > 1:
        #     return False

        parent_weight = parent_node.weights['weight']
        parent_bias = parent_node.weights['bias']

        bn_scale = node.weights['scale']
        bn_bias = node.weights['bias']

        fused_weight = bn_scale.data * parent_weight.data
        fused_bias = bn_scale.data * parent_bias.data + bn_bias.data

        model.remove_node(node, rewire=True)
        parent_weight.data = fused_weight
        parent_bias.data = fused_bias
        if not parent_node.get_attr('use_bias', True):
            parent_bias.update_precision(bn_bias.type.precision)

        return True
