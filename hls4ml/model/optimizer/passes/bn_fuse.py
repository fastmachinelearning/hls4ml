import numpy as np

from hls4ml.model.layers import BatchNormalization, Conv1D, Conv2D, Dense
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType, UnspecifiedPrecisionType


class FuseBatchNormalization(OptimizerPass):
    """
    OptimizerPass to merge a BatchNormalization layer with Dense or Conv layer, only if the Dense or Conv layer does not
    have the output type specified. There is a further check on the compatibility to merge: except in cases when merging a
    weight/scale of 1 or a bias of 0, this optimizer does not merge nodes when both the weight and scale or both biases
    are quantized.

    Note:  Consider restricting this to ApplyAlpha.  Batch Normalization quantization seems to be ignored.

    Note:  This optimizer may not be safe if weights are updateable. May need to turn off.
    """

    def match(self, node):
        prev_node = node.get_input_node()
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
                or ((s0 == np.ones_like(s0)).all() and prev_node.get_attr('weight_quantizer') is None)
                or ((s1 == np.ones_like(s1)).all() and node.get_attr('scale_quantizer') is None)
            )
            bias_compatible = (
                (prev_node.get_attr('bias_quantizer') is None and node.get_attr('bias_quantizer') is None)
                or ((b0 == np.zeros_like(b0)).all() and prev_node.get_attr('bias_quantizer') is None)
                or ((b1 == np.zeros_like(b1)).all() and node.get_attr('bias_quantizer') is None)
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

        parent_weight = parent_node.weights['weight']
        parent_bias = parent_node.weights['bias']

        bn_scale = node.weights['scale']
        bn_bias = node.weights['bias']

        allowed_precisions = (IntegerPrecisionType, FixedPrecisionType, UnspecifiedPrecisionType)

        # only merge if the types are integer or fixed
        if (
            not isinstance(parent_weight.type.precision, allowed_precisions)
            or not isinstance(parent_bias.type.precision, allowed_precisions)
            or not isinstance(bn_scale.type.precision, allowed_precisions)
            or not isinstance(bn_bias.type.precision, allowed_precisions)
        ):
            return False

        fused_weight = bn_scale.data * parent_weight.data
        fused_bias = bn_scale.data * parent_bias.data + bn_bias.data

        w_quantizer = (
            node.get_attr('scale_quantizer')
            if node.get_attr('scale_quantizer') is not None
            else parent_node.get_attr('weight_quantizer')
        )
        b_quantizer = (
            node.get_attr('bias_quantizer')
            if node.get_attr('bias_quantizer') is not None
            else parent_node.get_attr('bias_quantizer')
        )

        node.set_attr('weight_quantizer', w_quantizer)
        node.set_attr('bias_quantizer', b_quantizer)

        # call function so that quantizer would be called if needed
        parent_node.add_weights_variable(name='weight', var_name='w{index}', data=fused_weight, quantizer=w_quantizer)
        parent_node.add_weights_variable(name='bias', var_name='b{index}', data=fused_bias, quantizer=b_quantizer)

        model.remove_node(node, rewire=True)

        return True
