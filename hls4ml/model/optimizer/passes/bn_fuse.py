import numpy as np

from hls4ml.model.layers import BatchNormalization, Conv1D, Conv2D, Dense
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType, UnspecifiedPrecisionType


class FuseBatchNormalization(OptimizerPass):
    """
    OptimizerPass to merge BatchNormalization layers,
    only if the earlier one does not have quantization specified

    Note:  Consider restricting this to ApplyAlpha.  Batch Normalization quantization seems to be ignored.

    Note:  This optimizer may not be safe if weights are updateable. May need to turn off.
    """

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

        # only merge if the types are integer or fixed
        if (
            not isinstance(parent_weight.type, (IntegerPrecisionType, FixedPrecisionType))
            or not isinstance(parent_bias.type, (IntegerPrecisionType, FixedPrecisionType))
            or not isinstance(bn_scale.type, (IntegerPrecisionType, FixedPrecisionType))
            or not isinstance(bn_bias.type, (IntegerPrecisionType, FixedPrecisionType))
        ):
            return False

        fused_weight = bn_scale.data * parent_weight.data
        fused_bias = bn_scale.data * parent_bias.data + bn_bias.data

        w_quantizer = (
            node.get_attr('scale_quantizer')
            if (parent_weight.data == np.ones_like(parent_weight.data)).all()
            else parent_node.get_attr('weight_quantizer')
        )
        b_quantizer = (
            node.get_attr('bias_quantizer')
            if (parent_bias.data == np.zeros_like(parent_bias.data)).all()
            else parent_node.get_attr('bias_quantizer')
        )

        node.set_attr('weight_quantizer', w_quantizer)
        node.set_attr('bias_quantizer', b_quantizer)

        # Not sure if this setting of this is useful
        w_prec = None
        if w_quantizer is None and (fused_weight == np.ones_like(fused_weight)).all():
            if (
                isinstance(parent_weight.type, IntegerPrecisionType)
                and isinstance(bn_scale.type, IntegerPrecisionType)
                and parent_weight.type.width == 1
                and bn_scale.type.width == 1
            ):
                w_prec = node.weights['scale'].type

        b_prec = None
        if b_quantizer is None and (fused_bias == np.zeros_like(fused_bias)).all():
            if (
                isinstance(parent_bias.type, IntegerPrecisionType)
                and isinstance(bn_bias.type, IntegerPrecisionType)
                and parent_bias.type.width == 1
                and bn_bias.type.width == 1
            ):
                b_prec = node.weights['bias'].type

        # call function so that quantizer would be called if needed
        node.add_weights_variable(
            name='weight', var_name='w{index}', data=fused_weight, quantizer=w_quantizer, precision=w_prec
        )
        node.add_weights_variable(name='bias', var_name='b{index}', data=fused_bias, quantizer=b_quantizer, precision=b_prec)

        model.remove_node(node, rewire=True)

        return True
