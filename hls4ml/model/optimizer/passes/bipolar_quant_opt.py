"""
This file includes optimizations related to BipolarQuant nodes.

As a first step, QuantConstantParameters converts the extra inputs to attributes.

The next step differs between the case of (1) (positive) power-of-2 scale and zero offset, or (2) other cases. In the first
case no explicit scaling is required, so a Quant node logically becomes a linear activation. (Cases when the scale is a
power of 2 not equal to one are implicitly scaled with fixed precision types.) When the activation is applied to a constant
weight, the activation is immediately merged with the weight, quantizing the weights. In case (2), we need to explicitly
scale and unscale, so the Quant node becomes 3 nodes, an ApplyAlpha node to apply a scale/shift, a Linear node to apply the
quantization, and another ApplyAlpha to unscale/shift. We depend on optimization steps to move the unscaling ApplyAlpha
down as needed so that we can do integer or fixed-point calculations. When the Quant is a applied to a weight, the scaling
and Linear nodes are immediately merged into the Constant.

"""

import numpy as np

from hls4ml.model.layers import Activation, BipolarQuant, Constant
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.quantizers import BinaryQuantizer
from hls4ml.model.types import XnorPrecisionType

_ALSO_MATCH_PO2 = True


class BipolarQuantConstantParameters(OptimizerPass):
    """Remove Constant from the Qaunt node parameters (but not input[0])"""

    def match(self, node):
        is_match = (
            isinstance(node, BipolarQuant)
            and len(node.inputs) == 2
            and (node.get_input_node(node.inputs[1]) and isinstance(node.get_input_node(node.inputs[1]), Constant))
        )

        return is_match

    def transform(self, model, node):
        """
        Remove Constant from the Quant node parameters (but not input[0])
        """
        if node.get_input_node(node.inputs[1]):
            scale_node = node.get_input_node(node.inputs[1])
            if isinstance(scale_node, Constant):
                node.set_attr('scale', scale_node.get_attr('value'))
                node.inputs[1] = ''
                model.remove_node(scale_node)

        node.inputs = [inp for inp in node.inputs if inp]
        if len(node.inputs) != 1:
            raise RuntimeError("hls4ml only supports constant scale")

        return True


class BipolarQuantToActivation(OptimizerPass):
    """
    This is for the case when scale is a (positive) power of 2 and zeropt is 0. It is a a 1:1 transformation of
    a BipolarQuant to an Activation.

    As an optimization, this is not called when the input is constant.
    """

    def match(self, node):
        # only matches after the other inputs are already folded

        is_match = (
            isinstance(node, BipolarQuant)
            and len(node.inputs) == 1
            and not isinstance(node.get_input_node(node.inputs[0]), Constant)
        )

        # Only match if the scale is power of 2 and the zero-point is 0s
        if is_match:  # to make sure this is a quant node with inputs
            scale = node.get_attr('scale')
            # check if scale is ones-like or a power of two
            scale_unit_or_po2 = (scale == np.ones_like(scale)).all()
            is_match = scale_unit_or_po2

        return is_match

    def transform(self, model, node):
        """
        Change quant node to Activation
        """
        scale = node.get_attr('scale')
        assert np.all(scale == 1.0)  # TODO: Is this required?

        precision = XnorPrecisionType()
        quantizer = BinaryQuantizer(bits=1)

        attributes = {'activation': 'linear', 'quantizer': quantizer}

        # update the configuration
        config = model.config.get_layer_config(node)
        prec_config = config.setdefault('Precision', {})
        prec_config['result'] = str(precision)
        new_name = f'{node.name}_act'
        model.config.set_name_config(new_name, config)
        model.config.parse_name_config(new_name, config)

        new_node = model.make_node(Activation, new_name, attributes, [node.inputs[0]], [x for x in node.outputs])
        model.replace_node(node, new_node)
        return True


class FuseBipolarQuantWithConstant(OptimizerPass):
    """
    This is for the case when scale is a positive power of 2 and zeropt is 0.
    """

    def match(self, node):
        # only matches after the other inputs are already folded
        is_match = (
            isinstance(node, BipolarQuant)
            and len(node.inputs) == 1
            and isinstance(node.get_input_node(node.inputs[0]), Constant)
        )

        # Only match if the scale is power of 2 and the zero-point is 0s
        if is_match:  # to make sure this is a quant node with inputs
            scale = node.get_attr('scale')

            # check if scale is ones-like or a power of two
            scale_unit_or_po2 = (scale == np.ones_like(scale)).all()
            is_match = scale_unit_or_po2

        return is_match

    def transform(self, model, node):
        """
        Fuse Quant with Constant.
        """

        scale = node.get_attr('scale')
        assert np.all(scale == 1.0)  # TODO: Is this required?

        precision = XnorPrecisionType()
        quantizer = BinaryQuantizer(bits=1)

        const_node = node.get_input_node(node.inputs[0])
        const_node.set_attr('quantizer', quantizer)
        const_node.get_output_variable().type.precision = precision

        # Should we update the configuration to reflect the new precision? I don't think it's necessary

        # remove the Quant node
        model.remove_node(node)

        return True
