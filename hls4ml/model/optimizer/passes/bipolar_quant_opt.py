"""
This file includes optimizations related to BipolarQuant nodes.

"""

import numpy as np

from hls4ml.model.layers import Activation, BipolarQuant, Constant
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.quantizers import BinaryQuantizer
from hls4ml.model.types import XnorPrecisionType


class BipolarQuantConstantParameters(OptimizerPass):
    """Remove Constant from the BipolarQaunt node parameters (but not input[0])"""

    def match(self, node):
        is_match = (
            isinstance(node, BipolarQuant)
            and len(node.inputs) == 2
            and (node.get_input_node(node.inputs[1]) and isinstance(node.get_input_node(node.inputs[1]), Constant))
        )

        return is_match

    def transform(self, model, node):
        """
        Remove Constant from the BipolarQuant node parameters (but not input[0])
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
    This is for the case when scale is 1. It is a a 1:1 transformation of a BipolarQuant to an Activation.
    This is not called when the input is constant.
    """

    def match(self, node):
        # only matches after the other inputs are already folded
        is_match = (
            isinstance(node, BipolarQuant)
            and len(node.inputs) == 1
            and not isinstance(node.get_input_node(node.inputs[0]), Constant)
        )

        # Only match if the scale is 1
        if is_match:  # to make sure this is a quant node with inputs
            scale = node.get_attr('scale')
            is_match = (scale == 1.0).all()

        return is_match

    def transform(self, model, node):
        """
        Change BipolarQuant node to Activation
        """
        precision = XnorPrecisionType()
        quantizer = BinaryQuantizer(bits=1)

        attributes = {'activation': 'binary_tanh', 'quantizer': quantizer, 'precision': precision}

        # don't update the configuration because we can't manually set
        # the precision as xnor type
        new_name = f'{node.name}_act'
        new_node = model.make_node(Activation, new_name, attributes, [node.inputs[0]], list(node.outputs))
        model.replace_node(node, new_node)
        return True


class FuseBipolarQuantWithConstant(OptimizerPass):
    """
    This is for the case when scale is po2.
    """

    def match(self, node):

        # only matches after the other inputs are already folded
        # and scale is unit
        is_match = (
            isinstance(node, BipolarQuant)
            and len(node.inputs) == 1
            and isinstance(node.get_input_node(node.inputs[0]), Constant)
        )

        # Only match if the scale is po2
        if is_match:  # to make sure this is a quant node with inputs
            scale = node.get_attr('scale')
            scale_unit_or_po2 = (scale == 1.0).all()
            # This optimization only works if all scales are the same
            if np.all(scale[0] == scale):
                mantissa, _ = np.frexp(scale[0])
                scale_unit_or_po2 = mantissa == 0.5
            is_match = scale_unit_or_po2

        return is_match

    def transform(self, model, node):
        """
        Fuse BipolarQuant with Constant.
        """
        precision = XnorPrecisionType()
        quantizer = BinaryQuantizer(bits=1)

        const_node = node.get_input_node(node.inputs[0])
        const_node.set_attr('quantizer', quantizer)
        const_node.get_output_variable().type.precision = precision

        # remove the Quant node
        model.remove_node(node)
        return True
