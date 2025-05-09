"""
This file includes optimizations related to BipolarQuant nodes.

"""

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
    This is for the case when scale is a (positive) 1 and zeropt is 0.
    It is a a 1:1 transformation of a BipolarQuant to an Activation.
    As an optimization, this is not called when the input is constant.
    """

    def match(self, node):
        # only matches after the other inputs are already folded
        is_match = (
            isinstance(node, BipolarQuant)
            and len(node.inputs) == 1
            and not isinstance(node.get_input_node(node.inputs[0]), Constant)
        )

        # Only match if the scale is 1 and the zero-point is 0s
        if is_match:  # to make sure this is a quant node with inputs
            scale = node.get_attr('scale')
            scale_unit = (scale == 1.0).all()
            is_match = scale_unit

        return is_match

    def transform(self, model, node):
        """
        Change quant node to Activation
        """
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

        new_node = model.make_node(Activation, new_name, attributes, list(node.inputs[0]), list(node.outputs))
        model.replace_node(node, new_node)
        return True


class FuseBipolarQuantWithConstant(OptimizerPass):
    """
    This is for the case when scale is 1 and zeropt is 0.
    """

    def match(self, node):

        # only matches after the other inputs are already folded
        # and scale is unit
        is_match = (
            isinstance(node, BipolarQuant)
            and len(node.inputs) == 1
            and isinstance(node.get_input_node(node.inputs[0]), Constant)
        )

        # Only match if the scale is 1 and the zero-point is 0s
        if is_match:  # to make sure this is a quant node with inputs
            scale = node.get_attr('scale')
            scale_unit = (scale == 1.0).all()
            is_match = scale_unit

        return is_match

    def transform(self, model, node):
        """
        Fuse Quant with Constant.
        """
        precision = XnorPrecisionType()
        quantizer = BinaryQuantizer(bits=1)

        const_node = node.get_input_node(node.inputs[0])
        const_node.set_attr('quantizer', quantizer)
        const_node.get_output_variable().type.precision = precision

        # remove the Quant node
        model.remove_node(node)
        return True
