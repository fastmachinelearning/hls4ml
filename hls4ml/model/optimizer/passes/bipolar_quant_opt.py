"""
This file includes optimizations related to BipolarQuant nodes.

"""

import copy

import numpy as np

from hls4ml.model.layers import Activation, ApplyAlpha, BipolarQuant, Constant
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.quantizers import BinaryQuantizer
from hls4ml.model.types import XnorPrecisionType


class BipolarQuantConstantParameters(OptimizerPass):
    """Remove Constant from the BipolarQuant node parameters (but not input[0])"""

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

        attributes = {'activation': 'binary_tanh', 'quantizer': quantizer, 'quantizer_precision': precision}

        # update the configuration (not setting the precision since can't specify xnor type)
        config = model.config.get_layer_config(node)
        new_name = f'{node.name}_act'
        model.config.set_name_config(new_name, config)
        model.config.parse_name_config(new_name, config)

        new_node = model.make_node(Activation, new_name, attributes, [node.inputs[0]], list(node.outputs))
        model.replace_node(node, new_node)
        return True


class FuseBipolarQuantWithConstant(OptimizerPass):
    """
    This is for the case when scale is 1 and the input is a constant
    """

    def match(self, node):

        # only matches after the other inputs are already folded
        # and scale is unit
        is_match = (
            isinstance(node, BipolarQuant)
            and len(node.inputs) == 1
            and isinstance(node.get_input_node(node.inputs[0]), Constant)
        )

        # Only match if the scale is 1
        if is_match:  # to make sure this is a quant node with inputs
            scale = node.get_attr('scale')
            is_match = (scale == 1.0).all()

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


class BipolarQuantToAlphaActivationAlpha(OptimizerPass):
    """
    This is for the case when scale is not 1. It is a a 1:3 transformation of
    a BipolarQuant to an ApplyAlpha (to scale), Activation, ApplyAlpha (to rescale).
    Since there is no zero offset, the initial ApplyAlpha has no effect, so it is omitted.

    NOTE:  It needs to be scheduled after BipolarQuantToActivation (or we need to make the match criteria stricter)
    """

    def match(self, node):
        # only matches after the other inputs are already folded
        is_match = (
            isinstance(node, BipolarQuant)
            and len(node.inputs) == 1
            and not isinstance(node.get_input_node(node.inputs[0]), Constant)
        )
        return is_match

    def transform(self, model, node):
        """
        Change quant node to ApplyAlhpa, Activation, ApplyAlpha
        """

        # Do the Activation as in the simple case

        precision = XnorPrecisionType()
        quantizer = BinaryQuantizer(bits=1)

        activation_attributes = {'activation': 'binary_tanh', 'quantizer': quantizer, 'quantizer_precision': precision}

        # update the configuration (not setting the precision since can't specify xnor type)
        config = model.config.get_layer_config(node)
        act_config = copy.deepcopy(config)
        act_name = f'{node.name}_act'
        model.config.set_name_config(act_name, act_config)
        model.config.parse_name_config(act_name, act_config)

        new_node = model.make_node(Activation, act_name, activation_attributes, [node.inputs[0]], [x for x in node.outputs])
        model.replace_node(node, new_node)

        # but now add the ApplyAlphas before and after. Because of no zero offset,
        # the initial ApplyAlpha is omitted, since it has no effect.

        inshape = node.get_input_variable().shape

        scale = node.get_attr('scale')
        bias = np.array(0)

        attributes_rescale = {'n_filt': -1}

        rescale_config = config  # no need to deep copy the last
        rescale_name = f'{node.name}_rescale'
        model.config.set_name_config(rescale_name, rescale_config)
        model.config.parse_name_config(rescale_name, rescale_config)

        rescale = scale
        rebias = -bias * scale
        attributes_rescale['scale_data'] = np.broadcast_to(rescale, inshape)
        attributes_rescale['bias_data'] = np.broadcast_to(rebias, inshape)

        rescale_node = model.make_node(ApplyAlpha, rescale_name, attributes_rescale, [new_node.outputs[0]])
        model.insert_node(rescale_node)

        return True


class ConstBipolarQuantToConstAlpha(OptimizerPass):
    """
    This is for the case when scale is not 1. It is a a 1:3 transformation of
    a BipolarQuant to an ApplyAlpha (to scale), Activation, ApplyAlpho (to unscale), but an input
    consts allows for optimization, so the ApplyAlpha (to scale), Activation are
    optimized away right away.
    """

    def match(self, node):
        # only matches after the other inputs are already folded
        is_match = (
            isinstance(node, BipolarQuant)
            and len(node.inputs) == 1
            and isinstance(node.get_input_node(node.inputs[0]), Constant)
        )

        if is_match:  # to make sure this is a quant node with inputs
            scale = node.get_attr('scale')
            is_match = is_match and ((scale != np.ones_like(scale)).any())
        return is_match

    def transform(self, model, node):
        """
        Change Constant + Quant node to Constant, ApplyAlpha
        """

        precision = XnorPrecisionType()
        quantizer = BinaryQuantizer(bits=1)

        const_node = node.get_input_node(node.inputs[0])

        scale = node.get_attr('scale')
        bias = np.array(0)  # zeropt not defined for bipolar quants

        # Would logically calculate the new value here, but it is not needed because
        # bias == 0, so after quantization the result would be unchanged.

        const_node.set_attr('quantizer', quantizer)
        const_node.get_output_variable().type.precision = precision

        inshape = node.get_input_variable().shape

        attributes_rescale = {'n_filt': -1}

        rescale_config = copy.deepcopy(model.config.get_layer_config(node))
        rescale_name = f'{node.name}_rescale'
        model.config.set_name_config(rescale_name, rescale_config)
        model.config.parse_name_config(rescale_name, rescale_config)

        rescale = scale
        rebias = -bias * scale
        attributes_rescale['scale_data'] = np.broadcast_to(rescale, inshape)
        attributes_rescale['bias_data'] = np.broadcast_to(rebias, inshape)

        rescale_node = model.make_node(
            ApplyAlpha, rescale_name, attributes_rescale, [x for x in node.inputs], [x for x in node.outputs]
        )
        model.replace_node(node, rescale_node)

        return True
