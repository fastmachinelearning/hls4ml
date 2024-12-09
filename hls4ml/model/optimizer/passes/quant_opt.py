"""
This file includes optimizations related to quant nodes.

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

import copy
import math  # prefer to use math.ceil for scalar values

import numpy as np

from hls4ml.model.layers import Activation, ApplyAlpha, Constant, Quant
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.quantizers import QuantNodeQuantizer
from hls4ml.model.types import FixedPrecisionType

_ALSO_MATCH_PO2 = True


class QuantConstantParameters(OptimizerPass):
    """Remove Constant from the Qaunt node parameters (but not input[0])"""

    def match(self, node):
        is_match = (
            isinstance(node, Quant)
            and len(node.inputs) == 4
            and (
                (node.get_input_node(node.inputs[1]) and isinstance(node.get_input_node(node.inputs[1]), Constant))
                or (node.get_input_node(node.inputs[2]) and isinstance(node.get_input_node(node.inputs[2]), Constant))
                or (node.get_input_node(node.inputs[3]) and isinstance(node.get_input_node(node.inputs[3]), Constant))
            )
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
                model.remove_node(scale_node, rewire=False)

        if node.get_input_node(node.inputs[2]):
            zeropt_node = node.get_input_node(node.inputs[2])
            if isinstance(zeropt_node, Constant):
                node.set_attr('zeropt', zeropt_node.get_attr('value'))
                node.inputs[2] = ''
                model.remove_node(zeropt_node, rewire=False)

        if node.get_input_node(node.inputs[3]):
            bitwidth_node = node.get_input_node(node.inputs[3])
            if isinstance(bitwidth_node, Constant):
                bitwidth = bitwidth_node.get_attr('value')
                if bitwidth.size != 1:
                    raise RuntimeError('Only scalar bitwidth values are supporeted by the Quant node')
                node.set_attr('bitwidth', bitwidth[0])
                node.inputs[3] = ''
                model.remove_node(bitwidth_node, rewire=False)

        node.inputs = [inp for inp in node.inputs if inp]
        if len(node.inputs) != 1:
            raise RuntimeError("hls4ml only supports constant scale, zeropt, and bitwidth values")

        return True


class QuantToActivation(OptimizerPass):
    """
    This is for the case when scale is a (positive) power of 2 and zeropt is 0. It is a a 1:1 transformation of
    a Quant to an Activation.

    As an optimization, this is not called when the input is constant.
    """

    def match(self, node):
        # only matches after the other inputs are already folded

        is_match = (
            isinstance(node, Quant)
            and len(node.inputs) == 1
            and not isinstance(node.get_input_node(node.inputs[0]), Constant)
        )

        # Only match if the scale is power of 2 and the zero-point is 0s
        if is_match:  # to make sure this is a quant node with inputs
            scale = node.get_attr('scale')
            bias = node.get_attr('zeropt')
            is_match = is_match and (bias == np.zeros_like(bias)).all()

            # check if scale is ones-like or a power of two
            scale_unit_or_po2 = (scale == np.ones_like(scale)).all()
            if not scale_unit_or_po2 and _ALSO_MATCH_PO2:
                # This optimization only works if all scales are the same
                if np.all(scale[0] == scale):
                    mantissa, _ = np.frexp(scale[0])
                    scale_unit_or_po2 = mantissa == 0.5

            is_match = scale_unit_or_po2

        return is_match

    def transform(self, model, node):
        """
        Change quant node to Activation
        """

        rounding_mode = node.get_attr('rounding_mode')
        narrow = node.get_attr('narrow')
        signed = node.get_attr('signed')
        bitwidth = node.get_attr('bitwidth')
        integer = bitwidth
        scale = node.get_attr('scale')
        if _ALSO_MATCH_PO2 and not (scale == np.ones_like(scale)).all():
            _, exp = np.frexp(scale[0])
            integer = bitwidth + exp - 1

        precision, quantizer = _calculate_precision_quantizer(bitwidth, integer, signed, narrow, rounding_mode)

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


class FuseQuantWithConstant(OptimizerPass):
    """
    This is for the case when scale is a positive power of 2 and zeropt is 0.
    """

    def match(self, node):
        # only matches after the other inputs are already folded
        is_match = (
            isinstance(node, Quant) and len(node.inputs) == 1 and isinstance(node.get_input_node(node.inputs[0]), Constant)
        )

        # Only match if the scale is power of 2 and the zero-point is 0s
        if is_match:  # to make sure this is a quant node with inputs
            scale = node.get_attr('scale')
            bias = node.get_attr('zeropt')
            is_match = is_match and (bias == np.zeros_like(bias)).all()

            # check if scale is ones-like or a power of two
            scale_unit_or_po2 = (scale == np.ones_like(scale)).all()
            if not scale_unit_or_po2 and _ALSO_MATCH_PO2:
                # This optimization only works if all scales are the same
                if np.all(scale.item(0) == scale):
                    mantissa, _ = np.frexp(scale.item(0))
                    scale_unit_or_po2 = mantissa == 0.5

            is_match = scale_unit_or_po2

        return is_match

    def transform(self, model, node):
        """
        Fuse Quant with Constant.
        """

        rounding_mode = node.get_attr('rounding_mode')
        narrow = node.get_attr('narrow')
        signed = node.get_attr('signed')
        bitwidth = node.get_attr('bitwidth')
        integer = bitwidth
        scale = node.get_attr('scale')
        if _ALSO_MATCH_PO2 and not (scale == np.ones_like(scale)).all():
            _, exp = np.frexp(scale.item(0))  # know that np.all(scale.item(0) == scale) must be true
            integer = bitwidth + exp - 1

        precision, quantizer = _calculate_precision_quantizer(bitwidth, integer, signed, narrow, rounding_mode)

        const_node = node.get_input_node(node.inputs[0])
        const_node.set_attr('quantizer', quantizer)
        const_node.get_output_variable().type.precision = precision

        # Should we update the configuration to reflect the new precision? I don't think it's necessary

        # remove the Quant node
        model.remove_node(node, rewire=True)

        return True


class QuantToAlphaActivationAlpha(OptimizerPass):
    """
    This is for the case when scale is not power-of-2 or zeropt is not 0. It is a a 1:3 transformation of
    a Quant to an ApplyAlpha (to scale), Activatio, ApplyAlpho (to rescale).

    NOTE:  It needs to be scheduled after QuantToActivation (or we need to make the match criteria stricter)
    """

    def match(self, node):
        # only matches after the other inputs are already folded
        is_match = (
            isinstance(node, Quant)
            and len(node.inputs) == 1
            and not isinstance(node.get_input_node(node.inputs[0]), Constant)
        )
        return is_match

    def transform(self, model, node):
        """
        Change quant node to ApplyAlhpa, Activation, ApplyAlpha
        """

        # Do the Activation as in the simple case

        rounding_mode = node.get_attr('rounding_mode')
        narrow = node.get_attr('narrow')
        signed = node.get_attr('signed')
        bitwidth = node.get_attr('bitwidth')

        precision, quantizer = _calculate_precision_quantizer(bitwidth, bitwidth, signed, narrow, rounding_mode)

        activation_attributes = {'activation': 'linear', 'quantizer': quantizer}

        # update the configuration
        config = model.config.get_layer_config(node)
        act_config = copy.deepcopy(config)
        prec_config = act_config.setdefault('Precision', {})
        prec_config['result'] = str(precision)
        act_name = f'{node.name}_act'
        model.config.set_name_config(act_name, act_config)
        model.config.parse_name_config(act_name, act_config)

        new_node = model.make_node(Activation, act_name, activation_attributes, [node.inputs[0]], [x for x in node.outputs])
        model.replace_node(node, new_node)

        # but now add the ApplyAlhpas before and after

        inshape = node.get_input_variable().shape

        scale = node.get_attr('scale')
        bias = node.get_attr('zeropt')

        attributes_scale = {'n_filt': -1}
        attributes_rescale = {'n_filt': -1}

        scale_config = copy.deepcopy(config)
        scale_name = f'{node.name}_scale'
        model.config.set_name_config(scale_name, scale_config)
        model.config.parse_name_config(scale_name, scale_config)

        rescale_config = config  # no need to deep copy the last
        rescale_name = f'{node.name}_rescale'
        model.config.set_name_config(rescale_name, rescale_config)
        model.config.parse_name_config(rescale_name, rescale_config)

        firstscale = 1 / scale
        firstbias = bias
        attributes_scale['scale_data'] = np.broadcast_to(firstscale, inshape)
        attributes_scale['bias_data'] = np.broadcast_to(firstbias, inshape)

        scale_node = model.make_node(ApplyAlpha, scale_name, attributes_scale, [node.inputs[0]])
        model.insert_node(scale_node)

        rescale = scale
        rebias = -bias * scale
        attributes_rescale['scale_data'] = np.broadcast_to(rescale, inshape)
        attributes_rescale['bias_data'] = np.broadcast_to(rebias, inshape)

        rescale_node = model.make_node(ApplyAlpha, rescale_name, attributes_rescale, [new_node.outputs[0]])
        model.insert_node(rescale_node)

        return True


class ConstQuantToConstAlpha(OptimizerPass):
    """
    This is for the case when scale is not power-of-2 or zeropt is not 0. It is a a 1:3 transformation of
    a Quant to an ApplyAlpha (to scale), Activation, ApplyAlpho (to unscale), but an input
    consts allows for optimization, so the ApplyAlpha (to scale), Activation are
    optimized away right away.
    """

    def match(self, node):
        # only matches after the other inputs are already folded
        is_match = (
            isinstance(node, Quant) and len(node.inputs) == 1 and isinstance(node.get_input_node(node.inputs[0]), Constant)
        )

        if is_match:  # to make sure this is a quant node with inputs
            scale = node.get_attr('scale')
            bias = node.get_attr('zeropt')
            is_match = is_match and ((scale != np.ones_like(scale)).any() or (bias != np.zeros_like(bias)).any())
        return is_match

    def transform(self, model, node):
        """
        Change Constant + Quant node to Constant, ApplyAlpha
        """

        rounding_mode = node.get_attr('rounding_mode')
        narrow = node.get_attr('narrow')
        signed = node.get_attr('signed')
        bitwidth = node.get_attr('bitwidth')

        precision, quantizer = _calculate_precision_quantizer(bitwidth, bitwidth, signed, narrow, rounding_mode)

        const_node = node.get_input_node(node.inputs[0])

        scale = node.get_attr('scale')
        bias = node.get_attr('zeropt')

        # caclucate the new value
        new_val = const_node.get_attr('value') / scale + bias
        const_node.set_attr('value', new_val)
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


def _calculate_precision_quantizer(bitwidth, integer, signed, narrow, rounding_mode):
    """
    A function to determine the precision and quantizer
    """
    if rounding_mode == 'ROUND':
        bn_round = 'AP_RND_CONV'
    elif rounding_mode == 'FLOOR':
        bn_round = 'AP_TRN'
    else:
        raise NotImplementedError(
            f'Rounding mode {rounding_mode} not supported in Quant node. Only ROUND and FLOOR supported.'
        )

    if narrow and not signed:
        raise NotImplementedError('Narrow mode is only supported for singed numbers.')

    if narrow:
        bn_sat = 'AP_SAT_SYM'
    else:
        bn_sat = 'AP_SAT'

    bitwidth = math.ceil(bitwidth)
    integer = math.ceil(integer)

    precision = FixedPrecisionType(bitwidth, integer, signed, bn_round, bn_sat)
    quantizer = QuantNodeQuantizer(precision)
    return (precision, quantizer)
