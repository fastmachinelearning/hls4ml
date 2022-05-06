'''
This file includes optimizations related to quant nodes.

As a first step, QuantConstantParameters converts the extra inputs to attributes. It is always the first step

The next step differs between the case of (1) unitary scale and zero offset, or (2) nonunitary scale and/or nonzero offset.
In the first case no scaling is required, so a Quant node effectively becomes a linear activation. For the common case when this
is applied on a constant weight, the activation is immediately merged with the weight, qantizing the weights. In case 2,
we need to explictly scale and unscale, so the Quant node becomes 3 nodes, an ApplyAlpha node to apply a scale/shift, a
Linear node to apply the quantization, and another ApplyAlpha to unscale/shift. We depend on optimization steps to move the
unscaling ApplyAlpha down as needed. Again, when the Quant is a applied ot a Constant, the scaling and Linear nodes are
immediately merged into the Constant. This is done because it simplifies some of the other optimizations.

UPDATE:  Case 1 is loosened to also include power of 2 scalar scales, not just unitary scale, if 
    _ALSO_INCLUDE_PO2 is set to true (the default)

'''
from copy import deepcopy
import numpy as np
import math  # prefer to use math.ceil for scalar values
from hls4ml.model.types import FixedPrecisionType
from hls4ml.model.layers import Quant, Constant, Activation, ApplyAlpha
from hls4ml.converters.onnx.quantizer import QuantNodeQuantizer
from hls4ml.model.optimizer import OptimizerPass

_ALSO_MATCH_PO2 = True

_base_attributes = ('Trace', 'reuse_factor')

class QuantConstantParameters(OptimizerPass):
    """ Remove Constant from the Qaunt node parameters (but not input[0]) """
    def match(self, node):
        is_match = (isinstance(node, Quant)
                    and ((node.get_input_node(node.inputs[1])
                          and isinstance(node.get_input_node(node.inputs[1]), Constant))
                         or (node.get_input_node(node.inputs[2])
                             and isinstance(node.get_input_node(node.inputs[2]), Constant))
                         or (node.get_input_node(node.inputs[3])
                             and isinstance(node.get_input_node(node.inputs[3]), Constant))))

        return is_match

    def transform(self, model, node):
        """
        Remove Constant from the Qaunt node parameters (but not input[0])
        """
        if node.get_input_node(node.inputs[1]):
            scale_node = node.get_input_node(node.inputs[1])
            if isinstance(scale_node, Constant):
                node.set_attr('scale', scale_node.value)
                node.inputs[1] = ''
                model.remove_node(scale_node, rewire=False)

        if node.get_input_node(node.inputs[2]):
            zeropt_node = node.get_input_node(node.inputs[2])
            if isinstance(zeropt_node, Constant):
                node.set_attr('zeropt', zeropt_node.value)
                node.inputs[2] = ''
                model.remove_node(zeropt_node, rewire=False)

        if node.get_input_node(node.inputs[3]):
            bitwidth_node = node.get_input_node(node.inputs[3])
            if isinstance(bitwidth_node, Constant):
                if np.squeeze(bitwidth_node.value).shape:
                    raise RuntimeError("Only scalar bitwidth values are supporeted by the Quant node")
                node.set_attr('bitwidth', bitwidth_node.value)
                node.inputs[3] = ''
                model.remove_node(bitwidth_node, rewire=False)

        return True


class QuantToActivation(OptimizerPass):
    '''
    This is for the case when scale is 1 and zeropt is 0. It is a a 1:1 transformation of
    a Quant to an Activation.

    As an optimization, this is not called when the input is constant.

    UPDATE:  this is also called when scale is scalar and power of 2, not just 1.
    '''
    def match(self, node):
        # only matches after the other inputs are already folded

        is_match = (isinstance(node, Quant)
                    and not isinstance(node.get_input_node(node.inputs[0]), Constant)
                    and not node.get_input_node(node.inputs[1])
                    and not node.get_input_node(node.inputs[2])
                    and not node.get_input_node(node.inputs[3]))

        # Only match if the scale is 1s and the zero-point is 0s
        if is_match: # to make sure this is a quant node with inputs
            scale = node.get_attr("scale")
            bias = node.get_attr("zeropt")
            is_match = is_match and (bias == np.zeros_like(bias)).all()

            # check if scale is ones-like or a power of two
            scale_unit_or_po2 = (scale == np.ones_like(scale)).all()
            if not scale_unit_or_po2 and _ALSO_MATCH_PO2:
                sqscale = np.squeeze(scale)
                if not sqscale.shape:
                    # not an array
                    mantissa, _ = np.frexp(sqscale)
                    scale_unit_or_po2 = mantissa == 0.5

            is_match = is_match and scale_unit_or_po2

        return is_match

    def transform(self, model, node):
        '''
        Change quant node to Activation
        '''
        input_shape = node.get_input_variable().shape

        n_in = np.prod(input_shape)

        rounding_mode = node.get_attr("rounding_mode")
        narrow = node.get_attr("narrow")
        signed = node.get_attr("signed")
        bitwidth = node.get_attr("bitwidth")
        integer = bitwidth
        scale = node.get_attr("scale")
        if _ALSO_MATCH_PO2 and not (scale == np.ones_like(scale)).all():
            _, exp = np.frexp(np.squeeze(scale))
            integer = bitwidth + exp - 1

        precision, quantizer = _calculate_precision_quantizer(bitwidth, integer, signed, narrow, rounding_mode)

        attributes = {k: node.attributes.get(k, None) for k in _base_attributes}
        attributes.update({
            'activation' : 'linear',
            'quant_precision'  : precision,
            'quantizer'  : quantizer,
            'n_in'       : n_in
        })

        new_node = model.make_node(Activation, f'{node.name}_act',
                                   attributes, [node.inputs[0]], [x for x in node.outputs])
        new_node.get_output_variable().type.precision = precision
        model.replace_node(node, new_node)

        return True


class FuseQuantWithConstant(OptimizerPass):
    '''
    This is for the case when scale is 1 and zeropt is 0. It directly applies the quantization to a constant.
    UPDATE:  this is also called when scale is scalar and power of 2, not just 1.
    '''
    def match(self, node):
        # only matches after the other inputs are already folded
        is_match = (isinstance(node, Quant)
                    and isinstance(node.get_input_node(node.inputs[0]), Constant)
                    and not node.get_input_node(node.inputs[1])
                    and not node.get_input_node(node.inputs[2])
                    and not node.get_input_node(node.inputs[3]))

        # Only match if the scale is 1s and the zero-point is 0s
        if is_match: # to make sure this is a quant node with inputs
            input_shape = node.get_input_variable().shape
            scale = node.get_attr("scale")
            bias = node.get_attr("zeropt")
            is_match = is_match and (bias == np.zeros_like(bias)).all()

            # check if scale is ones-like or a power of two
            scale_unit_or_po2 = (scale == np.ones_like(scale)).all()
            if not scale_unit_or_po2 and _ALSO_MATCH_PO2:
                sqscale = np.squeeze(scale)
                if not sqscale.shape:
                    # not an array
                    mantissa, _ = np.frexp(sqscale)
                    scale_unit_or_po2 = mantissa == 0.5

            is_match = is_match and scale_unit_or_po2

        return is_match

    def transform(self, model, node):
        '''
        Fuse Quant with Constant.
        '''

        rounding_mode = node.get_attr("rounding_mode")
        narrow = node.get_attr("narrow")
        signed = node.get_attr("signed")
        bitwidth = node.get_attr("bitwidth")
        integer = bitwidth
        scale = node.get_attr("scale")
        if _ALSO_MATCH_PO2 and not (scale == np.ones_like(scale)).all():
            _, exp = np.frexp(np.squeeze(scale))
            integer = bitwidth + exp - 1

        precision, quantizer = _calculate_precision_quantizer(bitwidth, integer, signed, narrow, rounding_mode)

        const_node = node.get_input_node(node.inputs[0])
        const_node.set_attr("quant_precision", precision)
        const_node.set_attr("quantizer", quantizer)

        # reinitialize (which also runs quantization if quantizer exists)
        const_node.initialize()

        # remove the Quant node
        model.remove_node(node, rewire=True)

        return True


class QuantToAlphaActivationAlpha(OptimizerPass):
    '''
    This is for the case when scale is not 1 or zeropt is not 0. It is a a 1:3 transformation of
    a Quant to an ApplyAlpha (to scale), Activatio, ApplyAlpho (to rescale).

    As an optimization, this is not called when the input is constant.
    '''
    def match(self, node):
        # only matches after the other inputs are already folded
        is_match = (isinstance(node, Quant)
                    and not isinstance(node.get_input_node(node.inputs[0]), Constant)
                    and not node.get_input_node(node.inputs[1])
                    and not node.get_input_node(node.inputs[2])
                    and not node.get_input_node(node.inputs[3]))

        if is_match: # to make sure this is a quant node with inputs
            input_shape = node.get_input_variable().shape
            scale = node.get_attr("scale")
            bias = node.get_attr("zeropt")
            is_match = is_match and ((scale != np.ones_like(scale)).any() or (bias != np.zeros_like(bias)).any())
        return is_match

    def transform(self, model, node):
        '''
        Change quant node to ApplyAlhpa, Activation, ApplyAlpha
        '''

        # Do the Activation as in the simple case

        input_shape = node.get_input_variable().shape

        n_in = np.prod(input_shape)

        rounding_mode = node.get_attr("rounding_mode")
        narrow = node.get_attr("narrow")
        signed = node.get_attr("signed")
        bitwidth = node.get_attr("bitwidth")

        precision, quantizer = _calculate_precision_quantizer(bitwidth, bitwidth, signed, narrow, rounding_mode)

        attributes = {k: node.attributes.get(k, None) for k in _base_attributes}
        attributes.update({
            'activation' : 'linear',
            'quant_precision'  : precision,
            'quantizer'  : quantizer,
            'n_in'       : n_in
        })

        new_node = model.make_node(Activation, f'{node.name}_act',
                                   attributes, [node.inputs[0]], [x for x in node.outputs])
        new_node.get_output_variable().type.precision = precision
        model.replace_node(node, new_node)

        # but now add the ApplyAlhpas before and after

        scale = node.get_attr("scale")
        bias = node.get_attr("zeropt")

        attributes_scale = {k: node.attributes.get(k, None) for k in _base_attributes}
        attributes_scale.update({
            'n_in': n_in,
            'n_out': n_in,
            'n_filt': -1
        })

        attributes_rescale = {k: node.attributes.get(k, None) for k in _base_attributes}
        attributes_rescale.update({
            'n_in': n_in,
            'n_out': n_in,
            'n_filt': -1
        })

        firstscale = 1/scale
        firstbias = bias
        attributes_scale["scale_data"] = firstscale
        attributes_scale["bias_data"] = firstbias

        scale_node = model.make_node(ApplyAlpha, node.name + '_scale', attributes_scale, [node.inputs[0]])
        model.insert_node(scale_node)

        rescale = scale
        rebias = -bias*scale
        attributes_rescale["scale_data"] = rescale
        attributes_rescale["bias_data"] = rebias

        rescale_node = model.make_node(ApplyAlpha, node.name + '_rescale', attributes_rescale, [new_node.outputs[0]])
        model.insert_node(rescale_node)

        return True


class ConstQuantToConstAlpha(OptimizerPass):
    '''
    This is for the case when scale is not 1 or zeropt is not 0. It is a a 1:3 transformation of
    a Quant to an ApplyAlpha (to scale), Activation, ApplyAlpho (to unscale), but an input
    consts allows for optimization, so the ApplyAlpha (to scale), Activation are
    optimized away right away.
    '''
    def match(self, node):
        # only matches after the other inputs are already folded
        is_match = (isinstance(node, Quant)
                    and isinstance(node.get_input_node(node.inputs[0]), Constant)
                    and not node.get_input_node(node.inputs[1])
                    and not node.get_input_node(node.inputs[2])
                    and not node.get_input_node(node.inputs[3]))

        if is_match: # to make sure this is a quant node with inputs
            input_shape = node.get_input_variable().shape
            scale = node.get_attr("scale")
            bias = node.get_attr("zeropt")
            is_match = is_match and ((scale != np.ones_like(scale)).any() or (bias != np.zeros_like(bias)).any())
        return is_match

    def transform(self, model, node):
        '''
        Change Constant + Quant node to Constant, ApplyAlpha
        '''

        # Do the Activation as in the simple case

        input_shape = node.get_input_variable().shape

        n_in = np.prod(input_shape)

        rounding_mode = node.get_attr("rounding_mode")
        narrow = node.get_attr("narrow")
        signed = node.get_attr("signed")
        bitwidth = node.get_attr("bitwidth")

        precision, quantizer = _calculate_precision_quantizer(bitwidth, bitwidth, signed, narrow, rounding_mode)

        const_node = node.get_input_node(node.inputs[0])

        scale = node.get_attr("scale")
        bias = node.get_attr("zeropt")

        # caclucate the new value
        new_val = const_node.value / scale + bias
        const_node.set_attr('value', new_val)
        const_node.set_attr("quant_precision", precision)
        const_node.set_attr("quantizer", quantizer)

        # reinitialize (which also runs quantization if quantizer exists)
        const_node.initialize()

        attributes_rescale = {k: node.attributes.get(k, None) for k in _base_attributes}
        attributes_rescale.update({
            'n_in': n_in,
            'n_out': n_in,
            'n_filt': -1
        })

        rescale = scale
        rebias = -bias*scale
        attributes_rescale["scale_data"] = rescale
        attributes_rescale["bias_data"] = rebias

        rescale_node = model.make_node(ApplyAlpha, node.name + '_rescale', attributes_rescale,
             [x for x in node.inputs], [x for x in node.outputs])
        model.replace_node(node, rescale_node)

        return True


def _calculate_precision_quantizer(bitwidth, integer, signed, narrow, rounding_mode):
    '''
    A function to determine the precision and quantizer
    '''
    if rounding_mode == "ROUND":
        bn_round = "AP_RND_CONV"
    elif rounding_mode == "FLOOR":
        bn_round =  "AP_TRN"
    else:
        raise NotImplementedError(f"Rounding mode {rounding_mode} not supported in Quant node. Only ROUND and FLOOR supported.")

    if narrow and not signed:
        raise NotImplementedError("Narrow mode is only supported for singed numbers.")

    if narrow:
        bn_sat = "AP_SAT_SYM"
    else:
        bn_sat = "AP_SAT"

    bitwidth = math.ceil(bitwidth)
    integer = math.ceil(integer)

    precision = FixedPrecisionType(bitwidth, integer, signed, bn_round, bn_sat)
    quantizer = QuantNodeQuantizer(precision)
    return (precision, quantizer)
