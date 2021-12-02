import numpy as np
from hls4ml.model.hls_layers import FixedPrecisionType, Constant
from hls4ml.converters.onnx.quantizer import QuantNodeQuantizer
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.optimizer.passes.qkeras import ApplyAlpha

class QuantConstantParameters(OptimizerPass):
    """ Remove Constant from the Qaunt node parameters (but not input[0]) """
    def match(self, node):
        is_match = (node.__class__.__name__ == 'Quant'
                    and ((node.get_input_node(node.inputs[1])
                          and node.get_input_node(node.inputs[1]).__class__.__name__ == 'Constant')
                         or (node.get_input_node(node.inputs[2])
                             and node.get_input_node(node.inputs[2]).__class__.__name__ == 'Constant')
                         or (node.get_input_node(node.inputs[3])
                             and node.get_input_node(node.inputs[3]).__class__.__name__ == 'Constant')))

        return is_match

    def transform(self, model, node):
        """
        Remove Constant from the Qaunt node parameters (but not input[0])
        """
        if node.get_input_node(node.inputs[1]):
            scale_node = node.get_input_node(node.inputs[1])
            if scale_node.__class__.__name__ == 'Constant':
                node.set_attr('scale', scale_node.value)
                node.inputs[1] = ''
                node.attributes["scale_precision"] = scale_node.get_attr("quant_precision")
                model.remove_node(scale_node, rewire=False)

        if node.get_input_node(node.inputs[2]):
            zeropt_node = node.get_input_node(node.inputs[2])
            if zeropt_node.__class__.__name__ == 'Constant':
                node.set_attr('zeropt', zeropt_node.value)
                node.inputs[2] = ''
                node.attributes["bias_precision"] = zeropt_node.get_attr("quant_precision")
                model.remove_node(zeropt_node, rewire=False)

        if node.get_input_node(node.inputs[3]):
            bitwidth_node = node.get_input_node(node.inputs[3])
            if bitwidth_node.__class__.__name__ == 'Constant':
                node.set_attr('bitwidth', bitwidth_node.value)
                node.inputs[3] = ''
                model.remove_node(bitwidth_node, rewire=False)

        return True

class QuantFactorizeScale(OptimizerPass):
    '''
    Extract scale and zero-point from Quant Node
    '''
    def match(self, node):
        # only matches after the other inputs are already folded

        is_match = (node.__class__.__name__ == 'Quant'
                    and not node.get_input_node(node.inputs[1])
                    and not node.get_input_node(node.inputs[2])
                    and not node.get_input_node(node.inputs[3]))
        
        # Only match if the scale is not 1s and the zero-point is not 0s
        if is_match and node.get_input_variable() is not None: # to make sure this is a quant node with inputs
            input_shape = node.get_input_variable().shape
            scale = np.broadcast_to(1/node.get_attr("scale"), input_shape)
            bias = np.broadcast_to(node.get_attr("zeropt"), input_shape)
            is_match = is_match and (scale != np.ones_like(scale)).any()
            is_match = is_match and (bias != np.zeros_like(bias)).any()
        return is_match

    def transform(self, model, node):
        '''
        Insert an ApplyAlpha layer to factorize the scales
        '''
        input_shape = node.get_input_variable().shape

        scale = np.broadcast_to(1/node.get_attr('scale'), input_shape)
        bias = np.broadcast_to(node.get_attr('zeropt'), input_shape)
        # Unset the scale and zero-point so we don't try to factorize again
        node.set_attr('scale', 1)
        node.set_attr('zeropt', 0)

        # TODO derive these
        scale_precision = FixedPrecisionType()
        scale_quantizer = QuantNodeQuantizer(scale_precision)
        bias_precision = FixedPrecisionType()

        attrs = {
            'name' : node.get_attr('name') + '_alpha',
            'class_name' : 'Alpha',
            'inputs' : node.outputs,
            'n_in' : node.get_attr('n_out'),
            'n_filt' : node.get_attr('n_filt', -1),
            'reuse_factor' : node.get_attr('reuse_factor'),
            'bias_t' : bias_precision, 
            'scale_t' : scale_precision,
            'Trace' : node.get_attr('Trace', False) 
        }
        alpha_layer = model.make_node('ApplyAlpha', node.name + '_alpha', attrs, node.outputs)

        alpha_layer.add_weights(scale, quantizer=scale_quantizer)
        alpha_layer.add_bias(bias, quantizer=None)
        model.insert_node(alpha_layer)
 
        return True

class QuantToActivation(OptimizerPass):
    ''' Change Quant node to Activation input[0]'''
    def match(self, node):
        # only matches after the other inputs are already folded
        is_match = (node.__class__.__name__ == 'Quant'
                    and not isinstance(node.get_input_node(), Constant)
                    and not node.get_input_node(node.inputs[1])
                    and not node.get_input_node(node.inputs[2])
                    and not node.get_input_node(node.inputs[3]))
        
        # Only match if the scale is 1s and the zero-point is 0s
        if is_match: # to make sure this is a quant node with inputs
            input_shape = node.get_input_variable().shape
            scale = np.broadcast_to(1/node.get_attr("scale"), input_shape)
            bias = np.broadcast_to(node.get_attr("zeropt"), input_shape)
            is_match = is_match and (scale == np.ones_like(scale)).all()
            is_match = is_match and (bias == np.zeros_like(bias)).all()
        return is_match

    def transform(self, model, node):
        '''
        Change quant node to Activation
        '''
        input_shape = node.get_input_variable().shape

        n_in = np.prod(input_shape)

        rounding_mode = node.get_attr("rounding_mode")
        if rounding_mode == "ROUND":
            bn_round = "AP_RND_CONV"
        elif rounding_mode == "FLOOR":
            bn_round =  "AP_TRN"
        else:
            raise NotImplementedError(f"Rounding mode {rounding_mode} not supported in Quant node. Only ROUND and FLOOR supported.")

        if node.get_attr("narrow") and not node.get_attr("signed"):
            raise NotImplementedError("Narrow mode is only supported for singed numbers.")

        if node.get_attr("narrow"):
            bn_sat = "AP_SAT_SYM"
        else:
            bn_sat = "AP_SAT"

        bitwidth = node.get_attr("bitwidth")
        if np.squeeze(bitwidth).shape:
            raise RuntimeError("Only scalar bitwidth values are supporeted by the Quant node")
        bitwidth = int(bitwidth)

        precision = FixedPrecisionType(bitwidth, bitwidth, node.get_attr("signed"), bn_round, bn_sat)
        quantizer = QuantNodeQuantizer(precision)

        attributes = {
            'activation' : 'linear',
            'precision'  : precision,
            'n_in'       : n_in,
            'n_out'      : n_in,
            'n_filt'     : -1
        }

        new_node = model.make_node('Activation', f'{node.name}_act',
                                   attributes, [node.inputs[0]], node.outputs)
        new_node.get_output_variable().type.precision = precision
        model.replace_node(node, new_node)

        return True

class QuantToConstant(OptimizerPass):
    '''
    Remove a Quant node that is quantizing a constant.
    Update the attributes of the constant according to the quantization.
    '''

    def match(self, node):
        is_match = (node.__class__.__name__ == 'Quant'
                    and isinstance(node.get_input_node(node.inputs[0]), Constant))
        return is_match

    def transform(self, model, node):
        const_node = node.get_input_node(node.inputs[0])

        new_val = const_node.value * node.get_attr('scale') + node.get_attr('zeropt')
        quantizer = node.get_attr('quantizer')  # None if not defined
        if quantizer:
            const_node.set_attr('quantizer', quantizer)
        const_node.set_attr('value', new_val)

        quant_precision = node.get_attr('quant_precision')
        if quant_precision:
            const_node.set_attr('quant_precision', quant_precision)

        # reinitialize (which also runs quantization if quantizer exists)
        const_node.initialize()

        # remove the Quant node
        model.remove_node(node, rewire=True)
       
        return True