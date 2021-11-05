import numpy as np
from hls4ml.model.hls_layers import FixedPrecisionType
from hls4ml.converters.onnx.quantizer import QuantNodeQuantizer
from hls4ml.model.optimizer import OptimizerPass

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
                model.remove_node(scale_node, rewire=False)

        if node.get_input_node(node.inputs[2]):
            zeropt_node = node.get_input_node(node.inputs[2])
            if zeropt_node.__class__.__name__ == 'Constant':
                node.set_attr('zeropt', zeropt_node.value)
                node.inputs[2] = ''
                model.remove_node(zeropt_node, rewire=False)

        if node.get_input_node(node.inputs[3]):
            bitwidth_node = node.get_input_node(node.inputs[3])
            if bitwidth_node.__class__.__name__ == 'Constant':
                node.set_attr('bitwidth', bitwidth_node.value)
                node.inputs[3] = ''
                model.remove_node(bitwidth_node, rewire=False)

        return True


class QuantToBatchNorm(OptimizerPass):
    """ Change Quant node to BaseBatchNormalization input[0]"""
    def match(self, node):
        is_match = (node.__class__.__name__ == 'Quant'
                    and not node.get_input_node(node.inputs[1])
                    and not node.get_input_node(node.inputs[2])
                    and not node.get_input_node(node.inputs[3]))

        # only matches after the other inputs are already folded
        return is_match

    def transform(self, model, node):
        """
        Change quant node to BaseBatchNormalization
        """
        bn_scale = 1/node.get_attr("scale")
        bn_bias = node.get_attr("zeropt")

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

        bn_precision = FixedPrecisionType(bitwidth, bitwidth, node.get_attr("signed"), bn_round, bn_sat)
        bn_quantizer = QuantNodeQuantizer(bn_precision)

        bn_layer = model.make_node("BaseBatchNormalization", f"bn_{node.name}",
                                   {"scale": bn_scale, "bias": bn_bias, "quant_precision": bn_precision, "quantizer": bn_quantizer},
                                   [node.inputs[0]], node.outputs)
        model.replace_node(node, bn_layer)

        return True
