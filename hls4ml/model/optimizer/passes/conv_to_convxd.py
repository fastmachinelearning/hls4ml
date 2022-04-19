import numpy as np
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import FixedPrecisionType
from hls4ml.model.layers import Conv, Constant

class ConvToConvXD(OptimizerPass):
    """ Convert Conv with constant to a Conv1D or Conv2D layer """
    def match(self, node):
        is_match = (isinstance(node, Conv)
                    and ((len(node.inputs) == 2 and isinstance(node.get_input_node(node.inputs[1]), Constant))
                          or (len(node.inputs) == 3
                              and isinstance(node.get_input_node(node.inputs[1]), Constant)
                              and isinstance(node.get_input_node(node.inputs[2]), Constant))))

        return is_match

    def transform(self, model, node):
        """ Convert Conv with constant to a Conv1D or Conv2D layer """

        input_node = node.get_input_node(node.inputs[0])
        input_precision = input_node.get_attr("quant_precision")
        weight_node = node.get_input_node(node.inputs[1])
        weight_precision = weight_node.get_attr("quant_precision")
        bias_node = None
        bias_precision = None
        if len(node.inputs) == 3:
            bias_node = node.get_input_node(node.inputs[2])
            bias_precision = bias_node.get_attr("quant_precision")

        # copy the attributes to the new node. (No need to explictily copy since the old node is deleted)
        attributes = node.attributes

        quant_precision = None

        if weight_precision and input_precision and (bias_precision or not bias_node):
            if (weight_precision.width != weight_precision.integer
                or input_precision.width != input_precision.integer):
                raise ValueError("quant_precisions must always have the same width and integer parameters")

            num_feature_maps = weight_node.value.shape[0]
            Nacc = attributes['filt_width'] * attributes.get('filt_height', 1) * num_feature_maps
            bitwidth = weight_precision.width + input_precision.width + int(np.ceil(np.log2(Nacc)))
            signed = weight_precision.signed or input_precision.signed
            # copy staruation and rounding from "other"
            rounding_mode = input_precision.rounding_mode
            saturation_mode = input_precision.saturation_mode

            # correct if bias
            if bias_node:
                bitwidth = max(bitwidth + (bias_precision.signed and not signed),
                               bias_precision.width + (signed and not bias_precision.signed)) + 1
                signed = signed or bias_precision.signed
            quant_precision = FixedPrecisionType(bitwidth, bitwidth, signed, rounding_mode, saturation_mode)

        #creating the attributes

        # The ConvxD nodes expect the weight data to be in a different format, not (M, k1.., C)
        if attributes['n_dim'] == 1:
            nodetype = "Conv1D"
            attributes["weight_data"] =  np.transpose(weight_node.value, (1, 2, 0))
        else:
            nodetype = "Conv2D"
            attributes["weight_data"] =  np.transpose(weight_node.value, (1, 2, 3, 0))
        attributes["weight_precision"] = weight_precision
        attributes["weight_quantizer"] =  weight_node.get_attr("quantizer")
        attributes["quant_precision"] = quant_precision

        if bias_node:
            attributes["bias_data"] =  bias_node.value
            attributes["bias_precision"] = bias_precision,
            attributes["bias_quantizer"] =  bias_node.get_attr("quantizer")

        #making new node
        new_node = model.make_node(nodetype, f"{nodetype}_{node.name}", attributes,
            [node.inputs[0]], [x for x in node.outputs])

        #removing and replacing old nodes
        model.remove_node(weight_node, rewire=False)
        if bias_node:
            model.remove_node(bias_node, rewire=False)
        model.replace_node(node, new_node)

        return True
