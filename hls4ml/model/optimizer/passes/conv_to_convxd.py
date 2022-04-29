import numpy as np
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import IntegerPrecisionType
from hls4ml.model.layers import Conv, Constant, Conv1D, Conv2D
from hls4ml.model.optimizer.passes.quant_opt import propagete_type_conv

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
            quant_precision = propagete_type_conv(input_precision, weight_precision, bias_precision,
                num_feature_maps=weight_node.value.shape[0], filt_width=attributes['filt_width'],
                filt_height=attributes.get('filt_height', 1))

        #creating the attributes

        # The ConvxD nodes expect the weight data to be in a different format, not (M, k1.., C)
        if attributes['n_dim'] == 1:
            nodetype = Conv1D
            weight_data =  np.transpose(weight_node.value, (1, 2, 0))
        else:
            nodetype = Conv2D
            weight_data =  np.transpose(weight_node.value, (1, 2, 3, 0))
        attributes["weight_precision"] = weight_precision
        attributes["weight_quantizer"] =  weight_node.get_attr("quantizer")
        attributes["quant_precision"] = quant_precision

        node.add_weights_variable(name='weight', var_name='w{index}', data=weight_data,
                                  precision=weight_precision, quantizer=attributes['weight_quantizer'])
 
        if bias_node:
            attributes["bias_precision"] = bias_precision,
            attributes["bias_quantizer"] =  bias_node.get_attr("quantizer")
            node.add_weights_variable(name='bias', var_name='b{index}', data=bias_node.value,
                                      precision=bias_precision, quantizer=attributes['bias_quantizer'])
        else:
            node.add_weights_variable(name='bias', var_name='b{index}', data=np.zeros(node.get_output_variable().shape),
                                      precision=IntegerPrecisionType(1, False))



        #making new node
        new_node = model.make_node(nodetype, f"{nodetype}_{node.name}", attributes,
            [node.inputs[0]], [x for x in node.outputs])

        #removing and replacing old nodes
        model.remove_node(weight_node, rewire=False)
        if bias_node:
            model.remove_node(bias_node, rewire=False)
        model.replace_node(node, new_node)

        return True
