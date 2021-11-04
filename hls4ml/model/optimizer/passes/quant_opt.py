import numpy as np
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