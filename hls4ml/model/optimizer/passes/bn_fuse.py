import numpy as np

from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.layers import BatchNormalization, Dense, Conv1D, Conv2D

class FuseBatchNormalization(OptimizerPass):
    def match(self, node):
        prev_node = node.get_input_node(node.inputs[0])
        basic_match = (isinstance(node, BatchNormalization)
            and isinstance(prev_node, (Dense, Conv1D, Conv2D))
            and not prev_node.get_attr("quant_precision"))

        if basic_match:
            s0 = prev_node.weights['weight'].data_unquantized
            b0 = prev_node.weights['bias'].data_unquantized
            s1 = node.weights['scale'].data_unquantized
            b1 = node.weights['bias'].data_unquantized
            scale_compatible = (
                (prev_node.get_attr("weight_quantizer") is None
                 and node.get_attr("scale_quantizer") is None)
                or (s0 == np.ones_like(s0)).all()
                or (s1 == np.ones_like(s1)).all())
            bias_compatible = (
                (prev_node.get_attr("bias_quantizer") is None
                 and node.get_attr("bias_quantizer") is None)
                or (b0 == np.zeros_like(b0)).all()
                or (b1 == np.zeros_like(b1)).all())
            return scale_compatible and bias_compatible
        else:
            return False


    def transform(self, model, node):
        """ Fuse weight and bias of Dense/Conv1D/Conv2D layer with BN values
        """
        parent_node = node.get_input_node()
        parent_map = parent_node.get_output_use_map()
        node_map = node.get_output_use_map()

        if (len(parent_map.keys()) != 1
            or len(tuple(parent_map.values())[0]) != 1
            or len(node_map.keys()) != 1
            or len(tuple(node_map.values())[0]) > 1):
            # This checks that output of both the parent and the current node
            # is used at most one time for this optimzation. (For the parent, of course it can't be 0)
            # JM:  I understand the requirement on the parent, but not on the current node.
            return False

        # copying much of the logic from FuseConsecutiveBatchNormalization
        # (hence weight = scale in the variable names)

        prev_node = node.get_input_node(node.inputs[0])

        s0 = prev_node.weights['weight'].data_unquantized
        b0 = prev_node.weights['bias'].data_unquantized
        s1 = node.weights['scale'].data_unquantized
        b1 = node.weights['bias'].data_unquantized

        s_quantizer = (node.get_attr("scale_quantizer") if (s0 == np.ones_like(s0)).all()
                       else prev_node.get_attr("weight_quantizer"))
        b_quantizer = (node.get_attr("bias_quantizer") if (b0 == np.zeros_like(b0)).all()
                       else prev_node.get_attr("bias_quantizer"))

        prev_node.set_attr("weight_quantizer", s_quantizer)
        prev_node.set_attr("bias_quantizer", b_quantizer)
        if s_quantizer:
            prev_node.set_attr("weight_precision", s_quantizer.hls_type)
        if b_quantizer:
            prev_node.set_attr("bias_precision", b_quantizer.hls_type)

        scale_new = s0 * s1
        bias_new = s1 * b0 + b1

        prev_node.set_attr("quant_precision", node.get_attr("quant_precision"))

        model.remove_node(node, rewire=True)

        prev_node.add_weights_variable(name='weight', var_name='w{index}', data=scale_new, quantizer=s_quantizer)
        prev_node.add_weights_variable(name='bias', var_name='b{index}', data=bias_new, quantizer=b_quantizer)

        return True
