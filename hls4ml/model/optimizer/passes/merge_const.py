import numpy as np
from hls4ml.model.optimizer import OptimizerPass


class MergeTwoConstant(OptimizerPass):
    """ Merge of two constants makes another constant """
    def match(self, node):
        is_match = (node.__class__.__name__ == 'Merge'
                    and node.get_input_node(node.inputs[0]).__class__.__name__ == 'Constant'
                    and node.get_input_node(node.inputs[1]).__class__.__name__ == 'Constant')

        return is_match

    def transform(self, model, node):
        """
        Merge of two constants makes another constant
        """
        const_node0 = node.get_input_node(node.inputs[0])
        const_node1 = node.get_input_node(node.inputs[1])

        val0 = const_node0.value
        val1 = const_node1.value

        op = node.attributes["op"]
        if op in ('add', 'sum'):
            new_val = val0 + val1
        elif op == 'sub':
            new_val = val0 - val1
        elif op == 'mul':
            new_val = val0 * val1
        elif op == 'div':
            new_val = val0 / val1
        elif op == 'average':
            new_val = np.mean( np.array([val0, val1]), axis=0 )
        elif op == 'max':
            new_val = np.maximum(val0, val1)
        elif op == 'min':
            new_val = np.minimum(val0, val1)
        else:
            raise RuntimeError(f"Unexpected op_type: {op}")

        quantizer = node.get_attr("quantizer")  # None if not defined
        if quantizer:
            const_node0.set_attr("quantizer", quantizer)
        const_node0.set_attr("value", new_val)

        quant_precision = node.get_attr("quant_precision")
        if quant_precision:
            const_node0.set_attr("quant_precision", quant_precision)

        # reinitialize (which also runs quantization if quantizer exists)
        const_node0.initialize()

        model.remove_node(const_node1, rewire=False)

        # remove the batch norm node
        model.remove_node(node, rewire=True)

        return True

class MergeToBaseBatchNormalization(OptimizerPass):
    """ Convert Add, Sub, Mul, or Div Merges with consant to BaseBatchNormalization """
    def match(self, node):
        is_match = (node.__class__.__name__ == 'Merge'
                    and node.attributes["op"] in ("add", "sum", "sub", "mul")  # Div is separate
                    and ((node.get_input_node(node.inputs[0]).__class__.__name__ == 'Constant')
                         != (node.get_input_node(node.inputs[1]).__class__.__name__ == 'Constant')))
        # note: != for booleans is xor.
        return is_match

    def transform(self, model, node):
        node1 = node.get_input_node(node.inputs[1])

        node1const = node1.__class__.__name__ == 'Constant'
        if node1const:
            const_node = node1
            input_node_idx = 0
        else:
            const_node = node.get_input_node(node.inputs[0])
            input_node_idx = 1
        op = node.attributes["op"]
        if op in ('add', 'sum'):
            scale = np.array(1)
            bias = const_node.value
        elif op == 'sub':
            if node1const:
                scale = np.array(1)
                bias = -const_node.value
            else:
                scale = np.array(-1)
                bias = const_node.value

        elif op == 'mul':
            scale = const_node.value
            bias = np.array(0)

        bn_layer = model.make_node("BaseBatchNormalization", f"bn_{node.name}",
                                   {"scale": scale, "bias": bias,
                                   "quant_precision": node.get_attr("quant_precision"), "quantizer": node.get_attr("quantizer")},
                                   [node.inputs[input_node_idx]], node.outputs)

        model.remove_node(const_node, rewire=False)
        model.replace_node(node, bn_layer)

        return True

class MergeToBaseBatchNormalizationDiv(OptimizerPass):
    """ Convert Add, Sub, Mul, or Div Merges with consant to BaseBatchNormalization """
    def match(self, node):
        is_match = (node.__class__.__name__ == 'Merge'
                    and node.attributes["op"] == 'div'
                    and node.get_input_node(node.inputs[1]).__class__.__name__ == 'Constant')  # only second can be const

        return is_match

    def transform(self, model, node):
        const_node = node.get_input_node(node.inputs[1])
        scale = 1/const_node.value
        bias = np.array(0)

        bn_layer = model.make_node("BaseBatchNormalization", f"bn_{node.name}",
                                   {"scale": scale, "bias": bias,
                                   "quant_precision": node.get_attr("quant_precision"), "quantizer": node.get_attr("quantizer")},
                                   [node.inputs[0]], node.outputs)

        model.remove_node(const_node, rewire=False)
        model.replace_node(node, bn_layer)

        return True
