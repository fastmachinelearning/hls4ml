import numpy as np

from hls4ml.converters.onnx.quantizer import QuantNodeQuantizer
from hls4ml.model.layers import BatchNormalization, Constant, Merge
from hls4ml.model.optimizer import OptimizerPass

_base_attributes = ('Trace', 'reuse_factor', 'n_in')

# TODO This doesn't yet support quantization in the constants


class MergeTwoConstants(OptimizerPass):
    """Merge of two constants makes another constant"""

    def match(self, node):
        is_match = (
            isinstance(node, Merge)
            and isinstance(node.get_input_node(node.inputs[0]), Constant)
            and isinstance(node.get_input_node(node.inputs[1]), Constant)
        )

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
            new_val = np.mean(np.array([val0, val1]), axis=0)
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


class MergeToBatchNormalization(OptimizerPass):
    """Convert Add, Sub, Mul, or Div Merges with consant to BatchNormalization"""

    def match(self, node):
        is_match = (
            isinstance(node, Merge)
            and node.attributes["op"] in ("add", "sum", "sub", "mul")  # Div is separate
            and (
                isinstance(node.get_input_node(node.inputs[0]), Constant)
                != isinstance(node.get_input_node(node.inputs[1]), Constant)
            )
        )
        # note: != for booleans is xor.
        return is_match

    def transform(self, model, node):
        node1 = node.get_input_node(node.inputs[1])

        node1const = isinstance(node1, Constant)
        if node1const:
            const_node = node1
            input_node_idx = 0
        else:
            const_node = node.get_input_node(node.inputs[0])
            input_node_idx = 1

        input_shape = node.get_input_variable(node.inputs[input_node_idx]).shape
        n_in = np.prod(input_shape)

        scale_precision = None
        scale_quantizer = None
        bias_precision = None
        bias_quantizer = None

        op = node.attributes["op"]
        if op in ('add', 'sum'):
            scale = np.array(1)
            bias = const_node.value
            bias_precision = const_node.get_attr("quant_precision")
            bias_quantizer = const_node.get_attr("quantizer")
        elif op == 'sub':
            if node1const:
                scale = np.array(1)
                bias = -const_node.value
            else:
                scale = np.array(-1)
                bias = const_node.value
            bias_precision = const_node.get_attr("quant_precision")
            bias_quantizer = const_node.get_attr("quantizer")
            if bias_precision and not bias_precision.signed:
                # need to add a bit
                bias_precision.signed = 1
                bias_precision.width += 1
                bias_precision.integer += 1
                bias_quantizer = QuantNodeQuantizer(bias_precision)

        elif op == 'mul':
            scale = const_node.value
            bias = np.array(0)
            scale_precision = const_node.get_attr("quant_precision")
            scale_quantizer = const_node.get_attr("quantizer")

        attributes = {k: node.attributes.get(k, None) for k in _base_attributes}
        attributes.update(
            {
                "scale_data": scale,
                "bias_data": bias,
                "n_in": n_in,
                "n_out": n_in,
                "n_filt": -1,
                "scale_precision": scale_precision,
                "scale_quantizer": scale_quantizer,
                "bias_precision": bias_precision,
                "bias_quantizer": bias_quantizer,
            }
        )

        bn_layer = model.make_node(
            BatchNormalization, f"bn_{node.name}", attributes, [node.inputs[input_node_idx]], [x for x in node.outputs]
        )

        model.remove_node(const_node, rewire=False)
        model.replace_node(node, bn_layer)

        return True


class MergeToBatchNormalizationDiv(OptimizerPass):
    """
    Convert Div Merges with consant to BatchNormalization

    TODO:  propagate precision
    """

    def match(self, node):
        is_match = (
            isinstance(node, Merge)
            and node.attributes["op"] == 'div'
            and isinstance(node.get_input_node(node.inputs[1]), Constant)
        )  # only second can be const

        return is_match

    def transform(self, model, node):
        input_shape = node.get_input_variable().shape
        n_in = np.prod(input_shape)
        const_node = node.get_input_node(node.inputs[1])
        scale = 1 / const_node.value
        bias = np.array(0)

        attributes = {k: node.attributes.get(k, None) for k in _base_attributes}
        attributes.update({"scale_data": scale, "bias_data": bias, "n_in": n_in, "n_out": n_in, "n_filt": -1})

        bn_layer = model.make_node(
            "BatchNormalization", f"bn_{node.name}", attributes, [node.inputs[0]], [x for x in node.outputs]
        )

        model.remove_node(const_node, rewire=False)
        model.replace_node(node, bn_layer)

        return True
