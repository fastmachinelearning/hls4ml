import numpy as np

from hls4ml.model.layers import Constant, Dense, MatMul
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import IntegerPrecisionType

_base_attributes = ('Trace', 'reuse_factor', 'weight', 'weight_t', 'bias', 'bias_t')


class MatmulConstToDense(OptimizerPass):
    """
    Convert MatMul with constant to a dense layer. Note, this only supports the second input
    being the constant. If needed, one could add transposes to make that be the case in
    other yet to be written optimizers.
    """

    def match(self, node):
        is_match = (
            isinstance(node, MatMul) and len(node.inputs) == 2 and isinstance(node.get_input_node(node.inputs[1]), Constant)
        )
        return is_match

    def transform(self, model, node):
        """Substitute Matmul + Constant for a single dense"""
        # determining Constant layer input
        const_node = node.get_input_node(node.inputs[1])
        other_var = node.get_input_variable(node.inputs[0])

        weight_precision = const_node.get_attr("quant_precision")
        weight_quantizer = const_node.get_attr("quantizer")

        in_shape = other_var.shape
        n_in = np.prod(in_shape)
        out_shape = list(in_shape[:-1]) + [const_node.value.shape[-1]]
        n_out = np.prod(out_shape)

        # creating the attributes
        attributes = {k: node.attributes.get(k, None) for k in _base_attributes}
        attributes.update(
            {
                "weight_data": const_node.value,
                "weight_precision": weight_precision,
                "weight_quantizer": weight_quantizer,
                "bias_data": np.zeros(out_shape),
                "bias_precision": IntegerPrecisionType(1, False),
                "n_in": n_in,
                "n_out": n_out,
            }
        )

        # making new node
        new_dense = model.make_node(Dense, f"Dense_{node.name}", attributes, [node.inputs[0]], [x for x in node.outputs])

        # removing and replacing old nodes
        model.remove_node(const_node, rewire=False)
        model.replace_node(node, new_dense)

        return True
