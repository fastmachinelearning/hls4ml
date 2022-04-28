import numpy as np
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import IntegerPrecisionType, NamedType
from hls4ml.model.layers import MatMul, Constant, Dense
from hls4ml.model.optimizer.passes.quant_opt import propagete_type_mult

class MatmulConstToDense(OptimizerPass):
    """
    Convert MatMul with constant to a dense layer. Note, this only supports the second input
    being the constant. If needed, one could add transposes to make that be the case in
    other yet to be written optimizers.
    """
    def match(self, node):
        is_match = (isinstance(node, MatMul) and len(node.inputs) == 2
                    and isinstance(node.get_input_node(node.inputs[1]), Constant))
        return is_match

    def transform(self, model, node):
        """ Substitute Matmul + Constant for a single dense """
        #determining Constant layer input
        const_node = node.get_input_node(node.inputs[1])
        other_node = node.get_input_node(node.inputs[0])
        other_var = node.get_input_variable(node.inputs[0])

        quant_precision = None
        weight_precision = const_node.get_attr("quant_precision")
        weight_quantizer = const_node.get_attr("quantizer")
        other_precision = other_node.get_attr("quant_precision")

        in_shape = other_var.shape
        node.set_attr('n_in', np.prod(in_shape))
        out_shape = list(in_shape[:-1]) + [const_node.value.shape[-1]]
        node.set_attr('n_out', np.prod(out_shape))

        node.set_attr('trace', True)

        quant_precision = propagete_type_mult(other_precision, weight_precision, in_shape[-1])

        node.add_weights_variable(name='weight', var_name='w{index}', data=const_node.value,
                                  precision=weight_precision, quantizer=weight_quantizer)
        # add a dummy bias
        # (A real one can be added after with bn_fuse)
        node.add_weights_variable(name='bias', var_name='b{index}', data=np.zeros(out_shape),
                                  precision=IntegerPrecisionType(1, False))

        #creating the attributes
        node.attributes.update({
            "weight_precision": weight_precision,
            "weight_quantizer": weight_quantizer,
            "quant_precision": quant_precision,
        })

        #making new node
        new_dense = model.make_node(Dense, f"Dense_{node.name}", node.attributes,
            [node.inputs[0]], [x for x in node.outputs])

        if quant_precision:
            accum_t = NamedType('layer{}_accum_t'.format(new_dense.index), quant_precision)
            new_dense.set_attr('accum_t', accum_t)

        #removing and replacing old nodes
        model.remove_node(const_node, rewire=False)
        model.replace_node(node, new_dense)

        return True
