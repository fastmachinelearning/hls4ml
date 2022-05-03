import numpy as np
import math  # prefer to use math.ceil for scalar values (returns int)
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import IntegerPrecisionType, NamedType, FixedPrecisionType
from hls4ml.model.layers import MatMul, Constant, Dense
from numbers import Integral

_base_attributes = ('Trace', 'reuse_factor', 'weight', 'weight_t', 'bias', 'bias_t')

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
        n_in =  np.prod(in_shape)
        out_shape = list(in_shape[:-1]) + [const_node.value.shape[-1]]
        n_out = np.prod(out_shape)

        quant_precision = propagate_type_mult(other_precision, weight_precision, in_shape[-1])

        #creating the attributes
        attributes = {k: node.attributes.get(k, None) for k in _base_attributes}
        attributes.update({
            "weight_data": const_node.value,
            "weight_precision": weight_precision,
            "weight_quantizer": weight_quantizer,
            "bias_data": np.zeros(out_shape),
            "bias_precision": IntegerPrecisionType(1, False),
            "quant_precision": quant_precision,
            "n_in": n_in,
            "n_out": n_out
        })

        #making new node
        new_dense = model.make_node(Dense, f"Dense_{node.name}", attributes,
            [node.inputs[0]], [x for x in node.outputs])

        if quant_precision:
            accum_t = NamedType('layer{}_accum_t'.format(new_dense.index), quant_precision)
            new_dense.set_attr('accum_t', accum_t)

        #removing and replacing old nodes
        model.remove_node(const_node, rewire=False)
        model.replace_node(node, new_dense)

        return True

def propagate_type_mult(in1: FixedPrecisionType, in2: FixedPrecisionType, num_acc: Integral):
    '''
    Propagate the precion type across a multiply. Currently only "quant_precision" types (with no fractional bits)
    are supported. Rounding modes are propagated from in1
    '''
    if in2 and in1:
        if (in2.width != in2.integer
            or in1.width != in1.integer):
            raise ValueError("quant_precisions must always have the same width and integer parameters")

        bitwidth = in2.width + in1.width + math.ceil(np.log2(num_acc))
        signed = in2.signed or in1.signed
        # copy staruation and rounding from "in1"
        rounding_mode = in1.rounding_mode
        saturation_mode = in1.saturation_mode
        return FixedPrecisionType(bitwidth, bitwidth, signed, rounding_mode, saturation_mode)
    else:
        return None
