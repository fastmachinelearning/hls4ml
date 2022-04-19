import numpy as np
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import FixedPrecisionType
from hls4ml.model.layers import MatMul, Constant, Dense

class MatmulConstToDense(OptimizerPass):
    """ Convert MatMul with constant to a dense layer """
    def match(self, node):
        is_match = (isinstance(node, MatMul) and len(node.inputs) == 2
                    and (isinstance(node.get_input_node(node.inputs[0]), Constant)
                         or isinstance(node.get_input_node(node.inputs[1]), Constant)))
        return is_match

    def transform(self, model, node):
        """ Substitute Matmul + Constant for a single dense """
        #determining Constant layer input
        matmul_node = node
        const_node = None
        const_inp_idx = 0
        other_inp_idx = 1
        if isinstance(matmul_node.get_input_node(matmul_node.inputs[0]), Constant):
            const_node = matmul_node.get_input_node(matmul_node.inputs[0])
            other_node = matmul_node.get_input_node(matmul_node.inputs[1])
        else:
            const_node = matmul_node.get_input_node(matmul_node.inputs[1])
            other_node = matmul_node.get_input_node(matmul_node.inputs[0])
            const_inp_idx = 1
            other_inp_idx = 0

        quant_precision = None
        weight_precision = const_node.get_attr("quant_precision")
        other_precision = other_node.get_attr("quant_precision")

        if weight_precision and other_precision:
            if (weight_precision.width != weight_precision.integer
                or other_precision.width != other_precision.integer):
                raise ValueError("quant_precisions must always have the same width and integer parameters")

            Nacc = matmul_node.get_input_variable(matmul_node.inputs[0]).shape[-1]
            bitwidth = weight_precision.width + other_precision.width + int(np.ceil(np.log2(Nacc)))
            signed = weight_precision.signed or other_precision.signed
            # copy staruation and rounding from "other"
            rounding_mode = other_precision.rounding_mode
            saturation_mode = other_precision.saturation_mode
            quant_precision = FixedPrecisionType(bitwidth, bitwidth, signed, rounding_mode, saturation_mode)

        #creating the attributes
        attributes = matmul_node.attributes
        attributes.update({
            "weight_data": const_node.value,
            "weight_precision": weight_precision,
            "weight_quantizer": const_node.get_attr("quantizer"),
            "quant_precision": quant_precision,
            "omit_bias": True,
            "n_in": const_node.value.shape[0],
            "n_out": const_node.value.shape[1]
        })

        #making new node
        new_dense = model.make_node(Dense, f"Dense_{matmul_node.name}", attributes,
            [matmul_node.inputs[other_inp_idx]], [x for x in matmul_node.outputs])

        #removing and replacing old nodes
        model.remove_node(const_node, rewire=False)
        model.replace_node(matmul_node, new_dense)

        return True
