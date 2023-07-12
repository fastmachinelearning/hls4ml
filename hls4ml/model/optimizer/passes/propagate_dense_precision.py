import math  # prefer to use math.ceil for scalar values (returns int)

import numpy as np

from hls4ml.model.layers import Dense
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import FixedPrecisionType, NamedType


class PropagateDensePrecision(OptimizerPass):
    """
    Propagate precision for Dense nodes. Restrict it to only cases where
    the precision is set by a quant node, since otherwise the values get huge.
    """

    def match(self, node):
        is_match = isinstance(node, Dense)
        return is_match

    def transform(self, model, node):
        input_precision = node.get_input_node().get_attr("quant_precision")
        weight_precision = node.get_attr("weight_precision")
        if not input_precision or not weight_precision:
            return False

        bias_precision = node.get_attr("bias_precision")
        input_variable = node.get_input_variable()
        num_acc = input_variable.shape[-1]

        accum_precision = _propagate_type_dense(input_precision, weight_precision, bias_precision, num_acc)

        accum_t = NamedType(f'layer{node.index}_accum_t', accum_precision)
        node.set_attr('accum_t', accum_t)

        if not node.get_attr("quant_precision"):
            # output precision not set by quant node
            node.update_output_precision(accum_precision)

        return False


def _propagate_type_dense(input_precision, weight_precision, bias_precision, num_acc):
    '''
    Propagate the precion type across a multiply. Rounding modes are propagated from input_precision
    '''

    # check to make sure none are None
    bitwidth = weight_precision.width + input_precision.width + math.ceil(np.log2(num_acc))
    integer = weight_precision.integer + input_precision.integer + math.ceil(np.log2(num_acc))
    signed = weight_precision.signed or input_precision.signed

    # Because calculating precision, no need to round or sautration
    rounding_mode = None
    saturation_mode = None

    frac = bitwidth - integer

    # correct for bias
    if bias_precision:
        integer = (
            max(
                integer + (bias_precision.signed and not signed),
                bias_precision.integer + (signed and not bias_precision.signed),
            )
            + 1
        )
        bitwidth = integer + max(frac, bias_precision.width - bias_precision.integer)
        signed = signed or bias_precision.signed

    return FixedPrecisionType(bitwidth, integer, signed, rounding_mode, saturation_mode)
