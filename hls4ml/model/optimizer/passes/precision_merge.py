from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import FixedPrecisionType


def get_concat_type(itype1, itype2):
    newwidth = max(itype1.width, itype2.width)
    newint = max(itype1.integer, itype2.integer)
    if itype1.signed ^ itype2.signed:  # XOR
        newint += 1
        newwidth += 1
    newrmode = itype1.rounding_mode if itype1.rounding_mode is not None else itype2.rounding_mode
    newsmode = itype1.saturation_mode if itype1.saturation_mode is not None else itype2.saturation_mode
    newsbits = itype1.saturation_bits if itype1.saturation_bits is not None else itype2.saturation_bits

    newtype = FixedPrecisionType(newwidth, newint, itype1.signed or itype2.signed, newrmode, newsmode, newsbits)
    return newtype


class SetPrecisionConcat(OptimizerPass):
    def match(self, node):
        if node.__class__.__name__ == 'Concatenate':
            otype = node.get_output_variable().type.precision
            itype1 = node.get_input_variable(node.inputs[0]).type.precision
            itype2 = node.get_input_variable(node.inputs[1]).type.precision
            if isinstance(otype, FixedPrecisionType) and otype != get_concat_type(itype1, itype2):
                return True
        return False

    def transform(self, model, node):
        """
        Set concat output precision
        """
        otype = node.get_output_variable().type.precision
        itype1 = node.get_input_variable(node.inputs[0]).type.precision
        itype2 = node.get_input_variable(node.inputs[1]).type.precision
        newtype = get_concat_type(itype1, itype2)
        print(f"Found {node.name} in the model, optimizing {otype} to {newtype}...")
        node.get_output_variable().type.precision = newtype

        return True
