from hls4ml.model.layers import SymbolicExpression
from hls4ml.model.optimizer import ConfigurableOptimizerPass


class ValidateUserLookupTable(ConfigurableOptimizerPass):
    '''Validates the precision of user-defined LUTs is adequate'''

    def __init__(self):
        self.raise_exception = False

    def match(self, node):
        return isinstance(node, SymbolicExpression) and len(node.get_attr('lut_functions', [])) > 0

    def transform(self, model, node):
        precision = node.get_output_variable().type.precision
        range = 2 ** (precision.integer - precision.signed)
        frac_step = 1 / 2**precision.fractional

        for lut_fn in node.get_attr('lut_functions'):
            lut_range = lut_fn.range_end - lut_fn.range_start
            lut_step = lut_range / lut_fn.table_size

            if lut_step < frac_step:
                msg = f'LUT function {lut_fn.name} requires more fractional bits.'
                if self.raise_exception:
                    raise Exception(msg)
                else:
                    print('WARNING:', msg)

            if lut_range > range:
                msg = f'LUT function {lut_fn.name} requires more integer bits.'
                if self.raise_exception:
                    raise Exception(msg)
                else:
                    print('WARNING:', msg)

        return False
