from hls4ml.model.attributes import Attribute
from hls4ml.model.layers import Layer, register_layer
from hls4ml.model.types import IntegerPrecisionType

SIDEBAND_SHAPE = 2


class SidebandExtraction(Layer):
    """This layer extract the sideband and sends it to a different strem"""

    _expected_attributes = [
        Attribute('n_in'),
    ]

    def initialize(self):
        inp = self.get_input_variable()
        self.set_attr('n_in', inp.size())

        # I think the order of these must be as stated because they each set the result_t type.
        # We want the second one to be the actual result_t.
        self.add_output_variable(
            SIDEBAND_SHAPE,
            out_name='sideband',
            var_name='sideband_out',
            type_name='sideband_t',
            precision=IntegerPrecisionType(1, False),
        )
        self.add_output_variable(inp.shape, precision=inp.type.precision)


class SidebandMerging(Layer):
    """This layer gets the sideband from a different input and merges it"""

    _expected_attributes = [
        Attribute('n_in'),
    ]

    def initialize(self):
        inp = self.get_input_variable()
        self.set_attr('n_in', inp.size())
        self.add_output_variable(inp.shape, precision=inp.type.precision)


# register the layers
register_layer('SidebandExtraction', SidebandExtraction)
register_layer('SidebandMerging', SidebandMerging)
