from hls4ml.model.attributes import Attribute
from hls4ml.model.layers import Layer, register_layer

SIDEBAND_SHAPE = 2


class SidebandExtraction(Layer):
    """This layer extract the sideband and sends it to a different strem"""

    _expected_attributes = [Attribute('n_in')]

    def initialize(self):
        inp = self.get_input_variable()
        self.set_attr('n_in', inp.size())
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
