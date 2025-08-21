import numpy as np

from hls4ml.model.attributes import Attribute, ConfigurableAttribute, TypeAttribute
from hls4ml.model.layers import Layer
from hls4ml.model.types import IntegerPrecisionType


class ExtractSideband(Layer):
    '''This layer extract the sideband and sends it to a different strem
    '''

    SIDEBAND_SHAPE = 2

    def initialize(self):
        inp = self.get_input_variable()

        # I think the order of these must be as stated because they each set the result_t type.
        # We want the second one to be the actual result_t.
        self.add_output_variable(SIDEBAND_SHAPE,
                                 out_name='sideband',
                                 var_name='sideband_out',
                                 type_name='sideband_t',
                                 precision=IntegerPrecisionType(1, False))
        self.add_output_variable(inp.shape, precision=inp.precision)
