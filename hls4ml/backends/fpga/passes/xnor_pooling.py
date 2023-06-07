from hls4ml.model.layers import GlobalPooling1D, GlobalPooling2D, Pooling1D, Pooling2D
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import XnorPrecisionType


class XnorPooling(OptimizerPass):
    '''
    For correct behavior, for MaxPooling and similar, for XnorPrecisionType, have to propagate
    the type to the output.
    '''

    def match(self, node):
        if isinstance(node, (Pooling1D, Pooling2D, GlobalPooling1D, GlobalPooling2D)) and node.get_attr('pool_op') == 'Max':
            return isinstance(node.get_input_variable().type.precision, XnorPrecisionType) and not isinstance(
                node.get_output_variable().type.precision, XnorPrecisionType
            )
        return False

    def transform(self, model, node):
        outvar = node.get_output_variable()
        outvar.type.precision = XnorPrecisionType()
        return True
