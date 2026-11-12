from hls4ml.backends.fpga.passes.clone import Clone
from hls4ml.backends.fpga.passes.repack_stream import Repack
from hls4ml.model.layers import (
    Concatenate,
    Cropping1D,
    Cropping2D,
    GlobalPooling1D,
    GlobalPooling2D,
    Merge,
    Pooling1D,
    Pooling2D,
    Reshape,
    Resize,
    Transpose,
)
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import XnorPrecisionType


class XnorPooling(OptimizerPass):
    """
    For correct behavior, for MaxPooling and similar, for XnorPrecisionType, have to propagate
    the type to the output. The same holds for the layers that only move data, or pick one of their inputs: they copy
    binary data as it is stored, with 0 for -1. When padding has turned binary data into a 2-bit type, the layers after
    it get that type back.
    """

    def match(self, node):
        if not self.moves_data(node):
            return False
        inputs, outputs = self.precisions(node)
        if all(isinstance(p, XnorPrecisionType) for p in inputs):
            return not all(isinstance(p, XnorPrecisionType) for p in outputs)
        return len(inputs) == 1 and any(isinstance(p, XnorPrecisionType) for p in outputs)

    def transform(self, model, node):
        inputs, _ = self.precisions(node)
        precision = XnorPrecisionType() if isinstance(inputs[0], XnorPrecisionType) else inputs[0]
        for x in node.outputs:
            node.get_output_variable(x).type.precision = precision
        accum_t = node.types.get('accum_t')
        if accum_t is not None and XnorPrecisionType in (type(precision), type(accum_t.precision)):
            # Some pooling kernels compute the maximum in the accumulator type, whose name other layers may share
            accum_t.name = node.name + '_accum_t'
            accum_t.precision = precision
        return True

    def moves_data(self, node):
        if isinstance(node, (Pooling1D, Pooling2D, GlobalPooling1D, GlobalPooling2D)):
            return node.get_attr('pool_op') == 'Max'
        if isinstance(node, Resize):
            return node.get_attr('algorithm') == 'nearest'
        if isinstance(node, Merge) and not isinstance(node, Concatenate):
            return node.get_attr('op') in ('maximum', 'minimum')
        return isinstance(node, (Reshape, Repack, Transpose, Cropping1D, Cropping2D, Concatenate, Clone))

    def precisions(self, node):
        # A Reshape can keep an empty second input name after its shape input is removed
        names = node.inputs if isinstance(node, Merge) else node.inputs[:1]
        inputs = [node.get_input_variable(x).type.precision for x in names]
        outputs = [node.get_output_variable(x).type.precision for x in node.outputs]
        return inputs, outputs
