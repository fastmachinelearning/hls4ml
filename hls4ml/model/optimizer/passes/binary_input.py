from hls4ml.model.layers import (
    GRU,
    LSTM,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Conv1DTranspose,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dot,
    Einsum,
    EinsumDense,
    SeparableConv1D,
    SeparableConv2D,
    SimpleRNN,
    ZeroPadding1D,
    ZeroPadding2D,
)
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import XnorPrecisionType


class ValidateBinaryInput(OptimizerPass):
    """Binary data stores -1 as 0. It can go into the layers that keep it binary, which copy or pick it as it is stored,
    and into the layers that multiply it through the product classes, which take 0 as -1. Any other layer would take
    the 0 as 0, so conversion stops with an error.
    """

    # The layers that take 0 as -1: through the product classes, or (ZeroPadding) in their kernels
    reads_binary = (
        Dense,
        Conv1D,
        Conv2D,
        SeparableConv1D,
        SeparableConv2D,
        Conv1DTranspose,
        Conv2DTranspose,
        BatchNormalization,
        Dot,
        EinsumDense,
        Einsum,
        SimpleRNN,
        LSTM,
        GRU,
        Bidirectional,
        ZeroPadding1D,
        ZeroPadding2D,
    )

    def match(self, node):
        if isinstance(node, self.reads_binary) or node.class_name == 'BatchNormalizationQuantizedTanh':
            return False
        # A layer whose output is binary only moves or picks binary data
        if node.outputs and all(self.is_binary(node.get_output_variable(x)) for x in node.outputs):
            return False
        return any(x and self.is_binary(node.get_input_variable(x)) for x in node.inputs)

    def transform(self, model, node):
        raise RuntimeError(
            f'{node.class_name} layer {node.name} has a binary input, which is not supported. Binary data stores -1 as 0, '
            'and this layer would take the 0 as 0. Binary data can only go into layers that keep it binary, such as max '
            'pooling, or that multiply it with weights, such as Dense or Conv2D. Move the binary activation after this '
            'layer, or use a quantizer with more values, such as ternary.'
        )

    def is_binary(self, var):
        return var is not None and isinstance(var.type.precision, XnorPrecisionType)
