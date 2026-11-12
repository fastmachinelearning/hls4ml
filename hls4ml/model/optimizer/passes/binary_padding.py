from hls4ml.model.layers import Conv1D, Conv2D, ZeroPadding1D, ZeroPadding2D
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import IntegerPrecisionType, XnorPrecisionType

# Binary data stores -1 as 0, so it has no value for the zeros that padding adds. Padding binary data therefore gives a
# 2-bit type that holds -1, 0 and +1, and convolutions with a binary input get their padding from a padding layer.


class BinaryZeroPadding(OptimizerPass):
    """Padding of binary data outputs -1, 0 and +1 in a 2-bit type. The padding kernel copies the data as -1 and +1."""

    def match(self, node):
        return (
            isinstance(node, (ZeroPadding1D, ZeroPadding2D))
            and isinstance(node.get_input_variable().type.precision, XnorPrecisionType)
            and node.get_output_variable().type.precision != IntegerPrecisionType(width=2, signed=True)
        )

    def transform(self, model, node):
        node.get_output_variable().type.precision = IntegerPrecisionType(width=2, signed=True)
        return True


class InsertZeroPaddingBeforeBinaryConv(OptimizerPass):
    """A convolution with a binary input would pad it with 0, which stands for -1 in binary data. Its padding moves to
    a ZeroPadding layer in front of it."""

    def match(self, node):
        return (
            isinstance(node, (Conv1D, Conv2D))
            and isinstance(node.get_input_variable().type.precision, XnorPrecisionType)
            and any(node.get_attr(f'pad_{side}', 0) != 0 for sides in self.dims(node).values() for side in sides)
        )

    def transform(self, model, node):
        dims = self.dims(node)
        attrs = {'n_chan': node.get_attr('n_chan'), 'data_format': node.get_attr('data_format', 'channels_last')}
        for dim, (before, after) in dims.items():
            attrs[f'pad_{before}'] = node.get_attr(f'pad_{before}')
            attrs[f'pad_{after}'] = node.get_attr(f'pad_{after}')
            attrs[f'in_{dim}'] = node.get_attr(f'in_{dim}')
            attrs[f'out_{dim}'] = attrs[f'pad_{before}'] + attrs[f'in_{dim}'] + attrs[f'pad_{after}']

            # The convolution itself pads nothing
            node.set_attr(f'pad_{before}', 0)
            node.set_attr(f'pad_{after}', 0)
            node.set_attr(f'in_{dim}', attrs[f'out_{dim}'])

        padding = model.make_node(f'ZeroPadding{len(dims)}D', f'zp{len(dims)}d_{node.name}', attrs, node.inputs.copy())
        model.insert_node(padding, before=node)

        return True

    def dims(self, node):
        # The padded dimensions of the convolution, with the sides they are padded on
        if isinstance(node, Conv1D):
            return {'width': ('left', 'right')}
        return {'height': ('top', 'bottom'), 'width': ('left', 'right')}
