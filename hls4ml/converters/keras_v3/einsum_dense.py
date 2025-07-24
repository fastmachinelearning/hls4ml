import typing
from collections.abc import Sequence

from ._base import KerasV3LayerHandler, register

if typing.TYPE_CHECKING:
    import keras
    from keras import KerasTensor


def strip_batch_dim(equation: str, einsum_dense: bool = True):
    """Remove the batch dimension from the equation.

    Args:
        equation (str): The einsum equation.
        einsum_dense (bool): Whether the equation is for EinsumDense layer.

    Returns:
        str: The einsum equation without the batch dimension.
    """

    _inps, out = equation.split('->')
    inp0, inp1 = _inps.split(',')
    if einsum_dense:
        if inp0.startswith('...'):
            assert out.startswith('...'), f'Error in eq: {equation}: Batch dim mismatch for the input and output.'
        else:
            assert inp0[0] == out[0], f'Error in eq: {equation}: Batch dim mismatch for the input and output.'
            assert inp0[0] not in inp1, f'Error in eq: {equation}: Batch dim is used in the kernel.'
            inp0, out = inp0[1:], out[1:]
    else:
        if inp0.startswith('...'):
            # fmt: off
            assert inp1.startswith('...') and out.startswith('...'), (
                f'Error in eq: {equation}: Batch dim mismatch for the inputs and output.'
            )
            # fmt: on
        else:
            assert inp0[0] == inp1[0] == out[0], f'Error in eq: {equation}: Batch dim mismatch for the inputs and output.'
            inp0, inp1, out = inp0[1:], inp1[1:], out[1:]
    return f'{inp0},{inp1}->{out}'


@register
class EinsumDenseHandler(KerasV3LayerHandler):
    handles = ('keras.src.layers.core.einsum_dense.EinsumDense',)

    def handle(
        self,
        layer: 'keras.layers.EinsumDense',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        assert len(in_tensors) == 1, 'EinsumDense layer must have exactly one input tensor'
        assert len(out_tensors) == 1, 'EinsumDense layer must have exactly one output tensor'

        inp_shape: tuple[int, ...] = in_tensors[0].shape[1:]  # type: ignore
        out_shape: tuple[int, ...] = out_tensors[0].shape[1:]  # type: ignore

        # fmt: off
        assert all(d is not None for d in inp_shape), \
            f'Error when processing {layer.name}: EinsumDense layer requires fully inp shapes'
        assert all(d is not None for d in out_shape), \
            f'Error when processing {layer.name}: EinsumDense layer requires fully out shapes'
        # fmt: on

        equation = strip_batch_dim(layer.equation, True)

        kernel = self.load_weight(layer, 'kernel')

        bias = None
        if layer.bias_axes:
            bias = self.load_weight(layer, 'bias')

        return {
            'class_name': 'EinsumDense',
            'equation': equation,
            'weight_data': kernel,
            'bias_data': bias,
            'inp_shape': inp_shape,
            'out_shape': out_shape,
        }
