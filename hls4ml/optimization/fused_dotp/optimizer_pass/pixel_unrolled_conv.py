import warnings
from typing import Sequence

import numpy as np

from hls4ml.backends.fpga.fpga_types import NamedType
from hls4ml.model.graph import ModelGraph
from hls4ml.model.layers import Layer
from hls4ml.model.optimizer.passes.hgq_proxy_model import FixedPointQuantizer
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType

from ..config import _global_config


def _im2col(kernel_size: Sequence[int], arr: np.ndarray, buffer: np.ndarray, axis: int):
    w = kernel_size[0]
    if len(kernel_size) == 3:
        for i in range(arr.shape[axis] - w + 1):
            patch = np.take(arr, range(i, i + w), axis=axis)
            buffer[i] = patch.flatten()
    else:
        for i in range(arr.shape[axis] - w + 1):
            patch = arr[i : i + w]
            _im2col(kernel_size[1:], patch, buffer[i], axis + 1)


def im2col(kernel_size: Sequence[int], arr: np.ndarray):
    if len(kernel_size) < 3:
        return arr
    shape = [inp_d - ker_d + 1 for inp_d, ker_d in zip(arr.shape, kernel_size[:-2])]
    shape.append(np.prod(kernel_size[:-1]))  # type: ignore
    buf = np.empty(shape, dtype=arr.dtype)
    _im2col(kernel_size, arr, buf, 0)
    return buf


def ims2cols(kernel_size: Sequence[int], *arrs: np.ndarray):
    return [im2col(kernel_size, arr) for arr in arrs]


def pad_and_stride_inp_arr(node: Layer, arr: np.ndarray, pad_val=0):
    if node.class_name.endswith('Conv2D'):
        pad_top = node.attributes.attributes['pad_top']
        pad_bottom = node.attributes.attributes['pad_bottom']
        pad_left = node.attributes.attributes['pad_left']
        pad_right = node.attributes.attributes['pad_right']
        st_h = node.attributes.attributes['stride_height']
        st_w = node.attributes.attributes['stride_width']
        return np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), constant_values=pad_val)[::st_h, ::st_w]
    if node.class_name.endswith('Conv1D'):
        pad_left = node.attributes.attributes['pad_left']
        pad_right = node.attributes.attributes['pad_right']
        st_w = node.attributes.attributes['stride_width']
        return np.pad(arr, ((pad_left, pad_right), (0, 0)), constant_values=pad_val)[::st_w]
    return arr


def pad_and_stride_inp_arrs(node: Layer, *arrs: np.ndarray, pad_val=0):
    return [pad_and_stride_inp_arr(node, arr, pad_val) for arr in arrs]


def get_inp_shape(node: Layer):
    if node.class_name.endswith('Conv1D'):
        in_width = node.attributes.attributes['in_width']
        n_chan = node.attributes.attributes['n_chan']
        return (in_width, n_chan)
    if node.class_name.endswith('Conv2D'):
        in_height = node.attributes.attributes['in_height']
        in_width = node.attributes.attributes['in_width']
        n_chan = node.attributes.attributes['n_chan']
        return (in_height, in_width, n_chan)
    if node.class_name == 'Dense':
        n_in = node.attributes.attributes['n_in']
        return (n_in,)
    raise ValueError(f'Unsupported node type {node.class_name}')


t_KIF = tuple[tuple[np.ndarray, ...], ...]


def get_input_KIF_idxs(model: ModelGraph, node: Layer) -> tuple[t_KIF, list[list[int]] | None]:
    """Get input precision per-channel, in the form of (k, i, f) each of shape (in_channels,)"""

    assert 'weight_data' in node.attributes, 'No weight data found'
    kernel = node.attributes['weight_data']
    inp_node: Layer = model.graph[node.inputs[0]]
    input_named_t: NamedType = inp_node.attributes['result_t']

    # Get input precision per-element
    *ker_inp_shape, n_out_chan = kernel.shape
    pf = node.attributes.attributes.get('parallelization_factor', 1)
    n_partition = node.attributes.attributes.get('n_partitions', 1)

    if not _global_config.enable_pixel_unroll:
        pf = 1
    if n_partition != 1 and pf != 1:
        warnings.warn(
            f'Parallelization factor {pf}!= 1 is not fully optimized for n_partition {n_partition}>1. Using one unrolled kernel for all partitions.',  # noqa: E501
            stacklevel=2,
        )
        pf = 1
    if model.config.get_config_value('IOType') == 'io_stream':
        if pf != 1:
            warnings.warn(
                f'Parallelization factor {pf} is not supported for io_stream. Ignoring.', stacklevel=2  # noqa: E501
            )
        pf = 1

    index = None

    inp_shape = get_inp_shape(node)
    if pf > 1:
        index = np.arange(np.prod(inp_shape)).reshape(inp_shape)
        index = pad_and_stride_inp_arr(node, index, -1)
        index = im2col(kernel.shape, index)
        index = index.reshape(pf, index.shape[-1])

    if isinstance(inp_node, FixedPointQuantizer):
        assert inp_node.mask_kbi is not None
        K, B, I = inp_node.mask_kbi  # noqa: E741
        K, B, I = K.squeeze(0), B.squeeze(0), I.squeeze(0)  # noqa: E741
        K, I, F = K, I - K, B - I  # noqa: E741
        K, I, F = np.broadcast_to(K, inp_shape), np.broadcast_to(I, inp_shape), np.broadcast_to(F, inp_shape)  # noqa: E741
        K, I, F = pad_and_stride_inp_arrs(node, K, I, F)  # noqa: E741
        K, I, F = ims2cols(kernel.shape, K, I, F)  # noqa: E741
        K, I, F = (x.reshape(-1, K.shape[-1]) for x in (K, I, F))  # noqa: E741
        assert K.shape == I.shape == F.shape  # noqa: E741
        assert (
            len(K) % pf == 0
        ), f'Number of kernel operands ({len(K)}) must be divisible by n_partitions ({pf})'  # noqa: E741
        K, I, F = np.split(K, pf, axis=0), np.split(I, pf, axis=0), np.split(F, pf, axis=0)
        K, I, F = np.max(K, axis=1), np.max(I, axis=1), np.max(F, axis=1)
    else:
        assert isinstance(input_named_t.precision, (FixedPrecisionType, IntegerPrecisionType))
        input_t = input_named_t.precision
        k, i, f = input_t.signed, input_t.integer, input_t.fractional
        i -= k
        dim = np.prod(ker_inp_shape)
        K, I, F = np.full((pf, dim), k), np.full((pf, dim), i), np.full((pf, dim), f)
    KIFs_in = tuple(tuple(x) for x in np.array([K, I, F]).transpose(1, 0, 2))
    idx = index.tolist() if index is not None else None
    return KIFs_in, idx
