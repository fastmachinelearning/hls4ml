# k, i, f = keep_negative, integers (excluding sign), fractionals
# b, B, I = width (no sign), width (including sign), integers (including sign)

import re
import typing
from collections.abc import Sequence
from copy import copy
from functools import reduce, singledispatch
from math import ceil, log2, prod
from warnings import warn

import numpy as np
from numpy.typing import NDArray

from hls4ml.model.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Conv2D,
    Dense,
    Einsum,
    EinsumDense,
    GlobalPooling1D,
    GlobalPooling2D,
    Input,
    Layer,
    Merge,
    ParametrizedActivation,
    Pooling1D,
    Pooling2D,
    Reshape,
    Softmax,
    Transpose,
)
from hls4ml.model.optimizer import ModelOptimizerPass, OptimizerPass
from hls4ml.model.optimizer.passes.hgq_proxy_model import FixedPointQuantizer, UnaryLUT
from hls4ml.model.types import FixedPrecisionType, NamedType, RoundingMode, SaturationMode, WeightVariable
from hls4ml.utils.qinterval import QIntervalArray, einsum, minimal_kif

if typing.TYPE_CHECKING:
    from hls4ml.model import ModelGraph


KIF_t = tuple[NDArray[np.int16], NDArray[np.int16], NDArray[np.int16]]
rm_cpy = re.compile(r'(?P<name>.+)_cpy\d*')


def to_hls4ml_fixed(k, i, f, name, *args):
    signed, B, I = k != 0, k + i + f, int(k + i)
    args = [arg.upper() for arg in args]
    if B >= 1:
        ptype = FixedPrecisionType(B, I, signed, *args)
    else:
        ptype = FixedPrecisionType(2, 32, False, 'TRN', 'WRAP')
    return NamedType(name, ptype)


def get_input_layers(layer: Layer) -> list[Layer]:
    model: 'ModelGraph' = layer.model
    inp_names = layer.inputs
    ret = []
    for name in inp_names:
        if name not in model.graph.keys():  # in stream_io, <name>_cpt\d+ may be used instead of <name>
            matched = rm_cpy.match(name)
            assert matched, f'Layer {layer.name} has input {name} which is not in the model (keys: {model.graph.keys()})'
            name = matched.group('name')
        ret.append(model.graph[name])
    return ret


def get_output_layers(layer: Layer) -> list[Layer]:
    model: 'ModelGraph' = layer.model
    return [l for l in model.graph.values() if layer.name in l.inputs]


def get_output_shape(layer: Layer) -> tuple[int, ...]:
    return tuple(layer.get_output_variable().shape)


def get_input_shapes(layer: Layer) -> list[tuple[int, ...]]:
    return [get_output_shape(inp) for inp in get_input_layers(layer)]


def _maximum_kif_at_shape(shape: tuple[int, ...]):
    k = np.ones(shape, dtype=np.int16)
    i = np.full(shape, 126, dtype=np.int16)
    f = np.full(shape, 126, dtype=np.int16)
    return k, i, f


@singledispatch
def _request_kif(layer: Layer) -> tuple[KIF_t, ...]:
    input_shapes = get_input_shapes(layer)
    return tuple(_maximum_kif_at_shape(shape) for shape in input_shapes)


@_request_kif.register
def _(layer: FixedPointQuantizer):
    assert layer.mask_kbi is not None
    k, B, I = layer.mask_kbi
    k, i, f = k, I - k, B - I

    if k.ndim > 0:
        k, i, f = k[0], i[0], f[0]

    out_shape = get_output_shape(layer)
    k = np.broadcast_to(k, out_shape).astype(np.int16)
    i = np.broadcast_to(i, out_shape).astype(np.int16)
    f = np.broadcast_to(f, out_shape).astype(np.int16)

    if layer.SAT != 'WRAP':
        k[:] = 1
        i[:] = 126
    if layer.RND == 'TRN':
        pass
    elif layer.RND == 'RND':
        f += 1
    else:
        f[:] = 126
    return ((k, i, f),)


@_request_kif.register
def _(layer: Reshape):
    inp_shape = get_input_shapes(layer)[0]
    k, i, f = requested_kif(layer)
    k = k.reshape(inp_shape)
    i = i.reshape(inp_shape)
    f = f.reshape(inp_shape)
    return ((k, i, f),)


@_request_kif.register
def _(layer: Activation):
    fn_name = layer.attributes.get('activation')

    if layer.attributes.get('trusted', False):
        result_t = layer.get_output_variable().type.precision
        if fn_name in ('linear', 'relu'):
            output_shape = get_output_shape(layer)
            k, w, f = result_t.signed, result_t.width, result_t.fractional
            i = w - k - f
            k = np.full(output_shape, k, dtype=np.int16)
            i = np.full(output_shape, i, dtype=np.int16)
            f = np.full(output_shape, f, dtype=np.int16)
            if result_t.rounding_mode == RoundingMode.RND:
                f += 1
            elif result_t.rounding_mode != RoundingMode.TRN:
                f = np.full(output_shape, 126, dtype=np.int16)
            if result_t.saturation_mode != SaturationMode.WRAP:
                k = np.ones(output_shape, dtype=np.int16)
                i = np.full(output_shape, 126, dtype=np.int16)
            if fn_name == 'linear':
                return ((k, i, f),)
            else:
                k = np.ones(output_shape, dtype=np.int16)
                i = np.full(output_shape, 126, dtype=np.int16)
                return ((k, i, f),)

    if fn_name == 'linear':
        return (requested_kif(layer),)
    if fn_name == 'relu':
        _, _, f = requested_kif(layer)
        k = np.ones(f.shape, dtype=np.int16)
        i = np.full(f.shape, 126, dtype=np.int16)
        return ((k, i, f),)
    inp_shape = get_input_shapes(layer)[0]
    return (_maximum_kif_at_shape(inp_shape),)


@_request_kif.register
def _(layer: Concatenate):
    inp_shape0, inp_shape1 = get_input_shapes(layer)
    k, i, f = requested_kif(layer)
    ax = layer.attributes['axis']
    n_split = inp_shape0[ax]

    k0, k1 = np.split(k, [n_split], axis=ax)
    i0, i1 = np.split(i, [n_split], axis=ax)
    f0, f1 = np.split(f, [n_split], axis=ax)

    return ((k0, i0, f0), (k1, i1, f1))


@_request_kif.register
def _(layer: Transpose):
    k, i, f = requested_kif(layer)
    perm = layer.attributes['perm']
    inv_perm = np.argsort(perm)
    k = np.transpose(k, inv_perm)
    i = np.transpose(i, inv_perm)
    f = np.transpose(f, inv_perm)
    return ((k, i, f),)


def requested_kif(layer: Layer) -> KIF_t:
    out_layers = get_output_layers(layer)
    out_shape = get_output_shape(layer)
    if not out_layers:
        return _maximum_kif_at_shape(out_shape)

    k = np.zeros(out_shape, dtype=np.int16)
    i = np.full(out_shape, -127, dtype=np.int16)
    f = i.copy()
    for out_layer in out_layers:
        _kif_s = request_kif(out_layer)
        out_layer_inp_layers = get_input_layers(out_layer)
        idx = out_layer_inp_layers.index(layer)
        k = np.maximum(k, _kif_s[idx][0])
        i = np.maximum(i, _kif_s[idx][1])
        f = np.maximum(f, _kif_s[idx][2])

    return k, i, f


@singledispatch
def _produce_kif(layer: Layer) -> KIF_t:
    raise NotImplementedError(f'No implementation of produce_kif for {layer.__class__} ({layer.class_name})')


@_produce_kif.register
def _(layer: Input):
    shape = get_output_shape(layer)
    if layer.attributes.get('trusted', False):
        precision: FixedPrecisionType = layer.get_output_variable().type.precision
        k, i, f = precision.signed, precision.integer - precision.signed, precision.fractional
        k = np.full(shape, k, dtype=np.int16)
        i = np.full(shape, i, dtype=np.int16)
        f = np.full(shape, f, dtype=np.int16)
    else:
        k = np.ones(shape, dtype=np.int16)
        i = f = np.full(shape, 126, dtype=np.int16)
    return k, i, f


def get_input_kifs(layer: Layer):
    return [_produce_kif(l) for l in get_input_layers(layer)]


@_produce_kif.register
def _(layer: FixedPointQuantizer):
    assert layer.mask_kbi is not None

    _k, _B, _I = layer.mask_kbi
    shape0 = _k.shape[1:]
    k, i, f = _k, _I - _k, _B - _I
    last_layer = get_input_layers(layer)[0]
    lk, li, lf = produce_kif(last_layer)

    k, i, f = k[0], i[0], f[0]
    shape0 = k.shape
    ndim = k.ndim

    if ndim > 0:
        # Make sure input kbi masks are in good shape
        assert k.ndim == lk.ndim == i.ndim == li.ndim == f.ndim == lf.ndim

    # Bitwidth reduction
    _k = np.minimum(k, lk)
    _i = np.minimum(i, li)
    _f = np.minimum(f, lf)

    # Compansate for round-up/downs that may need extra bits for representing (ufixed<2,0> -> ufixed<2,1,RND>, 0.75->1.0)
    if layer.RND != 'TRN':
        _i += ((lf > f) & (i > li)).astype(np.int16)
    else:
        _i += ((lf > f) & (i > li) & k).astype(np.int16)

    if layer.SAT in ('SAT', 'SAT_SM'):
        k, i, f = _k, _i, _f
    else:
        # Perserve repr boundaries unless overflow never happens
        mask = (2.0**i - 2.0**-f >= 2.0**li - 2.0**-lf) & (k >= lk)
        i = np.where(mask, _i, i)
        f = np.where(mask, _f, f)
        k = np.where(mask, _k, k)
    # Set zeros to zero
    idx_zeros = np.where(k + i + f <= 0)
    k[idx_zeros] = 0
    i[idx_zeros] = 0
    f[idx_zeros] = 0

    if ndim > 0:  # Shrink to the original shape
        contract_axis = np.where(np.array(shape0) == 1)[0]
    else:  # Shrink to [1]*N instead of scaler; no real difference
        contract_axis = np.arange(k.ndim)

    _k = k
    _B = k + i + f
    _I = k + i

    for ax in contract_axis:
        _k = np.max(_k, axis=ax, keepdims=True)
        _B = np.max(_B, axis=ax, keepdims=True)
        _I = np.max(_I, axis=ax, keepdims=True)

    _k, _B, _I = _k[None], _B[None], _I[None]
    layer.mask_kbi = (_k, _B, _I)
    return k, i, f


@_produce_kif.register
def _(layer: Reshape):
    out_shape = get_output_shape(layer)
    k, i, f = produce_kif(get_input_layers(layer)[0])
    return k.reshape(out_shape), i.reshape(out_shape), f.reshape(out_shape)


@_produce_kif.register
def _(layer: Merge):
    op = layer.attributes['op'].lower()
    kif_ins = get_input_kifs(layer)
    match op:
        case 'add':
            qint_ins = [QIntervalArray.from_kif(*kif) for kif in kif_ins]
            k, i, f = reduce(lambda a, b: a + b, qint_ins).to_kif()  # type: ignore
        case 'subtract':
            qint_in0, qint_in1 = (QIntervalArray.from_kif(*kif) for kif in kif_ins)
            k, i, f = (qint_in0 - qint_in1).to_kif()
        case 'concatename':
            axis = layer.attributes['axis']
            _ks, _is, _fs = zip(*[kif for kif in kif_ins])
            k = np.concatenate(_ks, axis=axis)
            i = np.concatenate(_is, axis=axis)
            f = np.concatenate(_fs, axis=axis)
        case 'maximum':
            k, i, f = map(np.maximum, *kif_ins)
        case 'minimum':
            k, i, f = map(np.maximum, *kif_ins)
        case 'multiply':
            qint_ins = [QIntervalArray.from_kif(*kif) for kif in kif_ins]
            k, i, f = reduce(lambda a, b: a * b, qint_ins).to_kif()
        case 'average':
            qint_ins = [QIntervalArray.from_kif(*kif) for kif in kif_ins]
            k, i, f = reduce(lambda a, b: a + b, qint_ins).to_kif()  # type: ignore
            scale = layer.attributes.get('scale', 1 / len(qint_ins))
            shift = -int(log2(scale))
            if int(log2(scale)) == log2(scale):
                f = f + shift
            else:
                f[:] = 126
            i = i - shift
        case 'dot1d':
            qint_in0 = QIntervalArray.from_kif(*kif_ins[0])
            qint_in1 = QIntervalArray.from_kif(*kif_ins[1])
            k, i, f = (qint_in0 @ qint_in1).to_kif()
            if k.shape == ():
                k, i, f = k[None], i[None], f[None]
        case _:
            raise NotImplementedError(f'No implementation of Merge for {op}')
    return k.astype(np.int16), i, f


@_produce_kif.register
def _(layer: EinsumDense):
    kernel = layer.attributes['weight'].data
    _bias = layer.attributes['bias']
    eq = layer.attributes['equation']
    k_in, i_in, f_in = get_input_kifs(layer)[0]
    qint_in = QIntervalArray.from_kif(k_in, i_in, f_in)
    qint_out = einsum(eq, qint_in, kernel)
    if _bias is not None:
        qint_out = qint_out + _bias.data
    k, i, f = qint_out.to_kif()
    return k.astype(np.int16), i, f


@_produce_kif.register
def _(layer: Einsum):
    kif_in1, kif_in2 = get_input_kifs(layer)
    qint_in1 = QIntervalArray.from_kif(*kif_in1)
    qint_in2 = QIntervalArray.from_kif(*kif_in2)
    eq = layer.attributes['equation']
    qint_out = einsum(eq, qint_in1, qint_in2)
    k, i, f = qint_out.to_kif()
    return k.astype(np.int16), i, f


@_produce_kif.register
def _(layer: Dense):
    kernel = layer.attributes['weight'].data
    _bias = layer.attributes['bias']
    k_in, i_in, f_in = get_input_kifs(layer)[0]
    qint_in = QIntervalArray.from_kif(k_in, i_in, f_in)
    qint_out = qint_in @ kernel
    if _bias is not None:
        qint_out = qint_out + _bias.data
    k, i, f = qint_out.to_kif()
    return k.astype(np.int16), i, f


@_produce_kif.register
def _(layer: Transpose):
    k, i, f = get_input_kifs(layer)[0]
    perm = layer.attributes['perm']
    k = np.transpose(k, perm)
    i = np.transpose(i, perm)
    f = np.transpose(f, perm)
    return k, i, f


def r_im2col(kernel_size: Sequence[int], arr: np.ndarray, buffer: np.ndarray, axis: int):
    w = kernel_size[0]
    if len(kernel_size) == 3:  # 1D
        for i in range(arr.shape[axis] - w + 1):
            patch = np.take(arr, range(i, i + w), axis=axis)
            buffer[i] = patch.flatten()
    else:  # 2D+
        for i in range(arr.shape[axis] - w + 1):
            patch = arr[i : i + w]
            r_im2col(kernel_size[1:], patch, buffer[i], axis + 1)


def _im2col(kernel_size: Sequence[int], arr: np.ndarray):
    if len(kernel_size) < 3:
        return arr
    shape = [inp_d - ker_d + 1 for inp_d, ker_d in zip(arr.shape, kernel_size[:-2])]
    shape.append(np.prod(kernel_size[:-1]))  # type: ignore
    buf = np.empty(shape, dtype=arr.dtype)
    r_im2col(kernel_size, arr, buf, 0)
    return buf


def im2col(kernel_size: Sequence[int], *arrs: np.ndarray):
    """im2col for multidimensional arrays. Assumes Channel Last format.

    Args:
        kernel_size (Sequence[int]): The size of the kernel, in the form (*kernel_shape, ch_in, ch_out).
        *arrs (np.ndarray): The input arrays to be transformed.

    Returns:
        list[np.ndarray]: The transformed arrays.
    """
    return [_im2col(kernel_size, arr) for arr in arrs]


def pad_arrs(node: Layer, pad_val: float = 0, *arrs: np.ndarray):
    out_arrs = []
    if node.class_name.endswith('2D'):
        pad_top = node.attributes['pad_top']
        pad_bottom = node.attributes['pad_bottom']
        pad_left = node.attributes['pad_left']
        pad_right = node.attributes['pad_right']
        for arr in arrs:
            r = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), constant_values=pad_val)
            out_arrs.append(r)
    elif node.class_name.endswith('1D'):
        pad_left = node.attributes['pad_left']
        pad_right = node.attributes['pad_right']
        for arr in arrs:
            r = np.pad(arr, ((pad_left, pad_right), (0, 0)), constant_values=pad_val)
            out_arrs.append(r)
    else:
        raise ValueError(f'Layer {node.class_name} is not supported for pad_arrs')
    return tuple(out_arrs)


def stride_arrs(node: Layer, *arrs: np.ndarray):
    if node.class_name.endswith('2D'):
        st_h = node.attributes['stride_height']
        st_w = node.attributes['stride_width']
        return tuple(arr[::st_h, ::st_w] for arr in arrs)
    if node.class_name.endswith('1D'):
        st_w = node.attributes['stride_width']
        return tuple(arr[::st_w] for arr in arrs)
    raise ValueError(f'Layer {node.class_name} is not supported for stride_arrs')


@_produce_kif.register(Conv1D)
@_produce_kif.register(Conv2D)
def _(layer: Conv1D | Conv2D):
    assert layer.attributes['data_format'] == 'channels_last', 'Only channels_last format is supported'
    kernel = layer.attributes['weight'].data
    _bias = layer.attributes['bias']
    bias = _bias.data if _bias is not None else 0
    k_in, i_in, f_in = get_input_kifs(layer)[0]
    k_in, i_in, f_in = pad_arrs(layer, 0, k_in, i_in, f_in)
    k_in, i_in, f_in = im2col(kernel.shape, k_in, i_in, f_in)
    k_in, i_in, f_in = stride_arrs(layer, k_in, i_in, f_in)
    kernel = kernel.reshape(-1, kernel.shape[-1])
    qint_in = QIntervalArray.from_kif(k_in, i_in, f_in)
    qint_out = qint_in @ kernel
    qint_out = qint_out + bias
    k, i, f = qint_out.to_kif()
    return k.astype(np.int16), i, f


@_produce_kif.register(Pooling1D)
@_produce_kif.register(Pooling2D)
@_produce_kif.register(GlobalPooling1D)
@_produce_kif.register(GlobalPooling2D)
def _(layer: Pooling1D | Pooling2D | GlobalPooling1D | GlobalPooling2D):
    px_shape = _get_px_shape(layer)
    ch_out = ch_in = layer.attributes['n_filt']

    im2col_shape = *px_shape, ch_in, ch_out  # conv kernel shape
    k_in, i_in, f_in = get_input_kifs(layer)[0]
    count = np.ones_like(k_in, dtype=np.uint32)
    if isinstance(layer, (Pooling1D, Pooling2D)):
        k_in, i_in, f_in, count = pad_arrs(layer, 0, k_in, i_in, f_in, count)
    k_in, i_in, f_in, count = im2col(im2col_shape, k_in, i_in, f_in, count)
    if isinstance(layer, (Pooling1D, Pooling2D)):
        k_in, i_in, f_in, count = stride_arrs(layer, k_in, i_in, f_in, count)

    k_out = k_in.reshape(*k_in.shape[:-1], -1, ch_in).max(axis=-2).astype(np.int16)
    i_out = i_in.reshape(*i_in.shape[:-1], -1, ch_in).max(axis=-2).astype(np.int16)
    f_out = f_in.reshape(*f_in.shape[:-1], -1, ch_in).max(axis=-2).astype(np.int16)
    count = count.reshape(*count.shape[:-1], -1, ch_in).sum(axis=-2)

    pool_op = layer.attributes['pool_op']
    if pool_op == 'Average':
        f_add = minimal_kif(1 / count)[2]
        f_out += f_add

    if isinstance(layer, (GlobalPooling1D, GlobalPooling2D)):
        k_out, i_out, f_out = k_out[0], i_out[0], f_out[0]
    return k_out, i_out, f_out


@_produce_kif.register
def _(layer: BatchNormalization):
    k_in, i_in, f_in = get_input_kifs(layer)[0]
    qint_in = QIntervalArray.from_kif(k_in, i_in, f_in)
    scale = layer.attributes['scale'].data

    _bias = layer.attributes['bias']
    bias = _bias.data if _bias is not None else 0

    qint_out = qint_in * scale + bias
    k, i, f = qint_out.to_kif()
    return k.astype(np.int16), i, f


@_produce_kif.register
def _(layer: Softmax):
    out_shape = get_output_shape(layer)

    inv_table_t: FixedPrecisionType = layer.attributes['inv_table_t'].precision
    exp_table_t: FixedPrecisionType = layer.attributes['exp_table_t'].precision

    b_exp, I_exp = exp_table_t.width, exp_table_t.integer
    b_inv, I_inv = inv_table_t.width, inv_table_t.integer

    i_exp, f_exp = I_exp, b_exp - I_exp
    i_inv, f_inv = I_inv, b_inv - I_inv
    k = np.zeros(out_shape, dtype=np.int16)

    i = np.full(out_shape, i_exp + i_inv, dtype=np.int16)
    f = np.full(out_shape, f_exp + f_inv, dtype=np.int16)

    return k, i, f


@_produce_kif.register
def _(layer: Concatenate):
    kifs_in = get_input_kifs(layer)
    ks, is_, fs = zip(*kifs_in)
    ax = layer.attributes['axis']
    k = np.concatenate(ks, axis=ax)
    i = np.concatenate(is_, axis=ax)
    f = np.concatenate(fs, axis=ax)
    return k, i, f


@_produce_kif.register
def _(layer: Activation):
    fn_name = layer.attributes['activation'].lower()
    if layer.attributes.get('trusted', False):
        output_shape = get_output_shape(layer)
        result_t = layer.get_output_variable().type.precision
        k, w, f = result_t.signed, result_t.width, result_t.fractional
        i = w - k - f
        k = np.full(output_shape, k, dtype=np.int16)
        i = np.full(output_shape, i, dtype=np.int16)
        f = np.full(output_shape, f, dtype=np.int16)
        return k, i, f

    k, i, f = get_input_kifs(layer)[0]

    match fn_name:
        case 'linear':
            pass
        case 'relu':
            k = np.zeros_like(k, dtype=np.int16)
        case 'tanh':
            i = np.minimum(i, 1)
            f = np.full_like(f, 126, dtype=np.int16)
        case 'sigmoid':
            k = np.zeros_like(k, dtype=np.int16)
            i = np.minimum(i, 1)
            f = np.full_like(f, 126, dtype=np.int16)
        case _:
            k = np.ones(k, dtype=np.int16)
            i = np.full_like(i, 126, dtype=np.int16)
            f = np.full_like(f, 126, dtype=np.int16)
    return k, i, f


@_produce_kif.register
def _(layer: ParametrizedActivation):
    fn_name = layer.attributes['activation'].lower()

    k, i, f = get_input_kifs(layer)[0]
    p = layer.attributes['activ_param']
    _k, _i, _f = minimal_kif(np.array(p))
    match fn_name:
        case 'leakyrelu':
            k = k & np.int16(p > 0)
            i += np.maximum(0, _i - 1)
            f += np.maximum(0, _f)
        case 'thresholdedrelu':
            i = np.maximum(i, _i)
            f = np.maximum(f, _f)
            k = k & _k
        case 'elu':
            k = k & np.int16(p > 0)
            f = np.full_like(f, 126, dtype=np.int16)
            i = np.maximum(i, _i)
        case _:
            k = np.ones(k, dtype=np.int16)
            i = np.full_like(i, 126, dtype=np.int16)
            f = np.full_like(f, 126, dtype=np.int16)
    return k, i, f


@_produce_kif.register
def _(layer: UnaryLUT):
    k, i, f = minimal_kif(layer.attributes['table'].data)
    shape = get_output_shape(layer)
    k = np.full(shape, np.max(k), dtype=np.int16)
    i = np.full(shape, np.max(i), dtype=np.int16)
    f = np.full(shape, np.max(f), dtype=np.int16)
    return k, i, f


def kif_arrs_to_ints(arr: tuple[np.ndarray, np.ndarray, np.ndarray]):
    return tuple(int(np.max(a)) for a in arr)


def produce_kif(layer: Layer, force_reset=False) -> KIF_t:
    if layer.attributes.get('_produce_kif') and not force_reset:
        return layer.attributes['_produce_kif']
    kif = _produce_kif(layer)
    layer.attributes['_produce_kif'] = kif
    return kif


def request_kif(layer: Layer) -> tuple[KIF_t, ...]:
    if layer.attributes.get('_request_kif'):
        return layer.attributes['_request_kif']
    kif = _request_kif(layer)
    layer.attributes['_request_kif'] = kif
    return kif


def requested_by_non_saturating_quantizer(layer: Layer) -> bool:
    """Check if the current requested kif is from a quantizer.

    Args:
        layer (Layer): The layer to check.

    Returns:
        bool: True if requested by a non-saturating quantizer, False otherwise.
    """
    for n in get_output_layers(layer):
        if isinstance(n, FixedPointQuantizer) and n.SAT not in ('SAT', 'SAT_SYM'):
            return True
        if isinstance(n, Reshape):
            return requested_by_non_saturating_quantizer(n)
    return False


def default_register_precision(layer: Layer):
    if layer.attributes.get('trusted', False):
        # Trusted layers have their precision already set
        return

    _pk, _pi, _pf = produce_kif(layer)  # Maximum possible k,i,f output from this layer
    _rk, _ri, _rf = requested_kif(layer)  # Maximum possible k,i,f may be utilized by the next layer
    _oi, _of = np.minimum(_pi, _ri), np.minimum(_pf, _rf)

    if requested_by_non_saturating_quantizer(layer):
        _ok = _rk
    else:
        _ok = np.minimum(_pk, _rk)

    ok, oi, of = kif_arrs_to_ints((_ok, _oi, _of))

    result_t = to_hls4ml_fixed(ok, oi, of, f'{layer.name}_t')
    layer.attributes['result_t'] = result_t
    layer.get_output_variable().type = result_t

    overrides = {}

    # Set accum_t, if exists ONLY for layers with accum_t directly at output (in general, linear DSP operations)
    if 'accum_t' in layer.attributes:
        accum_kif = kif_arrs_to_ints((_pk, _pi, _pf))
        accum_t = to_hls4ml_fixed(*accum_kif, f'{layer.name}_accum_t')
        overrides['accum_t'] = accum_t

    # Set precision for fixed array (weight_t, bias_t, table_t, etc.)
    for w_name_t, v in layer.attributes.items():
        if not isinstance(v, NamedType) or not w_name_t.endswith('_t'):
            continue  # Not a precision, skip

        w_name = w_name_t[:-2]
        if w_name not in layer.attributes:
            continue  # No matching data found, skip

        weight_var: WeightVariable = layer.attributes[w_name]
        if weight_var is None:  # Corresponding weight not exist, precision to be used nowhere. Put dummy.
            precision = to_hls4ml_fixed(0, 0, 1, f'{layer.name}_{w_name_t}')
        else:
            data = weight_var.data
            if not isinstance(data, np.ndarray):
                raise ValueError(f'Expected data to be np.ndarray, got {type(data)} on layer {layer.name}')
            k, i, f = kif_arrs_to_ints(minimal_kif(data))
            precision = to_hls4ml_fixed(k, i, f, f'{layer.name}_{w_name_t}')
        overrides[w_name_t] = precision

    # Apply overrides
    for w_name_t, v in overrides.items():
        layer.attributes[w_name_t] = v
        if w_name_t[:-2] in layer.attributes:
            # weight variables need extra steps to update precision
            weight_var: WeightVariable = layer.attributes[w_name_t[:-2]]
            weight_var.type = v
            weight_var.update_precision(v.precision)
            layer.model.config.layer_name_precision[f'{layer.name}_{w_name_t[:-2]}'] = str(v.precision)

    return (_pk, _pi, _pf), (_rk, _ri, _rf), (_ok, _oi, _of)


@singledispatch
def register_precision(node: Layer):
    default_register_precision(node)


@register_precision.register
def _(node: Activation):
    default_register_precision(node)
    act_fn = node.attributes['activation'].lower()
    _k, _i, _f = get_input_kifs(node)[0]
    k, i, f = kif_arrs_to_ints((_k, _i, _f))
    table_size = int(2 ** (k + i + f))

    # Temporary workaround for sigmoid and tanh activations, which scale the input by constant factors
    # TODO: Rewrite tanh and sigmoid fn templates
    if act_fn == 'tanh':
        table_size = int(8 / 2.0**-f)  # LUT Range hardcoded to -4 ~ 4, match #fractional bits
    elif act_fn == 'sigmoid':
        table_size = int(16 / 2.0**-f)  # LUT Range hardcoded to -8 ~ 8, match #fractional bits

    node.attributes['table_size'] = table_size


@register_precision.register
def _(node: Softmax):
    if not node.attributes.get('_bit_exact', False):
        # Softmax is not bit-exact by default
        warn(f'Softmax layer {node.name} is converted from a frontend not supporting bit-exact softmax.')
        accum_t = node.attributes['accum_t']
        default_register_precision(node)
        node.attributes['accum_t'] = accum_t
        return

    inv_inp_t: FixedPrecisionType = node.attributes['inv_inp_t'].precision
    exp_table_t: FixedPrecisionType = node.attributes['exp_table_t'].precision
    accum_t = copy(inv_inp_t)
    n_slice = node.attributes['n_in'] // node.attributes.get('n_inner', 1) // node.attributes.get('n_outer', 1)
    scale = ceil(log2(n_slice))
    f_exp = exp_table_t.width - exp_table_t.integer
    f = f_exp + scale
    accum_t.width = f + inv_inp_t.integer
    accum_t.width += scale
    if inv_inp_t.saturation_mode != SaturationMode.WRAP:
        accum_t.saturation_mode = SaturationMode.WRAP
        accum_t.width += scale
        accum_t.integer += scale
    accum_t.width = max(accum_t.width, 1)  # Prevent crashes on absurd bw configurations
    accum_t.rounding_mode = RoundingMode.TRN
    default_register_precision(node)
    impl = node.attributes['implementation']
    match impl:
        case 'latency':
            k, i, f = get_input_kifs(node)[0]
            B = np.max(k) + np.max(i) + np.max(f)
        case 'stable':
            inp_norm_t: FixedPrecisionType = node.attributes['inp_norm_t'].precision
            B = inp_norm_t.width
        case 'lagency':
            raise ValueError('lagency softmax is not supported')
        case 'argmax':
            B = 0
        case _:
            raise ValueError(f'Unknown softmax implementation {impl}')

    exp_table_size = 2 ** int(B)
    node.attributes['exp_table_size'] = exp_table_size
    node.attributes['accum_t'] = NamedType(f'{node.name}_accum_t', accum_t)


@register_precision.register
def _(node: UnaryLUT):
    k, i, f = minimal_kif(node.attributes['table'].data)  # type: ignore
    k, i, f = bool(np.max(k)), int(np.max(i)), int(np.max(f))
    table_t = to_hls4ml_fixed(k, i, f, f'{node.name}_table_t')
    node.attributes['table_t'] = table_t
    default_register_precision(node)


def _get_px_shape(node: Layer):
    if isinstance(node, Pooling1D):
        px_shape = (node.attributes['pool_width'],)
    elif isinstance(node, GlobalPooling1D):
        inp_shape = get_input_shapes(node)[0]
        px_shape = (inp_shape[0],)
    elif isinstance(node, Pooling2D):
        px_shape = (node.attributes['pool_height'], node.attributes['pool_width'])
    elif isinstance(node, GlobalPooling2D):
        inp_shape = get_input_shapes(node)[0]
        px_shape = (inp_shape[0], inp_shape[1])
    else:
        raise ValueError(f'Layer {node.class_name} is not supported for pooling precision derivation')
    return px_shape


@register_precision.register(Pooling1D)
@register_precision.register(Pooling2D)
@register_precision.register(GlobalPooling1D)
@register_precision.register(GlobalPooling2D)
def _(node: Pooling1D | Pooling2D | GlobalPooling1D | GlobalPooling2D):
    default_register_precision(node)
    pool_op = node.attributes['pool_op']
    if pool_op != 'Average':
        return
    px_shape = _get_px_shape(node)
    # Used before division, also more int bits
    i_add = ceil(log2(prod(px_shape)))
    node.attributes['accum_t'].precision.width += i_add
    node.attributes['accum_t'].precision.integer += i_add


@register_precision.register(ParametrizedActivation)
def _(node: ParametrizedActivation):
    default_register_precision(node)
    param = node.attributes['activ_param']
    k, i, f = map(int, minimal_kif(np.array(param)))
    param_t = to_hls4ml_fixed(k, i, f, f'{node.name}_param_t')
    node.attributes['param_t'] = param_t


class BitExact(ModelOptimizerPass):
    """Model-wide bitwidth flow to ensure bit-exactness. Triggered by the presence of FixedPointQuantizer for now.
    On the high level:
    1. (forward flow) Starting from the model input, forward flow down the required bitwidth and shrink
    FixedQuantizer bits when possible. Register the generated bw by each layer as "produced_kif"
    2. (backward flow) For each layer, find the maximum bitwidth it can handle (till which point more bits makes no sense)
    For example, a ap_fixed<8,4> in the downstream has won't take advantage of >4 fractional bits on the input.
    3. (combine) Use the "minimal" of the produced and requested bitwidths on each layer

    In all cases, the process is (supposed to be) bit-exact. Both forward and backward flows use quantized
    interval arithmetic to determine the bitwidths semi-symbolically. BW>=128 are unhandled.
    """

    def __init__(self):
        pass

    def has_fixed_quantizer(self, model: 'ModelGraph'):
        if not any(isinstance(node, FixedPointQuantizer) for node in model.graph.values()):
            return False
        return True

    def _match(self, model: 'ModelGraph'):
        enabled = model.config.config['HLSConfig']['Model'].get('BitExact', None)
        if enabled is None:
            # Enable by default if any FixedPointQuantizer is present
            enabled = self.has_fixed_quantizer(model)
        return enabled

    def transform(self, model: 'ModelGraph'):
        if not self._match(model):
            return False

        if self.has_fixed_quantizer(model):
            # For HGQ-proxy model, no explicit linear layers will be reqired.
            for k in list(model.graph.keys()):
                v = model.graph[k]
                if isinstance(v, Activation) and v.attributes.get('activation') == 'linear':
                    model.remove_node(v)

        for node in model.graph.values():
            if node.attributes.get('bit_exact_transformed'):
                continue
            produce_kif(
                node, force_reset=True
            )  # Shrink FixedPointQuantizer bits when possible to be used in backward flow (requested_kif).

        for node in model.graph.values():
            if node.attributes.get('bit_exact_transformed'):
                continue
            register_precision(node)
            node.attributes['bit_exact_transformed'] = True

        for node in model.graph.values():
            if '_produce_kif' in node.attributes:
                del node.attributes['_produce_kif']
            if '_request_kif' in node.attributes:
                del node.attributes['_request_kif']

        return True


def get_output_layers_and_quantizers(
    node: Layer, layers: list | None = None, quantizers: list | None = None
) -> tuple[list[Layer], list[FixedPointQuantizer]]:

    layers = layers if layers is not None else []
    quantizers = quantizers if quantizers is not None else []
    for _node in get_output_layers(node):
        if isinstance(_node, FixedPointQuantizer):
            quantizers.append(_node)
        elif isinstance(_node, (Reshape, Transpose, Concatenate)):
            layers.append(_node)
            get_output_layers_and_quantizers(_node, layers, quantizers)
        else:
            raise ValueError(f'Layer {node.name} ({node.class_name}) unexpected input layer chain.')
    return layers, quantizers


class FixInputPrecision(OptimizerPass):
    def match(self, node: Layer):
        if not isinstance(node, Input):
            return False

        # Unhandled input precision, usually by a heterogeneous quantizer with non-WRAP saturation
        return node.get_output_variable().type.precision.width > 100

    def transform(self, model, node: Layer):
        layers, out_quantizers = get_output_layers_and_quantizers(node)

        if len(out_quantizers) == 0:  # Input connected to nothing
            new_type = to_hls4ml_fixed(0, 0, 1, f'{node.name}_t')
            node.get_output_variable().type = new_type
            node.model.config.layer_name_precision[node.name] = str(new_type)
            return False

        sat_modes = [l.SAT for l in out_quantizers]
        sat_modes_set = set(sat_modes)
        rnd_modes = [l.RND for l in out_quantizers]
        rnd_modes_set = set(rnd_modes)
        illegal_sat_modes = sat_modes_set - {'WRAP', 'SAT', 'SAT_SYM'}
        illegal_rnd_modes = rnd_modes_set - {'TRN', 'RND'}
        if illegal_sat_modes:
            warn(f'Saturation mode {illegal_sat_modes} may compromise bit-exactness.')
        if len(sat_modes_set) > 1 and not all('SAT' in s for s in sat_modes):
            warn(f'Inconsistent saturation modes {sat_modes_set} in the input quantizers may compromise bit-exactness.')
        if illegal_rnd_modes:
            warn(f'Saturation mode {illegal_rnd_modes} may compromise bit-exactness. Forcing at maximum 24 fractional bits.')

        kifs = [_produce_kif(l) for l in out_quantizers]
        i = np.max([np.max(i) for _, i, _ in kifs])
        k = np.max([np.max(k) for k, _, _ in kifs])
        if illegal_rnd_modes:
            f = 24
        else:
            f = node.get_output_variable().type.precision.fractional
        new_type = to_hls4ml_fixed(k, i, f, f'{node.name}_t')
        if not all('SAT' in s for s in sat_modes):
            # If any of the quantizers are not in SAT mode, set the input to WRAP mode
            new_type.precision.saturation_mode = 'WRAP'
        else:
            new_type.precision.saturation_mode = 'SAT'
        node.get_output_variable().type = new_type
        node.model.config.layer_name_precision[node.name] = str(new_type)
        node.attributes['trusted'] = True

        for layer in layers:
            produce_kif(layer, force_reset=True)
        for layer in layers:
            register_precision(layer)
        for layer in layers:
            if '_produce_kif' in layer.attributes:
                del layer.attributes['_produce_kif']
            if '_request_kif' in layer.attributes:
                del layer.attributes['_request_kif']
        return False
