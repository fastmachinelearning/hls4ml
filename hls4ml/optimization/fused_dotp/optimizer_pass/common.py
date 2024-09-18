from functools import lru_cache
from typing import Sequence

import numpy as np

from hls4ml.backends.fpga.fpga_types import NamedType
from hls4ml.model.graph import ModelGraph
from hls4ml.model.layers import Layer
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.optimizer.passes.hgq_proxy_model import FixedPointQuantizer
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType, Source

from ..codegen_backends import VitisCodegenBackend, code_gen
from ..config import _global_config
from ..dotp_unroll import compile_conv
from ..precision import FixedPointPrecision
from ..symbolic_variable import Variable
from ..utils import Singleton
from .pixel_unrolled_conv import get_input_KIF_idxs


def nn_codegen(
    kernel: np.ndarray,
    bias: np.ndarray | None,
    KIFs_in: Sequence[Sequence[np.ndarray]] | np.ndarray,
    inp_name: str,
    out_name: str,
    backend='vivado',
    index: Sequence[Sequence[int]] | None = None,
):
    """
    Codegen for conv/dense layer
    As individual kernel application in conv layers can be regarded as affine transformation, this function serves as a unified routine for both conv and dense layers. (Or, one may consider dense as conv0d xD)

    Args:
        kernel (np.ndarray): kernel, shape (..., out_channels)
        bias (np.ndarray): bias, shape (out_channels)
        KIF_in (tuple[np.ndarray, ...]): input precision in per-channel: (keep_negative, integer, fractional_bits), each of shape (in_channels,)
        inp_name (str): input variable name
        out_name (str): output variable name
        backend (str, optional): backend. Defaults to 'vivado'. Can be 'vivado' or 'vitis', and the behavior is the same for now.
    """  # noqa: E501

    if backend.lower() in ('vivado', 'vitis'):
        _backend = VitisCodegenBackend()
    else:
        raise ValueError(f'Backend {backend} not supported')
    if index is None:
        assert len(KIFs_in) == 1
        index = [range(len(KIFs_in[0][0]))]

    R = []

    def to_symbol(p, i):
        if i >= 0:
            if isinstance(p, FixedPointPrecision):
                return Variable(p, id=f'{inp_name}[{i}]')
            else:
                return p
        else:
            return 0

    for KIF_in, idxs in zip(KIFs_in, index):
        assert len(KIF_in) == 3

        precisions = [FixedPointPrecision.from_kif(k, i, f) for k, i, f in zip(*KIF_in)]
        inp = np.array([to_symbol(p, _i) for _i, p in zip(idxs, precisions)])
        r = compile_conv(kernel, inp)
        if bias is not None:
            r = r + bias
        R.extend(r)
    return code_gen(R, _backend, out_name), R


def get_input_KIF(model: ModelGraph, node: Layer):
    """Get input precision per-channel, in the form of (k, i, f) each of shape (in_channels,)"""

    assert 'weight_data' in node.attributes, 'No weight data found'
    kernel = node.attributes['weight_data']
    inp_node: Layer = model.graph[node.inputs[0]]
    input_named_t: NamedType = inp_node.attributes['result_t']

    # Get input precision per-element
    *inp_shape, out_shape = kernel.shape

    if isinstance(inp_node, FixedPointQuantizer):
        assert inp_node.mask_kbi is not None
        K, B, I = inp_node.mask_kbi  # noqa: E741
        K, B, I = K[0], B[0], I[0]  # noqa: E741
        K, I, F = K, I - K, B - I  # noqa: E741
        if K.ndim > 1:  # pixel-direction heterogeneous conv quantization not supported by the kernel (channel-wise only)
            axs_rm = tuple(np.arange(K.ndim - 1))
            K, I, F = np.max(K, axis=axs_rm), np.max(I, axis=axs_rm), np.max(F, axis=axs_rm)  # noqa: E741
        K, I, F = (np.broadcast_to(K, inp_shape), np.broadcast_to(I, inp_shape), np.broadcast_to(F, inp_shape))
    else:
        assert isinstance(input_named_t.precision, (FixedPrecisionType, IntegerPrecisionType))
        input_t = input_named_t.precision
        k, i, f = input_t.signed, input_t.integer, input_t.fractional
        i -= k
        K, I, F = np.full(inp_shape, k), np.full(inp_shape, i), np.full(inp_shape, f)
    return tuple(x.ravel() for x in (K, I, F))


def get_output_KIF(model: ModelGraph, node: Layer):
    """Get output precision per-channel, in the form of (k, i, f) each of shape (out_channels,)"""
    assert 'weight_data' in node.attributes, 'No weight data found'
    kernel = node.attributes['weight_data']
    out_nodes: list[Layer] = [layer for layer in model.graph.values() if node.name in layer.inputs]  # type: ignore
    output_named_t: NamedType = node.attributes['result_t']

    # Get input precision per-element
    *inp_shape, out_shape = kernel.shape

    if len(out_nodes) > 0 and all(isinstance(out_node, FixedPointQuantizer) for out_node in out_nodes):
        out_nodes: list[FixedPointQuantizer]
        Ks, Is, Fs = [], [], []
        for out_node in out_nodes:
            assert out_node.mask_kbi is not None
            K, B, I = out_node.mask_kbi  # noqa: E741
            K, B, I = K[0], B[0], I[0]  # noqa: E741
            K, I, F = K, I - K, B - I  # noqa: E741
            if K.ndim > 1:  # pixel-direction heterogeneous conv quantization not supported by the kernel (channel-wise only)
                axs_rm = tuple(np.arange(K.ndim - 1))
                K, I, F = np.max(K, axis=axs_rm), np.max(I, axis=axs_rm), np.max(F, axis=axs_rm)
            Ks.append(K)
            Is.append(I)
            Fs.append(F)
        K, I, F = np.max(Ks, axis=0), np.max(Is, axis=0), np.max(Fs, axis=0)
        output_t = output_named_t.precision
    else:
        assert isinstance(output_named_t.precision, (FixedPrecisionType, IntegerPrecisionType))
        output_t = output_named_t.precision
        k, i, f = output_t.signed, output_t.integer, output_t.fractional
        i -= k
        K, I, F = np.full(out_shape, k), np.full(out_shape, i), np.full(out_shape, f)
    return tuple(x.ravel() for x in (K, I, F))


@lru_cache
def latency_mat_vec_mul_fn_gen(model: ModelGraph, node: Layer):
    """Codegen for optimized matmul-like layer (conv/dense) fully unrolled
    Args:
        model (ModelGraph): model graph
        node (Layer): layer

    Returns:
        str: code
        list: resource variables
    """
    kernel = node.attributes['weight_data']

    # Get input precision per-element

    if 'Conv' in node.class_name and np.prod(node.attributes.attributes.get('dilation', 1)) > 1:
        KIFs_in = [get_input_KIF(model, node)]
        index = None
    else:
        KIFs_in, index = get_input_KIF_idxs(model, node)
    backend = model.config.config['Backend']
    oprs, r_variables = nn_codegen(kernel, node.attributes['bias_data'], KIFs_in, 'inp', 'out', backend, index=index)
    opr_code = '\n    '.join(oprs)
    return opr_code, r_variables


class UnrollCodeGenPass(OptimizerPass, metaclass=Singleton):
    """Unroll codegen pass.
    Works with Dense and ConvXD layers out of the box.
    """

    def __init__(self, *targets: str):
        self.target = targets
        self.backend = None

    def match(self, node: Layer):
        if not _global_config.enabled:
            return False
        return any(node.class_name == target for target in self.target)

    def get_stream_type_name(self, name: str) -> str:
        assert self.backend is not None, 'Backend not set'
        return self.backend.get_stream_type_name(name)

    def transform(self, model: ModelGraph, node: Layer):

        assert len(node.inputs) == 1, 'Dense/Conv should have exactly one input'
        opr_code, r_variables = latency_mat_vec_mul_fn_gen(model, node)

        inp_node: Layer = model.graph[node.inputs[0]]

        input_named_t: NamedType = inp_node.attributes['result_t']
        output_named_t: NamedType = node.attributes['result_t']
        input_t_name: str = input_named_t.name
        output_t_name: str = output_named_t.name

        io_type: str = model.config.get_config_value('IOType')
        assert io_type in ('io_stream', 'io_parallel'), f'io_type {io_type} is unknown.'
        if io_type == 'io_stream':
            input_t_name = self.get_stream_type_name(input_t_name)
            output_t_name = self.get_stream_type_name(output_t_name)

        fn_name = f'unrolled_fn_{node.index}'
        code = f'''\n
// {node.name}\nvoid {fn_name}({input_t_name} *inp, {output_t_name} *out) {{
    #pragma HLS INLINE
    {opr_code}
}}'''
        node.attributes.attributes['unrolled_codegen'] = Source(code)
        node.attributes.attributes['r_variables'] = r_variables
