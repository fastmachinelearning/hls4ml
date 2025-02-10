import typing
from functools import singledispatch
from math import prod

import numpy as np

from hls4ml.model.layers import Conv1D, Conv2D, Dense, EinsumDense, Layer
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.optimizer.passes.bit_exact import get_input_layers, get_output_layers, im2col, pad_arrs, stride_arrs
from hls4ml.model.optimizer.passes.hgq_proxy_model import FixedPointQuantizer
from hls4ml.model.types import FixedPrecisionType, Source
from hls4ml.utils.dependency import requires
from hls4ml.utils.einsum_utils import parse_einsum  # noqa: F401

if typing.TYPE_CHECKING:
    from hls4ml.model import ModelGraph


def add_kernel_wrapper(index: int, n_in: int, n_out: int):

    wrapper = f'''template <typename inp_t, typename out_t, typename DUMMY> struct dense_da_wrapper_{index} {{
static void dense(inp_t inp[{n_in}], out_t out[{n_out}], void *weights=nullptr, void *biases=nullptr) {{
    dense_da_{index}(inp, out);
}}
}};'''
    return wrapper


def _get_input_kif(node: Layer):
    """Get the input k, i, f to a layer.
    Use the results from the last FixedPointQuantzer if available, fallback to the result_t of the input variable.
    """
    result_t = node.get_input_variable().type.precision
    inp_shape = node.get_input_variable().shape
    if not isinstance(result_t, FixedPrecisionType):
        raise ValueError(f'Input to layer {node.name} is not a fixed point type - DA optimization not supported.')
    inp_layer = get_input_layers(node)[0]
    inp_shape = node.get_input_variable().shape
    if isinstance(inp_layer, FixedPointQuantizer):
        Ks, _Bs, _Is = inp_layer.mask_kbi
        Is, Fs = _Is - Ks, _Bs - _Is
    else:
        Ks = np.ones(inp_shape, dtype=np.int8)
        Is = Fs = np.full(inp_shape, 126, dtype=np.int8)

    _k, _B, _I = result_t.signed, result_t.width, result_t.integer
    _k, _i, _f = _k, _I - _k, _B - _I
    k, i, f = np.minimum(Ks, _k), np.minimum(Is, _i), np.minimum(Fs, _f)
    return k, i, f


@singledispatch
def get_kernel_inp_kif(node: Layer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the input k, i, f to a kernel. Supports Dense, Conv1/2D, and EinsumDense layers."""
    raise NotImplementedError(f'Layer {node.name} of type {node.__class__.__name__} is not supported by DA optimizer.')


@get_kernel_inp_kif.register
def _(node: Dense):
    k, i, f = _get_input_kif(node)
    n_ch = k.shape[-1]
    return k.reshape(-1, n_ch).max(axis=0), i.reshape(-1, n_ch).max(axis=0), f.reshape(-1, n_ch).max(axis=0)


@get_kernel_inp_kif.register(Conv1D)
@get_kernel_inp_kif.register(Conv2D)
def _(layer: Conv1D | Conv2D):
    assert layer.attributes.attributes['data_format'] == 'channels_last', 'Only channels_last format is supported'
    kernel = layer.attributes.attributes['weight'].data
    k_in, i_in, f_in = _get_input_kif(layer)
    k_in, i_in, f_in = pad_arrs(layer, 0, k_in, i_in, f_in)
    k_in, i_in, f_in = im2col(kernel.shape, k_in, i_in, f_in)
    k_in, i_in, f_in = stride_arrs(layer, k_in, i_in, f_in)
    n_ker_in: int = k_in.shape[-1]
    return (
        k_in.reshape(-1, n_ker_in).max(axis=0),
        i_in.reshape(-1, n_ker_in).max(axis=0),
        f_in.reshape(-1, n_ker_in).max(axis=0),
    )


@get_kernel_inp_kif.register
def _(node: EinsumDense):
    inp_tpose_idx = node.attributes.attributes['inp_tpose_idxs'][0]
    L = node.attributes['n_free_data']
    I = node.attributes['n_inplace']  # noqa: E741
    C = node.attributes['n_contract']

    k, i, f = _get_input_kif(node)
    k, i, f = k.transpose(inp_tpose_idx), i.transpose(inp_tpose_idx), f.transpose(inp_tpose_idx)
    k, i, f = k.reshape(I, L, C)  # noqa: E741
    return k.max(axis=1), i.max(axis=1), f.max(axis=1)


class DistributedArithmeticCodegen(OptimizerPass):
    '''Generates C++ code for distributed arithmetic implementation of Dense layers'''

    def match(self, node):
        if not node.get_attr('strategy', None) == 'distributed_arithmetic':
            return False
        if 'da_codegen' in node.attributes:
            return False
        supported = (Dense, Conv1D, Conv2D)
        # EinsumDense support is standalone as it requires additional configuration
        # RNN and depthwise conv families are not supported for now
        if not isinstance(node, supported):
            raise Exception(f'Layer {node.name} of type {node.__class__.__name__} is not supported by DA optimizer.')

        rf = node.get_attr('reuse_factor', 1)
        if rf != 1:
            raise Exception(f'Layer {node.name} has rf = {rf} != 1, but has strategy = DA.')

        return True

    @requires('da')
    def transform(self, model: 'ModelGraph', node: Layer):
        from da4ml import VitisCodegenBackend, compile_kernel, graph_compile_states
        from da4ml.cmvm.codegen import Namer

        kernel: np.ndarray = node.attributes['weight'].data
        kernel = kernel.reshape(-1, kernel.shape[-1])
        n_in, n_out = kernel.shape
        fn_name = f'dense_da_{node.index}'

        k, i, f = get_kernel_inp_kif(node)
        k, b, i = list(k), list(i + f + k), list(i + k)
        codegen_backend = VitisCodegenBackend(fn_name=fn_name)

        with Namer().tmp_scope():
            states = compile_kernel(kernel, k, b, i, [False] * n_in, [0] * n_in, 1, 0, 128, 64)  # type: ignore
            inp, out = graph_compile_states(states, False)
            if node.attributes['bias'] is not None:
                bias = node.attributes['bias'].data.ravel()
                assert len(bias) == n_out
                for i, b in enumerate(bias):
                    out[i] = out[i] + b
            _fn, fn_str = codegen_backend(inp, out)
        fn_str = fn_str.replace('{', '{\n    #pragma HLS INLINE', 1)  # add pragma to first line

        io_type = node.model.config.get_config_value("IOType")
        if io_type != 'io_parallel':
            fn_str += '\n\n' + add_kernel_wrapper(node.index, n_in, n_out)

        node.set_attr('da_codegen', Source(fn_str))


dense_da_stream_template = '''struct config{index} {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::distributed_arithmetic;
    constexpr static auto dense_da = nnet::dense_da_{index}<typename {inp_t}::value_type, typename {out_t}::value_type>;
}};\n'''


class FuseQuantizerIntoDALayers(OptimizerPass):
    def match(self, node: Layer):
        if not isinstance(node, FixedPointQuantizer):
            return False
        next_layers = get_output_layers(node)
        if not next_layers:  # Output quantizer
            return False
        allow = (Dense,)
        if all(n == 1 for n in node.mask_kbi[0].shape[:-1]):
            allow += (Conv1D, Conv2D)
        for next_layer in next_layers:
            if next_layer.get_attr('strategy', None) != 'distributed_arithmetic':
                return False
        return len(next_layers) == 1 or (node.RND == 'RND' and node.SAT == 'WRAP')  # avoid resource overhead

    def transform(self, model: 'ModelGraph', node: FixedPointQuantizer):
        for out_layer in get_output_layers(node):
            k, i, f = get_kernel_inp_kif(out_layer)
            B, I = i + f + k, i + k  # noqa: E741

            quantization_lines, replaces = [], []
            for i, (_k, _B, _I) in enumerate(zip(k, B, I)):
                u = '' if _k else 'u'
                _src = f'inp[{i}]'
                _dst = f'inp_q_{i}'
                if _B > 0:
                    var_def = f'ap_{u}fixed<{_B}, {_I}, AP_{node.RND}, AP_{node.SAT}> {_dst} = {_src};'
                else:
                    var_def = f'ap_ufixed<1, 0> {_dst} = 0;'
                quantization_lines.append(var_def)
                replaces.append((_src, _dst))

            replaces.append(('#pragma HLS INLINE', '#pragma HLS INLINE\n  ' + '\n    '.join(quantization_lines)))

            da_source: Source = out_layer.attributes['da_codegen']
            code: str = da_source.code
            for src, dst in replaces:
                code = code.replace(src, dst)
            out_layer.attributes['da_codegen'] = Source(code)
        model.remove_node(node)
        return True


class DALatencyDenseTemplate(OptimizerPass):
    # For Dense, distributed arithmetic do not call the original, regardless of the io_type
    # FOr io_stream, a minimal config will still be generated
    def match(self, node: Layer):
        if node.class_name != 'Dense':
            return False
        if 'function_cpp' in node.attributes:
            return False
        return node.get_attr('strategy', None) == 'distributed_arithmetic'

    def transform(self, model: 'ModelGraph', node: Layer):
        inp_t: str = node.get_input_variable().type.name
        out_t: str = node.get_output_variable().type.name
        inp_name: str = node.get_input_variable().name
        out_name: str = node.get_output_variable().name

        # override function_cpp
        io_type = node.model.config.get_config_value("IOType")
        if io_type == 'io_parallel':
            fn_name = f'dense_da_{node.index}<{inp_t}, {out_t}>'
            function_cpp = f'nnet::{fn_name}({inp_name}, {out_name});'
            node.attributes.attributes['function_cpp'] = function_cpp
        else:
            assert io_type == 'io_stream'
            config_cpp = dense_da_stream_template.format(inp_t=inp_t, out_t=out_t, **node.attributes.attributes)
            function_cpp = f'nnet::dense<{inp_t}, {out_t}, config{node.index}>({inp_name}, {out_name});'
            node.attributes['config_cpp'] = config_cpp
            node.attributes['function_cpp'] = function_cpp
            node.attributes['include_header'] = ['nnet_utils/nnet_da_wrappers.h']

        # avoid output weights and bias; alternatie entry point does not use them
        del node.attributes['weight_data']
        del node.attributes['bias_data']
        del node.attributes['weight']
        del node.attributes['weight_t']
        del node.attributes['bias']
        del node.attributes['bias_t']


conv_da_parallel_template = """struct config{index} {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
    static const unsigned n_filt = {n_filt};
    static const unsigned filt_height = {filt_height};
    static const unsigned filt_width = {filt_width};
    static const unsigned stride_height = {stride_height};
    static const unsigned stride_width = {stride_width};

    static const unsigned strategy = nnet::distributed_arithmetic;
    static const unsigned n_partitions = {n_partitions};
    static const unsigned n_pixels = {n_pixels};
    constexpr static auto dense_da = nnet::dense_da_{index}<{inp_t}, {out_t}>;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_{index}<data_T, CONFIG_T>;
}};\n"""


class DALatencyConvTemplate(OptimizerPass):
    def match(self, node: Layer):
        if not node.get_attr('strategy', None) == 'distributed_arithmetic':
            return False
        if 'function_cpp' in node.attributes:
            return False
        if node.class_name not in (
            'Conv1D',
            'Conv2D',
            'PointwiseConv1D',
            'PointwiseConv2D',
        ):
            return False
        if node.get_attr('implementation') != 'linebuffer':
            return False
        io_type = node.model.config.get_config_value("IOType")
        return io_type == 'io_parallel'

    def transform(self, model: 'ModelGraph', node: Layer):
        fmt = node.get_attr('data_format')
        assert (
            fmt == 'channels_last'
        ), f'At layer {node.name}, data_format must be "channels_last" for DA optimization. Got {fmt}.'
        inp_t: str = node.get_input_variable().type.name
        out_t: str = node.get_output_variable().type.name
        inp_name: str = node.get_input_variable().name
        out_name: str = node.get_output_variable().name

        ker_shape = node.attributes['weight'].data.shape

        # function call generation
        class_name = node.class_name
        if class_name.startswith('Pointwise'):
            class_name = class_name[9:]

        ndim = len(ker_shape) - 2
        function_cpp = f'nnet::conv{ndim}d_da_cl<config{node.index}, {inp_t}, {out_t}>({inp_name}, {out_name});'
        node.attributes.attributes['function_cpp'] = function_cpp

        # config generation
        params = node.attributes.attributes.copy()
        n_pixels = prod(node.get_output_variable().shape[:-1]) // node.attributes['n_partitions']

        # conv 1d case, set dummy values for heights
        params.setdefault('in_height', -1)
        params.setdefault('out_height', -1)
        params.setdefault('filt_height', -1)
        params.setdefault('stride_height', -1 if ndim == 1 else 1)

        config_cpp = conv_da_parallel_template.format(inp_t=inp_t, out_t=out_t, n_pixels=n_pixels, **params)
        node.attributes.attributes['config_cpp'] = config_cpp

        # Only unrolled header is required for io_parallel
        include_headers = [
            'nnet_utils/nnet_da_wrappers.h',
            f'nnet_utils/nnet_{class_name.lower()}.h',
            'nnet_utils/nnet_conv_stream.h',  # some properties defined in config need this
        ]
        node.attributes.attributes['include_header'] = include_headers

        # avoid output weights and bias; alternatie entry point does not use them
        del node.attributes['weight_data']
        del node.attributes['bias_data']
        del node.attributes['weight']
        del node.attributes['bias']
        del node.attributes['weight_t']
        del node.attributes['bias_t']


# einsum_dense_da_parallel_template = '''
# struct config{index} {{
#     typedef config{index}_tpose_inp tpose_inp_conf;
#     typedef config{index}_tpose_out tpose_out_conf;
#     typedef void dense_conf;

#     // Layer Sizes
#     static const unsigned n_free_data = {n_free_data};
#     static const unsigned n_free_kernel = {n_free_kernel};
#     static const unsigned n_contract = {n_contract};
#     static const unsigned n_inplace = {n_inplace};

#     // Resource reuse info
#     static const unsigned io_type = nnet::op_parallel;
#     static const unsigned strategy = nnet::distributed_arithmetic;
#     static const unsigned reuse_factor = 1;
#     static const unsigned parallelization_factor = {parallelization_factor}; // Only useful when n_inplace > 1
# }};
# '''


# class DAEinsumDense(OptimizerPass):
#     def match(self, node: Layer):
#         if not node.get_attr('strategy', None) == 'distributed_arithmetic':
#             return False
#         return node.class_name == 'EinsumDense'

#     def coded_gen(self, node: EinsumDense):

#     def transform(self, model: 'ModelGraph', node: EinsumDense):
#         inp_shape = node.attributes.attributes['inp_shape']
#         out_interpert_shape = node.attributes.attributes['out_interpert_shape']
#         inp_tpose_idxs = node.attributes.attributes['inp_tpose_idxs']
#         out_tpose_idxs = node.attributes.attributes['out_tpose_idxs']
#         tpose_inp_conf_name = f'config{node.index}_tpose_inp'
#         tpose_out_conf_name = f'config{node.index}_tpose_out'
