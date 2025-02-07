import typing
from math import prod

import numpy as np

from hls4ml.model.layers import Conv1D, Conv2D, Dense, Layer
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.optimizer.passes.bit_exact import get_input_kifs
from hls4ml.model.types import FixedPrecisionType, Source
from hls4ml.utils.dependency import requires

if typing.TYPE_CHECKING:
    from hls4ml.model import ModelGraph


class DistributedArithmeticCodegen(OptimizerPass):
    '''Generates C++ code for distributed arithmetic implementation of Dense layers'''

    def match(self, node):
        # if not node.get_attr('strategy', 'latency').lower() in ('da', 'distributed_arithmetic'):
        #     return False

        supported = (Dense, Conv1D, Conv2D)
        # TODO: EinsumDense support
        # RNN and depthwise conv families are not supported for now
        if not isinstance(node, supported):
            return False
            raise Exception(f'Layer {node.name} of type {node.__class__.__name__} is not supported by DA optimizer.')

        rf = node.get_attr('reuse_factor', 1)
        if rf != 1:
            raise Exception(f'Layer {node.name} has rf = {rf} != 1, but has strategy = DA.')

        return True

    @requires('da')
    def transform(self, model: 'ModelGraph', node: Layer):
        from da4ml import VitisCodegenBackend, compile_kernel, graph_compile_states

        kernel: np.ndarray = node.attributes['weight'].data
        kernel = kernel.reshape(-1, kernel.shape[-1])
        n_in, n_out = kernel.shape
        fn_name = f'dense_da_{node.index}'

        Ks, Is, Fs = get_input_kifs(node)[0]
        if np.all(Is == 126) and np.all(Fs == 126):
            # No fixed quantizer before this layer to produce reasonable bw
            # Use result_t from prev layer instead
            result_t = node.get_input_variable().type.precision
            if not isinstance(result_t, FixedPrecisionType):
                k = [True] * n_in
                b = [7] * n_in
                i = [0] * n_in
            else:
                _k, _b, _i = result_t.signed, result_t.width, result_t.integer
                k = [_k] * n_in
                b = [_b - _k] * n_in
                i = [_i - _k] * n_in
        else:
            Ks = np.any(Ks.reshape(-1, n_in), axis=0)
            Is = np.max(Is.reshape(-1, n_in), axis=0)
            Fs = np.max(Fs.reshape(-1, n_in), axis=0)
            k, b, i = Ks, Is + Fs, Is
            k, b, i = list(k), list(b), list(i)

        codegen_backend = VitisCodegenBackend(fn_name=fn_name)

        states = compile_kernel(kernel, k, b, i, [False] * n_in, [0] * n_in, 1, 0, 128, 64)  # type: ignore

        inp, out = graph_compile_states(states, False)
        if node.attributes['bias'] is not None:
            bias = node.attributes['bias'].data.ravel()
            assert len(bias) == n_out
            for i, b in enumerate(bias):
                out[i] += b
        _fn, fn_str = codegen_backend(inp, out)

        node.set_attr('da_codegen', Source(fn_str))


class DALatencyDenseTemplate(OptimizerPass):
    def match(self, node: Layer):
        if not node.get_attr('da_codegen') or node.class_name != 'Dense':
            return False
        io_type = node.model.config.get_config_value("IOType")
        return io_type == 'io_parallel'

    def transform(self, model: 'ModelGraph', node: Layer):
        inp_t: str = node.get_input_variable().type.name
        out_t: str = node.get_output_variable().type.name
        inp_name: str = node.get_input_variable().name
        out_name: str = node.get_output_variable().name

        # override function_cpp
        fn_name = f'dense_da_{node.index}<{inp_t}, {out_t}>'
        function_cpp = f'nnet::{fn_name}({inp_name}, {out_name});'
        node.attributes.attributes['function_cpp'] = function_cpp

        # avoid output weights and bias; alternatie entry point does not use them
        del node.attributes['weight_data']
        del node.attributes['bias_data']
        del node.attributes['weight']
        del node.attributes['weight_t']
        del node.attributes['bias']
        del node.attributes['bias_t']


conv_da_template = """struct config{index} {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
    static const unsigned n_filt = {n_filt};
    static const unsigned filt_height = {filt_height};
    static const unsigned filt_width = {filt_width};

    static const unsigned strategy = nnet::latency;
    static const unsigned n_partitions = {n_partitions};
    static const unsigned n_pixels = {n_pixels};;
    constexpr static auto dense_da = nnet::dense_da_{index}<{inp_t}, {out_t}>;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_{index}<data_T, CONFIG_T>;
}};\n"""


class DALatencyConvTemplate(OptimizerPass):
    def match(self, node: Layer):
        if not node.get_attr('da_codegen') and node.class_name in (
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
        n_pixels = prod(node.get_input_variable().shape[:-1]) // node.attributes['n_partitions']

        # conv 1d case, set dummy values for heights
        params.setdefault('in_height', -1)
        params.setdefault('out_height', -1)
        params.setdefault('filt_height', -1)

        config_cpp = conv_da_template.format(inp_t=inp_t, out_t=out_t, n_pixels=n_pixels, **params)
        node.attributes.attributes['config_cpp'] = config_cpp

        # Only unrolled header is required for io_parallel
        include_headers = [
            'nnet_utils/nnet_conv_da.h',
            'nnet_utils/nnet_dense_latency.h',
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
