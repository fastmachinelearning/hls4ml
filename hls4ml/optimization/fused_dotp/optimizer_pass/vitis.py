from hls4ml.backends import get_backend
from hls4ml.backends.fpga.fpga_backend import FPGABackend
from hls4ml.model.graph import ModelGraph
from hls4ml.model.layers import Layer
from hls4ml.model.optimizer import OptimizerPass

from .common import UnrollCodeGenPass

conf_template = """struct config{index}{postfix} {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned strategy = nnet::latency;
    constexpr static auto unrolled_fn = nnet::unrolled_fn_{index};
}};\n"""


class VitisUnrollCodeGen(UnrollCodeGenPass):
    def __init__(self):
        super().__init__('Dense', 'Conv1D', 'Conv2D', 'PointwiseConv1D', 'PointwiseConv2D')

    def get_stream_type_name(self, name: str) -> str:
        return f'{name}::value_type'


class VitisDensePreTemplate(OptimizerPass):
    def match(self, node: Layer):
        return node.get_attr('unrolled_codegen') and node.class_name == 'Dense'

    def transform(self, model: ModelGraph, node: Layer):
        io_type = model.config.get_config_value("IOType")

        node.attributes['unrolled_fn_name'] = f'nnet::unrolled_fn_{node.index}'

        if io_type != 'io_parallel':
            return

        self.latency_transform(model, node)

    def latency_transform(self, model: ModelGraph, node: Layer):
        inp_name: str = node.get_input_variable().name
        out_name: str = node.get_output_variable().name

        n_in = node.get_attr('n_in')
        n_out = node.get_attr('n_out')
        name_postfix = ''

        # override config_cpp
        config_cpp = conf_template.format(n_in=n_in, n_out=n_out, name=node.name, index=node.index, postfix=name_postfix)
        node.attributes.attributes['dense_config'] = config_cpp

        # override function_cpp
        fn_name = f'dense<config{node.index}>'
        function_cpp = f'nnet::{fn_name}({inp_name}, {out_name});'
        node.attributes.attributes['function_cpp'] = function_cpp

        # Only unrolled header is required for io_parallel
        include_header = ['nnet_utils/nnet_unrolled.h', 'nnet_utils/nnet_dense.h']
        node.attributes.attributes['include_header'] = include_header

        # avoid output weights and bias; alternatie entry point does not use them
        del node.attributes.attributes['weight_data']
        del node.attributes.attributes['bias_data']


class VitisConvPreTemplate(OptimizerPass):
    def match(self, node: Layer):
        if node.get_attr('implementation') != 'linebuffer':
            return False
        return node.get_attr('unrolled_codegen') and node.class_name in (
            'Conv1D',
            'Conv2D',
            'PointwiseConv1D',
            'PointwiseConv2D',
        )

    def transform(self, model: ModelGraph, node: Layer):
        io_type = model.config.get_config_value("IOType")
        node.attributes['unrolled_fn_name'] = f'nnet::unrolled_fn_{node.index}'

        if io_type != 'io_parallel':
            return

        self.latency_transform(model, node)

    def latency_transform(self, model: ModelGraph, node: Layer):

        inp_name: str = node.get_input_variable().name
        out_name: str = node.get_output_variable().name

        n_chan: int = node.attributes.attributes['n_chan']
        ker_width: int = node.attributes.attributes['filt_width']
        ker_height: int = node.attributes.attributes.get('filt_height', 1)
        n_in = n_chan * ker_width * ker_height
        n_out = n_chan
        name_postfix = '_mult'

        # override config_cpp::mult
        config_cpp = conf_template.format(n_in=n_in, n_out=n_out, name=node.name, index=node.index, postfix=name_postfix)
        node.attributes.attributes['dense_config'] = config_cpp

        # override function_cpp
        class_name = node.class_name
        if class_name.startswith('Pointwise'):
            class_name = class_name[9:]

        if class_name == 'Conv1D':
            fn_name = f'conv_1d<config{node.index}>'
        elif class_name == 'Conv2D':
            fn_name = f'conv_2d<config{node.index}>'
        else:
            raise ValueError(f'Unsupported layer type {node.class_name}')
        function_cpp = f'nnet::{fn_name}({inp_name}, {out_name});'
        node.attributes.attributes['function_cpp'] = function_cpp

        # Only unrolled header is required for io_parallel
        include_headers = [
            'nnet_utils/nnet_unrolled.h',
            'nnet_utils/nnet_dense_latency.h',
            f'nnet_utils/nnet_{class_name.lower()}.h',
            'nnet_utils/nnet_conv_stream.h',  # some properties defined in config need this
        ]
        node.attributes.attributes['include_header'] = include_headers

        # avoid output weights and bias; alternatie entry point does not use them
        del node.attributes.attributes['weight_data']
        del node.attributes.attributes['bias_data']


unrolled_codegen = VitisUnrollCodeGen()
vitis_dense_pre_template = VitisDensePreTemplate()
vitis_conv_pre_template = VitisConvPreTemplate()
# vivado_backend: FPGABackend = get_backend('vivado')
vitis_backend: FPGABackend = get_backend('vitis')
# Optimizer flow is shared
vitis_backend.register_pass('unrolled_codegen', unrolled_codegen, flow='vivado:specific_types')
vitis_backend.register_pass('dense_pre_template', vitis_dense_pre_template, flow='vivado:specific_types')
vitis_backend.register_pass('conv_pre_template', vitis_conv_pre_template, flow='vivado:specific_types')
