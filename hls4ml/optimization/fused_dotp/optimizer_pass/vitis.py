import numpy as np

from hls4ml.backends import get_backend
from hls4ml.backends.fpga.fpga_backend import FPGABackend
from hls4ml.model.graph import ModelGraph
from hls4ml.model.layers import Layer
from hls4ml.model.optimizer import OptimizerPass

from ..codegen_backends import VitisCodegenBackend
from ..config import _global_config
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
        self.backend = VitisCodegenBackend()


class VitisFullyUnrolledConvToDense(OptimizerPass):
    def match(self, node: Layer):
        if not (_global_config.enabled and _global_config.enable_pixel_unroll):
            return False
        return (
            node.get_attr('unrolled_codegen')
            and node.class_name in ('Conv1D', 'Conv2D', 'PointwiseConv1D', 'PointwiseConv2D')
            and node.get_attr('n_partitions') == 1
            and node.model.config.get_config_value("IOType") == 'io_parallel'
        )

    def transform(self, model: ModelGraph, node: Layer):
        class_name = 'Dense'

        name = node.name
        attrs = {
            'n_out': node.get_attr('out_width', 1) * node.get_attr('out_height', 1) * node.get_attr('n_filt'),  # type: ignore # noqa: E501
            'n_in': np.prod(node.get_input_variable().shape),
            'result_t': node.get_attr('result_t'),
            'unrolled_codegen': node.get_attr('unrolled_codegen'),
            'weight_data': node.get_attr('weight_data'),
            'bias_data': node.get_attr('weight_data'),
            'r_variables': node.get_attr('r_variables'),
        }
        new_node = model.make_node(class_name, node.name, attrs, node.inputs.copy())
        new_node.attributes[name] = node.attributes[name]
        new_node.attributes['result_t'] = node.attributes['result_t']
        new_node.attributes['index'] = node.attributes['index']
        new_node.index = node.index
        del new_node.attributes.attributes['accum_t']
        del new_node.attributes.attributes['weight_t']
        del new_node.attributes.attributes['bias_t']
        model.replace_node(node, new_node)


class VitisDensePreTemplate(OptimizerPass):
    def match(self, node: Layer):
        if not _global_config.enabled:
            return False
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
        del node.attributes.attributes['weight']
        del node.attributes.attributes['bias']


class VitisConvPreTemplate(OptimizerPass):
    def match(self, node: Layer):
        if not _global_config.enabled:
            return False
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
        # del node.attributes.attributes['weight']
        # del node.attributes.attributes['bias']


unrolled_codegen = VitisUnrollCodeGen()
vitis_dense_pre_template = VitisDensePreTemplate()
vitis_conv_pre_template = VitisConvPreTemplate()
vitis_fully_unrolled_conv = VitisFullyUnrolledConvToDense()
vitis_backend: FPGABackend = get_backend('vitis')
# Optimizer flow is shared
vitis_backend.register_pass('unrolled_codegen', unrolled_codegen, flow='vivado:specific_types')
vitis_backend.register_pass('fully_unrolled_conv_to_dense', vitis_fully_unrolled_conv, flow='vivado:specific_types')
vitis_backend.register_pass('dense_pre_template', vitis_dense_pre_template, flow='vivado:specific_types')
vitis_backend.register_pass('conv_pre_template', vitis_conv_pre_template, flow='vivado:specific_types')
