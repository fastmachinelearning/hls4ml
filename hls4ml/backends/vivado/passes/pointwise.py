from hls4ml.backends.fpga.fpga_layers import PointwiseConv1D, PointwiseConv2D
from hls4ml.backends.vivado.passes.convolution_templates import (
    Conv1DConfigTemplate,
    Conv1DFunctionTemplate,
    Conv2DConfigTemplate,
    Conv2DFunctionTemplate,
    conv1d_config_template,
    conv2d_config_template,
    conv_mult_config_template,
)
from hls4ml.model.layers import register_layer
from hls4ml.model.optimizer import OptimizerPass

pointwise_conv1d_function_template = (
    'nnet::pointwise_conv_1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
)
pointwise_conv2d_function_template = (
    'nnet::pointwise_conv_2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
)

sepconv1d_include_list = ['nnet_utils/nnet_conv1d.h', 'nnet_utils/nnet_sepconv1d_stream.h']
sepconv2d_include_list = ['nnet_utils/nnet_conv2d.h', 'nnet_utils/nnet_sepconv2d_stream.h']


class PointwiseConv1DConfigTemplate(Conv1DConfigTemplate):
    def __init__(self):
        super(Conv1DConfigTemplate, self).__init__(PointwiseConv1D)
        self.template = conv1d_config_template
        self.mult_template = conv_mult_config_template


class PointwiseConv1DFunctionTemplate(Conv1DFunctionTemplate):
    def __init__(self):
        super(Conv1DFunctionTemplate, self).__init__(PointwiseConv1D, include_header=sepconv1d_include_list)
        self.template = pointwise_conv1d_function_template


class PointwiseConv2DConfigTemplate(Conv2DConfigTemplate):
    def __init__(self):
        super(Conv2DConfigTemplate, self).__init__(PointwiseConv2D)
        self.template = conv2d_config_template
        self.mult_template = conv_mult_config_template


class PointwiseConv2DFunctionTemplate(Conv2DFunctionTemplate):
    def __init__(self):
        super(Conv2DFunctionTemplate, self).__init__(PointwiseConv2D, include_header=sepconv2d_include_list)
        self.template = pointwise_conv2d_function_template


def register_pointwise(backend):
    # Register the layer types to the layer map
    register_layer('PointwiseConv1D', PointwiseConv1D)
    register_layer('PointwiseConv2D', PointwiseConv2D)

    # Register the optimization passes
    backend.register_pass('optimize_pointwise_conv', OptimizePointwiseConv)

    # Register template passes
    backend.register_template(PointwiseConv1DConfigTemplate)
    backend.register_template(PointwiseConv1DFunctionTemplate)
    backend.register_template(PointwiseConv2DConfigTemplate)
    backend.register_template(PointwiseConv2DFunctionTemplate)


class OptimizePointwiseConv(OptimizerPass):
    def match(self, node):
        if node.get_attr('strategy') == 'distributed_arithmetic':
            if node.class_name == 'Conv1D':
                return False
        return (
            node.class_name in ('Conv1D', 'Conv2D')
            and node.get_attr('filt_height', 1) == 1
            and node.get_attr('filt_width') == 1
        )

    def transform(self, model, node):
        dim = node.__class__.__name__[-2:]  # '1D' or '2D'
        # to remove warning, since these get set again
        new_attrs = {k: v for k, v in node.attributes.items() if k not in ('trace', 'precision', 'reuse_factor')}
        pw_node = model.make_node(
            'PointwiseConv' + dim, node.name, new_attrs, node.inputs.copy(), outputs=node.outputs.copy()
        )
        # Set strategy to ensure lowercase string is passed to the template
        pw_node.set_attr('strategy', node.get_attr('strategy'))
        model.replace_node(node, pw_node)
        return True
