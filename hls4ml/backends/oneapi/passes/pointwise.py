from copy import copy

import numpy as np

from hls4ml.backends.fpga.fpga_layers import PointwiseConv1D, PointwiseConv2D
from hls4ml.backends.quartus.passes.convolution_templates import (
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

'''
Custom hls4ml layer implementation for 1x1 Conv filters using im2col
Allows lower latency andresource usage, due to less loop invocations
'''

pointwise_conv1d_function_template = (
    'nnet::pointwise_conv_1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
)
pointwise_conv2d_function_template = (
    'nnet::pointwise_conv_2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
)

sepconv1d_include_list = ['nnet_utils/nnet_conv1d.h']
sepconv2d_include_list = ['nnet_utils/nnet_conv2d.h']


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
        return (
            node.class_name in ('Conv1D', 'Conv2D')
            and node.get_attr('filt_height', 1) == 1
            and node.get_attr('filt_width') == 1
            and node.model.config.get_config_value('IOType') == 'io_parallel'
        )

    def transform(self, model, node):
        dim = node.__class__.__name__[-2:]  # '1D' or '2D'
        pw_node = model.make_node(
            'PointwiseConv' + dim, node.name, copy(node.attributes), node.inputs.copy(), outputs=node.outputs.copy()
        )
        if len(node.weights['weight'].data.shape) == 2:  # This can happen if we assign weights of Dense layer to 1x1 Conv2D
            expand_axis = tuple(range(int(dim[0])))
            pw_node.weights['weight'].data = np.expand_dims(node.weights['weight'].data, axis=expand_axis)
        pw_node.weights['bias'].data = node.weights['bias'].data
        model.replace_node(node, pw_node)

        return True
