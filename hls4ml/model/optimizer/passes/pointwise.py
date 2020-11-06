import numpy as np
import re

from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.hls_model import Conv2D, register_layer
from hls4ml.templates import templates

class PointwiseConv2D(Conv2D):
    ''' Optimized Conv2D implementation for 1x1 kernels. '''

    # Nothing to do, will pick up function and config from class name
    pass

pointwise_conv2d_function_template = 'nnet::pointwise_conv_2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'

sepconv2d_include_list = ['nnet_utils/nnet_conv2d.h', 'nnet_utils/nnet_sepconv2d_stream.h']

# Register the layer types to the layer map
register_layer('PointwiseConv2D', PointwiseConv2D)

# Register the templates for config and function
templates.get_backend('Vivado').register_templates(
    'PointwiseConv2D',
    pointwise_conv2d_function_template,
    templates.get_backend('Vivado').get_config_template('Conv2D')
)

class OptimizePointwiseConv2D(OptimizerPass):
    def match(self, node):
        return node.__class__.__name__ == 'Conv2D' and \
            node.get_attr('filt_height') == 1 and \
            node.get_attr('filt_width') == 1

    def transform(self, model, node):
        pw_node = model.make_node('PointwiseConv2D', node.name, node.attributes.copy(), node.inputs.copy())
        model.replace_node(node, pw_node)
        
        return True
