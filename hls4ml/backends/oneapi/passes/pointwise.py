from hls4ml.backends.fpga.fpga_layers import PointwiseConv1D, PointwiseConv2D
from hls4ml.backends.oneapi.oneapi_template import StreamFunctionCallTemplate, TaskSequenceTemplate
from hls4ml.backends.oneapi.passes.convolution_templates import (
    Conv1DConfigTemplate,
    Conv2DConfigTemplate,
    conv1d_config_template,
    conv2d_config_template,
    conv_mult_config_template,
)
from hls4ml.backends.template import FunctionCallTemplate
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

pointwise_conv1d_task_sequence_template = (
    'task_sequence<nnet::pintwise_conv_1d_{data_format}_stream<{input_pipe}, {output_pipe}, {config}>> {name};'
)

pointwise_conv2d_task_sequence_template = (
    'task_sequence<nnet::pintwise_conv_2d_{data_format}_stream<{input_pipe}, {output_pipe}, {config}>> {name};'
)

pointwise_conv_stream_function_template = '{name}.async({w}, {b});'

sepconv1d_include_list = ['nnet_utils/nnet_conv1d.h']
sepconv2d_include_list = ['nnet_utils/nnet_conv2d.h']


class PointwiseConv1DConfigTemplate(Conv1DConfigTemplate):
    def __init__(self):
        super(Conv1DConfigTemplate, self).__init__(PointwiseConv1D)
        self.template = conv1d_config_template
        self.mult_template = conv_mult_config_template


class PointwiseConv1DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(PointwiseConv1D, include_header=sepconv1d_include_list)
        self.template = pointwise_conv1d_function_template

    def format(self, node):
        params = self._default_function_params(node)
        if node.get_attr('data_format') == 'channels_first':
            raise RuntimeError('channels_first not supported on oneAPI')
        params['data_format'] = 'cl'
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name

        return self.template.format(**params)


class PointwiseConv1DTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(PointwiseConv1D)
        self.template = pointwise_conv1d_function_template

    def format(self, node):
        params = self._default_function_params(node)
        if node.get_attr('data_format') == 'channels_first':
            raise RuntimeError('channels_first not supported on oneAPI')
        params['data_format'] = 'cl'
        return self.template.format(**params)


class PointwiseConv2DConfigTemplate(Conv2DConfigTemplate):
    def __init__(self):
        super(Conv2DConfigTemplate, self).__init__(PointwiseConv2D)
        self.template = conv2d_config_template
        self.mult_template = conv_mult_config_template


class PointwiseConv2DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(PointwiseConv2D, include_header=sepconv2d_include_list)
        self.template = pointwise_conv2d_function_template

    def format(self, node):
        params = self._default_function_params(node)
        if node.get_attr('data_format') == 'channels_first':
            raise RuntimeError('channels_first not supported on oneAPI')
        params['data_format'] = 'cl'
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name

        return self.template.format(**params)


class PointwiseConv2DTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(PointwiseConv2D)
        self.template = pointwise_conv1d_function_template

    def format(self, node):
        params = self._default_function_params(node)
        if node.get_attr('data_format') == 'channels_first':
            raise RuntimeError('channels_first not supported on oneAPI')
        params['data_format'] = 'cl'
        return self.template.format(**params)


class PointwiseConvStreamFunctionTemplate(StreamFunctionCallTemplate):
    def __init__(self):
        super().__init__((PointwiseConv1D, PointwiseConv2D))

    def format(self, node):
        params = self._default_function_params(node)
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name

        return self.template.format(**params)


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
        new_attrs = {k: v for k, v in node.attributes.items() if k not in ('trace', 'precision', 'reuse_factor')}
        pw_node = model.make_node(
            'PointwiseConv' + dim, node.name, new_attrs, node.inputs.copy(), outputs=node.outputs.copy()
        )
        model.replace_node(node, pw_node)

        return True
