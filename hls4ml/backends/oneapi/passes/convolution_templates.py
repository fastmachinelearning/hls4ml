from hls4ml.backends.backend import get_backend
from hls4ml.backends.oneapi.oneapi_template import StreamFunctionCallTemplate, TaskSequenceTemplate
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Conv1D, Conv2D, Conv2DBatchnorm, DepthwiseConv1D, DepthwiseConv2D

# TODO - Dilation rate ?

""" Shared mutliplication config """
conv_mult_config_template = """struct config{index}_mult : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};

    static const unsigned rf_pad = {rfpad};
    static const unsigned bf_pad = {bfpad};

    static const unsigned reuse_factor = {reuse};
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

""" 1D Conv """
conv1d_config_template = """struct config{index} : nnet::conv1d_config {{
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};

    static const unsigned filt_width = {filt_width};
    static const unsigned impl_filt_width = {impl_filt_width};
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = {n_filt};
    static const unsigned out_width = {out_width};

    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned stride_width = {stride_width};
    static const unsigned dilation = {dilation};

    static const unsigned reuse_factor = {reuse};
    static const unsigned parallelization_factor = {parallelization};
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::{implementation};

    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    typedef {config_t} mult_config;
}};
"""

conv1d_function_template = 'nnet::conv_1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'

conv1d_task_sequence_template = (
    'task_sequence<nnet::conv_1d_{data_format}_stream<{input_pipe}, {output_pipe}, {config}>> {name};'
)

conv_stream_function_template = '{name}.async({w}, {b});'

conv1d_include_list = ['nnet_utils/nnet_conv1d.h', 'nnet_utils/nnet_conv1d_stream.h']


depthconv1d_function_template = (
    'nnet::depthwise_conv_1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
)
depthconv1d_include_list = [
    'nnet_utils/nnet_conv1d.h',
    'nnet_utils/nnet_conv1d_resource.h',
    'nnet_utils/nnet_depthconv1d.h',
    'nnet_utils/nnet_depthconv1d_resource.h',
]


class Conv1DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((Conv1D, DepthwiseConv1D))
        self.template = conv1d_config_template
        self.mult_template = conv_mult_config_template

    def format(self, node):
        conv_params = self._default_config_params(node)
        conv_params['dilation'] = node.get_attr('dilation', 1)
        if conv_params['dilation'] != 1:
            raise RuntimeError('dilation != 1 not supported yet')
        conv_params['config_t'] = f'config{node.index}_mult'
        conv_config = self.template.format(**conv_params)

        mult_params = self._default_config_params(node)
        mult_params['n_in'] = node.get_attr('n_chan') * node.get_attr('filt_width')
        mult_params['n_out'] = node.get_attr('n_filt')
        mult_params['product_type'] = get_backend('oneAPI').product_type(
            node.get_input_variable().type.precision, node.get_weights('weight').type.precision
        )
        mult_config = self.mult_template.format(**mult_params)

        return mult_config + '\n' + conv_config


class Conv1DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Conv1D, include_header=conv1d_include_list)
        self.template = conv1d_function_template

    def format(self, node):
        params = self._default_function_params(node)
        if node.get_attr('data_format') == 'channels_first':
            raise RuntimeError('channels_first not supported on oneAPI')
        params['data_format'] = 'cl'
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name

        return self.template.format(**params)


class Conv1DTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(Conv1D)
        self.template = conv1d_task_sequence_template

    def format(self, node):
        params = self._default_function_params(node)
        if node.get_attr('data_format') == 'channels_first':
            raise RuntimeError('channels_first not supported on oneAPI')
        params['data_format'] = 'cl'
        return self.template.format(**params)


class ConvStreamFunctionTemplate(StreamFunctionCallTemplate):
    def __init__(self):
        super().__init__((Conv1D, Conv2D, Conv2DBatchnorm))
        self.template = conv_stream_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name

        return self.template.format(**params)


class DepthwiseConv1DFunctionTemplate(Conv1DFunctionTemplate):
    def __init__(self):
        super(Conv1DFunctionTemplate, self).__init__(DepthwiseConv1D, include_header=depthconv1d_include_list)
        self.template = depthconv1d_function_template


""" 2D Conv """
conv2d_config_template = """struct config{index} : nnet::conv2d_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};

    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};

    static const unsigned n_filt = {n_filt};
    static const unsigned filt_height = {filt_height};
    static const unsigned filt_width = {filt_width};
    static const unsigned impl_filt_height = {impl_filt_height};
    static const unsigned impl_filt_width = {impl_filt_width};
    static const unsigned kernel_size = filt_height * filt_width;

    static const unsigned pad_top = {pad_top};
    static const unsigned pad_bottom = {pad_bottom};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned stride_height = {stride_height};
    static const unsigned stride_width = {stride_width};

    static const unsigned reuse_factor = {reuse};
    static const unsigned parallelization_factor = {parallelization};
    static const bool store_weights_in_bram = false;

    static const nnet::conv2d_implementation implementation = nnet::conv2d_implementation::{implementation};

    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    typedef {config_t} mult_config;
}};\n"""

conv2d_function_template = 'nnet::conv_2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'

conv2d_task_sequence_template = (
    'task_sequence<nnet::conv_2d_{data_format}_stream<{input_pipe}, {output_pipe}, {config}>> {name};'
)

conv2d_include_list = ['nnet_utils/nnet_conv2d.h', 'nnet_utils/nnet_conv2d_stream.h']


class Conv2DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((Conv2D, Conv2DBatchnorm, DepthwiseConv2D))
        self.template = conv2d_config_template
        self.mult_template = conv_mult_config_template

    def format(self, node):
        conv_params = self._default_config_params(node)
        conv_params['dilation'] = node.get_attr('dilation', 1)
        if conv_params['dilation'] != 1:
            raise RuntimeError('dilation != 1 not supported yet')
        conv_params['config_t'] = f'config{node.index}_mult'
        conv_config = self.template.format(**conv_params)

        mult_params = self._default_config_params(node)
        mult_params['n_in'] = node.get_attr('n_chan') * node.get_attr('filt_height') * node.get_attr('filt_width')
        mult_params['n_out'] = node.get_attr('n_filt')
        mult_params['product_type'] = get_backend('oneAPI').product_type(
            node.get_input_variable().type.precision, node.get_weights('weight').type.precision
        )
        mult_config = self.mult_template.format(**mult_params)

        return mult_config + '\n' + conv_config


class Conv2DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((Conv2D, Conv2DBatchnorm), include_header=conv2d_include_list)
        self.template = conv2d_function_template

    def format(self, node):
        params = self._default_function_params(node)
        if node.get_attr('data_format') == 'channels_first':
            raise RuntimeError('channels_first not supported for oneAPI')
        params['data_format'] = 'cl'
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name

        return self.template.format(**params)


class Conv2DTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__((Conv2D, Conv2DBatchnorm))
        self.template = conv2d_task_sequence_template

    def format(self, node):
        params = self._default_function_params(node)
        if node.get_attr('data_format') == 'channels_first':
            raise RuntimeError('channels_first not supported on oneAPI')
        params['data_format'] = 'cl'
        return self.template.format(**params)


depthconv2d_function_template = (
    'nnet::depthwise_conv_2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
)
depthconv2d_include_list = [
    'nnet_utils/nnet_conv2d.h',
    'nnet_utils/nnet_conv2d_resource.h',
    'nnet_utils/nnet_depthconv2d.h',
    'nnet_utils/nnet_depthconv2d_resource.h',
]


class DepthwiseConv2DFunctionTemplate(Conv2DFunctionTemplate):
    def __init__(self):
        super(Conv2DFunctionTemplate, self).__init__(DepthwiseConv2D, include_header=depthconv2d_include_list)
        self.template = depthconv2d_function_template
