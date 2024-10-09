from hls4ml.backends.backend import get_backend
from hls4ml.backends.oneapi.oneapi_template import StreamFunctionCallTemplate, TaskSequenceTemplate
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import (
    Conv1D,
    Conv2D,
    Conv2DBatchnorm,
    DepthwiseConv2D,
    SeparableConv2D,
)
# TODO - Dilation rate ?

''' Shared mutliplication config '''
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

''' 1D Conv '''
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


class Conv1DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Conv1D)
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


''' 2D Conv '''
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

sepconv2d_function_template = (
    'nnet::separable_conv_2d_{data_format}<{input_t}, {dw_output_t}, {output_t}, {config}>('
    '{input}, {output}, {d}, {p}, {z}, {b});'
)

sepconv_config_template = """struct config{index} {{
    typedef {depthwise_config} depthwise_config;
    typedef {pointwise_config} pointwise_config;
}};\n"""

sepconv2d_include_list = ['nnet_utils/nnet_conv2d.h', 'nnet_utils/nnet_conv2d_resource.h', 'nnet_utils/nnet_depthconv2d.h', 'nnet_utils/nnet_depthconv2d_resource.h']


class DepthwiseConv2DFunctionTemplate(Conv2DFunctionTemplate):
    def __init__(self):
        super(Conv2DFunctionTemplate, self).__init__(DepthwiseConv2D, include_header=sepconv2d_include_list)
        self.template = depthconv2d_function_template





class SeparableConv2DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(SeparableConv2D)
        self.template = sepconv_config_template
        self.depthwise_template = conv2d_config_template
        self.pointwise_template = conv2d_config_template
        self.depthwise_mult_template = conv_mult_config_template
        self.pointwise_mult_template = conv_mult_config_template

    def format(self, node):
        # Separable master config
        params = {}
        params['index'] = node.index
        params['depthwise_config'] = f'config{node.index}_depthwise'
        params['pointwise_config'] = f'config{node.index}_pointwise'
        sep_config = self.template.format(**params)

        # Depthwise config
        params = self._default_config_params(node)
        # Override bias and bias_t since these are zeros in depthwise step of SepConv2D
        params['bias'] = params['zero_bias']
        params['bias_t'] = params['zero_bias_t']
        params['n_filt'] = params['n_chan']  # In depthwise step n_chan == n_filt
        params['dilation'] = node.get_attr('dilation', 1)
        params['nzeros'] = node.get_weights('depthwise').nzeros
        params['index'] = str(node.index) + '_depthwise'
        params['weight_t'] = node.get_weights('depthwise').type
        if node.model.config.get_config_value('IOType') == 'io_parallel':
            params['fill_fn'] = f'fill_buffer_{node.index}_dw'
        else:
            params['fill_fn'] = 'FillConv2DBuffer'

        if node.get_attr('unscaled_h'):
            params['scale_index_height_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_height_type'] = 'scale_index_regular'

        if node.get_attr('unscaled_w'):
            params['scale_index_width_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_width_type'] = 'scale_index_regular'

        params['config_t'] = f'config{node.index}_depthwise_mult'
        depthwise_config = self.depthwise_template.format(**params)

        # Depthwise mult config
        mult_params = self._default_config_params(node)
        mult_params['index'] = str(node.index) + '_depthwise'
        mult_params['n_in'] = node.get_attr('n_chan') * node.get_attr('filt_height') * node.get_attr('filt_width')
        mult_params['n_out'] = node.get_attr('n_chan')
        mult_params['nzeros'] = node.get_weights('depthwise').nzeros
        mult_params['weight_t'] = node.get_weights('depthwise').type
        mult_params['product_type'] = get_backend('oneAPI').product_type(
            node.get_input_variable().type.precision, node.get_weights('depthwise').type.precision
        )
        depthwise_mult_config = self.depthwise_mult_template.format(**mult_params)

        # Pointwise config
        params = self._default_config_params(node)
        if node.get_attr('data_format') == 'channels_last':
            params['in_height'] = node.get_output_variable().shape[0]
            params['in_width'] = node.get_output_variable().shape[1]
        else:
            params['in_height'] = node.get_output_variable().shape[1]
            params['in_width'] = node.get_output_variable().shape[2]

        params['filt_height'] = params['filt_width'] = 1
        params['stride_height'] = params['stride_width'] = 1
        params['pad_left'] = params['pad_right'] = 0
        params['pad_top'] = params['pad_bottom'] = 0
        params['dilation'] = node.get_attr('dilation', 1)
        params['nzeros'] = node.get_weights('pointwise').nzeros
        params['index'] = str(node.index) + '_pointwise'
        params['weight_t'] = node.get_weights('pointwise').type
        params['min_height'] = params['in_height']
        params['min_width'] = params['in_width']
        params['instructions'] = '0'
        if node.model.config.get_config_value('IOType') == 'io_parallel':
            params['fill_fn'] = f'fill_buffer_{node.index}_pw'
        else:
            params['fill_fn'] = 'FillConv2DBuffer'

        if node.get_attr('unscaled_h'):
            params['scale_index_height_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_height_type'] = 'scale_index_regular'

        if node.get_attr('unscaled_w'):
            params['scale_index_width_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_width_type'] = 'scale_index_regular'
        params['config_t'] = f'config{node.index}_pointwise_mult'
        pointwise_config = self.pointwise_template.format(**params)

        # Pointwise mult config
        mult_params = self._default_config_params(node)
        mult_params['index'] = str(node.index) + '_pointwise'
        mult_params['n_in'] = node.get_attr('n_chan')
        mult_params['n_out'] = node.get_attr('n_filt')
        mult_params['nzeros'] = node.get_weights('pointwise').nzeros
        mult_params['weight_t'] = node.get_weights('pointwise').type
        mult_params['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_weights('pointwise').type.precision
        )
        pointwise_mult_config = self.pointwise_mult_template.format(**mult_params)

        return (
            depthwise_mult_config
            + '\n'
            + depthwise_config
            + '\n'
            + pointwise_mult_config
            + '\n'
            + pointwise_config
            + '\n'
            + sep_config
        )


class SeparableConv2DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(SeparableConv2D, include_header=sepconv2d_include_list)
        self.template = sepconv2d_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['dw_output_t'] = node.get_attr('dw_output_t').name
        params['data_format'] = 'cf' if node.get_attr('data_format') == 'channels_first' else 'cl'
        params['d'] = node.get_weights('depthwise').name
        params['p'] = node.get_weights('pointwise').name
        params['b'] = node.get_weights('bias').name
        params['z'] = node.get_weights('zero_bias').name

        return self.template.format(**params)
