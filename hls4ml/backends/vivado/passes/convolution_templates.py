from hls4ml.backends.backend import get_backend
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import (
    Conv1D,
    Conv1DTranspose,
    Conv2D,
    Conv2DBatchnorm,
    Conv2DTranspose,
    DepthwiseConv1D,
    DepthwiseConv2D,
    SeparableConv1D,
    SeparableConv2D,
)

# Shared multiplication template

conv_mult_config_template = """struct config{index}_mult : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned reuse_factor = {reuse};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned n_zeros = {nzeros};
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = {dense_function}<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

# Conv1D templates

conv1d_config_template = """struct config{index} : nnet::conv1d_config {{
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned filt_width = {filt_width};
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = {n_filt};
    static const unsigned stride_width = {stride_width};
    static const unsigned dilation = {dilation};
    static const unsigned out_width = {out_width};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::{strategy};
    static const nnet::conv_implementation implementation = nnet::conv_implementation::{implementation};
    static const unsigned min_width = {min_width};
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = {n_partitions};
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = {fill_fn}<data_T, CONFIG_T>;
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    typedef {config_t} mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::{scale_index_type}<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = {conv_fn}<data_T, res_T, CONFIG_T>;
}};
const ap_uint<config{index}::filt_width> config{index}::pixels[] = {{{instructions}}};\n"""

conv1d_function_template = 'nnet::conv_1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
depthconv1d_function_template = (
    'nnet::depthwise_conv_1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
)

conv1d_include_list = ['nnet_utils/nnet_conv1d.h', 'nnet_utils/nnet_conv1d_stream.h']


class Conv1DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((Conv1D, DepthwiseConv1D))
        self.template = conv1d_config_template
        self.mult_template = conv_mult_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['dilation'] = node.get_attr('dilation', 1)
        params['nzeros'] = node.get_weights('weight').nzeros

        params['config_t'] = f'config{node.index}_mult'
        if node.get_attr('in_width') == node.get_attr('min_width'):
            params['scale_index_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_type'] = 'scale_index_regular'

        namespace = params['namespace']
        if node.model.config.get_config_value('IOType') == 'io_parallel':
            params['fill_fn'] = f'{namespace}::fill_buffer_{node.index}'
        else:
            params['fill_fn'] = 'nnet::FillConv1DBuffer'

        is_pointwise_parallel_latency = (
            node.get_attr('filt_width') == 1
            and node.get_attr('strategy').lower() == 'latency'
            and node.model.config.get_config_value('IOType') == 'io_parallel'
        )

        n_partitions = node.attributes['n_partitions']

        if is_pointwise_parallel_latency and n_partitions == 1:
            params['conv_fn'] = 'nnet::BatchedDenseForConv1D'
        else:
            if node.get_attr('strategy').lower() == 'latency':
                params['conv_fn'] = 'nnet::Conv1DLatency'
            else:
                params['conv_fn'] = 'nnet::Conv1DResource'

        params['min_width'] = node.get_attr('min_width', node.get_attr('in_width'))
        params['instructions'] = node.get_attr('instructions', '0')

        conv_config = self.template.format(**params)

        mult_params = self._default_config_params(node)
        if is_pointwise_parallel_latency and n_partitions == 1:
            mult_params['n_in'] = (
                node.get_attr('in_width') * node.get_attr('n_chan') * node.get_attr('filt_width') // n_partitions
            )
            mult_params['n_out'] = node.get_attr('in_width') * node.get_attr('n_filt') // n_partitions
        else:
            mult_params['n_in'] = node.get_attr('n_chan') * node.get_attr('filt_width')
            mult_params['n_out'] = node.get_attr('n_filt')
        mult_params['nzeros'] = node.get_weights('weight').nzeros
        mult_params['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_weights('weight').type.precision
        )

        namespace = params['namespace']

        if node.get_attr('strategy').lower() == 'latency':
            if isinstance(node, DepthwiseConv1D):
                mult_params['dense_function'] = 'nnet::DepthwiseDenseLatency'
            else:
                mult_params['dense_function'] = 'nnet::DenseLatency'
        elif node.get_attr('strategy').lower() == 'resource':
            if isinstance(node, DepthwiseConv1D):
                if int(mult_params['reuse_factor']) <= int(mult_params['n_out']):
                    mult_params['dense_function'] = 'nnet::DepthwiseDenseResource_rf_leq_nout'
                else:
                    if int(mult_params['reuse_factor']) % int(mult_params['n_out']) == 0:
                        mult_params['dense_function'] = 'nnet::DepthwiseDenseResource_rf_gt_nout_rem0'
                    else:
                        mult_params['dense_function'] = 'nnet::DepthwiseDenseResource_rf_gt_nout'
            else:
                if int(mult_params['reuse_factor']) <= int(mult_params['n_in']):
                    mult_params['dense_function'] = 'nnet::DenseResource_rf_leq_nin'
                else:
                    if int(mult_params['reuse_factor']) % int(mult_params['n_in']) == 0:
                        mult_params['dense_function'] = 'nnet::DenseResource_rf_gt_nin_rem0'
                    else:
                        mult_params['dense_function'] = 'nnet::DenseResource_rf_gt_nin'
        elif node.get_attr('strategy').lower() == 'resource_unrolled':
            mult_params['dense_function'] = f'{namespace}::dense_resource_unrolled_{node.index}'
        elif node.get_attr('strategy').lower() == 'distributed_arithmetic':
            mult_params['dense_function'] = f'{namespace}::dense_da_wrapper_{node.index}'

        mult_config = self.mult_template.format(**mult_params)

        return mult_config + '\n' + conv_config

    def match(self, node):
        if node.get_attr('strategy') == 'distributed_arithmetic':
            io_type = node.model.config.get_config_value('IOType')
            if io_type == 'io_parallel':
                # DA impl use alternate entry point for io_parallel conv
                return False
        return super().match(node)


class Conv1DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Conv1D, include_header=conv1d_include_list)
        self.template = conv1d_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['data_format'] = 'cf' if node.get_attr('data_format') == 'channels_first' else 'cl'
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name

        return self.template.format(**params)

    def match(self, node):
        if node.get_attr('strategy') == 'distributed_arithmetic':
            io_type = node.model.config.get_config_value('IOType')
            if io_type == 'io_parallel':
                # DA impl use alternate entry point for io_parallel conv
                return False
        return super().match(node)


class DepthwiseConv1DFunctionTemplate(Conv1DFunctionTemplate):
    def __init__(self):
        super(Conv1DFunctionTemplate, self).__init__(DepthwiseConv1D, include_header=sepconv1d_include_list)
        self.template = depthconv1d_function_template


# Conv2D Templates

conv2d_config_template = """struct config{index} : nnet::conv2d_config {{
    static const unsigned pad_top = {pad_top};
    static const unsigned pad_bottom = {pad_bottom};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned filt_height = {filt_height};
    static const unsigned filt_width = {filt_width};
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = {n_filt};
    static const unsigned stride_height = {stride_height};
    static const unsigned stride_width = {stride_width};
    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::{strategy};
    static const nnet::conv_implementation implementation = nnet::conv_implementation::{implementation};
    static const unsigned min_height = {min_height};
    static const unsigned min_width = {min_width};
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = {n_partitions};
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = {fill_fn}<data_T, CONFIG_T>;
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    typedef {config_t} mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::{scale_index_height_type}<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::{scale_index_width_type}<K, S, W>;
}};
const ap_uint<config{index}::filt_height * config{index}::filt_width> config{index}::pixels[] = {{{instructions}}};\n"""

conv2d_function_template = 'nnet::conv_2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
depthconv2d_function_template = (
    'nnet::depthwise_conv_2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
)

conv2d_include_list = ['nnet_utils/nnet_conv2d.h', 'nnet_utils/nnet_conv2d_stream.h']


class Conv2DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((Conv2D, Conv2DBatchnorm, DepthwiseConv2D))
        self.template = conv2d_config_template
        self.mult_template = conv_mult_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['dilation'] = node.get_attr('dilation', 1)
        params['nzeros'] = node.get_weights('weight').nzeros

        params['config_t'] = f'config{node.index}_mult'

        if node.get_attr('in_height') == node.get_attr('min_height'):
            params['scale_index_height_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_height_type'] = 'scale_index_regular'

        if node.get_attr('in_width') == node.get_attr('min_width'):
            params['scale_index_width_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_width_type'] = 'scale_index_regular'

        if node.model.config.get_config_value('IOType') == 'io_parallel':
            namespace = params['namespace']
            params['fill_fn'] = f'{namespace}::fill_buffer_{node.index}'
        else:
            params['fill_fn'] = 'nnet::FillConv2DBuffer'

        params['min_height'] = node.get_attr('min_height', node.get_attr('in_height'))
        params['min_width'] = node.get_attr('min_width', node.get_attr('in_width'))
        params['instructions'] = node.get_attr('instructions', '0')

        conv_config = self.template.format(**params)

        mult_params = self._default_config_params(node)
        mult_params['n_in'] = node.get_attr('n_chan') * node.get_attr('filt_height') * node.get_attr('filt_width')
        mult_params['n_out'] = node.get_attr('n_filt')
        mult_params['nzeros'] = node.get_weights('weight').nzeros
        mult_params['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_weights('weight').type.precision
        )

        namespace = params['namespace']
        if node.get_attr('strategy').lower() == 'latency':
            if isinstance(node, DepthwiseConv2D):
                mult_params['dense_function'] = 'nnet::DepthwiseDenseLatency'
            else:
                mult_params['dense_function'] = 'nnet::DenseLatency'
        elif node.get_attr('strategy').lower() == 'resource':
            if isinstance(node, DepthwiseConv2D):
                if int(mult_params['reuse_factor']) <= int(mult_params['n_out']):
                    mult_params['dense_function'] = 'nnet::DepthwiseDenseResource_rf_leq_nout'
                else:
                    if int(mult_params['reuse_factor']) % int(mult_params['n_out']) == 0:
                        mult_params['dense_function'] = 'nnet::DepthwiseDenseResource_rf_gt_nout_rem0'
                    else:
                        mult_params['dense_function'] = 'nnet::DepthwiseDenseResource_rf_gt_nout'
            else:
                if int(mult_params['reuse_factor']) <= int(mult_params['n_in']):
                    mult_params['dense_function'] = 'nnet::DenseResource_rf_leq_nin'
                else:
                    if int(mult_params['reuse_factor']) % int(mult_params['n_in']) == 0:
                        mult_params['dense_function'] = 'nnet::DenseResource_rf_gt_nin_rem0'
                    else:
                        mult_params['dense_function'] = 'nnet::DenseResource_rf_gt_nin'
        elif node.get_attr('strategy').lower() == 'resource_unrolled':
            mult_params['dense_function'] = f'{namespace}::dense_resource_unrolled_{node.index}'
        elif node.get_attr('strategy').lower() == 'distributed_arithmetic':
            mult_params['dense_function'] = f'{namespace}::dense_da_wrapper_{node.index}'

        mult_config = self.mult_template.format(**mult_params)

        return mult_config + '\n' + conv_config

    def match(self, node):
        if node.get_attr('strategy') == 'distributed_arithmetic':
            io_type = node.model.config.get_config_value('IOType')
            if io_type == 'io_parallel':
                # DA impl use alternate entry point for io_parallel conv
                return False
        return super().match(node)


class Conv2DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((Conv2D, Conv2DBatchnorm), include_header=conv2d_include_list)
        self.template = conv2d_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['data_format'] = 'cf' if node.get_attr('data_format') == 'channels_first' else 'cl'
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name

        return self.template.format(**params)

    def match(self, node):
        if node.get_attr('strategy') == 'distributed_arithmetic':
            io_type = node.model.config.get_config_value('IOType')
            if io_type == 'io_parallel':
                # DA impl use alternate entry point for io_parallel conv
                return False
        return super().match(node)


class DepthwiseConv2DFunctionTemplate(Conv2DFunctionTemplate):
    def __init__(self):
        super(Conv2DFunctionTemplate, self).__init__(DepthwiseConv2D, include_header=sepconv2d_include_list)
        self.template = depthconv2d_function_template


# SeparableConv1D/2D Templates

sepconv_config_template = """struct config{index} {{
    typedef {depthwise_config} depthwise_config;
    typedef {pointwise_config} pointwise_config;
}};\n"""

sepconv1d_function_template = (
    'nnet::separable_conv_1d_{data_format}<{input_t}, {dw_output_t}, {output_t}, {config}>('
    '{input}, {output}, {d}, {p}, {z}, {b});'
)
sepconv2d_function_template = (
    'nnet::separable_conv_2d_{data_format}<{input_t}, {dw_output_t}, {output_t}, {config}>('
    '{input}, {output}, {d}, {p}, {z}, {b});'
)

sepconv1d_include_list = ['nnet_utils/nnet_conv1d.h', 'nnet_utils/nnet_sepconv1d.h', 'nnet_utils/nnet_sepconv1d_stream.h']
sepconv2d_include_list = ['nnet_utils/nnet_conv2d.h', 'nnet_utils/nnet_sepconv2d.h', 'nnet_utils/nnet_sepconv2d_stream.h']


class SeparableConv1DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(SeparableConv1D)
        self.template = sepconv_config_template
        self.depthwise_template = conv1d_config_template
        self.pointwise_template = conv1d_config_template
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
        # Override bias and bias_t since these are zeros in depthwise step of SepConv1D
        params['bias'] = params['zero_bias']
        params['bias_t'] = params['zero_bias_t']
        params['n_filt'] = params['n_chan'] * node.get_attr('depth_multiplier')  # In depthwise step n_chan == n_filt
        params['dilation'] = node.get_attr('dilation', 1)
        params['nzeros'] = node.get_weights('depthwise').nzeros
        params['index'] = str(node.index) + '_depthwise'
        params['weight_t'] = node.get_weights('depthwise').type
        params['bias_t'] = node.get_weights('zero_bias').type
        if node.model.config.get_config_value('IOType') == 'io_parallel':
            namespace = params['namespace']
            params['fill_fn'] = f'{namespace}::fill_buffer_{node.index}_dw'
        else:
            params['fill_fn'] = 'nnet::FillConv1DBuffer'

        if node.get_attr('unscaled'):
            params['scale_index_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_type'] = 'scale_index_regular'

        params['config_t'] = f'config{node.index}_depthwise_mult'
        # TODO - Extend unrolled Dense Resource
        params['unrolled_function'] = 'DenseResourceUnrolled'
        depthwise_config = self.depthwise_template.format(**params)

        # Depthwise mult config
        mult_params = self._default_config_params(node)
        mult_params['index'] = str(node.index) + '_depthwise'
        mult_params['n_in'] = node.get_attr('n_chan') * node.get_attr('filt_width')
        mult_params['n_out'] = node.get_attr('n_chan')
        mult_params['nzeros'] = node.get_weights('depthwise').nzeros
        mult_params['weight_t'] = node.get_weights('depthwise').type
        mult_params['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_weights('depthwise').type.precision
        )
        # TODO - Extend unrolled Dense Resource to depthwise Conv1D
        mult_params['unrolled_function'] = 'DenseResourceUnrolled'

        depthwise_mult_config = self.depthwise_mult_template.format(**mult_params)

        # Pointwise config
        params = self._default_config_params(node)
        if node.get_attr('data_format') == 'channels_last':
            params['in_width'] = node.get_output_variable().shape[0]
        else:
            params['in_width'] = node.get_output_variable().shape[1]

        params['filt_width'] = 1
        params['stride_width'] = 1
        params['pad_left'] = params['pad_right'] = 0
        params['dilation'] = node.get_attr('dilation', 1)
        params['nzeros'] = node.get_weights('pointwise').nzeros
        params['index'] = str(node.index) + '_pointwise'
        params['weight_t'] = node.get_weights('pointwise').type
        params['min_width'] = params['in_width']
        params['instructions'] = '0'
        if node.model.config.get_config_value('IOType') == 'io_parallel':
            namespace = params['namespace']
            params['fill_fn'] = f'{namespace}::fill_buffer_{node.index}_pw'
        else:
            params['fill_fn'] = 'nnet::FillConv1DBuffer'

        if node.get_attr('unscaled'):
            params['scale_index_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_type'] = 'scale_index_regular'

        params['config_t'] = f'config{node.index}_pointwise_mult'
        # TODO - Extend unrolled Dense Resource
        params['unrolled_function'] = 'DenseResourceUnrolled'
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
        # TODO - Extend unrolled Dense Resource to separable Conv1D
        mult_params['unrolled_function'] = 'DenseResourceUnrolled'

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


class SeparableConv1DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(SeparableConv1D, include_header=sepconv1d_include_list)
        self.template = sepconv1d_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['dw_output_t'] = node.get_attr('dw_output_t').name
        params['data_format'] = 'cf' if node.get_attr('data_format') == 'channels_first' else 'cl'
        params['d'] = node.get_weights('depthwise').name
        params['p'] = node.get_weights('pointwise').name
        params['b'] = node.get_weights('bias').name
        params['z'] = node.get_weights('zero_bias').name

        return self.template.format(**params)


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
            namespace = params['namespace']
            params['fill_fn'] = f'{namespace}::fill_buffer_{node.index}_dw'
        else:
            params['fill_fn'] = 'nnet::FillConv2DBuffer'

        if node.get_attr('unscaled_h'):
            params['scale_index_height_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_height_type'] = 'scale_index_regular'

        if node.get_attr('unscaled_w'):
            params['scale_index_width_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_width_type'] = 'scale_index_regular'

        params['config_t'] = f'config{node.index}_depthwise_mult'
        # TODO - Extend unrolled Dense Resource
        params['unrolled_function'] = 'DenseResourceUnrolled'
        depthwise_config = self.depthwise_template.format(**params)

        # Depthwise mult config
        mult_params = self._default_config_params(node)
        mult_params['index'] = str(node.index) + '_depthwise'
        mult_params['n_in'] = node.get_attr('n_chan') * node.get_attr('filt_height') * node.get_attr('filt_width')
        mult_params['n_out'] = node.get_attr('n_chan')
        mult_params['nzeros'] = node.get_weights('depthwise').nzeros
        mult_params['weight_t'] = node.get_weights('depthwise').type
        mult_params['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_weights('depthwise').type.precision
        )
        # TODO - Extend unrolled Dense Resource to depthwise Conv2D
        mult_params['unrolled_function'] = 'DenseResourceUnrolled'
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
            namespace = params['namespace']
            params['fill_fn'] = f'{namespace}::fill_buffer_{node.index}_pw'
        else:
            params['fill_fn'] = 'nnet::FillConv2DBuffer'

        if node.get_attr('unscaled_h'):
            params['scale_index_height_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_height_type'] = 'scale_index_regular'

        if node.get_attr('unscaled_w'):
            params['scale_index_width_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_width_type'] = 'scale_index_regular'
        params['config_t'] = f'config{node.index}_pointwise_mult'
        # TODO - Extend unrolled Dense Resource
        params['unrolled_function'] = 'DenseResourceUnrolled'
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
        # TODO - Extend unrolled Dense Resource to separable Conv2D
        mult_params['unrolled_function'] = 'DenseResourceUnrolled'
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


# Conv1DTranspose/Conv2DTranspose templates

conv_transpose_config_template = """struct config{index} : nnet::conv_transpose_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
    static const unsigned n_filt = {n_filt};
    static const unsigned stride_height = {stride_height};
    static const unsigned stride_width = {stride_width};
    static const unsigned crop_top = {crop_top};
    static const unsigned crop_left = {crop_left};
    static const unsigned first_block_height = {first_block_height};
    static const unsigned first_block_width = {first_block_width};
    static const unsigned n_blocks_height = {n_blocks_height};
    static const unsigned n_blocks_width = {n_blocks_width};
    static const unsigned grid_height = {grid_height};
    static const unsigned grid_width = {grid_width};
    static const unsigned window_height = {window_height};
    static const unsigned window_width = {window_width};
    static const unsigned first_group_height = {first_group_height};
    static const unsigned first_group_width = {first_group_width};
    static const unsigned block_fifo_depth = {block_fifo_depth};
    static const unsigned n_buf = {n_buf};
    static const unsigned group_of[n_buf];
    static const unsigned slot_shift_height[{n_slots}];
    static const unsigned slot_shift_width[{n_slots}];
    static const unsigned chunk_first_group[{n_chunks}];
    static const unsigned chunk_last_group[{n_chunks}];
    static const unsigned reuse_factor = {reuse};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned n_partitions = {n_partitions};
    static const unsigned n_pixels = n_blocks_height * n_blocks_width / n_partitions;
    static const unsigned multiplier_limit = n_pixels * {config_t}::multiplier_limit;
    template<class data_T, class CONFIG_T>
    using fill_buffer = {fill_fn}<data_T, CONFIG_T>;
    template<class data_T, class res_T, class CONFIG_T>
    using place_block = {place_fn}<data_T, res_T, CONFIG_T>;
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    typedef {config_t} mult_config;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = {conv_fn}<data_T, res_T, CONFIG_T>;
}};
const unsigned config{index}::group_of[] = {{{group_of}}};
const unsigned config{index}::slot_shift_height[] = {{{slot_shift_height}}};
const unsigned config{index}::slot_shift_width[] = {{{slot_shift_width}}};
const unsigned config{index}::chunk_first_group[] = {{{chunk_first_group}}};
const unsigned config{index}::chunk_last_group[] = {{{chunk_last_group}}};"""

conv_transpose_function_template = 'nnet::conv_transpose_cl<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'

conv_transpose_include_list = ['nnet_utils/nnet_conv_transpose.h', 'nnet_utils/nnet_conv_transpose_stream.h']


class ConvTransposeConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((Conv1DTranspose, Conv2DTranspose))
        self.template = conv_transpose_config_template
        self.mult_template = conv_mult_config_template

    def format(self, node):
        params = self._default_config_params(node)
        # A 1D layer is computed as a 2D layer of height 1
        for key, value in (
            ('in_height', 1),
            ('out_height', 1),
            ('stride_height', 1),
            ('crop_top', 0),
            ('first_block_height', 0),
            ('n_blocks_height', 1),
        ):
            params.setdefault(key, value)
        group_of = [g for g in node.get_attr('slot_groups') for _ in range(node.get_attr('n_chan'))]
        params['group_of'] = ','.join(map(str, group_of))
        # Groups of the first and last input of each chunk of reuse_factor inputs; only the Resource strategy reads them
        rf = node.get_attr('reuse_factor')
        chunks = [group_of[j : j + rf] for j in range(0, len(group_of), rf)]
        params['n_chunks'] = len(chunks)
        params['chunk_first_group'] = ','.join(str(chunk[0]) for chunk in chunks)
        params['chunk_last_group'] = ','.join(str(chunk[-1]) for chunk in chunks)
        params['config_t'] = f'config{node.index}_mult'
        self._add_stream_params(params, node.get_attr('slot_shifts'))
        strategy = 'Latency' if node.get_attr('strategy').lower() == 'latency' else 'Resource'
        if node.model.config.get_config_value('IOType') == 'io_parallel':
            params['fill_fn'] = f'{params["namespace"]}::fill_buffer_{node.index}'
            params['place_fn'] = f'{params["namespace"]}::place_block_{node.index}'
            params['conv_fn'] = f'nnet::ConvTranspose{strategy}'
        else:
            params['fill_fn'] = 'nnet::FillConvTransposeBuffer'
            params['place_fn'] = 'nnet::PlaceConvTransposeBlock'
            params['conv_fn'] = f'nnet::ConvTransposeStream{strategy}'
        conv_config = self.template.format(**params)

        mult_params = self._default_config_params(node)
        mult_params['n_in'] = node.get_attr('n_buf')
        mult_params['n_out'] = node.get_attr('n_filt')
        mult_params['nzeros'] = node.get_weights('weight').nzeros
        mult_params['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_weights('weight').type.precision
        )
        if node.get_attr('strategy').lower() == 'latency':
            mult_params['dense_function'] = 'nnet::DenseLatency'
        else:
            mult_params['dense_function'] = 'nnet::DenseResource_rf_leq_nin'
        mult_config = self.mult_template.format(**mult_params)

        return mult_config + '\n' + conv_config

    def _add_stream_params(self, params, slot_shifts):
        # Only the io_stream version reads these. Its first process walks the input and, where blocks read past its end,
        # zeros. Its window holds the last (largest shift + 1) rows and columns, with the newest input last.
        for i_axis, (axis, side) in enumerate([('height', 'top'), ('width', 'left')]):
            first_block = params[f'first_block_{axis}']
            params[f'window_{axis}'] = max(shift[i_axis] for shift in slot_shifts) + 1
            params[f'grid_{axis}'] = max(params[f'in_{axis}'], first_block + params[f'n_blocks_{axis}'])
            params[f'first_group_{axis}'] = params[f'crop_{side}'] - first_block * params[f'stride_{axis}']
        params['n_slots'] = len(slot_shifts)
        params['slot_shift_height'] = ','.join(str(shift_h) for shift_h, _ in slot_shifts)
        params['slot_shift_width'] = ','.join(str(shift_w) for _, shift_w in slot_shifts)

        params['block_fifo_depth'] = self._block_fifo_depth(params)

    def _block_fifo_depth(self, params):
        # A cycle model of the two processes. The first takes reuse_factor cycles per position of its grid and writes a
        # block at the end of each block position, waiting while the FIFO is full. The writer writes one pixel per cycle
        # and takes the next block at the first column of each block in the first output row of a block row. The depth
        # is the smallest at which four images in a row reach the shortest interval, plus one for the pipeline delays
        # the model leaves out.
        rf, grid_w, out_w = params['reuse'], params['grid_width'], params['out_width']
        n_positions = params['grid_height'] * grid_w
        n_out = params['out_height'] * out_w
        blocks = [
            i_h * grid_w + i_w
            for i_h in range(params['first_block_height'], params['first_block_height'] + params['n_blocks_height'])
            for i_w in range(params['first_block_width'], params['first_block_width'] + params['n_blocks_width'])
        ]
        needs = [
            i_h * out_w + i_w
            for i_h in range(params['out_height'])
            for i_w in range(out_w)
            if (i_w == 0 or (i_w + params['crop_left']) % params['stride_width'] == 0)
            and (i_h == 0 or (i_h + params['crop_top']) % params['stride_height'] == 0)
        ]
        n_images = 4

        def interval(depth):
            consumed = []  # when the writer takes each block
            t_block = t_write = pixel = 0
            ends = []
            for i_image in range(n_images):
                next_pos = 0
                for block, need in zip(blocks, needs):
                    # The positions up to the block position, then a free place in the FIFO
                    t_block += (block + 1 - next_pos) * rf
                    next_pos = block + 1
                    if len(consumed) >= depth:
                        t_block = max(t_block, consumed[-depth])
                    # The pixels before the one that needs the block, then the block
                    t_write += i_image * n_out + need - pixel
                    pixel = i_image * n_out + need
                    t_write = max(t_write, t_block)
                    consumed.append(t_write)
                t_block += (n_positions - next_pos) * rf
                t_write += (i_image + 1) * n_out - pixel
                pixel = (i_image + 1) * n_out
                ends.append(max(t_block, t_write))
            return ends[-1] - ends[-2]

        shortest = interval(n_images * len(blocks))
        low, high = 1, n_images * len(blocks)
        while low < high:
            mid = (low + high) // 2
            if interval(mid) <= shortest:
                high = mid
            else:
                low = mid + 1
        return max(2, low + 1)


class ConvTransposeFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((Conv1DTranspose, Conv2DTranspose), include_header=conv_transpose_include_list)
        self.template = conv_transpose_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name

        return self.template.format(**params)
