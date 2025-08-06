from math import ceil, log2

from hls4ml.backends.backend import get_backend
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import (
    Activation,
    BatchNormalization,
    Dense,
    HardActivation,
    LayerNormalization,
    ParametrizedActivation,
    PReLU,
    Softmax,
)
from hls4ml.model.optimizer.passes.hgq_proxy_model import UnaryLUT

# Dense templates

dense_config_template = """struct config{index} : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const unsigned n_nonzeros = {nonzeros};
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    typedef {index_t.name} index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = {dense_function}<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

dense_function_template = 'nnet::dense<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'

dense_include_list = ['nnet_utils/nnet_dense.h', 'nnet_utils/nnet_dense_compressed.h', 'nnet_utils/nnet_dense_stream.h']


class DenseConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Dense)
        self.template = dense_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['nzeros'] = node.get_weights('weight').nzeros
        params['nonzeros'] = node.get_weights('weight').nonzeros
        params['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_weights('weight').type.precision
        )

        namespace = params['namespace']

        if node.get_attr('strategy').lower() == 'latency':
            params['dense_function'] = 'nnet::DenseLatency'
        elif node.get_attr('strategy').lower() == 'resource':
            if int(params['reuse_factor']) <= int(params['n_in']):
                params['dense_function'] = 'nnet::DenseResource_rf_leq_nin'
            else:
                params['dense_function'] = 'nnet::DenseResource_rf_gt_nin_rem0'
            # The 3rd case is never used
        elif node.get_attr('strategy').lower() == 'resource_unrolled':
            params['dense_function'] = f'{namespace}::dense_resource_unrolled_{node.index}'
        elif node.get_attr('strategy').lower() == 'distributed_arithmetic':
            # Only triggered in io_streaming mode
            params['dense_function'] = f'{namespace}::dense_da_wrapper_{node.index}'

        return self.template.format(**params)

    def match(self, node):
        if node.get_attr('strategy') == 'distributed_arithmetic':
            return False  # DA does not use common dense template
        return super().match(node)


class DenseFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Dense, include_header=dense_include_list)
        self.template = dense_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name

        return self.template.format(**params)

    def match(self, node):
        if node.get_attr('strategy') == 'distributed_arithmetic':
            return False  # DA does not use common dense template
        return super().match(node)


# BatchNormalization templates

batchnorm_config_template = """struct config{index} : nnet::batchnorm_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_filt = {n_filt};
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef {bias_t.name} bias_t;
    typedef {scale_t.name} scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

batchnorm_function_template = 'nnet::normalize<{input_t}, {output_t}, {config}>({input}, {output}, {scale}, {bias});'

batchnorm_include_list = ['nnet_utils/nnet_batchnorm.h', 'nnet_utils/nnet_batchnorm_stream.h']


class BatchNormalizationConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(BatchNormalization)
        self.template = batchnorm_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['n_in'] = node.get_input_variable().size_cpp()
        params['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_weights('scale').type.precision
        )

        return self.template.format(**params)


class BatchNormalizationFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(BatchNormalization, include_header=batchnorm_include_list)
        self.template = batchnorm_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['scale'] = node.get_weights('scale').name
        params['bias'] = node.get_weights('bias').name

        return self.template.format(**params)


# LayerNormalization templates

layernorm_config_template = """struct config{index} : nnet::layernorm_config {{
    static const unsigned n_in = {n_in};
    static const unsigned seq_len = {seq_len};
    static const unsigned axis = {axis};
    static const unsigned epsilon_power_of_10 = {epsilon_power_of_10};
    static const unsigned table_range_power2 = {table_range_power2};
    static const unsigned table_size = {table_size};
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {scale_t.name} scale_t;
    typedef {table_t.name} table_t;
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

layernorm_function_template = 'nnet::layernormalize<{input_t}, {output_t}, {config}>({input}, {output}, {scale}, {bias});'

layernorm_include_list = ['nnet_utils/nnet_layernorm.h']


class LayerNormalizationConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(LayerNormalization)
        self.template = layernorm_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['n_in'] = node.get_input_variable().size_cpp()
        params['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_weights('scale').type.precision
        )

        return self.template.format(**params)


class LayerNormalizationFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(LayerNormalization, include_header=layernorm_include_list)
        self.template = layernorm_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['scale'] = node.get_weights('scale').name
        params['bias'] = node.get_weights('bias').name

        return self.template.format(**params)


# Activation templates

activ_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    typedef {table_t.name} table_t;
}};\n"""

param_activ_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    typedef {table_t.name} table_t;
    typedef {param_t.name} param_t;
}};\n"""

hard_activ_config_template = """struct {type}_config{index} {{
    static const unsigned n_in = {n_in};
    static const {slope_t.name} slope;
    static const {shift_t.name} shift;
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
}};
const {slope_t.name} {type}_config{index}::slope = {slope};
const {shift_t.name} {type}_config{index}::shift = {shift};\n"""

softmax_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_slice = {n_slice};
    static const unsigned n_outer = {n_outer};
    static const unsigned n_inner = {n_inner};
    static const unsigned parallelization_factor = {parallelization_factor};
    static const unsigned exp_table_size = {exp_table_size};
    static const unsigned inv_table_size = {inv_table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    static const unsigned axis = {axis};
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::{implementation};
    static constexpr float exp_scale = {exp_scale};
    typedef {exp_table_t.name} exp_table_t;
    typedef {inv_table_t.name} inv_table_t;
    typedef {accum_t.name} accum_t;
    typedef {inv_inp_t.name} inv_inp_t;
    typedef {inp_norm_t_str} inp_norm_t;
}};\n"""

activ_function_template = 'nnet::{activation}<{input_t}, {output_t}, {config}>({input}, {output});'
param_activ_function_template = (
    'nnet::{activation}<{input_t}, {param_t.name}, {output_t}, {config}>({input}, {param}, {output});'
)

activ_include_list = ['nnet_utils/nnet_activation.h', 'nnet_utils/nnet_activation_stream.h']


class ActivationConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((Activation, UnaryLUT))
        self.template = activ_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['type'] = node.get_attr('activation')

        return self.template.format(**params)


class ParamActivationConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((ParametrizedActivation, PReLU))
        self.template = param_activ_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['type'] = node.get_attr('activation')

        return self.template.format(**params)


class HardActivationConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HardActivation)
        self.template = hard_activ_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['type'] = node.get_attr('activation')

        return self.template.format(**params)


class SoftmaxConfigTemplate(ActivationConfigTemplate):
    def __init__(self):
        super(ActivationConfigTemplate, self).__init__(Softmax)  # Skip ActivationConfigTemplate's __init__
        self.template = softmax_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['type'] = node.get_attr('activation')
        params.setdefault('exp_table_size', params['table_size'])
        params.setdefault('inv_table_size', params['table_size'])
        params.setdefault('n_inner', 1)
        params.setdefault('n_outer', 1)
        params.setdefault('exp_scale', 1.0)
        params.setdefault('parallelization_factor', -1)

        n_slice = params['n_in'] // params['n_inner'] // params['n_outer']  # type: ignore
        params['n_slice'] = n_slice

        if params['accum_t'].name == 'model_default_t':  # type: ignore
            scale = ceil(log2(n_slice))
            exp_table_t = node.attributes['exp_table_t'].precision
            signed, width, integers = exp_table_t.signed, exp_table_t.width, exp_table_t.integer
            params['accum_t_str'] = f'ap_{"" if signed else "u"}fixed<{width + scale}, {integers + scale}>'
        else:
            params['accum_t_str'] = params['accum_t'].name  # type: ignore
        if params['inv_inp_t'].name == 'model_default_t':  # type: ignore
            params['inv_inp_t'] = params['exp_table_t']

        if params['implementation'] == 'stable':
            if 'inp_norm_t' not in params:
                # Only used in stable (max-normalized) implementation
                input_t = node.get_input_variable().type.precision
                width, iwidth, signed = input_t.width, input_t.integer, input_t.signed  # noqa: F841
                width, iwidth = width - signed, iwidth - signed
                if signed:
                    # Fix table size if too large
                    exp_table_size = params['inv_table_size']
                    params['exp_table_size'] = str(min(int(exp_table_size), 2**width))
                params['inp_norm_t_str'] = f'ap_ufixed<{width}, {iwidth}>'
            else:
                params['inp_norm_t_str'] = params['inp_norm_t'].name  # type: ignore
        else:
            params['inp_norm_t_str'] = 'ap_fixed<1,0>'

        return self.template.format(**params)


class SoftmaxFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Softmax, include_header=activ_include_list)
        self.template = activ_function_template

    def format(self, node):
        params = self._default_function_params(node)
        use_multidim = node.get_attr('n_inner', 1) > 1 or node.get_attr('n_outer', 1) > 1
        use_multidim = use_multidim and node.model.config.get_config_value('IOType') == 'io_parallel'
        params['activation'] = 'softmax' if not use_multidim else 'softmax_multidim'
        params['config'] = f'softmax_config{node.index}'

        return self.template.format(**params)


class ActivationFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((Activation, HardActivation), include_header=activ_include_list)
        self.template = activ_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['activation'] = node.get_attr('activation').lower()
        params['config'] = '{}_config{}'.format(node.get_attr('activation'), node.index)

        return self.template.format(**params)


class ParametrizedActivationFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(ParametrizedActivation, include_header=activ_include_list)
        self.template = param_activ_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['activation'] = node._get_act_function_name()
        params['param'] = node.get_attr('activ_param', 1.0)
        params['config'] = '{}_config{}'.format(node.get_attr('activation'), node.index)

        return self.template.format(**params)


class PReLUFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(PReLU, include_header=activ_include_list)
        self.template = param_activ_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['activation'] = node.get_attr('activation').lower()
        params['param'] = node.get_weights('param').name
        params['config'] = '{}_config{}'.format(node.get_attr('activation'), node.index)

        return self.template.format(**params)
