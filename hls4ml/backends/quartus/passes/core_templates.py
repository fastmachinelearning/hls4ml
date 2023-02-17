from hls4ml.backends.backend import get_backend
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Activation, BatchNormalization, Dense, HardActivation, ParametrizedActivation, PReLU, Softmax

# Dense templates

dense_config_template = """struct config{index} : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned n_zeros = {nzeros};
    static const unsigned n_nonzeros = {nonzeros};
    static const bool store_weights_in_bram = false;

    static const unsigned rf_pad = {rfpad};
    static const unsigned bf_pad = {bfpad};

    static const unsigned reuse_factor = {reuse};
    static const unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    typedef {index_t.name} index_t;

    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

dense_function_template = 'nnet::dense_{strategy}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'

dense_include_list = ['nnet_utils/nnet_dense.h', 'nnet_utils/nnet_dense_compressed.h', 'nnet_utils/nnet_dense_stream.h']


class DenseConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Dense)
        self.template = dense_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['nzeros'] = node.get_weights('weight').nzeros
        params['nonzeros'] = node.get_weights('weight').nonzeros
        params['product_type'] = get_backend('quartus').product_type(
            node.get_input_variable().type.precision, node.get_weights('weight').type.precision
        )

        return self.template.format(**params)


class DenseFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Dense, include_header=dense_include_list)
        self.template = dense_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name

        return self.template.format(**params)


# BatchNormalization templates

batchnorm_config_template = """struct config{index} : nnet::batchnorm_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_filt = {n_filt};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
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
        params['product_type'] = get_backend('quartus').product_type(
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


# Activation templates

activ_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    typedef {table_t.name} table_t;
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
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::{implementation};
    typedef {exp_table_t.name} exp_table_t;
    typedef {inv_table_t.name} inv_table_t;
}};\n"""

activ_function_template = 'nnet::{activation}<{input_t}, {output_t}, {config}>({input}, {output});'
param_activ_function_template = 'nnet::{activation}<{input_t}, {output_t}, {config}>({input}, {param}, {output});'

activ_include_list = ['nnet_utils/nnet_activation.h', 'nnet_utils/nnet_activation_stream.h']


class ActivationConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((Activation, ParametrizedActivation, PReLU))
        self.template = activ_config_template

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


class ActivationFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((Activation, HardActivation, Softmax), include_header=activ_include_list)
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
        params['param'] = node.get_weights('alpha').name
        params['config'] = '{}_config{}'.format(node.get_attr('activation'), node.index)

        return self.template.format(**params)
