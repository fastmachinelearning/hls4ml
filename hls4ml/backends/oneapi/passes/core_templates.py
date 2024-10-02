from hls4ml.backends.backend import get_backend
from hls4ml.backends.oneapi.oneapi_template import StreamFunctionCallTemplate, TaskSequenceTemplate
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Activation, BatchNormalization, Dense, HardActivation, ParametrizedActivation, PReLU, Softmax

# Dense templates

dense_config_template = """struct config{index} : nnet::dense_config {{
    static constexpr unsigned n_in = {n_in};
    static constexpr unsigned n_out = {n_out};
    static constexpr unsigned io_type = nnet::{iotype};
    static constexpr unsigned n_zeros = {nzeros};
    static constexpr unsigned n_nonzeros = {nonzeros};
    static constexpr bool store_weights_in_bram = false;

    static constexpr unsigned rf_pad = {rfpad};
    static constexpr unsigned bf_pad = {bfpad};

    static constexpr unsigned reuse_factor = {reuse};
    static constexpr unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static constexpr unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static constexpr unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static constexpr unsigned block_factor_rounded = block_factor + bf_pad;
    static constexpr unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static constexpr unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static constexpr unsigned multiplier_scale = multiplier_limit/n_out;

    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    typedef {index_t.name} index_t;

    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

dense_function_template = 'nnet::dense_{strategy}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
dense_task_sequence_template = 'task_sequence<nnet::dense_{strategy}_stream<{input_pipe}, {output_pipe}, {config}>> {name};'
dense_stream_function_template = '{name}.async({w}, {b});'
dense_include_list = ['nnet_utils/nnet_dense.h', 'nnet_utils/nnet_dense_stream.h']


class DenseConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Dense)
        self.template = dense_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['nzeros'] = node.get_weights('weight').nzeros
        params['nonzeros'] = node.get_weights('weight').nonzeros
        params['product_type'] = get_backend('oneAPI').product_type(
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


class DenseTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(Dense)
        self.template = dense_task_sequence_template

    def format(self, node):
        params = self._default_function_params(node)

        return self.template.format(**params)


class DenseStreamFunctionTemplate(StreamFunctionCallTemplate):
    def __init__(self):
        super().__init__(Dense)
        self.template = dense_stream_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name

        return self.template.format(**params)


# BatchNormalization templates

batchnorm_config_template = """struct config{index} : nnet::batchnorm_config {{
    static constexpr unsigned n_in = {n_in};
    static constexpr unsigned n_filt = {n_filt};
    static constexpr unsigned io_type = nnet::{iotype};
    static constexpr unsigned reuse_factor = {reuse};
    static constexpr bool store_weights_in_bram = false;
    typedef {bias_t.name} bias_t;
    typedef {scale_t.name} scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

batchnorm_function_template = 'nnet::normalize<{input_t}, {output_t}, {config}>({input}, {output}, {scale}, {bias});'
batchnorm_task_sequence_template = 'task_sequence<nnet::normalize_stream<{input_pipe}, {output_pipe}, {config}>> {name};'
batchnorm_stream_function_template = '{name}.async({scale}, {bias});'
batchnorm_include_list = ['nnet_utils/nnet_batchnorm.h', 'nnet_utils/nnet_batchnorm_stream.h']


class BatchNormalizationConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(BatchNormalization)
        self.template = batchnorm_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['n_in'] = node.get_input_variable().size_cpp()
        params['product_type'] = get_backend('oneAPI').product_type(
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


class BatchNormalizationTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(BatchNormalization)
        self.template = batchnorm_task_sequence_template

    def format(self, node):
        params = self._default_function_params(node)

        return self.template.format(**params)


class BatchNormalizationStreamFunctionTemplate(StreamFunctionCallTemplate):
    def __init__(self):
        super().__init__(BatchNormalization)
        self.template = batchnorm_stream_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['scale'] = node.get_weights('scale').name
        params['bias'] = node.get_weights('bias').name

        return self.template.format(**params)


# Activation templates

activ_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static constexpr unsigned n_in = {n_in};
    static constexpr unsigned table_size = {table_size};
    static constexpr unsigned io_type = nnet::{iotype};
    static constexpr unsigned reuse_factor = {reuse};
    typedef {table_t.name} table_t;
}};\n"""

param_activ_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static constexpr unsigned n_in = {n_in};
    static constexpr unsigned table_size = {table_size};
    static constexpr unsigned io_type = nnet::{iotype};
    static constexpr unsigned reuse_factor = {reuse};
    typedef {table_t.name} table_t;
    typedef {param_t.name} param_t;
}};\n"""

hard_activ_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static constexpr unsigned n_in = {n_in};
    static constexpr {slope_t.name} slope = {slope};
    static constexpr {shift_t.name} shift = {shift};
    static constexpr unsigned io_type = nnet::{iotype};
    static constexpr unsigned reuse_factor = {reuse};
}};\n"""

softmax_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static constexpr unsigned n_in = {n_in};
    static constexpr unsigned table_size = {table_size};
    static constexpr unsigned io_type = nnet::{iotype};
    static constexpr unsigned reuse_factor = {reuse};
    static constexpr nnet::softmax_implementation implementation = nnet::softmax_implementation::{implementation};
    typedef {exp_table_t.name} exp_table_t;
    typedef {inv_table_t.name} inv_table_t;
}};\n"""

activ_function_template = 'nnet::{activation}<{input_t}, {output_t}, {config}>({input}, {output});'
param_activ_function_template = 'nnet::{activation}<{input_t}, {output_t}, {config}>({input}, {param}, {output});'

activ_task_sequence_template = 'task_sequence<nnet::{activation}_stream<{input_pipe}, {output_pipe}, {config}>> {name};'
activ_stream_function_template = '{name}.async();'
param_activ_stream_function_template = '{name}.async({param});'

activ_include_list = ['nnet_utils/nnet_activation.h', 'nnet_utils/nnet_activation_stream.h']


class ActivationConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Activation)
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


class ActivationFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((Activation, HardActivation, Softmax), include_header=activ_include_list)
        self.template = activ_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['activation'] = node.get_attr('activation').lower()
        params['config'] = f"{node.get_attr('activation')}_config{node.index}"

        return self.template.format(**params)


class ParametrizedActivationFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(ParametrizedActivation, include_header=activ_include_list)
        self.template = param_activ_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['activation'] = node._get_act_function_name()
        params['param'] = node.get_attr('activ_param', 1.0)
        params['config'] = f"{node.get_attr('activation')}_config{node.index}"

        return self.template.format(**params)


class PReLUFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(PReLU, include_header=activ_include_list)
        self.template = param_activ_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['activation'] = node.get_attr('activation').lower()
        params['param'] = node.get_weights('param').name
        params['config'] = f"{node.get_attr('activation')}_config{node.index}"

        return self.template.format(**params)


class ActivationTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__((Activation, HardActivation, Softmax, PReLU))
        self.template = activ_task_sequence_template

    def format(self, node):
        params = self._default_function_params(node)
        params['activation'] = node.get_attr('activation').lower()
        params['config'] = f"{node.get_attr('activation')}_config{node.index}"
        return self.template.format(**params)


class ParametrizedActivationTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(ParametrizedActivation)
        self.template = activ_task_sequence_template

    def format(self, node):
        params = self._default_function_params(node)
        params['activation'] = node._get_act_function_name()
        params['config'] = f"{node.get_attr('activation')}_config{node.index}"
        return self.template.format(**params)


class ActivationStreamFunctionTemplate(StreamFunctionCallTemplate):
    def __init__(self):
        super().__init__((Activation, HardActivation, Softmax))
        self.template = activ_stream_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)


class ParametrizedActivationStreamFunctionTemplate(StreamFunctionCallTemplate):
    def __init__(self):
        super().__init__(ParametrizedActivation)
        self.template = param_activ_stream_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['param'] = node.get_attr('activ_param', 1.0)
        return self.template.format(**params)


class PReLUActivationStreamFunctionTemplate(StreamFunctionCallTemplate):
    def __init__(self):
        super().__init__(PReLU)
        self.template = param_activ_stream_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['param'] = node.get_weights('param').name
        return self.template.format(**params)
