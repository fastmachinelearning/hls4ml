from hls4ml.backends.backend import get_backend
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Dense

# The kernel's name for each activation that FoldActivationIntoFused can move into a Dense layer
FUSED_ACTIVATIONS = {
    None: 'FUSED_LINEAR',
    'linear': 'FUSED_LINEAR',
    'relu': 'FUSED_RELU',
    'sigmoid': 'FUSED_SIGMOID',
    'tanh': 'FUSED_TANH',
    'softplus': 'FUSED_SOFTPLUS',
    'softsign': 'FUSED_SOFTSIGN',
    'selu': 'FUSED_SELU',
    'elu': 'FUSED_ELU',
    'leaky_relu': 'FUSED_LEAKY_RELU',
    'thresholded_relu': 'FUSED_THRESHOLDED_RELU',
    'hard_sigmoid': 'FUSED_HARD_SIGMOID',
    'hard_tanh': 'FUSED_HARD_TANH',
    'binary_tanh': 'FUSED_BINARY_TANH',
}

dense_fused_config_template = """struct config{index} : nnet::dense_fused_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned reuse_factor = {reuse};
    static const unsigned multiplier_limit = {multiplier_limit};
    static const unsigned pad_cycles = {pad_cycles};
    static const unsigned activation = nnet::{activation};
    static const unsigned table_size = {table_size};
    typedef {table_t} table_t;
    typedef {param_t} param_t;
    typedef {slope_t} slope_t;
    typedef {shift_t} shift_t;
    typedef {preact_t} preact_t;
    static const param_t activation_param;
    static const slope_t slope;
    static const shift_t shift;
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};
const config{index}::param_t config{index}::activation_param = {activation_param};
const config{index}::slope_t config{index}::slope = {slope};
const config{index}::shift_t config{index}::shift = {shift};\n"""

plain_function_template = 'nnet::dense_fused<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
dot_function_template = 'nnet::dense_fused_dot<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
axpy_function_template = 'nnet::dense_fused_axpy<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'

fused_include_list = ['nnet_utils/nnet_dense_fused.h']


def _is_fused(node):
    return str(node.get_attr('strategy', '')).lower() == 'fused'


class DenseFusedConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Dense)
        self.name = 'dense_fused_config_template'
        self.template = dense_fused_config_template

    def _type_name(self, node, attribute, default):
        named_type = node.get_attr(attribute)
        return named_type.name if named_type is not None else default

    def match(self, node):
        return _is_fused(node) and super().match(node)

    def format(self, node):
        params = self._default_config_params(node)
        params['multiplier_limit'] = node.get_attr('fused_multipliers', 1)
        params['pad_cycles'] = node.get_attr('fused_pad_cycles', 0) or 0
        params['activation'] = FUSED_ACTIVATIONS.get(node.get_attr('fused_activation'), 'FUSED_LINEAR')
        params['table_size'] = node.get_attr('fused_table_size') or 1024
        table_t = node.get_attr('fused_table_t')
        params['table_t'] = table_t.name if table_t is not None else 'ap_fixed<18,8>'
        # Each parameter keeps the type it had in the activation layer
        params['param_t'] = self._type_name(node, 'fused_param_t', 'ap_fixed<16,6>')
        params['slope_t'] = self._type_name(node, 'fused_slope_t', 'ap_ufixed<16,0>')
        params['shift_t'] = self._type_name(node, 'fused_shift_t', 'ap_ufixed<2,0>')
        # The type the activation is applied to: the output type before the fold, or the current output
        # type when nothing was folded
        preact_t = node.get_attr('fused_preact_t')
        params['preact_t'] = preact_t.name if preact_t is not None else node.get_output_variable().type.name
        params['activation_param'] = node.get_attr('fused_activation_param', 0.0)
        params['slope'] = node.get_attr('fused_activation_slope', 0.0)
        params['shift'] = node.get_attr('fused_activation_shift', 0.0)
        params['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_weights('weight').type.precision
        )

        return self.template.format(**params)


class DenseFusedFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Dense, include_header=fused_include_list)
        self.name = 'dense_fused_function_template'
        self.template = plain_function_template

    def match(self, node):
        return _is_fused(node) and super().match(node)

    def format(self, node):
        params = self._default_function_params(node)
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name

        template = {
            'dot': dot_function_template,
            'axpy': axpy_function_template,
        }.get(node.get_attr('fused_form'), self.template)

        return template.format(**params)
