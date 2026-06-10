from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import IFNeuron, LIFNeuron, SNNReadout

if_config_template = """struct config{index} : nnet::if_neuron_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned window_size = {window_size};
    static const bool threshold_is_vector = {threshold_is_vector};
    static constexpr float threshold = {threshold};
    static const nnet::snn_reset_mode reset_mode = nnet::snn_reset_mode::{reset_mechanism};
    typedef {threshold_t.name} threshold_t;
    typedef {membrane_t.name} membrane_t;
}};\n"""

if_function_template = 'nnet::if_neuron<{input_t}, {output_t}, {config}>({input}, {output}, {threshold});'
snn_include_list = ['nnet_utils/nnet_snn.h', 'nnet_utils/nnet_snn_stream.h']


class IFNeuronConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(IFNeuron)
        self.template = if_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['threshold_is_vector'] = 'true' if node.get_attr('threshold_mode', 'scalar') == 'vector' else 'false'
        return self.template.format(**params)


class IFNeuronFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(IFNeuron, include_header=snn_include_list)
        self.template = if_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['threshold'] = (
            node.get_weights('threshold_vec').name if node.get_attr('threshold_mode', 'scalar') == 'vector' else 'nullptr'
        )
        return self.template.format(**params)


lif_config_template = """struct config{index} : nnet::lif_neuron_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned window_size = {window_size};
    static const bool beta_is_vector = {beta_is_vector};
    static const bool threshold_is_vector = {threshold_is_vector};
    static constexpr float threshold = {threshold};
    static constexpr float beta = {beta};
    static const nnet::snn_reset_mode reset_mode = nnet::snn_reset_mode::{reset_mechanism};
    typedef {beta_t.name} beta_t;
    typedef {threshold_t.name} threshold_t;
    typedef {membrane_t.name} membrane_t;
}};\n"""

lif_function_template = 'nnet::lif_neuron<{input_t}, {output_t}, {config}>({input}, {output}, {beta}, {threshold});'


class LIFNeuronConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(LIFNeuron)
        self.template = lif_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['beta_is_vector'] = 'true' if node.get_attr('beta_mode', 'scalar') == 'vector' else 'false'
        params['threshold_is_vector'] = 'true' if node.get_attr('threshold_mode', 'scalar') == 'vector' else 'false'
        return self.template.format(**params)


class LIFNeuronFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(LIFNeuron, include_header=snn_include_list)
        self.template = lif_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['beta'] = node.get_weights('beta_vec').name if node.get_attr('beta_mode', 'scalar') == 'vector' else 'nullptr'
        params['threshold'] = (
            node.get_weights('threshold_vec').name if node.get_attr('threshold_mode', 'scalar') == 'vector' else 'nullptr'
        )
        return self.template.format(**params)


readout_config_template = """struct config{index} : nnet::snn_readout_config {{
    static const unsigned n_classes = {n_classes};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned window_size = {window_size};
    static const unsigned class_threshold = {class_threshold};
    static constexpr float beta = {beta};
    static const nnet::snn_readout_mode output_mode = nnet::snn_readout_mode::{output_mode};
    static const nnet::snn_decision_rule decision_rule = nnet::snn_decision_rule::{decision_rule};
    typedef {membrane_t.name} membrane_t;
}};\n"""

readout_function_template = 'nnet::snn_readout<{input_t}, {output_t}, {config}>({input}, {output});'


class SNNReadoutConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(SNNReadout)
        self.template = readout_config_template

    def format(self, node):
        params = self._default_config_params(node)
        return self.template.format(**params)


class SNNReadoutFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(SNNReadout, include_header=snn_include_list)
        self.template = readout_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)


def register_snn_templates(backend):
    backend.register_template(IFNeuronConfigTemplate)
    backend.register_template(IFNeuronFunctionTemplate)
    backend.register_template(LIFNeuronConfigTemplate)
    backend.register_template(LIFNeuronFunctionTemplate)
    backend.register_template(SNNReadoutConfigTemplate)
    backend.register_template(SNNReadoutFunctionTemplate)
