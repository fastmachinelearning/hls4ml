from hls4ml.model.layers import LSTM, SimpleRNN
from hls4ml.backends.template import LayerConfigTemplate, FunctionCallTemplate

#####################
# activation templates
#####################

rnn_activ_config_template = """struct lstm_activ_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
}};\n"""

#####################
# lstm templates
#####################

lstm_config_template = """struct config{index} : nnet::lstm_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned n_timestamp = {n_timestamp};
    static const unsigned sliding_window = {sliding_window};
    static const unsigned return_sequences = {return_sequences};
    typedef {config_t} activ_config;
}};\n"""
lstm_function_template = 'nnet::lstm_network<{input_t}, {output_t}, {config} ,{input_t}>({input}, {output}, {weights});'
lstm_include_list = ['nnet_utils/nnet_lstm_cell.h']


class LstmConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(LSTM)
        self.template = lstm_config_template
        self.activ_template = rnn_activ_config_template

    def format(self, node):
        lstm_params = self._default_config_params(node)
        lstm_params ['n_in'] = node.get_attr('n_in')
        lstm_params ['n_out'] = node.get_attr('n_out')
        lstm_params ['n_timestamp'] = node.get_attr('n_timestamp', 5)
        lstm_params ['sliding_window'] = str(node.get_attr('Sliding_window')).lower()
        lstm_params ['return_sequences'] = str(node.get_attr('return_sequences')).lower()
        lstm_params ['config_t'] = 'lstm_activ_config{}'.format(node.index)
        lstm_params ['table_size'] = node.get_attr('table_size', 1024)


        activ_params = self._default_config_params(node)
        activ_params ['n_in'] = node.get_attr('n_out')
        activ_params ['table_size'] = node.get_attr('table_size', 1024)

        return  self.activ_template.format(**activ_params) + "\n" + self.template.format(**lstm_params) 


class LstmFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(LSTM, include_header=lstm_include_list)
        self.template = lstm_function_template

    def format(self, node):
        params = self._default_function_params(node)
        _sliding_window_bool = str(node.get_attr('Sliding_window'))

        if _sliding_window_bool == 'True':
            params['input'] = params['input'] + '[0]'
        params['weights'] = ""
        for i in ["kernel", "recurrent_kernel", "bias"]:
            for j in ["i", "f", "c", "o"]:
                params['weights'] += "" + i + "_" + j + "_" + str(node.index)
                if not(i == "bias" and j == "o"):
                    params['weights'] += ","
        return self.template.format(**params)

#####################
# activation templates
#####################

simple_rnn_activ_config_template = """struct simple_rnn_activ_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
}};\n"""


#####################
# SimpleRNN templates
#####################

simple_rnn_config_template = """struct config{index} : nnet::simpleRNN_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned n_timestamp = {n_timestamp};
    static const unsigned sliding_window = {sliding_window};
    static const unsigned return_sequences = {return_sequences};
    typedef {config_t} activ_config;
}};\n"""
simple_rnn_function_template = 'nnet::simple_rnn_network<{input_t}, {output_t}, {config} ,{input_t}>({input}, {output}, {weights});'
simple_rnn_include_list = ['nnet_utils/nnet_simple_rnn_cell.h']


class SimpleRNNConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(SimpleRNN)
        self.template = simple_rnn_config_template
        self.activ_template = simple_rnn_activ_config_template

    def format(self, node):
        simple_rrn_params = self._default_config_params(node)
        simple_rrn_params['n_in'] = node.get_attr('n_in')
        simple_rrn_params ['n_out'] = node.get_attr('n_out')
        simple_rrn_params['n_timestamp'] = node.get_attr('n_timestamp', 5)
        simple_rrn_params['table_size'] = node.get_attr('table_size', 1024)
        simple_rrn_params['sliding_window'] = str(node.get_attr('Sliding_window')).lower()
        simple_rrn_params['return_sequences'] = str(node.get_attr('return_sequences')).lower()
        simple_rrn_params['config_t'] = 'simple_rnn_activ_config{}'.format(node.index)

        activ_params = self._default_config_params(node)
        activ_params ['n_in'] = node.get_attr('n_out')
        activ_params ['table_size'] = node.get_attr('table_size', 1024)

        return self.activ_template.format(**activ_params) + "\n" + self.template.format(**simple_rrn_params)        


class SimpleRNNFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(SimpleRNN, include_header=simple_rnn_include_list)
        self.template = simple_rnn_function_template

    def format(self, node):
        params = self._default_function_params(node)
        _sliding_window_bool = str(node.get_attr('Sliding_window'))

        if _sliding_window_bool == 'True':
            params['input'] = params['input'] + '[0]'
        params['weights'] = ""
        for i in ["kernel", "recurrent_kernel", "bias"]:
            params['weights'] += "" + i + "_" + str(node.index)
            if not(i == "bias"):
                params['weights'] += ","
        return self.template.format(**params)