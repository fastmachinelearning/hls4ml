
from hls4ml.backends.backend import get_backend
from hls4ml.model.layers import MultiHeadAttention
from hls4ml.backends.template import LayerConfigTemplate, FunctionCallTemplate
                                                                  
mha_config_template = """struct config{index} : nnet::multiheadattention_config {{ 
    typedef {accum_t.name} accum_t;
    typedef {attention_output_bias_t.name} bias_t;
    typedef {attention_output_weight_t.name} weight_t;
    
    static const unsigned num_heads = {num_heads};
    static const unsigned head_dim_key = {head_dim_key};
    static const unsigned head_dim_value = {head_dim_value};
    static const unsigned feature_dim = {feature_dim};
    static const unsigned seq_len = {seq_len};

    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    static const bool store_weights_in_bram = false;
}};\n"""

mha_function_template = 'nnet::multiheadattention<{input_t}, {output_t}, {config}>({input_q}, {input_kv}, {output}, {w_o}, {b_o}, {w_k}, {b_k}, {w_q}, {b_q}, {w_v}, {b_v});'

mha_include_list = ['nnet_utils/nnet_multiheadattention.h']

class MhaConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(MultiHeadAttention)
        self.template = mha_config_template
        # self.mult1_template = recr_mult_config_template
    
    def format(self, node):

        params = self._default_config_params(node)

        params['num_heads'] = node.get_attr('num_heads')
        params['head_dim_key'] = node.get_attr('head_dim_key')
        params['head_dim_value'] = node.get_attr('head_dim_value')
        params['feature_dim'] = node.get_attr('feature_dim')
        params['seq_len'] = node.get_attr('seq_len')

        mht_config = self.template.format(**params)

        return mht_config

class RecurrentFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(MultiHeadAttention, include_header=mha_include_list)
        self.template = mha_function_template

    def format(self, node):
        params = {}
        params.update(node.attributes)
        params['config'] = 'config{}'.format(node.index)
        params['input_t'] = node.get_input_variable().type.name
        params['output_t'] = node.get_output_variable().type.name
        
        params['input_q'] = node.model.get_layer_output_variable(node.inputs[0]).name
        params['input_kv'] = node.model.get_layer_output_variable(node.inputs[1]).name
        params['output'] = node.get_output_variable().name
        params['w_o'] = node.get_weights('attention_output_weight').name
        params['b_o'] = node.get_weights('attention_output_bias').name
        params['w_k'] = node.get_weights('key_weight').name
        params['b_k'] = node.get_weights('key_bias').name
        params['w_q'] = node.get_weights('query_weight').name
        params['b_q'] = node.get_weights('query_bias').name
        params['w_v'] = node.get_weights('value_weight').name
        params['b_v'] = node.get_weights('value_bias').name

        return self.template.format(**params)

