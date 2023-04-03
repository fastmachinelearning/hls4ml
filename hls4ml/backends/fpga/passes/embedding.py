from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Embedding

embed_config_template = """struct config{index} : nnet::embed_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned vocab_size = {vocab_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    typedef {embeddings_t.name} embeddings_t;
}};\n"""

embed_function_template = 'nnet::embedding<{input_t}, {output_t}, {config}>({input}, {output}, {e});'

embed_include_list = ['nnet_utils/nnet_embed.h', 'nnet_utils/nnet_embed_stream.h']


class EmbeddingConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Embedding)
        self.template = embed_config_template

    def format(self, node):
        params = self._default_config_params(node)
        return self.template.format(**params)


class EmbeddingFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Embedding, include_header=embed_include_list)
        self.template = embed_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['e'] = node.get_weights('embeddings').name

        return self.template.format(**params)
