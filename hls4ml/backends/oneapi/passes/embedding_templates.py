"""
These are the stream oneAPI templates for embedding layers. The io_parallel ones are in backends/fpga/passes/embedding.py.
"""

from hls4ml.backends.oneapi.oneapi_template import StreamFunctionCallTemplate, TaskSequenceTemplate
from hls4ml.model.layers import Embedding

embed_task_sequence_template = 'task_sequence<nnet::embedding_stream<{input_pipe}, {output_pipe}, {config}>> {name};'
embed_stream_function_template = '{name}.async({e});'


class EmbeddingTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(Embedding)
        self.template = embed_task_sequence_template

    def format(self, node):
        params = self._default_function_params(node)

        return self.template.format(**params)


class EmbeddingStreamFunctionTemplate(StreamFunctionCallTemplate):
    def __init__(self):
        super().__init__(Embedding)
        self.template = embed_stream_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['e'] = node.get_weights('embeddings').name

        return self.template.format(**params)
