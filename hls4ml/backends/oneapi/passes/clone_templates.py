""" The clone templates in the fpga backend are not enough for oneAPI, so this adds the missing parts
"""

from hls4ml.backends.fpga.passes.clone import Clone
from hls4ml.backends.oneapi.oneapi_template import StreamFunctionCallTemplate, TaskSequenceTemplate

clone_stream_function_template = '{name}.async();'


class CloneTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(Clone)

    def format(self, node):
        params = self._default_function_params(node)
        for i in range(len(node.outputs)):
            params[f'output{i + 1}_pipe'] = node.variables[node.outputs[i]].pipe_name

        output_pipes = ', '.join([f'{{output{i + 1}_pipe}}' for i in range(len(node.outputs))])

        template = f'task_sequence<nnet::clone_stream<{{input_pipe}}, {output_pipes}, {{size}}>> {{name}};'
        return template.format(**params)


class CloneStreamFunctionTemplate(StreamFunctionCallTemplate):
    def __init__(self):
        super().__init__(Clone)
        self.template = clone_stream_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)
