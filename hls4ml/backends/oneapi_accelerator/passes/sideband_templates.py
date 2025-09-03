"""The sideband handling templates are needed for oneAPI accelerator when using io_stream.
They are not used in io_paralle.
"""

from hls4ml.backends.oneapi.oneapi_template import StreamFunctionCallTemplate, TaskSequenceTemplate
from hls4ml.backends.oneapi_accelerator.oneapi_accelerator_layers import SidebandExtraction, SidebandMerging
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate

sideband_config_template = """struct config{index} : nnet::sideband_config {{
    static constexpr unsigned n_in = {n_in};
}};\n"""
sideband_stream_function_template = '{name}.async();'
sideband_extract_task_sequence_template = (
    'task_sequence<nnet::extract_sideband_stream<{input_pipe}, {output_pipe}, {skip_pipe}, {config}>> {name};'
)
sideband_merge_task_sequence_template = (
    'task_sequence<nnet::merge_sideband_stream<{input_pipe}, {output_pipe}, {skip_pipe}, {config}>> {name};'
)
sideband_include_list = ['nnet_utils/nnet_stream_beat.h']


class SidebandConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((SidebandExtraction, SidebandMerging))
        self.template = sideband_config_template

    def format(self, node):
        params = self._default_config_params(node)
        return self.template.format(**params)


class SidebandFunctionTemplate(FunctionCallTemplate):
    """Only used to add the include list"""

    def __init__(self):
        super().__init__((SidebandExtraction, SidebandMerging), include_header=sideband_include_list)

    def format(self, node):
        return ''


class SidebandStreamFunctionTemplate(StreamFunctionCallTemplate):
    def __init__(self):
        super().__init__((SidebandExtraction, SidebandMerging))
        self.template = sideband_stream_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)


class SidebandExtractionTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(SidebandExtraction)
        self.template = sideband_extract_task_sequence_template

    def format(self, node):
        params = self._default_function_params(node)
        params['skip_pipe'] = node.get_output_variable('sideband').pipe_name
        return self.template.format(**params)


class SidebandMergeTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(SidebandMerging)
        self.template = sideband_merge_task_sequence_template

    def format(self, node):
        params = self._default_function_params(node)
        params['skip_pipe'] = node.get_input_variable('sideband').pipe_name
        return self.template.format(**params)
