import numpy as np

from hls4ml.backends.oneapi.oneapi_template import StreamFunctionCallTemplate, TaskSequenceTemplate
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Reshape, Resize, Transpose, ZeroPadding1D, ZeroPadding2D

# ZeroPadding templates

zeropad1d_config_template = """struct config{index} : nnet::padding1d_config {{
    static const unsigned in_width = {in_width};
    static const unsigned out_width = {out_width};
    static const unsigned n_chan = {n_chan};

    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
}};\n"""

zeropad2d_config_template = """struct config{index} : nnet::padding2d_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
    static const unsigned n_chan = {n_chan};

    static const unsigned pad_top = {pad_top};
    static const unsigned pad_bottom = {pad_bottom};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
}};\n"""

zeropad1d_function_template = 'nnet::zeropad1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output});'
zeropad2d_function_template = 'nnet::zeropad2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output});'

zeropad1d_task_sequence_template = (
    'task_sequence<nnet::zeropad1d_{data_format}_stream<{input_pipe}, {output_pipe}, {config}>> {name};'
)
zeropad2d_task_sequence_template = (
    'task_sequence<nnet::zeropad2d_{data_format}_stream<{input_pipe}, {output_pipe}, {config}>> {name};'
)

reshaping_stream_function_template = '{name}.async();'

padding_include_list = ['nnet_utils/nnet_padding.h', 'nnet_utils/nnet_padding_stream.h']


class ZeroPaddingConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((ZeroPadding1D, ZeroPadding2D))
        self.templates = {
            'ZeroPadding1D': zeropad1d_config_template,
            'ZeroPadding2D': zeropad2d_config_template,
        }

    def format(self, node):
        params = self._default_config_params(node)
        return self.templates[node.class_name].format(**params)


class ZeroPaddingFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((ZeroPadding1D, ZeroPadding2D), include_header=padding_include_list)
        self.templates = {
            'ZeroPadding1D': zeropad1d_function_template,
            'ZeroPadding2D': zeropad2d_function_template,
        }

    def format(self, node):
        params = self._default_function_params(node)
        if node.get_attr('data_format') == 'channels_first':
            raise Exception('oneAPI only supports channels_last data format')
        params['data_format'] = 'cl'

        return self.templates[node.class_name].format(**params)


class ZeroPaddingTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__((ZeroPadding1D, ZeroPadding2D))
        self.templates = {
            'ZeroPadding1D': zeropad1d_task_sequence_template,
            'ZeroPadding2D': zeropad2d_task_sequence_template,
        }

    def format(self, node):
        params = self._default_function_params(node)
        if node.get_attr('data_format') == 'channels_first':
            raise RuntimeError('channels_first not supported on oneAPI')
        params['data_format'] = 'cl'

        return self.templates[node.class_name].format(**params)


class ReshapingStreamFunctionTemplate(StreamFunctionCallTemplate):
    def __init__(self):
        super().__init__((ZeroPadding1D, ZeroPadding2D, Resize, Reshape, Transpose))
        self.template = reshaping_stream_function_template

    def format(self, node):
        params = self._default_function_params(node)

        return self.template.format(**params)


# Resize templates

resize_config_template = """struct config{index} : nnet::resize_config {{
    static const unsigned height = {in_height};
    static const unsigned width = {in_width};

    static const unsigned new_height = {out_height};
    static const unsigned new_width = {out_width};

    static const unsigned n_chan = {n_chan};
}};\n"""

resize_function_template = 'nnet::resize_{algorithm}<{input_t}, {output_t}, {config}>({input}, {output});'
resize_task_sequence_template = (
    'task_sequence<nnet::resize_{algorithm}_stream<{input_pipe}, {output_pipe}, {config}>> {name};'
)
resize_include_list = ['nnet_utils/nnet_resize.h', 'nnet_utils/nnet_resize_stream.h']


class ResizeConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Resize)
        self.template = resize_config_template

    def format(self, node):
        params = self._default_config_params(node)

        return self.template.format(**params)


class ResizeFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Resize, include_header=resize_include_list)
        self.template = resize_function_template

    def format(self, node):
        params = self._default_function_params(node)
        if node.get_attr('algorithm') != 'nearest':
            raise Exception('Currently only supporting resize_nearest')
        params['algorithm'] = node.get_attr('algorithm')

        return self.template.format(**params)


class ResizeTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(Resize)
        self.template = resize_task_sequence_template

    def format(self, node):
        params = self._default_function_params(node)
        if node.get_attr('algorithm') != 'nearest':
            raise Exception('Currently only supporting resize_nearest')
        params['algorithm'] = node.get_attr('algorithm')

        return self.template.format(**params)


# Transpose templates

transpose_config_template = """struct config{index} : nnet::transpose_config {{
    static const unsigned depth = {depth};
    static const unsigned height = {height};
    static const unsigned width = {width};
    static constexpr unsigned perm[3] = {{{perm_str}}};
}};\n"""

transpose_function_template = 'nnet::transpose_{dim}<{input_t}, {output_t}, {config}>({input}, {output});'
transpose_task_sequence_template = (
    'task_sequence<nnet::transpose_{dim}_stream<{input_pipe}, {output_pipe}, {config}>> {name};'
)
transpose_include_list = ['nnet_utils/nnet_transpose.h', 'nnet_utils/nnet_transpose_stream.h']


class TransposeConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Transpose)
        self.template = transpose_config_template

    def format(self, node):
        params = self._default_config_params(node)

        return self.template.format(**params)


class TransposeFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Transpose, include_header=transpose_include_list)
        self.template = transpose_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['dim'] = node.get_attr('dim')

        return self.template.format(**params)


class TransposeTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(Transpose)
        self.template = transpose_task_sequence_template

    def format(self, node):
        params = self._default_function_params(node)
        params['dim'] = node.get_attr('dim')

        return self.template.format(**params)


# Reshape template (only used in streaming)
reshape_task_sequence_template = 'task_sequence<nnet::repack_stream<{input_pipe}, {output_pipe}, {size}>> {name};'
reshape_include_list = ['nnet_utils/nnet_stream.h']


class ReshapeConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Reshape)

    def format(self, node):
        return ''


class ReshapeFunctionTemplate(FunctionCallTemplate):
    """Only used to add the include list"""

    def __init__(self):
        super().__init__(Reshape, include_header=reshape_include_list)

    def format(self, node):
        return ''


class ReshapeTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(Reshape)
        self.template = reshape_task_sequence_template

    def format(self, node):
        params = self._default_function_params(node)
        params['size'] = np.prod(node.get_output_variable().shape)
        return self.template.format(**params)
