'''
This package includes oneAPI-specific templates
'''

from hls4ml.backends.template import Template


class StreamFunctionCallTemplate(Template):
    def __init__(self, layer_class):
        if isinstance(layer_class, (list, tuple, set)):
            name = '_'.join([cls.__name__.lower() for cls in layer_class])
        else:
            name = layer_class.__name__.lower()
        name += '_stream_function_template'
        super().__init__(name, layer_class, 'stream_function_cpp')

    def _default_function_params(self, layer):
        params = self._default_params(layer)
        return params

    def transform(self, model, node):
        return super().transform(model, node)


class TaskSequenceTemplate(Template):
    def __init__(self, layer_class):
        if isinstance(layer_class, (list, tuple, set)):
            name = '_'.join([cls.__name__.lower() for cls in layer_class])
        else:
            name = layer_class.__name__.lower()
        name += '_task_sequence_template'
        super().__init__(name, layer_class, 'tast_sequence_cpp')

    def _default_function_params(self, layer):
        params = self._default_params(layer)
        params['config'] = f'config{layer.index}'
        params['input_pipe'] = layer.get_input_variable().pipe_name
        params['output_pipe'] = layer.get_output_variable().pipe_name

        return params

    def transform(self, model, node):
        return super().transform(model, node)
