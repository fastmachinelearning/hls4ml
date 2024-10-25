'''
This package includes oneAPI-specific templates
'''

from hls4ml.backends.template import Template


class StreamFunctionCallTemplate(Template):
    """Base class for the streaming function call templates in oneAPI:  provides the 'stream_function_cpp' attribute.
    This generally provides the async call to the task sequence that executes the streaming function.

    Note:  the include header files are specified in the regular FunctionCallTemplate, not here.

    Args:
        layer_class (Layer or list, tuple, or set of Layers): The Layers that this template handles.
    """

    def __init__(self, layer_class):
        if isinstance(layer_class, (list, tuple, set)):
            name = '_'.join([cls.__name__.lower() for cls in layer_class])
        else:
            name = layer_class.__name__.lower()
        name += '_stream_function_template'
        super().__init__(name, layer_class, 'stream_function_cpp')

    def _default_function_params(self, layer):
        params = self._default_params(layer)
        params['name'] = layer.name
        return params

    def transform(self, model, node):
        return super().transform(model, node)


class TaskSequenceTemplate(Template):
    """Base class for the task sequence definition in oneAPI:  provides the 'task_sequence_cpp' attribute.
    This defines the task sequence that is then called by the StreamFunctionCallTemplate.

    Args:
        layer_class (Layer or list, tuple, or set of Layers): The Layers that this template handles.
    """

    def __init__(self, layer_class):
        if isinstance(layer_class, (list, tuple, set)):
            name = '_'.join([cls.__name__.lower() for cls in layer_class])
        else:
            name = layer_class.__name__.lower()
        name += '_task_sequence_template'
        super().__init__(name, layer_class, 'tast_sequence_cpp')

    def _default_function_params(self, layer):
        params = self._default_params(layer)
        params['name'] = layer.name
        params['config'] = f'config{layer.index}'
        params['input_pipe'] = layer.get_input_variable().pipe_name
        params['output_pipe'] = layer.get_output_variable().pipe_name

        return params

    def transform(self, model, node):
        return super().transform(model, node)
