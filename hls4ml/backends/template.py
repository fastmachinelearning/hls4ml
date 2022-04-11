
from hls4ml.model.optimizer.optimizer import OptimizerPass


class Template(OptimizerPass):
    def __init__(self, name, layer_class, attribute_name):
        self.name = name
        self.layer_class = layer_class
        if not isinstance(self.layer_class, (list, tuple, set)):
            self.layer_class = [self.layer_class]
        self.attribute_name = attribute_name
    
    def match(self, node):
        for layer_cls in self.layer_class:
            if node.class_name == layer_cls.__name__:
                return True
        return False

    def transform(self, model, node):
        formatted_template = self.format(node)
        node.set_attr(self.attribute_name, formatted_template)
        return False
    
    def format(self, node):
        raise NotImplementedError

    def get_name(self):
        return self.name
    
class LayerConfigTemplate(Template):
    def __init__(self, layer_class):
        if isinstance(layer_class, (list, tuple, set)):
            name = '_'.join([cls.__name__.lower() for cls in layer_class])
        else:
            name = layer_class.__name__.lower()
        name += '_config_template'
        super().__init__(name, layer_class, 'config_cpp')
    
    def _default_config_params(self, layer):
        params = {}
        params.update(layer.attributes)
        params['iotype'] = layer.model.config.get_config_value('IOType')
        params['reuse'] = layer.get_attr('reuse_factor')

        return params

class FunctionCallTemplate(Template):
    def __init__(self, layer_class, include_header=[]):
        if isinstance(layer_class, (list, tuple, set)):
            name = '_'.join([cls.__name__.lower() for cls in layer_class])
        else:
            name = layer_class.__name__.lower()
        name += '_function_template'
        super().__init__(name, layer_class, 'function_cpp')
        self.include_header = include_header
    
    def _default_function_params(self, layer):
        params = {}
        params.update(layer.attributes)
        params['config'] = 'config{}'.format(layer.index)
        params['input_t'] = layer.get_input_variable().type.name
        params['output_t'] = layer.get_output_variable().type.name
        params['input'] = layer.get_input_variable().name
        params['output'] = layer.get_output_variable().name

        return params

    def transform(self, model, node):
        node.set_attr('include_header', self.include_header)
        return super().transform(model, node)
