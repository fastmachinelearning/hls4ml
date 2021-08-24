import inspect
import os

from collections.abc import MutableMapping

from hls4ml.backends.template import Template
from hls4ml.model.hls_layers import Layer
from hls4ml.model.flow import get_backend_flows
from hls4ml.model.optimizer import LayerOptimizerPass, optimizer_pass, register_pass, extract_optimizers_from_path, extract_optimizers_from_object, get_backend_passes

class LayerDict(MutableMapping):
    def __init__(self):
        self.layer_dict = {}

    def __getitem__(self, key):
        if key in self.layer_dict:
            return self.layer_dict[key]
        else:
            return self.layer_dict[key.__bases__[0]]

    def __len__(self):
        return len(self.layer_dict)

    def __iter__(self):
        for key in self.layer_dict.keys():
            yield key

    def __setitem__(self, key, value):
        if not issubclass(key, Layer):
            raise KeyError('Keys must be instances of Layer class')
        self.layer_dict[key] = value

    def __delitem__(self, key):
        self.layer_dict.remove(key)

def custom_initializer(*args):
    def decorator(function):
        function.handles = [arg for arg in args]
        return function
    return decorator

def layer_optimizer(layer):
    def decorator(function):
        return optimizer_pass(layer)(function)
    return decorator

class Backend(object):
    def __init__(self, name):
        self.name = name
        # Templates
        self.config_templates = LayerDict()
        self.function_templates = LayerDict()
        self.include_lists = LayerDict()
        self.init_templates()
        # Optimizers
        self.layer_initializers = LayerDict()
        init_func_list = [getattr(self, func) for func in dir(self) if callable(getattr(self, func)) and hasattr(getattr(self, func), 'handles')]
        for func in init_func_list:
            for layer_class in func.handles:
                self.layer_initializers[layer_class] = func
        self._init_optimizers()

    def init_templates(self):
        raise NotImplementedError

    def _init_optimizers(self):
        optimizers = {}
        optimizers.update(self._init_class_optimizers())
        optimizers.update(self._init_file_optimizers())
        for opt_name, opt in optimizers.items():
            self.register_pass(opt_name, opt)

    def _init_class_optimizers(self):
        class_optimizers = extract_optimizers_from_object(self)
        return class_optimizers

    def _init_file_optimizers(self):
        opt_path = os.path.dirname(inspect.getfile(self.__class__)) + '/passes'
        module_path = self.__module__[:self.__module__.rfind('.')] + '.passes'
        file_optimizers = extract_optimizers_from_path(opt_path, module_path, self)
        return file_optimizers

    def _get_layer_initializers(self):
        all_initializers = { name:opt for name, opt in get_backend_passes(self.name) if isinstance(opt, LayerOptimizerPass) }

        # Sort through the initializers based on the base class (e.g., to apply 'Layer' optimizers before 'Dense')
        sorted_initializers = sorted(all_initializers.items(), key=lambda x: len(x[1].__class__.mro()))

        return sorted_initializers

    def _get_layer_templates(self):
        return { name:opt for name, opt in get_backend_passes(self.name) if isinstance(opt, Template) }

    def create_initial_config(self, **kwargs):
        raise NotImplementedError

    def create_layer_class(self, layer_class):
        raise NotImplementedError

    def get_config_template(self, kind):
        return self.config_templates.get(kind)

    def get_function_template(self, kind):
        return self.function_templates.get(kind)

    def get_include_list(self, kind):
        return self.include_lists.get(kind, [])

    def get_available_flows(self):
        return get_backend_flows(self.name)

    def get_available_optimizers(self):
        return get_backend_passes(self.name)
    
    def get_default_flow(self):
        raise NotImplementedError
    
    def get_available_flows(self):
        raise NotImplementedError

    def register_templates(self, cls, function_template, config_template, include_list=[]):
        self.function_templates[cls] = function_template
        self.config_templates[cls] = config_template
        self.include_lists[cls] = include_list

    def register_source(self, file_name, source, destination_dir='nnet_utils'):
        raise NotImplementedError
    
    def register_pass(self, name, opt_cls):
        register_pass(name, opt_cls, backend=self.name)

    def initialize_layer(self, layer):
        for cls, init_func in self.layer_initializers.items():
            if isinstance(layer, cls):
                init_func(layer)

backend_map = {}

def register_backend(name, backend_cls):
    if name.lower() in backend_map:
        raise Exception('Backend {} already registered'.format(name))
    
    backend_map[name.lower()] = backend_cls()

def get_backend(name):
    return backend_map[name.lower()]

def get_available_backends():
    return list(backend_map.keys())
