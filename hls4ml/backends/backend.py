import importlib
import inspect
import os

from collections.abc import MutableMapping

from hls4ml.model.hls_layers import Layer
from hls4ml.model.optimizer import OptimizerPass, optimizer_pass, extract_optimizers_from_object

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
        return optimizer_pass(lambda node: isinstance(node, layer))(function)
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
        self.optimizers = {}
        self._init_optimizers()

    def init_templates(self):
        raise NotImplementedError

    def _init_optimizers(self):
        self._init_class_optimizers()
        self._init_file_optimizers()

    def _init_class_optimizers(self):
        self.optimizers.update(extract_optimizers_from_object(self))

    def _init_file_optimizers(self):
        opt_path = os.path.dirname(inspect.getfile(self.__class__)) + '/passes'
        if not os.path.exists(opt_path):
            return

        for module in os.listdir(opt_path):
            if module == '__init__.py' or module[-3:] != '.py':
                continue
            try:
                lib = importlib.import_module(self.__module__[:self.__module__.rfind('.')] + '.passes.' + module[:-3])
                if 'register_' + module[:-3] in lib.__dict__:
                    opt_init_func = lib.__dict__['register_' + module[:-3]]
                    opt_init_func(self)
                else:
                    for func in list(lib.__dict__.values()):
                        # if 'func' is a class
                        # and it inherits from OptimizerPass
                        # and is defined in this module (i.e., not imported)
                        if inspect.isclass(func) and issubclass(func, OptimizerPass) and func.__module__ == lib.__name__:
                            self.register_pass(func.get_name(), func)

            except ImportError:
                print('WARN: Unable to import optiizer(s) from {}'.format(module))
                continue

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

    def get_available_optimizers(self):
        return list(self.optimizers.keys())

    def get_optimizers(self):
        return self.optimizers.values()

    def register_templates(self, cls, function_template, config_template, include_list=[]):
        self.function_templates[cls] = function_template
        self.config_templates[cls] = config_template
        self.include_lists[cls] = include_list

    def register_source(self, file_name, source, destination_dir='nnet_utils'):
        raise NotImplementedError
    
    def register_pass(self, name, opt_cls):
        if name in self.optimizers:
            raise Exception('Optimization pass {} already registered'.format(name))
    
        if inspect.isclass(opt_cls):
            opt = opt_cls()
        else:
            opt = opt_cls

        if type(name) in [list, tuple]:
            for n in name:
                self.optimizers[n] = opt
        else:    
            self.optimizers[name] = opt

    def initialize_layer(self, layer):
        init_func = self.layer_initializers.get(layer.__class__)
        if init_func is not None:
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
