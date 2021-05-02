import importlib
import inspect
import os

from hls4ml.model.optimizer import OptimizerPass, optimizer_pass, extract_optimizers_from_object

def custom_initializer(*args):
    def decorator(function):
        function.handles = [arg for arg in args]
        return function
    return decorator

def layer_optimizer(layer):
    def decorator(function):
        return optimizer_pass(lambda node: node.__class__.__name__ == layer)(function)
    return decorator

class Backend(object):
    def __init__(self, name):
        self.name = name
        # Templates
        self.config_templates = {}
        self.function_templates = {}
        self.include_lists = {}
        # Optimizers
        self.layer_initializers = {}
        init_func_list = [getattr(self, func) for func in dir(self) if callable(getattr(self, func)) and hasattr(getattr(self, func), 'handles')]
        for func in init_func_list:
            for layer_class in func.handles:
                self.layer_initializers[layer_class] = func
        self.optimizers = {}
        self._init_optimizers()

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

    def register_templates(self, name, function_template, config_template, include_list=[]):
        self.function_templates[name] = function_template
        self.config_templates[name] = config_template
        self.include_lists[name] = include_list

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
        init_func = self.layer_initializers.get(layer.__class__.__name__)
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
