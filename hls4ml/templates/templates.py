
def custom_initializer(*args):
    def decorator(function):
        function.handles = [arg for arg in args]
        return function
    return decorator

class Backend(object):
    def __init__(self, name):
        self.name = name
        self.config_templates = {}
        self.function_templates = {}
        self.include_lists = {}
        self.layer_initializers = {}
        init_func_list = [getattr(self, func) for func in dir(self) if callable(getattr(self, func)) and hasattr(getattr(self, func), 'handles')]
        for func in init_func_list:
            for layer_class in func.handles:
                self.layer_initializers[layer_class] = func

    def get_config_template(self, kind):
        return self.config_templates.get(kind)

    def get_function_template(self, kind):
        return self.function_templates.get(kind)

    def get_include_list(self, kind):
        return self.include_lists.get(kind, [])

    def register_templates(self, name, function_template, config_template, include_list=[]):
        self.function_templates[name] = function_template
        self.config_templates[name] = config_template
        self.include_lists[name] = include_list

    def register_source(self, file_name, source, destination_dir='nnet_utils'):
        raise NotImplementedError

    def initialize_layer(self, layer):
        init_func = self.layer_initializers.get(layer.__class__.__name__)
        if init_func is not None:
            init_func(layer)

backend_map = {}

def register_backend(name, backend_cls):
    if name in backend_map:
        raise Exception('Backend {} already registered'.format(name))
    
    backend_map[name] = backend_cls()

def get_backend(name):
    return backend_map[name]
