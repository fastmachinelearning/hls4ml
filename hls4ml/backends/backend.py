import inspect
import os

from hls4ml.backends.template import Template
from hls4ml.model.flow import get_backend_flows
from hls4ml.model.optimizer import LayerOptimizerPass, register_pass, extract_optimizers_from_path, extract_optimizers_from_object, get_backend_passes, get_optimizer


class Backend(object):
    def __init__(self, name):
        self.name = name
        self._init_optimizers()

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
        all_initializers = { name:get_optimizer(name) for name in get_backend_passes(self.name) if isinstance(get_optimizer(name), LayerOptimizerPass) }

        # Sort through the initializers based on the base class (e.g., to apply 'Layer' optimizers before 'Dense')
        sorted_initializers = sorted(all_initializers.items(), key=lambda x: len(x[1].layer_class.mro()))

        # Return only the names of the initializers
        return [opt[0] for opt in sorted_initializers]

    def _get_layer_templates(self):
        return [name for name in get_backend_passes(self.name) if isinstance(get_optimizer(name), Template)]

    def create_initial_config(self, **kwargs):
        raise NotImplementedError

    def create_layer_class(self, layer_class):
        raise NotImplementedError

    def get_available_flows(self):
        return get_backend_flows(self.name)

    def get_default_flow(self):
        raise NotImplementedError

    def register_source(self, file_name, source, destination_dir='nnet_utils'):
        raise NotImplementedError

    def register_pass(self, name, opt_cls):
        register_pass(name, opt_cls, backend=self.name)

    def register_template(self, template_cls):
        template = template_cls()
        register_pass(template.get_name(), template, backend=self.name)


backend_map = {}

def register_backend(name, backend_cls):
    if name.lower() in backend_map:
        raise Exception('Backend {} already registered'.format(name))

    backend_map[name.lower()] = backend_cls()

def get_backend(name):
    return backend_map[name.lower()]

def get_available_backends():
    return list(backend_map.keys())
