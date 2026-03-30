import inspect
import os
from pathlib import Path

from hls4ml.backends.template import Template
from hls4ml.model.flow import get_backend_flows, update_flow
from hls4ml.model.optimizer import (
    LayerOptimizerPass,
    extract_optimizers_from_object,
    extract_optimizers_from_path,
    get_backend_passes,
    get_optimizer,
    register_pass,
)


class Backend:
    def __init__(self, name):
        self.name = name
        self.custom_source = {}
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
        file_optimizers = {}
        for cls in [*self.__class__.__bases__, self.__class__]:
            opt_path = os.path.dirname(inspect.getfile(cls)) + '/passes'
            module_path = cls.__module__[: cls.__module__.rfind('.')] + '.passes'
            cls_optimizers = extract_optimizers_from_path(opt_path, module_path, self)
            file_optimizers.update(cls_optimizers)
        return file_optimizers

    def _get_layer_initializers(self):
        all_initializers = {
            name: get_optimizer(name)
            for name in get_backend_passes(self.name)
            if isinstance(get_optimizer(name), LayerOptimizerPass)
        }

        # Sort through the initializers based on the base class (e.g., to apply 'Layer' optimizers before 'Dense')
        sorted_initializers = sorted(all_initializers.items(), key=lambda x: len(x[1].layer_class.mro()))

        # Return only the names of the initializers
        return [opt[0] for opt in sorted_initializers]

    def _get_layer_templates(self):
        return [name for name in get_backend_passes(self.name) if isinstance(get_optimizer(name), Template)]

    def create_initial_config(self, **kwargs):
        """Create the minimal conversion config for the backend.

        Subclasses should implement this method to provide the initial configuration for the conversion.
        """
        raise NotImplementedError

    def create_layer_class(self, layer_class):
        """Wrap the original layer class into the backend-specific layer class.

        Backends should extend base layer classes with new attributes and variables as needed. These new classes are then
        used within the model.

        Args:
            layer_class (class): Base class to extend
        """
        raise NotImplementedError

    def get_available_flows(self):
        """Returns the list of flows registered for this backend.

        Returns:
            list: The list of registered flows.
        """
        return get_backend_flows(self.name)

    def get_default_flow(self):
        """The name of the default flow of the backend.

        Default flow is used as the conversion target if the target flow has not been specified.
        """
        raise NotImplementedError

    def get_custom_source(self):
        """Returns the registered custom source files.

        Returns:
            dict: Custom source files. Keys represent destination paths, values are absolute paths to registered source
                files.
        """
        return self.custom_source

    def register_source(self, source_file, destination_dir='nnet_utils'):
        """Register custom source that is not part of the backend's templates.

        Args:
            source_file (str or Path): Absolute path to the source file.
            destination_dir (str, optional): The sub-directory of the output project to write the source file to.
                Defaults to 'nnet_utils'.

        Raises:
            Exception: If the source file is not a str or Path, or if the path is not absolute
        """
        if isinstance(source_file, str):
            if not os.path.isabs(source_file):
                raise Exception(f'Expected absolute path to custom source file, got: "{source_file}"')
            source_path = Path(source_file)
        elif isinstance(source_file, Path):
            source_path = source_file
        else:
            raise Exception(f'Expected string or Path, got: "{type(source_file)}"')

        self.custom_source[destination_dir + os.path.sep + source_path.name] = source_path

    def register_pass(self, name, opt_cls, flow=None):
        """Register an optimizer path for the backend.

        Note that user-provided optimizers registered without specifying any flow will not be invoked.

        Args:
            name (str): Name of the optimizer
            opt_cls (class): Optimizer class
            flow (str, list or tuple, optional): Existing flow(s) to add the optimizer to. Defaults to None.
        """
        opt_name = register_pass(name, opt_cls, backend=self.name)
        if flow is not None:
            if not isinstance(flow, (list, tuple)):
                flow = [flow]

            for f in flow:
                update_flow(f, add_optimizers=[opt_name])

    def register_template(self, template_cls):
        """Register a template "optimizer".

        E.g., function call template or op configuration template.

        Args:
            template_cls (class): Template to register.
        """
        template = template_cls()
        register_pass(template.get_name(), template, backend=self.name)


backend_map = {}


def register_backend(name, backend_cls):
    """Create the backend instance and add it to the registry.

    Args:
        name (str): Name of the backend.
        backend_cls (class): Backend class to instantiate. Class must implement a constructor without parameters.

    Raises:
        Exception: If the backend has already been registered.
    """
    if name.lower() in backend_map:
        raise Exception(f'Backend {name} already registered')

    backend_map[name.lower()] = backend_cls()


def get_backend(name):
    return backend_map[name.lower()]


def get_available_backends():
    return list(backend_map.keys())
