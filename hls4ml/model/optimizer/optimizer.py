import importlib
import inspect
import os

from hls4ml.utils.string_utils import convert_to_snake_case


class OptimizerPass:
    """Base optimizer class from which all other optimizer types are derived."""

    name = None

    def __init__(self):
        pass

    def match(self, node):
        """Predicate to match on a given node.

        Args:
            node (Layer): Node in the model graph to try matching the optimizer on.
        """
        raise NotImplementedError

    def transform(self, model, node):
        """Transformation to apply if matching was successful.

        Transform should return a boolean value indicating if the model graph was altered (by adding/removing nodes).

        Args:
            model (ModelGraph): Model to optimize
            node (Layer): The matched node in the model graph.
        """
        raise NotImplementedError

    @classmethod
    def get_name(cls):
        if cls.name is None:
            return convert_to_snake_case(cls.__name__)  # OptimizerPass -> optimizer_pass
        else:
            return cls.name


class GlobalOptimizerPass(OptimizerPass):
    """Global optimizer that matches on every node in the model graph."""

    def match(self, node):
        return True  # Match everything


class WrappedOptimizerPass(OptimizerPass):
    """An optimizer class created by wrapping a function call.

    Users should generally not create any wrapped optimizer passes manually.
    """

    def __init__(self, name, condition, transform):
        self.name = name
        self.condition = condition
        self.transform_func = transform

    def match(self, node):
        return self.condition(node)

    def transform(self, model, node):
        retval = self.transform_func(node)
        return retval if retval is not None else False

    def get_name(self):
        return self.name


class LayerOptimizerPass(WrappedOptimizerPass):
    """An wrapper optimizer specific to a layer class.

    Commonly used by backends to add extra initialization to a layer instance.
    """

    def __init__(self, name, layer_class, transform):
        super().__init__(name, lambda node: isinstance(node, layer_class), transform)
        self.layer_class = layer_class


class ModelOptimizerPass(OptimizerPass):
    """A special optimizer that works with the model itself.

    Examples include writing the model to C++/HLS.
    """

    def __init__(self, name, transform):
        self.name = name
        self.transform_func = transform

    def transform(self, model):
        retval = self.transform_func(model)
        return retval if retval is not None else False


class ConfigurableOptimizerPass(OptimizerPass):
    """An optimizer that can be configured.

    Existing instances of this class in the registry can be configured with the configure() method. Multiple instances
    with different configuration can co-exist if registered with different names.
    """

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_config(self):
        attrs = vars(self)
        return attrs.copy()


# Decorator optimizers


def optimizer_pass(condition):
    def decorator(function):
        function._condition = condition
        return function

    return decorator


def layer_optimizer(layer):
    """Decorator to turn a function into the optimization pass.

    Example::

        @layer_optimizer(MyLayer)
        def init_mylayer(self, layer):
            layer.set_attr('new_attribute', 'some_value')

    Args:
        layer (_type_): _description_
    """

    def decorator(function):
        return optimizer_pass(layer)(function)

    return decorator


def model_optimizer():
    """Decorator to turn a function into a model optimizer."""

    def decorator(function):
        return optimizer_pass(None)(function)

    return decorator


# Helpers for extracting optimizers from objects


def extract_optimizers_from_path(opt_path, module_path, initializer=None):
    optimizers = {}

    if not os.path.exists(opt_path):
        return optimizers

    if not module_path.endswith('.'):
        module_path += '.'

    for module in os.listdir(opt_path):
        if module == '__init__.py' or module[-3:] != '.py':
            continue
        try:
            lib = importlib.import_module(module_path + module[:-3])
            if 'register_' + module[:-3] in lib.__dict__:
                opt_init_func = lib.__dict__['register_' + module[:-3]]
                if initializer is not None:
                    opt_init_func(initializer)
                else:
                    opt_init_func()
            else:
                for func in list(lib.__dict__.values()):
                    # if 'func' is a class
                    # and it inherits from OptimizerPass
                    # and is defined in this module (i.e., not imported)
                    if inspect.isclass(func) and issubclass(func, OptimizerPass) and func.__module__ == lib.__name__:
                        if inspect.ismethod(func.get_name):
                            optimizers[func.get_name()] = func
                        else:
                            func_instance = func()
                            optimizers[func_instance.get_name()] = func_instance

        except ImportError as e:
            print(f'WARN: Unable to import optimizer(s) from {module}: {e}')
            continue

    return optimizers


def extract_optimizers_from_object(clazz):
    optimizers = {}
    optimizer_list = [
        func for func in dir(clazz) if callable(getattr(clazz, func)) and hasattr(getattr(clazz, func), '_condition')
    ]
    for opt_name in optimizer_list:
        func = getattr(clazz, opt_name)
        if func._condition is None:
            opt = ModelOptimizerPass(name=opt_name, transform=func)
        elif inspect.isclass(func._condition):
            opt = LayerOptimizerPass(name=opt_name, layer_class=func._condition, transform=func)
        else:
            opt = WrappedOptimizerPass(name=opt_name, condition=func._condition, transform=func)
        optimizers[opt_name] = opt

    return optimizers


# Optimizer registry

optimizer_map = {}


def _get_backend_name_prefix(name, backend):
    if backend is not None and not name.startswith(backend.lower() + ':'):
        name = backend.lower() + ':' + name

    return name


def register_pass(name, opt_cls, backend=None):
    """Register a new optimizer pass.

    Args:
        name (str): Name of the optimizer
        opt_cls (class): The class of the optimizer.
        backend (str, optional): Optional backend to register the optimizer to. If not None, the name of the backend
            will be appended to the name of the registered flow. Defaults to None.

    Raises:
        Exception: If the optimization pass has already been registered with the given name.

    Returns:
        str: The name of the registered optimizer.
    """
    name = _get_backend_name_prefix(name, backend)

    if name in optimizer_map:
        raise Exception(f'Optimization pass {name} already registered')

    if inspect.isclass(opt_cls):
        opt = opt_cls()
    else:
        opt = opt_cls

    optimizer_map[name] = opt

    return name


def get_optimizer(name):
    """Return the optimizer instance registered with the given name.

    Args:
        name (str): Name of the optimizer in the registry.

    Raises:
        Exception: If the optimizer with the given name is not found in the registry.

    Returns:
        OptimizerPass: The optimizer from the registry.
    """
    if name in optimizer_map:
        return optimizer_map[name]
    else:
        raise Exception(f'Unknown optimizer: {name}')


def get_backend_passes(backend):
    """Returns the list of optimizer passes belonging to a backend

    Args:
        backend (str): Name of the backend.

    Returns:
        list: List of optimizer names registered with the given backend.
    """
    return [opt for opt in optimizer_map.keys() if opt.startswith(backend.lower() + ':')]


def get_available_passes():
    """Return the list of all registered optimizer passes.

    Returns:
        list: List of registered passes.
    """
    return list(optimizer_map.keys())


def optimize_model(model, passes):
    """Optimize a given model with the given passes.

    The passes are attempted until all passes no longer match or no changes to the model graph occur.

    Args:
        model (ModelGraph): The model to optimize.
        passes (list): List of passes to apply.

    Returns:
        set: The set of applied passes (the passes that matched the predicate).
    """
    optimizers = {opt_pass: get_optimizer(opt_pass) for opt_pass in passes}
    applied_passes = set()
    optimization_done = False
    while not optimization_done:
        for opt_name, opt in optimizers.items():
            if isinstance(opt, ModelOptimizerPass) and opt_name not in applied_passes:
                res = opt.transform(model)
                if res:
                    applied_passes.add(opt_name)
                continue
            for node in model.graph.values():
                if opt.match(node):
                    res = opt.transform(model, node)
                    applied_passes.add(opt_name)
                    if res:
                        break
            else:
                continue
            break
        else:
            optimization_done = True

    return applied_passes
