import inspect
import re

class OptimizerPass(object):
    name = None

    def __init__(self):
        pass

    def match(self, node):
        raise NotImplementedError
    
    def transform(self, model, node):
        raise NotImplementedError
    
    @classmethod
    def get_name(cls):
        if cls.name is None:
            return re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower() # OptimizerPass -> optimizer_pass
        else:
            return cls.name

class GlobalOptimizerPass(OptimizerPass):
    def match(self, node):
        return True # Match everything

class WrappedOptimizerPass(OptimizerPass):
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
    def __init__(self, name, layer_class, transform):
        super(LayerOptimizerPass, self).__init__(name, lambda node: isinstance(node, layer_class), transform)

def optimizer_pass(condition):
    def decorator(function):
        function._condition = condition
        return function
    return decorator

def extract_optimizers_from_object(clazz):
    optimizers = {}
    optimizer_list = [func for func in dir(clazz) if callable(getattr(clazz, func)) and hasattr(getattr(clazz, func), '_condition')]
    for opt_name in optimizer_list:
        func = getattr(clazz, opt_name)
        if inspect.isclass(func._condition):
            opt = LayerOptimizerPass(name=opt_name, layer_class=func._condition, transform=func)
        else:
            opt = WrappedOptimizerPass(name=opt_name, condition=func._condition, transform=func)
        optimizers[opt_name] = opt
    
    return optimizers

optimizer_map = {}

def register_pass(name, opt_cls):
    if name in optimizer_map:
        raise Exception('Optimization pass {} already registered'.format(name))
    
    if inspect.isclass(opt_cls):
        opt = opt_cls()
    else:
        opt = opt_cls

    if type(name) in [list, tuple]:
        for n in name:
            optimizer_map[n] = opt
    else:    
        optimizer_map[name] = opt

def get_optimizer(name):
    if name in optimizer_map:
        return optimizer_map[name]
    elif isinstance(name, OptimizerPass):
        return name
    else:
        raise Exception('Unknown optimizer: {}'.format(name))

def get_available_passes():
    return list(optimizer_map.keys())

def optimize_model(model, passes=None):
    if passes is None:
        passes = optimizer_map.keys()
    optimizers = [get_optimizer(opt_pass) for opt_pass in passes]
    optimization_done = False
    while not optimization_done:
        for opt in optimizers:
            for node in model.graph.values():
                if opt.match(node):
                    res = opt.transform(model, node)
                    if res:
                        break
            else:
                continue
            break
        else:
            optimization_done = True
