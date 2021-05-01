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
    def __init__(self, name, condition=None, transform=None):
        self.name = name
        self.transform_func = transform
        self.condition = condition
    
    def match(self, node):
        if self.condition is not None:
            return self.condition(node)
        else:
            raise NotImplementedError

    def transform(self, model, node):
        if self.transform_func is not None:
            retval = self.transform_func(node)
            return retval if retval is not None else False
        else:
            raise NotImplementedError
    
    def get_name(self):
        return self.name

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
