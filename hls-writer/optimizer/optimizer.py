
class OptimizerPass(object):
    def __init__(self):
        pass

    def match(self, node):
        raise NotImplementedError
    
    def transform(self, model, node):
        raise NotImplementedError

class OptimizerRegistry(object):
    optimizer_map = {}

    @classmethod
    def register_pass(cls, name, opt_cls):
        if name in cls.optimizer_map:
            raise Exception('Optimization pass {} already registered'.format(name))
        
        if type(name) in [list, tuple]:
            for n in name:
                cls.optimizer_map[n] = opt_cls
        else:    
            cls.optimizer_map[name] = opt_cls

    @classmethod
    def get(cls, name):
        return cls.optimizer_map[name]()

from passes.nop import EliminateLinearActivation

OptimizerRegistry.register_pass('eliminate_linear_activation', EliminateLinearActivation)

def optimize_model(model, passes):
    optimizers = [OptimizerRegistry.get(opt_pass) for opt_pass in passes]
    optimization_done = False
    while not optimization_done:
        for node in model.graph.values():
            for opt in optimizers:
                if opt.match(node):
                    res = opt.transform(model, node)
                    if res:
                        break
            else:
                continue
            break
        else:
            optimization_done = True

