from hls4ml.model.optimizer import optimize_model

class Flow(object):
    def __init__(self, name, optimizers, requires=None):
        self.name = name
        if optimizers is None:
            self.optimizers = []
        else:
            self.optimizers = optimizers
        self.optimizers = optimizers
        if requires is None:
            self.requires = []
        else:
            self.requires = requires

flow_map = {}

def register_flow(name, optimizers, requires=None):
    if name in flow_map:
        raise Exception('Flow {} already registered'.format(name))
    
    flow_map[name] = Flow(name, optimizers, requires)

def get_flow(name):
    if name in flow_map:
        return flow_map[name]
    else:
        raise Exception('Unknown flow: {}'.format(name))

def get_available_flows():
    return list(flow_map.keys())
