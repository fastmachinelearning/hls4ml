
class Flow(object):
    def __init__(self, name, optimizers, requires=None):
        self.name = name
        if optimizers is None:
            self.optimizers = []
        else:
            self.optimizers = optimizers
        if requires is None:
            self.requires = []
        else:
            self.requires = requires

flow_map = {}

def _get_backend_name_prefix(name, backend):
    if backend is not None and not name.startswith(backend.lower()):
        name = backend.lower() + ':' + name

    return name

def register_flow(name, optimizers, requires=None, backend=None):
    name = _get_backend_name_prefix(name, backend)

    if name in flow_map:
        raise Exception('Flow {} already registered'.format(name))

    flow_map[name] = Flow(name, optimizers, requires)

    return name

def get_flow(name):
    if name in flow_map:
        return flow_map[name]
    else:
        raise Exception('Unknown flow: {}'.format(name))

def get_backend_flows(backend):
    return [flow for flow in flow_map.keys() if flow.startswith(backend.lower() + ':')]

def get_available_flows():
    return list(flow_map.keys())
