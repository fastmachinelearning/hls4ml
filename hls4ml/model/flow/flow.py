
class Flow(object):
    def __init__(self, name, optimizers, requires=None):
        self.name = name
        if optimizers is None:
            self._optimizers = []
        else:
            self._optimizers = optimizers
        if requires is None:
            self.requires = []
        else:
            self.requires = requires

    @property
    def optimizers(self):
        return self._optimizers
    
    def _add_optimizer(self, opt_name):
        self._optimizers.append(opt_name)

    def _remove_optimizer(self, opt_name):
        self._optimizers.remove(opt_name)

class DynamicFlow(Flow):
    def __init__(self, name, optimizer_func, requires=None):
        self.name = name
        self._optimizer_func = optimizer_func
        self._added_optimizers = set()
        self._removed_optimizers = set()
        if requires is None:
            self.requires = []
        else:
            self.requires = requires

    @property
    def optimizers(self):
        optimizers = self._optimizer_func()
        optimizers.extend(self._added_optimizers)
        optimizers = [o for o in optimizers if o not in self._removed_optimizers]
        return optimizers
    
    def _add_optimizer(self, opt_name):
        self._added_optimizers.put(opt_name)

    def _remove_optimizer(self, opt_name):
        self._removed_optimizers.put(opt_name)

flow_map = {}

def _get_backend_name_prefix(name, backend):
    if backend is not None and not name.startswith(backend.lower()):
        name = backend.lower() + ':' + name

    return name

def register_flow(name, optimizers, requires=None, backend=None):
    name = _get_backend_name_prefix(name, backend)

    if name in flow_map:
        raise Exception('Flow {} already registered'.format(name))

    if callable(optimizers):
        flow = DynamicFlow(name, optimizer_func=optimizers, requires=requires)
    else:
        flow = Flow(name, optimizers=optimizers, requires=requires)

    flow_map[name] = flow

    return name

def update_flow(flow_name, add_optimizers=None, remove_optimizers=None):
    flow = get_flow(flow_name)
    if add_optimizers is not None:
        for opt in add_optimizers:
            flow._add_optimizer(opt)

    if remove_optimizers is not None:
        for opt in remove_optimizers:
            flow._remove_optimizer(opt)

def get_flow(name):
    if name in flow_map:
        return flow_map[name]
    else:
        raise Exception('Unknown flow: {}'.format(name))

def get_backend_flows(backend):
    return [flow for flow in flow_map.keys() if flow.startswith(backend.lower() + ':')]

def get_available_flows():
    return list(flow_map.keys())
