class _global_config:
    trace_depth = False
    fuse_associative_ops = True
    backend = 'generic'
    order_metrics = ()


class VariableOverrideContextManager:
    target = ''

    def __init__(self, value=None):
        self.value = value

    def __enter__(self):
        self.original = getattr(_global_config, self.target)
        setattr(_global_config, self.target, self.value)

    def __exit__(self, exc_type, exc_value, traceback):
        setattr(_global_config, self.target, self.original)

    def __new__(cls, *args, **kwargs):
        if cls is VariableOverrideContextManager:
            raise TypeError('VariableOverrideContextManager should not be instantiated directly')
        return super().__new__(cls)
