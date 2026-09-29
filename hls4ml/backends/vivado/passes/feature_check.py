from hls4ml.model.optimizer import OptimizerPass

# Flow that holds the passes of the fused strategy. A backend that does not register it cannot run it.
FUSION_FLOW = 'fuse_dense'


class ValidateFusedStrategy(OptimizerPass):
    """Stop a build that asks for the fused strategy on a backend that does not provide it.

    It reads the strategy from the configuration, not the layer attribute, so a layer type whose
    initializer ignores the setting is still caught.
    """

    def match(self, node):
        # Layers without a strategy, such as the input layer
        if node.get_attr('strategy') is None:
            return False
        if str(node.model.config.get_strategy(node)).lower() != 'fused':
            return False
        backend = node.model.config.backend
        return f'{backend.name.lower()}:{FUSION_FLOW}' not in backend.get_available_flows()

    def transform(self, model, node):
        raise Exception(
            f'Layer "{node.name}" ({node.class_name}) has strategy = "fused", which the '
            f'{model.config.backend.name} backend does not support. Use the Vitis backend, or one of the '
            'strategies this backend provides.'
        )
