from hls4ml.model.flow import get_flow
from hls4ml.model.optimizer import OptimizerPass


class ValidateFusedStrategy(OptimizerPass):
    """Stop a build that asks for the fused strategy on a backend that does not provide it.

    It reads the strategy from the configuration, not the layer attribute, so a layer type whose
    initializer ignores the setting is still caught.
    """

    # Flow that holds the passes of the fused strategy
    _fusion_flow = 'fuse_dense'

    def match(self, node):
        # Layers without a strategy, such as the input layer
        if node.get_attr('strategy') is None:
            return False
        if str(node.model.config.get_strategy(node)).lower() != 'fused':
            return False
        return not self._runs_fusion(node.model)

    def _runs_fusion(self, model):
        """Whether the flows of the model include the fusion flow, directly or through the flows they require.

        A backend that builds on Vitis, such as Coyote, runs it through the Vitis flow it requires.
        """

        seen, pending = set(), list(model.config.flows)
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            if name.split(':')[-1] == self._fusion_flow:
                return True
            pending.extend(get_flow(name).requires)
        return False

    def transform(self, model, node):
        raise Exception(
            f'Layer "{node.name}" ({node.class_name}) has strategy = "fused", which the '
            f'{model.config.backend.name} backend does not support. Use the Vitis backend, or one of the '
            'strategies this backend provides.'
        )
