from hls4ml.model.layers import Dense
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType


def get_reloadable_weights(model):
    """The reloadable weights of the model, in the order in which they are loaded."""
    return [w for layer in model.get_layers() for w in layer.get_weights() if w.storage == 'reloadable']


class RegisterReloadableWeights(OptimizerPass):
    """Moves the weights of layers with ``reloadable_weights`` set from constants to reloadable on-chip memory."""

    def match(self, node):
        return node.get_attr('reloadable_weights', False) and any(w.storage != 'reloadable' for w in node.get_weights())

    def transform(self, model, node):
        if not isinstance(node, Dense):
            raise Exception(f'Reloadable weights are only supported for Dense layers, not {node.class_name} ({node.name})')
        if model.config.get_compression(node):
            raise Exception(f'Reloadable weights cannot be compressed ({node.name})')
        weight_sets = model.config.get_config_value('WeightSets', 1)
        if not isinstance(weight_sets, int) or weight_sets < 1:
            raise Exception(f'WeightSets must be a positive integer, not {weight_sets}')

        for w in node.get_weights():
            if not isinstance(w.type.precision, (FixedPrecisionType, IntegerPrecisionType)):
                raise Exception(f'Reloadable weights must have a fixed-point or integer type ({w.name})')
            w.storage = 'reloadable'
        return False
