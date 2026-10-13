import numpy as np

from hls4ml.backends.fpga.fpga_types import BramWeightVariableConverter
from hls4ml.model.optimizer import OptimizerPass


class RegisterBramWeights(OptimizerPass):
    """Convert selected weight variables to external BRAM-interface variables."""

    def match(self, node):
        return len(node.weights) > 0

    def transform(self, model, node):
        bramport_size = model.config.get_bram_size(node)
        external = self._external_roles(model, node)
        for w_name, w_var in node.weights.items():
            has_storage = 'storage' in w_var.__dict__
            if has_storage and w_var.storage == 'bram':
                continue
            if w_name in external:
                if not has_storage:
                    raise ValueError(f'{node.name}: parameter {w_name!r} cannot be made external in this backend')
                node.set_attr(w_name, BramWeightVariableConverter.convert(w_var))
            elif has_storage and np.prod(w_var.shape) > bramport_size:
                node.set_attr(w_name, BramWeightVariableConverter.convert(w_var))

    @staticmethod
    def _external_roles(model, node):
        """``ExternalParameters``: a list of this layer's parameter roles, from
        LayerName or LayerType config. Roles belong to a layer, so a Model-level
        entry is a mistake rather than a default."""
        hls_config = model.config.get_config_value('HLSConfig', {})
        if 'ExternalParameters' in hls_config.get('Model', {}):
            raise ValueError("ExternalParameters must be set per layer (LayerName or LayerType), not under 'Model'")
        roles = model.config.get_layer_config(node).get('ExternalParameters')
        if roles is None:
            return set()
        if not isinstance(roles, (list, tuple)) or not all(isinstance(r, str) for r in roles):
            raise ValueError(f'{node.name}: ExternalParameters must be a list or tuple of parameter names, got {roles!r}')
        unknown = sorted(set(roles) - set(node.weights))
        if unknown:
            raise ValueError(
                f'{node.name}: ExternalParameters names {unknown}, but this layer has parameters {sorted(node.weights)}'
            )
        return set(roles)
