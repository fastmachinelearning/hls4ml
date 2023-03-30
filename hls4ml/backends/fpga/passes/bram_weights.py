import numpy as np

from hls4ml.backends.fpga.fpga_types import BramWeightVariableConverter
from hls4ml.model.optimizer import OptimizerPass


class RegisterBramWeights(OptimizerPass):
    def match(self, node):
        return len(node.weights) > 0

    def transform(self, model, node):
        bramport_size = model.config.get_bram_size(node)
        for w_name, w_var in node.weights.items():
            if ('storage' in w_var.__dict__ and w_var.storage != 'bram') and np.prod(w_var.shape) > bramport_size:
                new_weight = BramWeightVariableConverter.convert(w_var)
                node.set_attr(w_name, new_weight)
