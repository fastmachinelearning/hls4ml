import numpy as np


class IsolatedLayerReader:
    def __init__(self, layer):
        self.layer = layer

    def get_weights_data(self, layer_name, var_name):
        assert layer_name == self.layer.name, f'Processing {self.layer.name}, but handler tried to read {layer_name}'
        for w in self.layer.weights:
            if var_name in w.name:
                return np.array(w)
        return None
