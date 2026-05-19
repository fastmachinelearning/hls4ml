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


def set_default_config(hls_conf, default_config):
    for key, value in default_config.items():
        if key not in hls_conf.keys():
            hls_conf[key] = value
    return hls_conf
