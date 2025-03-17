from collections import defaultdict

import numpy as np
import pandas
import torch

from hls4ml.model.profiling import array_to_summary


def _torch_batchnorm(layer):
    weights = list(layer.parameters())
    epsilon = layer.eps

    gamma = weights[0]
    beta = weights[1]
    if layer.track_running_stats:
        mean = layer.running_mean
        var = layer.running_var
    else:
        mean = torch.tensor(np.ones(20))
        var = torch.tensor(np.zeros(20))

    scale = gamma / np.sqrt(var + epsilon)
    bias = beta - gamma * mean / np.sqrt(var + epsilon)

    return [scale, bias], ['s', 'b']


def _torch_layer(layer):
    return list(layer.parameters()), ['w', 'b']


def _torch_rnn(layer):
    return list(layer.parameters()), ['w_ih_l0', 'w_hh_l0', 'b_ih_l0', 'b_hh_l0']


torch_process_layer_map = defaultdict(
    lambda: _torch_layer,
    {
        'BatchNorm1d': _torch_batchnorm,
        'BatchNorm2d': _torch_batchnorm,
        'RNN': _torch_rnn,
        'LSTM': _torch_rnn,
        'GRU': _torch_rnn,
    },
)


class WeightsTorch:

    def __init__(self, model: torch.nn.Module, fmt: str = 'longform', plot: str = 'boxplot') -> None:
        self.model = model
        self.fmt = fmt
        self.plot = plot
        self.registered_layers = list()
        self._find_layers(self.model, self.model.__class__.__name__)

    def _find_layers(self, model, module_name):
        for name, module in model.named_children():
            if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
                self._find_layers(module, module_name + "." + name)
            elif isinstance(module, (torch.nn.Module)) and self._is_parameterized(module):
                if len(list(module.named_children())) != 0:
                    # custom nn.Module, continue search
                    self._find_layers(module, module_name + "." + name)
                else:
                    self._register_layer(module_name + "." + name)

    def _is_registered(self, name: str) -> bool:
        return name in self.registered_layers

    def _register_layer(self, name: str) -> None:
        if self._is_registered(name) is False:
            self.registered_layers.append(name)

    def _is_parameterized(self, module: torch.nn.Module) -> bool:
        return any(p.requires_grad for p in module.parameters())

    def _get_weights(self) -> pandas.DataFrame | list[dict]:
        if self.fmt == 'longform':
            data = {'x': [], 'layer': [], 'weight': []}
        elif self.fmt == 'summary':
            data = []
        for layer_name in self.registered_layers:
            layer = self._get_layer(layer_name, self.model)
            name = layer.__class__.__name__
            weights, suffix = torch_process_layer_map[layer.__class__.__name__](layer)
            for i, w in enumerate(weights):
                label = f'{name}/{suffix[i]}'
                w = weights[i].detach().numpy()
                w = w.flatten()
                w = abs(w[w != 0])
                n = len(w)
                if n == 0:
                    print(f'Weights for {name} are only zeros, ignoring.')
                    break
                if self.fmt == 'longform':
                    data['x'].extend(w.tolist())
                    data['layer'].extend([name] * n)
                    data['weight'].extend([label] * n)
                elif self.fmt == 'summary':
                    data.append(array_to_summary(w, fmt=self.plot))
                    data[-1]['layer'] = name
                    data[-1]['weight'] = label

        if self.fmt == 'longform':
            data = pandas.DataFrame(data)
        return data

    def get_weights(self) -> pandas.DataFrame | list[dict]:
        return self._get_weights()

    def get_layers(self) -> list[str]:
        return self.registered_layers

    def _get_layer(self, layer_name: str, module: torch.nn.Module) -> torch.nn.Module:
        for name in layer_name.split('.')[1:]:
            module = getattr(module, name)
        return module
