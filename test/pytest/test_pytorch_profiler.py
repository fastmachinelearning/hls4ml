import pytest

from hls4ml.model.profiling import numerical

try:
    import torch
    import torch.nn as nn

    __torch_profiling_enabled__ = True
except ImportError:
    __torch_profiling_enabled__ = False


class SubClassModel(torch.nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        for idx, layer in enumerate(layers):
            setattr(self, f'layer_{idx}', layer)


class ModuleListModel(torch.nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.layer = torch.nn.ModuleList(layers)


class NestedSequentialModel(torch.nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(*layers)


def count_bars_in_figure(fig):
    count = 0
    for ax in fig.get_axes():
        count += len(ax.patches)
    return count


# Reusable parameter list
test_layers = [
    (4, [nn.Linear(10, 20), nn.Linear(20, 5)]),
    (3, [nn.Linear(10, 20), nn.BatchNorm1d(20)]),
    (6, [nn.Linear(10, 20), nn.Linear(20, 5), nn.Conv1d(3, 16, kernel_size=3)]),
    (6, [nn.Linear(15, 30), nn.Linear(30, 15), nn.Conv2d(1, 32, kernel_size=3)]),
    (6, [nn.RNN(64, 128), nn.Linear(128, 10)]),
    (6, [nn.LSTM(64, 128), nn.Linear(128, 10)]),
    (6, [nn.GRU(64, 128), nn.Linear(128, 10)]),
]


@pytest.mark.parametrize("layers", test_layers)
def test_sequential_model(layers):
    if __torch_profiling_enabled__:
        param_count, layers = layers
        model = torch.nn.Sequential(*layers)
        wp, _, _, _ = numerical(model)
        assert count_bars_in_figure(wp) == param_count


@pytest.mark.parametrize("layers", test_layers)
def test_subclass_model(layers):
    if __torch_profiling_enabled__:
        param_count, layers = layers
        model = SubClassModel(layers)
        wp, _, _, _ = numerical(model)
        assert count_bars_in_figure(wp) == param_count


@pytest.mark.parametrize("layers", test_layers)
def test_modulelist_model(layers):
    if __torch_profiling_enabled__:
        param_count, layers = layers
        model = ModuleListModel(layers)
        wp, _, _, _ = numerical(model)
        assert count_bars_in_figure(wp) == param_count


@pytest.mark.parametrize("layers", test_layers)
def test_nested_model(layers):
    if __torch_profiling_enabled__:
        param_count, layers = layers
        model = NestedSequentialModel(layers)
        wp, _, _, _ = numerical(model)
        assert count_bars_in_figure(wp) == param_count
