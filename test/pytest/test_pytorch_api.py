from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model

test_root_path = Path(__file__).parent


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_linear(backend, io_type):

    model = LinearModel()
    model.eval()

    X_input = np.random.rand(1)

    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model)
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_linear_{backend}_{io_type}')

    hls_model = convert_from_pytorch_model(
        model, (None, 1), hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )

    hls_model.compile()

    hls_prediction = hls_model.predict(X_input)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)

    from torch.fx import symbolic_trace

    traced_model = symbolic_trace(model)

    nNodes = 0
    for _node in traced_model.graph.nodes:
        nNodes += 1

    assert nNodes - 1 == len(hls_model.get_layers())
    assert list(hls_model.get_layers())[0].attributes['class_name'] == "InputLayer"
    assert list(hls_model.get_layers())[1].attributes["class_name"] == "Dense"
    assert list(hls_model.get_layers())[0].attributes['input_shape'] == [1]
    assert list(hls_model.get_layers())[1].attributes['n_in'] == 1
    assert list(hls_model.get_layers())[1].attributes['n_out'] == 1


# TODO: add ThresholdedReLU test when it can be made to pass
@pytest.mark.parametrize(
    "activation_function",
    [
        nn.ReLU(),
        nn.LeakyReLU(negative_slope=1.0),
        nn.ELU(alpha=1.0),
        nn.PReLU(init=0.25),
        nn.Sigmoid(),
        nn.Threshold(threshold=1.0, value=0.0),
    ],
)
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_activations(activation_function, backend, io_type):

    model = torch.nn.Sequential(nn.Linear(1, 1), activation_function).to()
    model.eval()

    X_input = np.random.rand(1)

    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model)
    output_dir = str(
        test_root_path / f'hls4mlprj_pytorch_api_activations_{activation_function.__class__.__name__}_{backend}_{io_type}'
    )
    hls_model = convert_from_pytorch_model(
        model, (None, 1), hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    hls_model.compile()

    hls_prediction = hls_model.predict(X_input)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)

    from torch.fx import symbolic_trace

    traced_model = symbolic_trace(model)

    nNodes = 0
    for _node in traced_model.graph.nodes:
        nNodes += 1

    assert nNodes - 1 == len(hls_model.get_layers())

    if activation_function.__class__.__name__ == 'ReLU' or activation_function.__class__.__name__ == 'Sigmoid':
        assert list(hls_model.get_layers())[2].attributes['class_name'] == 'Activation'
    elif activation_function.__class__.__name__ == 'Threshold':
        assert list(hls_model.get_layers())[2].attributes['class_name'] == 'ThresholdedReLU'
    else:
        assert list(hls_model.get_layers())[2].attributes['class_name'] == activation_function.__class__.__name__


padds_options = [0, 1]


@pytest.mark.parametrize('padds', padds_options)
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_conv1d(padds, backend, io_type):

    model = torch.nn.Sequential(nn.Conv1d(16, 33, 3, padding=padds), nn.ReLU()).to()
    model.eval()

    X_input = np.random.rand(20, 16, 50)
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model)
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_conv1d_{padds}_{backend}_{io_type}')
    hls_model = convert_from_pytorch_model(
        model, (None, 16, 50), hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    hls_model.compile()

    hls_prediction = hls_model.predict(X_input)

    # 5e-2 might be too high
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=5e-2)
