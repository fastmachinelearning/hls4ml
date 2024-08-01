from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import hls4ml
from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model

test_root_path = Path(__file__).parent

in_shape = 16
atol = 5e-3


@pytest.fixture
def data():
    np.random.seed(0)
    X = np.random.rand(100, in_shape)
    return X


@pytest.fixture
def fusion_data():
    n_batch = 2
    n_in = 2
    size_in_height = 1024
    X = np.random.rand(n_batch, n_in, size_in_height)
    return X


class BatchNorm_w_Fusion(nn.Module):
    def __init__(self, filters, momentum):
        super().__init__()
        self.conv1 = nn.Conv1d(
            int(filters),
            filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(filters, momentum, track_running_stats=True)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'Catapult'])
def test_batchnorm(data, backend, io_type):
    model = nn.Sequential(
        nn.BatchNorm1d(in_shape),
    ).to()
    model.eval()

    default_precision = 'ac_fixed<32, 1, true>' if backend == 'Quartus' else 'ac_fixed<32, 1>'

    config = hls4ml.utils.config_from_pytorch_model(model, default_precision=default_precision, granularity='name')
    output_dir = str(test_root_path / f'hls4mlprj_batchnorm_{backend}_{io_type}')
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model, (None, in_shape), backend=backend, hls_config=config, io_type=io_type, output_dir=output_dir
    )
    hls_model.compile()

    # Predict
    pytorch_prediction = model(torch.Tensor(data)).detach().numpy()
    hls_prediction = hls_model.predict(data)
    np.testing.assert_allclose(pytorch_prediction, hls_prediction, rtol=0, atol=atol, verbose=True)


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_batchnorm_fusion(fusion_data, backend, io_type):
    n_in = 2
    momentum = 0.2
    size_in_height = 1024
    filters = n_in

    # see above for model definition
    model = BatchNorm_w_Fusion(filters, momentum)

    # generating config
    config = config_from_pytorch_model(
        model,
        default_reuse_factor=12,
        # granularity='name',
        channels_last_conversion="full",
        transpose_outputs=False,
    )
    config['Model']['Strategy'] = 'Resource'

    default_precision = 'ac_fixed<32, 1, true>' if backend == 'Quartus' else 'ac_fixed<32, 1>'

    config['Model']['Precision'] = default_precision

    # conversion
    output_dir = str(test_root_path / f'hls4mlprj_block_{backend}_{io_type}')
    hls_model = convert_from_pytorch_model(
        model,
        (None, n_in, size_in_height),
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
    )

    # compiling model
    hls_model.compile()

    """
    Makes predictions with pytorch and hls model. Currently fails with opaque error (core dump).
    When run outside pytest, np.testing.assert_allclose raises an AssertionError because tolerances
    are exceeded. This is likely the cause of the core dump.
    """
    pytorch_prediction = model(torch.Tensor(fusion_data)).detach().numpy()
    hls_prediction = hls_model.predict(fusion_data).reshape(pytorch_prediction.shape)
    np.testing.assert_allclose(pytorch_prediction, hls_prediction, rtol=0, atol=atol, verbose=True)
