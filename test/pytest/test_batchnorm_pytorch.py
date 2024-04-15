from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import hls4ml

test_root_path = Path(__file__).parent

in_shape = 16
atol = 5e-3


@pytest.fixture(scope='module')
def data():
    np.random.seed(0)
    X = np.random.rand(100, in_shape)
    return X


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
