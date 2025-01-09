from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import hls4ml

test_root_path = Path(__file__).parent

in_shape = 16
atol = 5e-3


@pytest.fixture
def data():
    np.random.seed(0)
    X = np.random.rand(100, in_shape)
    return X


@pytest.fixture(scope='module')
def fusion_data():
    n_batch = 2
    n_in = 2
    size_in_height = 32
    X = np.random.rand(n_batch, n_in, size_in_height)
    return X


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'Catapult'])
def test_batchnorm(data, backend, io_type):
    model = nn.Sequential(
        nn.BatchNorm1d(in_shape),
    ).to()
    model.eval()

    default_precision = 'ac_fixed<32, 1, true>' if backend == 'Quartus' else 'ac_fixed<32, 1>'

    config = hls4ml.utils.config_from_pytorch_model(
        model, (in_shape,), default_precision=default_precision, granularity='name', backend=backend
    )
    output_dir = str(test_root_path / f'hls4mlprj_batchnorm_{backend}_{io_type}')
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model, backend=backend, hls_config=config, io_type=io_type, output_dir=output_dir
    )
    hls_model.compile()

    # Predict
    pytorch_prediction = model(torch.Tensor(data)).detach().numpy()
    hls_prediction = hls_model.predict(data)
    np.testing.assert_allclose(pytorch_prediction, hls_prediction, rtol=0, atol=atol, verbose=True)


atol = 5e-2


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
        self.bn1 = nn.BatchNorm1d(filters)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'Catapult'])
def test_batchnorm_fusion(fusion_data, backend, io_type):
    n_in = 2
    momentum = 0.99
    size_in_height = 32
    filters = n_in

    # see above for model definition
    model = BatchNorm_w_Fusion(filters, momentum)
    # Important to set model to eval to fix batchnorm behavior
    model.eval()
    # generating config
    pytorch_prediction = model(torch.Tensor(fusion_data)).detach().numpy()

    # We do not have an implementation of a transpose for io_stream, need to transpose inputs and outputs outside of hls4ml
    if io_type == 'io_stream':
        fusion_data = np.ascontiguousarray(fusion_data.transpose(0, 2, 1))
        config = hls4ml.utils.config_from_pytorch_model(
            model, (n_in, size_in_height), channels_last_conversion='internal', transpose_outputs=False
        )
    else:
        config = hls4ml.utils.config_from_pytorch_model(
            model, (n_in, size_in_height), channels_last_conversion='full', transpose_outputs=True
        )

    config['Model']['Strategy'] = 'Resource'

    # conversion
    output_dir = str(test_root_path / f'hls4mlprj_block_{backend}_{io_type}')
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
    )

    # compiling model
    hls_model.compile()

    if io_type == 'io_stream':
        hls_prediction = np.transpose(
            np.reshape(
                hls_model.predict(fusion_data),
                (pytorch_prediction.shape[0], pytorch_prediction.shape[2], pytorch_prediction.shape[1]),
            ),
            (0, 2, 1),
        )
    else:
        hls_prediction = np.reshape(hls_model.predict(fusion_data), pytorch_prediction.shape)
    np.testing.assert_allclose(pytorch_prediction, hls_prediction, rtol=0, atol=atol, verbose=True)
