import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFixedPoint

import numpy as np
import pytest

from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model

test_root_path = Path(__file__).parent

class QuantModelConv2d(Module):
    def __init__(self):
        super(QuantModelConv2d, self).__init__()
        self.conv1 = qnn.QuantConv2d(3, 6, 5, bias=True, weight_quant=Int8WeightPerTensorFixedPoint)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        return out

class QuantModelLinear(Module):
    def __init__(self):
        super(QuantModelLinear, self).__init__()
        self.conv1 = qnn.QuantLinear(4, 4, bias=True, weight_quant=Int8WeightPerTensorFixedPoint)
        self.relu1 = qnn.QuantReLU()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        return out

@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_quantlinear(backend, io_type):
    model = QuantModelLinear()

    x = torch.tensor([1.,2.,3.,4.])

    pytorch_prediction = model(x).detach().numpy()
    config = config_from_pytorch_model(model)
    output_dir = str(test_root_path / f'hls4mlprj_brevitas_linear_{backend}_{io_type}')

    hls_model = convert_from_pytorch_model(
        model,
        (None, 4),
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
    )
    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict(x.detach().numpy()), pytorch_prediction.shape)
 
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)

@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_quantconv2d(backend, io_type):
    model = QuantModelConv2d()

    x = torch.randn(1,3,6,5)

    pytorch_prediction = model(x).detach().numpy()
    config = config_from_pytorch_model(model, inputs_channel_last=False,transpose_outputs=True)
    if io_type == 'io_stream':
        x = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
        config = config_from_pytorch_model(model, inputs_channel_last=True, transpose_outputs=False)
    else:
        config = config_from_pytorch_model(model, inputs_channel_last=False, transpose_outputs=True)

    output_dir = str(test_root_path / f'hls4mlprj_brevitas_linear_{backend}_{io_type}')

    hls_model = convert_from_pytorch_model(
        model,
        (None, 3,6,5),
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
    )
    hls_model.compile()

    if io_type == 'io_stream':
        hls_prediction = np.transpose(
            np.reshape(hls_model.predict(x.detach().numpy()), pytorch_prediction.shape), (0, 3, 1, 2)
        )
    else:
        hls_prediction = np.reshape(hls_model.predict(x.detach().numpy()), pytorch_prediction.shape)
  
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)
