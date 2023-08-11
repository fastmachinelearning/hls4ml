from pathlib import Path

import pytest
import torch.nn as nn

from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model

test_root_path = Path(__file__).parent

# simple model with unnamed sequential
model = nn.Sequential(nn.Conv2d(1, 20, 5), nn.ReLU(), nn.Conv2d(20, 64, 5), nn.ReLU())


# simple model with namend sequential
class SeqModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(nn.Conv2d(1, 20, 5), nn.ReLU(), nn.Conv2d(20, 64, 5), nn.ReLU())

    def forward(self, x):
        output = self.layer(x)
        return output


@pytest.mark.parametrize('backend', ['Vivado'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_named(backend, io_type):
    config = config_from_pytorch_model(model)
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_gru_{backend}_{io_type}')

    convert_from_pytorch_model(
        model, (None, 1, 5, 5), hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )


@pytest.mark.parametrize('backend', ['Vivado'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_unnnamed(backend, io_type):
    pytorch_model = SeqModel()
    config = config_from_pytorch_model(pytorch_model)
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_gru_{backend}_{io_type}')

    convert_from_pytorch_model(
        pytorch_model, (None, 1, 5, 5), hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )
