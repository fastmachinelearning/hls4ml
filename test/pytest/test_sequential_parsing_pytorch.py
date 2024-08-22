from collections import OrderedDict
from pathlib import Path

import pytest
import torch.nn as nn

from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model

test_root_path = Path(__file__).parent

# Model with unnamed Sequential and no named layers
seq_unnamed = nn.Sequential(nn.Conv2d(1, 20, 5), nn.ReLU(), nn.Conv2d(20, 64, 5), nn.ReLU())

# Model with unnamed Sequential and named layers
seq_named = nn.Sequential(
    OrderedDict(
        [('conv_1', nn.Conv2d(1, 20, 5)), ('relu_1', nn.ReLU()), ('conv_2', nn.Conv2d(20, 64, 5)), ('relu_2', nn.ReLU())]
    )
)


# Model with named Sequential and no named layers
class SeqModelUnnamedLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(nn.Conv2d(1, 20, 5), nn.ReLU(), nn.Conv2d(20, 64, 5), nn.ReLU())

    def forward(self, x):
        output = self.layer(x)
        return output


# Model with named Sequential and named layers
class SeqModelNamedLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            OrderedDict(
                [
                    ('conv_1', nn.Conv2d(1, 20, 5)),
                    ('relu_1', nn.ReLU()),
                    ('conv_2', nn.Conv2d(20, 64, 5)),
                    ('relu_2', nn.ReLU()),
                ]
            )
        )

    def forward(self, x):
        output = self.layer(x)
        return output


@pytest.mark.parametrize('backend', ['Vivado'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
@pytest.mark.parametrize('named_layers', [True, False])
def test_unnamed_seq(backend, io_type, named_layers):
    if named_layers:
        model = seq_named
    else:
        model = seq_unnamed
    config = config_from_pytorch_model(model, (1, 5, 5))
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_seq_unnamed_{backend}_{io_type}_{named_layers}')

    convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)


@pytest.mark.parametrize('backend', ['Vivado'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
@pytest.mark.parametrize('named_layers', [True, False])
def test_named_seq(backend, io_type, named_layers):
    if named_layers:
        model = SeqModelNamedLayers()
    else:
        model = SeqModelUnnamedLayers()
    config = config_from_pytorch_model(model, (1, 5, 5))
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_seq_named_{backend}_{io_type}_{named_layers}')

    convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)
