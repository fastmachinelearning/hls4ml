from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model

test_root_path = Path(__file__).parent


class GRUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(10, 20, num_layers=1, batch_first=True)

    def forward(self, x, h0):
        output, hnn = self.rnn(x, h0)
        return output


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_gru(backend, io_type):
    model = GRUNet()
    model.eval()

    X_input = torch.randn(1, 1, 10)
    h0 = torch.zeros(1, 1, 20)

    pytorch_prediction = model(torch.Tensor(X_input), torch.Tensor(h0)).detach().numpy()

    config = config_from_pytorch_model(model, inputs_channel_last=True, transpose_outputs=False)
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_linear_{backend}_{io_type}')

    hls_model = convert_from_pytorch_model(
        model, [(None, 10, 1), (None, 20, 1)], hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )

    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict([X_input.detach().numpy(), h0.detach().numpy()]), (1, 1, 20))

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=5e-2)
