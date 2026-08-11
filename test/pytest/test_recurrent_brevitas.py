from pathlib import Path

import brevitas.nn as qnn
import numpy as np
import pytest
import torch
from brevitas.quant import (
    Int8ActPerTensorFixedPoint,
    Int8BiasPerTensorFixedPointInternalScaling,
    Int8WeightPerTensorFixedPoint,
)
from torch import nn

from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model

test_root_path = Path(__file__).parent


class QuantRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = qnn.QuantRNN(
            input_size=10,
            hidden_size=20,
            bidirectional=False,
            shared_input_hidden_weights=False,
            batch_first=True,
            weight_quant=Int8WeightPerTensorFixedPoint,
            bias_quant=Int8BiasPerTensorFixedPointInternalScaling,
            io_quant=Int8ActPerTensorFixedPoint,
            gate_acc_quant=Int8ActPerTensorFixedPoint,
            return_quant_tensor=True,
            bias=True,
        )

    def forward(self, x, h0):
        # brevitas' recurrent layers take the initial states as separate arguments, not as the tuple
        # torch.nn.RNN/LSTM expect
        output, _ = self.rnn(x, h0)
        return output


@pytest.mark.parametrize('backend', ['Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_rnn(backend, io_type):
    model = QuantRNNModel()
    model.eval()

    X_input = torch.randn(1, 1, 10)
    X_input = np.round(X_input * 2**16) * 2**-16  # make it exact ap_fixed<32,16>
    h0 = torch.randn(1, 1, 20)
    h0 = np.round(h0 * 2**16) * 2**-16

    pytorch_prediction = model(torch.Tensor(X_input), torch.Tensor(h0)).detach().value.numpy()

    config = config_from_pytorch_model(
        model,
        [(None, 1, 10), (None, 1, 20)],
        channels_last_conversion='off',
        transpose_outputs=False,
        default_precision='fixed<32,16>',
    )
    output_dir = str(test_root_path / f'hls4mlprj_brevitas_rnn_{backend}_{io_type}')

    hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)

    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict([X_input.detach().numpy(), h0.detach().numpy()]), pytorch_prediction.shape)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, atol=0.3)


class QuantLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = qnn.QuantLSTM(
            input_size=10,
            hidden_size=20,
            bidirectional=False,
            batch_first=True,
            weight_quant=Int8WeightPerTensorFixedPoint,
            bias_quant=Int8BiasPerTensorFixedPointInternalScaling,
            io_quant=Int8ActPerTensorFixedPoint,
            gate_acc_quant=Int8ActPerTensorFixedPoint,
            sigmoid_quant=Int8ActPerTensorFixedPoint,
            tanh_quant=Int8ActPerTensorFixedPoint,
            cell_state_quant=Int8ActPerTensorFixedPoint,
            return_quant_tensor=True,
            bias=True,
        )

    def forward(self, x, h0, c0):
        # QuantLSTM.forward is (inp, hx=None, cx=None); passing a (h0, c0) tuple as torch.nn.LSTM
        # expects would silently bind the tuple to hx and leave cx unset
        output, _ = self.rnn(x, h0, c0)
        return output


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_lstm(backend, io_type):
    model = QuantLSTMModel()
    model.eval()

    X_input = torch.randn(1, 1, 10)
    X_input = np.round(X_input * 2**16) * 2**-16  # make it exact ap_fixed<32,16>
    h0 = torch.randn(1, 1, 20)
    h0 = np.round(h0 * 2**16) * 2**-16
    c0 = torch.randn(1, 1, 20)
    c0 = np.round(c0 * 2**16) * 2**-16

    pytorch_prediction = model(torch.Tensor(X_input), torch.Tensor(h0), torch.Tensor(c0)).detach().value.numpy()

    config = config_from_pytorch_model(
        model,
        [(None, 1, 10), (None, 1, 20), (None, 1, 20)],
        channels_last_conversion='off',
        transpose_outputs=False,
        default_precision='fixed<32,16>',
    )
    output_dir = str(test_root_path / f'hls4mlprj_brevitas_lstm_{backend}_{io_type}')

    hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)

    hls_model.compile()

    hls_prediction = np.reshape(
        hls_model.predict([X_input.detach().numpy(), h0.detach().numpy(), c0.detach().numpy()]),
        pytorch_prediction.shape,
    )

    # measured worst case is 0.023 (3 LSBs of the 8 bit io_quant) over several seeds and backends;
    # the tolerance matches the unquantized LSTM test in test_recurrent_pytorch.py
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, atol=1e-1)
