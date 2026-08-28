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
        self.rnn = nn.GRU(10, 20, num_layers=1, batch_first=True, bias=True)

    def forward(self, x, h0):
        output, hnn = self.rnn(x, h0)
        return output


class GRUNetStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(10, 20, num_layers=1, batch_first=True, bias=True)

    def forward(self, x):
        output, hnn = self.rnn(x)
        return output


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_gru(test_case_id, backend, io_type):
    model = GRUNet()
    model.eval()

    X_input = torch.randn(1, 1, 10)
    X_input = np.round(X_input * 2**16) * 2**-16  # make it exact ap_fixed<32,16>
    h0 = torch.randn(1, 1, 20)
    h0 = np.round(h0 * 2**16) * 2**-16

    pytorch_prediction = model(torch.Tensor(X_input), torch.Tensor(h0)).detach().numpy()

    config = config_from_pytorch_model(
        model,
        [(None, 1, 10), (None, 1, 20)],
        channels_last_conversion='off',
        transpose_outputs=False,
        default_precision='fixed<32,16>',
    )
    output_dir = str(test_root_path / test_case_id)

    hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)

    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict([X_input.detach().numpy(), h0.detach().numpy()]), (1, 1, 20))

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=1e-1)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_stream'])
def test_gru_stream(test_case_id, backend, io_type):
    model = GRUNetStream()
    model.eval()

    X_input = torch.randn(1, 1, 10)
    X_input = np.round(X_input * 2**16) * 2**-16  # make it exact ap_fixed<32,16>

    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(
        model, (None, 1, 10), channels_last_conversion='off', transpose_outputs=False, default_precision='fixed<32,16>'
    )
    output_dir = str(test_root_path / test_case_id)

    hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)

    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict(X_input.detach().numpy()), (1, 1, 20))

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=1e-1)


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(10, 20, num_layers=1, batch_first=True, bias=True)

    def forward(self, x, h0, c0):
        output, (_, _) = self.rnn(x, (h0, c0))
        return output


class LSTMStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(10, 20, num_layers=1, batch_first=True, bias=True)

    def forward(self, x):
        output, (_, _) = self.rnn(x)
        return output


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_lstm(test_case_id, backend, io_type):
    model = LSTM()
    model.eval()

    X_input = torch.randn(1, 1, 10)
    X_input = np.round(X_input * 2**16) * 2**-16  # make it exact ap_fixed<32,16>
    h0 = torch.randn(1, 1, 20)
    h0 = np.round(h0 * 2**16) * 2**-16
    c0 = torch.randn(1, 1, 20)
    c0 = np.round(c0 * 2**16) * 2**-16

    pytorch_prediction = model(torch.Tensor(X_input), torch.Tensor(h0), torch.tensor(c0)).detach().numpy()

    config = config_from_pytorch_model(
        model,
        [(None, 1, 10), (None, 1, 20), (None, 1, 20)],
        channels_last_conversion='off',
        transpose_outputs=False,
        default_precision='fixed<32,16>',
    )
    output_dir = str(test_root_path / test_case_id)

    hls_model = convert_from_pytorch_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
    )

    hls_model.compile()

    hls_prediction = np.reshape(
        hls_model.predict([X_input.detach().numpy(), h0.detach().numpy(), c0.detach().numpy()]), (1, 1, 20)
    )

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=1e-1)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_stream'])
def test_lstm_stream(test_case_id, backend, io_type):
    if not (backend in ('Quartus', 'oneAPI') and io_type == 'io_stream'):
        model = LSTMStream()
        model.eval()

        X_input = torch.randn(1, 1, 10)
        X_input = np.round(X_input * 2**16) * 2**-16  # make it exact ap_fixed<32,16>

        pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

        config = config_from_pytorch_model(
            model, [(None, 1, 10)], channels_last_conversion='off', transpose_outputs=False, default_precision='fixed<32,16>'
        )
        output_dir = str(test_root_path / test_case_id)

        hls_model = convert_from_pytorch_model(
            model,
            hls_config=config,
            output_dir=output_dir,
            backend=backend,
            io_type=io_type,
        )

        hls_model.compile()

        hls_prediction = np.reshape(hls_model.predict(X_input.detach().numpy()), (1, 1, 20))

        np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=1e-1)


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(10, 20, num_layers=1, batch_first=True, bias=True)

    def forward(self, x, h0):
        output, _ = self.rnn(x, h0)
        return output


@pytest.mark.parametrize('backend', ['Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_rnn(test_case_id, backend, io_type):
    if not (backend in ('Quartus', 'oneAPI') and io_type == 'io_stream'):
        model = RNN()
        model.eval()

        X_input = torch.randn(1, 1, 10)
        X_input = np.round(X_input * 2**16) * 2**-16  # make it exact ap_fixed<32,16>
        h0 = torch.zeros(1, 1, 20)

        pytorch_prediction = model(torch.Tensor(X_input), torch.Tensor(h0)).detach().numpy()

        config = config_from_pytorch_model(
            model,
            [(1, 10), (1, 20)],
            channels_last_conversion='off',
            transpose_outputs=False,
            default_precision='fixed<32,16>',
        )
        output_dir = str(test_root_path / test_case_id)

        hls_model = convert_from_pytorch_model(
            model,
            hls_config=config,
            output_dir=output_dir,
            backend=backend,
            io_type=io_type,
        )

        hls_model.compile()

        hls_prediction = np.reshape(hls_model.predict([X_input.detach().numpy(), h0.detach().numpy()]), (1, 1, 20))

        np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=1e-1)


# ---------------------------------------------------------------------------- #
# Which element of the RNN output tuple the model consumes decides between the
# full output sequence and the final hidden state; unrepresentable uses raise.
# ---------------------------------------------------------------------------- #

n_in = 8
n_hidden = 12
n_timesteps = 3


class SequenceModel(nn.Module):
    """Consumes the full output sequence (element 0 of the RNN output tuple)."""

    def __init__(self, rnn_cls):
        super().__init__()
        self.rnn = rnn_cls(n_in, n_hidden, num_layers=1, batch_first=True)

    def forward(self, x):
        output, _ = self.rnn(x)
        return output


class StackedSequenceModel(nn.Module):
    """Feeds the output sequence of one recurrent layer into another."""

    def __init__(self):
        super().__init__()
        self.rnn1 = nn.GRU(n_in, n_hidden, num_layers=1, batch_first=True)
        self.rnn2 = nn.GRU(n_hidden, n_hidden, num_layers=1, batch_first=True)

    def forward(self, x):
        output, _ = self.rnn1(x)
        output, _ = self.rnn2(output)
        return output


class GRUStateModel(nn.Module):
    """Consumes the final hidden state (element 1 of the RNN output tuple)."""

    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(n_in, n_hidden, num_layers=1, batch_first=True)

    def forward(self, x):
        _, state = self.rnn(x)
        return state


class LSTMStateModel(nn.Module):
    """Consumes the final hidden state of an LSTM (element [1][0] of the output tuple)."""

    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(n_in, n_hidden, num_layers=1, batch_first=True)

    def forward(self, x):
        _, (hidden_state, _) = self.rnn(x)
        return hidden_state


def _quantized_random(shape):
    x = np.random.rand(*shape) - 0.5
    return np.round(x * 2**16) * 2**-16  # make it exact fixed<32,16>


def _convert(model, test_case_id, backend, input_shape=(None, n_timesteps, n_in)):
    config = config_from_pytorch_model(
        model,
        input_shape,
        channels_last_conversion='off',
        transpose_outputs=False,
        default_precision='fixed<32,16>',
    )
    output_dir = str(test_root_path / test_case_id)
    return convert_from_pytorch_model(
        model, hls_config=config, output_dir=output_dir, backend=backend, io_type='io_parallel'
    )


@pytest.mark.parametrize(
    'rnn_cls,backend',
    [(nn.GRU, 'Vivado'), (nn.LSTM, 'Vivado'), (nn.RNN, 'Quartus')],
    ids=['gru', 'lstm', 'rnn'],
)
def test_rnn_sequence_output(test_case_id, rnn_cls, backend):
    model = SequenceModel(rnn_cls)
    model.eval()

    X_input = _quantized_random((5, n_timesteps, n_in))
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    hls_model = _convert(model, test_case_id, backend)
    assert list(hls_model.get_layers())[1].attributes['return_sequences'] is True
    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict(X_input), pytorch_prediction.shape)
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=2e-2)


def test_rnn_stacked_sequence_output(test_case_id):
    model = StackedSequenceModel()
    model.eval()

    X_input = _quantized_random((5, n_timesteps, n_in))
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    hls_model = _convert(model, test_case_id, 'Vivado')
    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict(X_input), pytorch_prediction.shape)
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=2e-2)


@pytest.mark.parametrize('state_model', [GRUStateModel, LSTMStateModel], ids=['gru', 'lstm'])
def test_rnn_state_output(test_case_id, state_model):
    model = state_model()
    model.eval()

    X_input = _quantized_random((5, n_timesteps, n_in))
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()[0]  # (num_layers, batch, hidden) -> (batch, hidden)

    hls_model = _convert(model, test_case_id, 'Vivado')
    assert list(hls_model.get_layers())[1].attributes['return_sequences'] is False
    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict(X_input), pytorch_prediction.shape)
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=2e-2)


def test_lstm_cell_state_rejected(test_case_id):
    class CellStateModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(n_in, n_hidden, num_layers=1, batch_first=True)

        def forward(self, x):
            _, (_, cell_state) = self.rnn(x)
            return cell_state

    model = CellStateModel()
    model.eval()

    with pytest.raises(NotImplementedError, match='cell state'):
        _convert(model, test_case_id, 'Vivado')


def test_rnn_sequence_and_state_rejected(test_case_id):
    class BothOutputsModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(n_in, n_hidden, num_layers=1, batch_first=True)

        def forward(self, x):
            output, state = self.rnn(x)
            return output, state

    model = BothOutputsModel()
    model.eval()

    with pytest.raises(NotImplementedError, match='both the output sequence and the final state'):
        _convert(model, test_case_id, 'Vivado')


def test_rnn_sliced_output_rejected(test_case_id):
    class SlicedOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(n_in, n_hidden, num_layers=1, batch_first=True)

        def forward(self, x):
            output, _ = self.rnn(x)
            return output[:, -1, :]

    model = SlicedOutputModel()
    model.eval()

    with pytest.raises(NotImplementedError, match='slicing'):
        _convert(model, test_case_id, 'Vivado')


def test_lstm_proj_size_rejected(test_case_id):
    class ProjectedLSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(n_in, n_hidden, num_layers=1, batch_first=True, proj_size=4)

        def forward(self, x):
            _, (hidden_state, _) = self.rnn(x)
            return hidden_state

    model = ProjectedLSTMModel()
    model.eval()

    with pytest.raises(NotImplementedError, match='proj_size'):
        _convert(model, test_case_id, 'Vivado')


@pytest.mark.parametrize('rnn_cls', [nn.RNN, nn.LSTM, nn.GRU], ids=['rnn', 'lstm', 'gru'])
def test_rnn_no_bias(test_case_id, rnn_cls):
    class NoBiasModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = rnn_cls(n_in, n_hidden, num_layers=1, batch_first=True, bias=False)

        def forward(self, x):
            output, _ = self.rnn(x)
            return output

    model = NoBiasModel()
    model.eval()

    X_input = _quantized_random((5, n_timesteps, n_in))
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    backend = 'Quartus' if rnn_cls is nn.RNN else 'Vivado'
    hls_model = _convert(model, test_case_id, backend)
    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict(X_input), pytorch_prediction.shape)
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=2e-2)


def test_rnn_sequence_functional_consumer(test_case_id):
    """A functional call consuming the RNN output tuple resolves to the sequence output."""

    class FunctionalConsumerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(n_in, n_hidden, num_layers=1, batch_first=True)

        def forward(self, x):
            output, _ = self.rnn(x)
            return torch.nn.functional.relu(output)

    model = FunctionalConsumerModel()
    model.eval()

    X_input = _quantized_random((5, n_timesteps, n_in))
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    hls_model = _convert(model, test_case_id, 'Vivado')
    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict(X_input), pytorch_prediction.shape)
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=2e-2)


def test_rnn_sequence_method_consumer(test_case_id):
    """A method call consuming the RNN output tuple resolves to the sequence output."""

    class MethodConsumerModel(nn.Module):
        # note: no Linear here on purpose — with channels_last_conversion='off' PyTorch Dense
        # weights are never transposed (pre-existing upstream bug, reported separately)
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(n_in, n_hidden, num_layers=1, batch_first=True)

        def forward(self, x):
            output, _ = self.rnn(x)
            return torch.nn.functional.relu(output.flatten(1))

    model = MethodConsumerModel()
    model.eval()

    X_input = _quantized_random((5, n_timesteps, n_in))
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    hls_model = _convert(model, test_case_id, 'Vivado')
    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict(X_input), pytorch_prediction.shape)
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=2e-2)


def test_rnn_state_as_initial_state(test_case_id):
    """The final state of one RNN used as the initial state of another parses with both inputs wired."""

    class StateChainModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru1 = nn.GRU(n_in, n_hidden, num_layers=1, batch_first=True)
            self.gru2 = nn.GRU(n_in, n_hidden, num_layers=1, batch_first=True)

        def forward(self, x):
            _, state = self.gru1(x)
            output, _ = self.gru2(x, state)
            return output

    model = StateChainModel()
    model.eval()

    hls_model = _convert(model, test_case_id, 'Vivado')

    gru2_layer = next(layer for layer in hls_model.get_layers() if layer.name == 'gru2')
    assert gru2_layer.attributes['pass_initial_states'] is True
    assert gru2_layer.attributes['return_sequences'] is True
    gru1_layer = next(layer for layer in hls_model.get_layers() if layer.name == 'gru1')
    assert gru1_layer.attributes['return_sequences'] is False


def test_rnn_output_tuple_returned(test_case_id):
    """A model that returns the RNN output tuple unchanged parses as a final-state output."""

    class TupleReturnModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(n_in, n_hidden, num_layers=1, batch_first=True)

        def forward(self, x):
            return self.rnn(x)

    model = TupleReturnModel()
    model.eval()

    hls_model = _convert(model, test_case_id, 'Vivado')
    assert list(hls_model.get_layers())[1].attributes['return_sequences'] is False
