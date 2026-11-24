from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import hls4ml

test_root_path = Path(__file__).parent

atol = 0.03
# LayerNorm uses lookup tables; 5e-2 matches test_layernorm_pytorch.py
atol_overrides = {'layernorm_on_seq': 5e-2}


class LinearOnSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(5, 3)
        self.l2 = nn.Linear(3, 2)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


class LNOnSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(8)

    def forward(self, x):
        return self.ln(x)


class GRUSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(5, 4, batch_first=True)

    def forward(self, x):
        return self.rnn(x)[0]


class GRUHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(5, 4, batch_first=True)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h[-1])


class Conv1dNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv1d(2, 3, 3)
        self.fc = nn.Linear(3 * 6, 2)

    def forward(self, x):
        return self.fc(torch.flatten(torch.relu(self.c(x)), 1))


class Conv2dOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(2, 3, 3)

    def forward(self, x):
        return torch.relu(self.c(x))


class Pool2dOnly(nn.Module):
    def forward(self, x):
        return nn.functional.max_pool2d(x, 2)


class ElementwiseND(nn.Module):
    def forward(self, x):
        return torch.relu(x) * 0.5 + 0.25


class ConvThenGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv1d(2, 3, 3)
        self.rnn = nn.GRU(3, 4, batch_first=True)

    def forward(self, x):
        return self.rnn(torch.relu(self.c(x)).permute(0, 2, 1))[0]


class ConvThenTokenLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv1d(2, 3, 3)
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(torch.relu(self.c(x)).permute(0, 2, 1))


class TokenLinearThenConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 3)
        self.c = nn.Conv1d(3, 2, 3)

    def forward(self, x):
        return self.c(torch.relu(self.fc(x)).permute(0, 2, 1))


# name: (constructor, input shape without batch, default precision)
models = {
    'mlp_1d': (lambda: nn.Sequential(nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 2)), (5,), 'ap_fixed<24,10>'),
    'linear_on_seq': (LinearOnSeq, (4, 5), 'ap_fixed<24,10>'),
    'layernorm_on_seq': (LNOnSeq, (4, 8), 'ap_fixed<24,10>'),
    'gru_seq': (GRUSeq, (6, 5), 'ap_fixed<32,16>'),
    'gru_head': (GRUHead, (6, 5), 'ap_fixed<32,16>'),
    'conv1d_flatten_linear': (Conv1dNet, (2, 8), 'ap_fixed<24,10>'),
    'conv2d': (Conv2dOnly, (2, 5, 5), 'ap_fixed<24,10>'),
    'maxpool2d': (Pool2dOnly, (2, 4, 4), 'ap_fixed<24,10>'),
    'elementwise_nd': (ElementwiseND, (3, 4), 'ap_fixed<24,10>'),
    'conv_then_gru': (ConvThenGRU, (2, 8), 'ap_fixed<32,16>'),
    'conv_then_token_linear': (ConvThenTokenLinear, (2, 8), 'ap_fixed<24,10>'),
    'token_linear_then_conv': (TokenLinearThenConv, (8, 5), 'ap_fixed<24,10>'),
}

# Known failures with the current ChannelsLastConverter, keyed by (model, mode, io_type).
known_failures = {}
for _name in ['conv_then_gru', 'conv_then_token_linear', 'token_linear_then_conv']:
    for _mode in ['off', 'internal', 'full']:
        for _io_type in ['io_parallel', 'io_stream']:
            known_failures[(_name, _mode, _io_type)] = 'parse error: Unsupported function permute'
for _mode in ['internal', 'full']:
    known_failures[('linear_on_seq', _mode, 'io_parallel')] = 'crash in im2col codegen: Linear input rotated'
    known_failures[('linear_on_seq', _mode, 'io_stream')] = 'wrong output size: Linear input rotated'
    for _name in ['gru_seq', 'gru_head']:
        for _io_type in ['io_parallel', 'io_stream']:
            known_failures[(_name, _mode, _io_type)] = 'wrong numbers: batch-first GRU input rotated'
for _name in ['conv1d_flatten_linear', 'conv2d']:
    known_failures[(_name, 'off', 'io_parallel')] = 'crash in im2col codegen: channels-first conv unsupported'
    known_failures[(_name, 'off', 'io_stream')] = 'compile error: no conv_*_cf kernel'
known_failures[('maxpool2d', 'off', 'io_stream')] = 'compile error: no channels-first streaming pooling'

vitis_models = ['linear_on_seq', 'gru_seq', 'conv1d_flatten_linear']
# LayerNorm has no io_stream kernel on any backend
io_parallel_only = ['layernorm_on_seq']


def to_channels_last(x):
    """(N, C, *S) -> (N, *S, C), the layout the old pass uses internally."""
    return np.moveaxis(x, 1, -1) if x.ndim > 2 else x


def count_transposes(hls_model):
    return sum(layer.class_name == 'Transpose' for layer in hls_model.get_layers())


def _cases():
    cases = []
    for backend in ['Vivado', 'Vitis']:
        for name in models:
            if backend == 'Vitis' and name not in vitis_models:
                continue
            for mode in ['off', 'internal', 'full']:
                for io_type in ['io_parallel', 'io_stream']:
                    if io_type == 'io_stream' and (backend == 'Vitis' or name in io_parallel_only):
                        continue
                    reason = known_failures.get((name, mode, io_type))
                    marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
                    cases.append(pytest.param(name, mode, io_type, backend, marks=marks))
    return cases


@pytest.mark.parametrize('name, mode, io_type, backend', _cases())
def test_layout_scoreboard(test_case_id, name, mode, io_type, backend):
    ctor, shape, precision = models[name]
    torch.manual_seed(0)
    model = ctor().eval()

    rng = np.random.default_rng(0)
    X = np.round(rng.uniform(-1, 1, (8, *shape)) * 2**8) * 2**-8
    y_ref = model(torch.tensor(X, dtype=torch.float32)).detach().numpy()

    config = hls4ml.utils.config_from_pytorch_model(
        model,
        shape,
        default_precision=precision,
        channels_last_conversion=mode,
        transpose_outputs=False,
        backend=backend,
    )
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model, hls_config=config, output_dir=str(test_root_path / test_case_id), backend=backend, io_type=io_type
    )
    hls_model.compile()

    x_in = to_channels_last(X) if mode == 'internal' else X
    y_hls = np.asarray(hls_model.predict(np.ascontiguousarray(x_in))).reshape(len(X), -1)

    # With transpose_outputs=False the old pass leaves outputs channels-last, so accept either layout.
    tol = atol_overrides.get(name, atol)
    y_source = y_ref.reshape(len(X), -1)
    y_channels_last = to_channels_last(y_ref).reshape(len(X), -1)
    assert np.allclose(y_hls, y_source, atol=tol) or np.allclose(y_hls, y_channels_last, atol=tol), (
        f'max abs error {np.abs(y_hls - y_source).max():.3g}'
    )
