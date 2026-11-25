from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import hls4ml
from hls4ml.model.optimizer.passes.layout_inference import transpose_perm
from hls4ml.utils.transpose_utils import to_native_layout, to_source_layout

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


class LSTMSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(5, 4, batch_first=True)

    def forward(self, x):
        return self.rnn(x)[0]


class LSTMHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(5, 4, batch_first=True)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        _, (h, _) = self.rnn(x)
        return self.fc(h[-1])


class ConvBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv1d(2, 3, 3)
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x):
        return self.bn(self.c(x))


class ConcatBranches(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(2, 3, 3)
        self.c2 = nn.Conv1d(2, 2, 3)

    def forward(self, x):
        return torch.cat([self.c1(x), self.c2(x)], dim=1)


class ConvSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv1d(2, 3, 3)

    def forward(self, x):
        return nn.functional.softmax(self.c(x), dim=-1)


class FanOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv1d(4, 3, 3)
        self.rnn = nn.GRU(6, 3, batch_first=True)

    def forward(self, x):
        return self.c(x), self.rnn(x)[0]


class TwoInputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(2, 3, 3)
        self.rnn = nn.GRU(5, 4, batch_first=True)
        self.fc = nn.Linear(3 * 3 * 3 + 4, 2)

    def forward(self, image, seq):
        _, h = self.rnn(seq)
        return self.fc(torch.cat([torch.flatten(self.c(image), 1), h[-1]], dim=1))


# name: (constructor, input shapes without batch, default precision)
models = {
    'mlp_1d': (lambda: nn.Sequential(nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 2)), [(5,)], 'ap_fixed<24,10>'),
    'linear_on_seq': (LinearOnSeq, [(4, 5)], 'ap_fixed<24,10>'),
    'layernorm_on_seq': (LNOnSeq, [(4, 8)], 'ap_fixed<24,10>'),
    'gru_seq': (GRUSeq, [(6, 5)], 'ap_fixed<32,16>'),
    'gru_head': (GRUHead, [(6, 5)], 'ap_fixed<32,16>'),
    'lstm_seq': (LSTMSeq, [(6, 5)], 'ap_fixed<32,16>'),
    'lstm_head': (LSTMHead, [(6, 5)], 'ap_fixed<32,16>'),
    'elementwise_nd': (ElementwiseND, [(3, 4)], 'ap_fixed<24,10>'),
    'conv1d_flatten_linear': (Conv1dNet, [(2, 8)], 'ap_fixed<24,10>'),
    'conv2d': (Conv2dOnly, [(2, 5, 5)], 'ap_fixed<24,10>'),
    'maxpool2d': (Pool2dOnly, [(2, 4, 4)], 'ap_fixed<24,10>'),
    'upsample2d': (lambda: nn.Upsample(scale_factor=2), [(2, 4, 4)], 'ap_fixed<24,10>'),
    'zeropad2d': (lambda: nn.ConstantPad2d(1, 0), [(2, 4, 4)], 'ap_fixed<24,10>'),
    'conv1d_bn': (ConvBN, [(2, 8)], 'ap_fixed<24,10>'),
    'concat_branches': (ConcatBranches, [(2, 8)], 'ap_fixed<24,10>'),
    'conv1d_softmax': (ConvSoftmax, [(2, 8)], 'ap_fixed<24,10>'),
    'conv_then_gru': (ConvThenGRU, [(2, 8)], 'ap_fixed<32,16>'),
    'conv_then_token_linear': (ConvThenTokenLinear, [(2, 8)], 'ap_fixed<24,10>'),
    'token_linear_then_conv': (TokenLinearThenConv, [(8, 5)], 'ap_fixed<24,10>'),
    'fan_out': (FanOut, [(4, 6)], 'ap_fixed<32,16>'),
    'two_inputs': (TwoInputs, [(2, 5, 5), (6, 5)], 'ap_fixed<32,16>'),
}

atol = 0.03
# LayerNorm and Softmax use lookup tables
atol_overrides = {'layernorm_on_seq': 5e-2, 'conv1d_softmax': 5e-2}

# Transpose layers left in the converted model with 'native' layouts at both ends, in io_parallel
expected_transposes = {name: 0 for name in models}
expected_transposes.update({'conv1d_softmax': 1, 'fan_out': 1})
# io_stream cannot fold the Transpose before Flatten into the next Dense layer
expected_transposes_stream = {'conv1d_flatten_linear': 1, 'two_inputs': 1}

# LayerNorm has no io_stream kernel on any backend
io_parallel_only = ['layernorm_on_seq']
vitis_models = ['conv1d_flatten_linear', 'gru_seq', 'conv_then_gru']


def count_transposes(hls_model):
    return sum(layer.class_name == 'Transpose' for layer in hls_model.get_layers())


def _cases():
    cases = []
    for name in models:
        for io_type in ['io_parallel', 'io_stream']:
            if io_type == 'io_stream' and name in io_parallel_only:
                continue
            for layout in ['source', 'native']:
                cases.append(pytest.param(name, io_type, layout, 'Vivado'))
    for name in vitis_models:
        cases.append(pytest.param(name, 'io_parallel', 'source', 'Vitis'))
    return cases


def _convert(model, input_shapes, precision, output_dir, backend='Vivado', io_type='io_parallel', **layout_args):
    config = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shapes if len(input_shapes) > 1 else input_shapes[0],
        default_precision=precision,
        backend=backend,
        **layout_args,
    )
    return hls4ml.converters.convert_from_pytorch_model(
        model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )


@pytest.mark.parametrize('name, io_type, layout, backend', _cases())
def test_layout_inference(test_case_id, name, io_type, layout, backend):
    ctor, input_shapes, precision = models[name]
    torch.manual_seed(0)
    model = ctor().eval()

    rng = np.random.default_rng(0)
    X = [np.round(rng.uniform(-1, 1, (8, *shape)) * 2**8) * 2**-8 for shape in input_shapes]
    y_ref = model(*[torch.tensor(x, dtype=torch.float32) for x in X])
    y_ref = [y.detach().numpy() for y in (y_ref if isinstance(y_ref, tuple) else (y_ref,))]

    hls_model = _convert(
        model,
        input_shapes,
        precision,
        str(test_root_path / test_case_id),
        backend=backend,
        io_type=io_type,
        input_layout=layout,
        output_layout=layout,
    )
    hls_model.compile()

    layouts = hls_model.get_io_layouts()
    x_in = [to_native_layout(x, entry['perm']) for x, entry in zip(X, layouts['inputs'])]
    y_hls = hls_model.predict(x_in if len(x_in) > 1 else x_in[0])
    y_hls = y_hls if isinstance(y_hls, list) else [y_hls]
    tol = atol_overrides.get(name, atol)
    for y, ref, entry in zip(y_hls, y_ref, layouts['outputs']):
        y_source = to_source_layout(np.asarray(y).reshape(len(ref), *entry['stored_shape']), entry['perm'])
        np.testing.assert_allclose(y_source, ref, rtol=0, atol=tol)

    expected = expected_transposes[name]
    if io_type == 'io_stream':
        expected = expected_transposes_stream.get(name, expected)
    if layout == 'source':
        expected += sum(entry['perm'] != sorted(entry['perm']) for entry in layouts['inputs'] + layouts['outputs'])
    assert count_transposes(hls_model) == expected
    if layout == 'source':
        assert all(entry['perm'] == sorted(entry['perm']) for entry in layouts['inputs'] + layouts['outputs'])


@pytest.mark.parametrize(
    'state, target, perm',
    [
        ((0, 1), (1, 0), [1, 0]),
        ((1, 0), (0, 1), [1, 0]),
        ((0, 1, 2), (1, 2, 0), [1, 2, 0]),
        ((1, 2, 0), (0, 1, 2), [2, 0, 1]),
        ((2, 0, 1), (1, 2, 0), [2, 0, 1]),
    ],
)
def test_transpose_perm(state, target, perm):
    assert transpose_perm(state, target) == perm
    shape = (2, 3, 4)[: len(state)]
    x = np.arange(np.prod(shape)).reshape(shape)
    stored = np.transpose(x, state)
    np.testing.assert_array_equal(np.transpose(stored, perm), np.transpose(x, target))


@pytest.mark.parametrize(
    'old_args, input_layout, output_layout',
    [
        ({'channels_last_conversion': 'full'}, 'source', 'source'),
        ({'channels_last_conversion': 'internal'}, 'native', 'source'),
        ({'channels_last_conversion': 'off'}, 'source', 'source'),
        ({'transpose_outputs': True}, 'source', 'source'),
        ({'transpose_outputs': False}, 'source', 'native'),
        ({'channels_last_conversion': 'internal', 'transpose_outputs': False}, 'native', 'native'),
    ],
)
def test_deprecated_layout_arguments(old_args, input_layout, output_layout):
    model = Conv2dOnly().eval()
    with pytest.warns(DeprecationWarning):
        config = hls4ml.utils.config_from_pytorch_model(model, (2, 5, 5), **old_args)
    assert config['Model']['InputLayout'] == input_layout
    assert config['Model']['OutputLayout'] == output_layout
    assert config['Model']['LayoutInference'] is True
    assert 'ChannelsLastConversion' not in config['Model']
    assert 'TransposeOutputs' not in config['Model']


def test_default_layout_arguments():
    config = hls4ml.utils.config_from_pytorch_model(Conv2dOnly().eval(), (2, 5, 5))
    assert config['Model']['InputLayout'] == 'source'
    assert config['Model']['OutputLayout'] == 'source'


def test_old_and_new_layout_arguments_conflict():
    with pytest.raises(ValueError):
        hls4ml.utils.config_from_pytorch_model(
            Conv2dOnly().eval(), (2, 5, 5), channels_last_conversion='full', input_layout='native'
        )


def test_hls_config_with_old_keys(test_case_id):
    model = Conv2dOnly().eval()
    config = hls4ml.utils.config_from_pytorch_model(model, (2, 5, 5))
    for key in ['InputLayout', 'OutputLayout', 'LayoutInference']:
        del config['Model'][key]
    config['Model']['ChannelsLastConversion'] = 'internal'
    config['Model']['TransposeOutputs'] = False
    with pytest.warns(DeprecationWarning):
        hls_model = hls4ml.converters.convert_from_pytorch_model(
            model, hls_config=config, output_dir=str(test_root_path / test_case_id)
        )
    layouts = hls_model.get_io_layouts()
    assert layouts['inputs'][0]['layout'] == 'native'
    assert layouts['inputs'][0]['perm'] == [1, 2, 0]
    assert layouts['outputs'][0]['layout'] == 'native'
    assert count_transposes(hls_model) == 0


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_old_pass_fallback(test_case_id, io_type):
    torch.manual_seed(0)
    model = Conv2dOnly().eval()
    X = np.round(np.random.default_rng(0).uniform(-1, 1, (8, 2, 5, 5)) * 2**8) * 2**-8
    y_ref = model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
    config = hls4ml.utils.config_from_pytorch_model(model, (2, 5, 5), default_precision='ap_fixed<24,10>')
    config['Model']['LayoutInference'] = False
    config['Model']['ChannelsLastConversion'] = 'full'
    config['Model']['TransposeOutputs'] = False
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model, hls_config=config, output_dir=str(test_root_path / test_case_id), io_type=io_type
    )
    hls_model.compile()
    y_hls = np.asarray(hls_model.predict(np.ascontiguousarray(X))).reshape(8, 3, 3, 3)
    np.testing.assert_allclose(y_hls, np.moveaxis(y_ref, 1, -1), rtol=0, atol=atol)


def test_old_pass_rejects_permute(test_case_id):
    model = ConvThenTokenLinear().eval()
    config = hls4ml.utils.config_from_pytorch_model(model, (2, 8))
    config['Model']['LayoutInference'] = False
    with pytest.raises(NotImplementedError):
        hls4ml.converters.convert_from_pytorch_model(model, hls_config=config, output_dir=str(test_root_path / test_case_id))


class Permute(nn.Module):
    def __init__(self, fn, n_in):
        super().__init__()
        self.fn = fn
        self.fc = nn.Linear(n_in, 3)

    def forward(self, x):
        return self.fc(self.fn(x))


permute_forms = {
    'method_args': (lambda x: x.permute(0, 2, 1), (4, 6), [1, 0]),
    'method_tuple': (lambda x: x.permute((0, 2, 1)), (4, 6), [1, 0]),
    'torch_permute': (lambda x: torch.permute(x, (0, 2, 1)), (4, 6), [1, 0]),
    'method_transpose': (lambda x: x.transpose(1, 2), (4, 6), [1, 0]),
    'torch_transpose': (lambda x: torch.transpose(x, 1, 2), (4, 6), [1, 0]),
    'negative_dims': (lambda x: x.permute(0, -1, -2), (4, 6), [1, 0]),
    'swapaxes': (lambda x: x.swapaxes(1, 2), (4, 6), [1, 0]),
    'rank3': (lambda x: x.permute(0, 3, 1, 2), (2, 3, 4), [2, 0, 1]),
}


@pytest.mark.parametrize('form', list(permute_forms))
def test_permute_parsing(test_case_id, form):
    fn, shape, perm = permute_forms[form]
    model = Permute(fn, shape[-2] if form != 'rank3' else shape[1]).eval()
    config = hls4ml.utils.config_from_pytorch_model(model, shape)
    config['Model']['LayoutInference'] = False
    config['Model']['ChannelsLastConversion'] = 'off'
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model, hls_config=config, output_dir=str(test_root_path / test_case_id)
    )
    assert [layer.get_attr('perm') for layer in hls_model.get_layers() if layer.class_name == 'Transpose'] == [perm]


def test_permute_of_batch_axis_rejected(test_case_id):
    model = Permute(lambda x: x.permute(1, 0, 2), 6).eval()
    with pytest.raises(NotImplementedError):
        _convert(model, [(4, 6)], 'ap_fixed<16,6>', str(test_root_path / test_case_id))


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_permute_numerics(test_case_id, io_type):
    torch.manual_seed(0)
    model = Permute(lambda x: x.permute(0, 2, 1), 4).eval()
    X = np.round(np.random.default_rng(0).uniform(-1, 1, (8, 4, 6)) * 2**8) * 2**-8
    y_ref = model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
    hls_model = _convert(model, [(4, 6)], 'ap_fixed<24,10>', str(test_root_path / test_case_id), io_type=io_type)
    hls_model.compile()
    y_hls = np.asarray(hls_model.predict(np.ascontiguousarray(X))).reshape(y_ref.shape)
    np.testing.assert_allclose(y_hls, y_ref, rtol=0, atol=atol)
    assert count_transposes(hls_model) == 1


def test_layout_helpers():
    x = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    perm = [1, 2, 0]
    native = to_native_layout(x, perm)
    assert native.shape == (2, 4, 5, 3)
    assert native.flags['C_CONTIGUOUS']
    np.testing.assert_array_equal(to_source_layout(native, perm), x)


def test_get_io_layouts(test_case_id):
    model = TwoInputs().eval()
    hls_model = _convert(
        model,
        [(2, 5, 5), (6, 5)],
        'ap_fixed<16,6>',
        str(test_root_path / test_case_id),
        input_layout='native',
        output_layout='native',
    )
    layouts = hls_model.get_io_layouts()
    image, seq = layouts['inputs']
    assert (image['layout'], image['source_shape'], image['stored_shape'], image['perm']) == (
        'native',
        [2, 5, 5],
        [5, 5, 2],
        [1, 2, 0],
    )
    assert (seq['source_shape'], seq['stored_shape'], seq['perm']) == ([6, 5], [6, 5], [0, 1])
    assert layouts['outputs'][0]['perm'] == [0]
