import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.nn import AvgPool1d, AvgPool2d, MaxPool1d, MaxPool2d

from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model

test_root_path = Path(__file__).parent


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'oneAPI', 'XLS'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_linear(test_case_id, backend, io_type):
    if backend == 'XLS' and io_type != 'io_parallel':
        pytest.skip(f'XLS backend only supports IOType: io_parallel, but got: {io_type}')
    model = LinearModel()
    model.eval()

    X_input = np.random.rand(1)

    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model, (1,))
    output_dir = str(test_root_path / test_case_id)

    hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)

    hls_model.compile()

    hls_prediction = hls_model.predict(X_input)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)

    from torch.fx import symbolic_trace

    traced_model = symbolic_trace(model)

    nNodes = 0
    for _node in traced_model.graph.nodes:
        nNodes += 1

    assert nNodes - 1 == len(hls_model.get_layers())
    assert list(hls_model.get_layers())[0].attributes['class_name'] == 'InputLayer'
    assert list(hls_model.get_layers())[1].attributes['class_name'] == 'Dense'
    assert list(hls_model.get_layers())[0].attributes['input_shape'] == [1]
    assert list(hls_model.get_layers())[1].attributes['n_in'] == 1
    assert list(hls_model.get_layers())[1].attributes['n_out'] == 1


# TODO: add ThresholdedReLU test when it can be made to pass
@pytest.mark.parametrize(
    'activation_function',
    [
        nn.Softmax(dim=-1),
        nn.ReLU(),
        nn.Tanh(),
        nn.LeakyReLU(negative_slope=1.0),
        nn.ELU(alpha=1.0),
        nn.PReLU(init=0.25),
        nn.Sigmoid(),
        nn.Threshold(threshold=1.0, value=0.0),
    ],
    ids=['softmax', 'relu', 'tanh', 'leaky_relu', 'elu', 'prelu', 'sigmoid', 'threshold'],
)
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'oneAPI', 'XLS'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_activations(test_case_id, activation_function, backend, io_type):
    if backend == 'XLS' and io_type != 'io_parallel':
        pytest.skip(f'XLS backend only supports IOType: io_parallel, but got: {io_type}')
    model = torch.nn.Sequential(nn.Linear(1, 1), activation_function).to()
    model.eval()

    X_input = np.random.rand(1)
    X_input = np.round(X_input * 2**10) * 2**-10  # make it an exact ap_fixed<16,6>

    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model, (1,), granularity='name')
    # XLS uses a custom algorithm for determining lookup table boundaries,
    # so we need to increase the table size for some activations
    # (note that other backends use a hardcoded range [-8; 8]).
    # See hls4ml/backends/xls/passes/build_tables.py
    if backend == 'XLS' and activation_function.__class__.__name__ == 'Tanh':
        config['LayerName']['_1']['TableSize'] = 4096
    output_dir = str(test_root_path / test_case_id)
    hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)
    hls_model.compile()

    hls_prediction = hls_model.predict(X_input)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)

    from torch.fx import symbolic_trace

    traced_model = symbolic_trace(model)

    nNodes = 0
    for _node in traced_model.graph.nodes:
        nNodes += 1

    assert nNodes - 1 == len(hls_model.get_layers())

    if activation_function.__class__.__name__ in ['ReLU', 'Sigmoid', 'Tanh']:
        assert list(hls_model.get_layers())[2].attributes['class_name'] == 'Activation'
    elif activation_function.__class__.__name__ == 'Threshold':
        assert list(hls_model.get_layers())[2].attributes['class_name'] == 'ThresholdedReLU'
    else:
        assert list(hls_model.get_layers())[2].attributes['class_name'] == activation_function.__class__.__name__


class ReLuModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.relu(x)


class SoftmaxModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.softmax(x, dim=-1)


class TanHModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.tanh(x)


class LeakyReLuModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.leaky_relu(x, negative_slope=1.0)


class EluModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.elu(x, alpha=1.0)


class ThresholdModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.threshold(x, threshold=1.0, value=0.0)


class SigmoidModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.sigmoid(x)


@pytest.mark.parametrize(
    'activation_function',
    [
        SoftmaxModel(),
        ReLuModel(),
        TanHModel(),
        LeakyReLuModel(),
        EluModel(),
        SigmoidModel(),
        ThresholdModel(),
    ],
    ids=['softmax', 'relu', 'tanh', 'leaky_relu', 'elu', 'sigmoid', 'threshold'],
)
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'oneAPI', 'XLS'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_activation_functionals(test_case_id, activation_function, backend, io_type):
    if backend == 'XLS' and io_type != 'io_parallel':
        pytest.skip(f'XLS backend only supports IOType: io_parallel, but got: {io_type}')
    model = activation_function
    model.eval()

    X_input = np.random.rand(1)

    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model, (1,))
    output_dir = str(test_root_path / test_case_id)
    hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)
    hls_model.compile()

    hls_prediction = hls_model.predict(X_input)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=0.05)

    from torch.fx import symbolic_trace

    traced_model = symbolic_trace(model)

    nNodes = 0
    for _node in traced_model.graph.nodes:
        nNodes += 1

    assert nNodes - 1 == len(hls_model.get_layers())


padds_options = [0, 1]


@pytest.mark.parametrize('padds', padds_options)
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'oneAPI', 'XLS'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_conv1d(test_case_id, padds, backend, io_type):
    if backend == 'XLS' and io_type != 'io_parallel':
        pytest.skip(f'XLS backend only supports IOType: io_parallel, but got: {io_type}')
    n_in = 2
    n_out = 2
    kernel_size = 3
    size_in = 4

    model = torch.nn.Sequential(nn.Conv1d(n_in, n_out, kernel_size, padding=padds), nn.ReLU()).to()
    model.eval()

    X_input = np.random.rand(1, n_in, size_in)
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    if io_type == 'io_stream':
        X_input = np.ascontiguousarray(X_input.transpose(0, 2, 1))
        config = config_from_pytorch_model(
            model, (n_in, size_in), channels_last_conversion='internal', transpose_outputs=False
        )
    else:
        config = config_from_pytorch_model(model, (n_in, size_in), channels_last_conversion='full', transpose_outputs=True)

    output_dir = str(test_root_path / test_case_id)
    hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)
    hls_model.compile()

    from torch.fx import symbolic_trace

    traced_model = symbolic_trace(model)
    nNodes = 0
    convNode = None
    reluNode = None
    for _node in traced_model.graph.nodes:
        nNodes += 1
        if nNodes == 2:
            convNode = _node
        if nNodes == 3:
            reluNode = _node

    if io_type == 'io_stream':
        # Vivado inserts and additional layer for 'same' padding in io_stream
        if (backend == 'Vivado' or backend == 'Vitis') and padds == 1:
            assert nNodes == len(hls_model.get_layers())
        else:
            assert nNodes - 1 == len(hls_model.get_layers())
    else:
        assert nNodes + 1 == len(hls_model.get_layers())

    children = {c[0]: c[1] for c in model.named_children()}
    class_object_conv = children[convNode.target]
    class_object_relu = children[reluNode.target]

    out_width = int(
        (size_in + 2 * padds - class_object_conv.dilation[0] * (class_object_conv.kernel_size[0] - 1) - 1)
        / class_object_conv.stride[0]
        + 1
    )  # following https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

    if io_type == 'io_stream':
        hls_prediction = np.transpose(np.reshape(hls_model.predict(X_input), (1, out_width, n_out)), (0, 2, 1))
    else:
        hls_prediction = np.reshape(hls_model.predict(X_input), (1, n_out, out_width))
    # results are not very good at the moment
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=5e-2)

    # if not (backend == 'Vivado' and io_type == 'io_stream' and padds == 1):
    conv_index = 2
    act_index = 3
    if io_type == 'io_stream' and not ((backend == 'Vivado' or backend == 'Vitis') and padds == 1):
        conv_index = 1
        act_index = 2
    assert list(hls_model.get_layers())[conv_index].attributes['name'] == convNode.name
    assert list(hls_model.get_layers())[conv_index].attributes['class_name'] == 'Conv1D'
    assert list(hls_model.get_layers())[act_index].attributes['activation'] == class_object_relu.__class__.__name__.lower()
    if io_type == 'io_stream' and (backend == 'Vivado' or backend == 'Vitis') and padds == 1:
        assert list(hls_model.get_layers())[conv_index].attributes['in_width'] == size_in + 2
    else:
        assert list(hls_model.get_layers())[conv_index].attributes['in_width'] == size_in
    assert list(hls_model.get_layers())[conv_index].attributes['filt_width'] == class_object_conv.kernel_size[0]
    assert list(hls_model.get_layers())[conv_index].attributes['n_chan'] == class_object_conv.in_channels
    assert list(hls_model.get_layers())[conv_index].attributes['n_filt'] == class_object_conv.out_channels
    assert list(hls_model.get_layers())[conv_index].attributes['stride_width'] == class_object_conv.stride[0]
    padding = padds
    if io_type == 'io_stream' and (backend == 'Vivado' or backend == 'Vitis') and padds == 1:
        padding = 1
        padds = 0

    assert padding == class_object_conv.padding[0]
    assert list(hls_model.get_layers())[conv_index].attributes['data_format'] == 'channels_last'
    assert list(hls_model.get_layers())[conv_index].attributes['out_width'] == out_width

    pad_along_width = max((out_width - 1) * class_object_conv.stride[0] + class_object_conv.kernel_size[0] - size_in, 0)
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    if padds == 1:
        assert list(hls_model.get_layers())[conv_index].attributes['pad_left'] == pad_left
        assert list(hls_model.get_layers())[conv_index].attributes['pad_right'] == pad_right
    elif padds == 0:
        assert list(hls_model.get_layers())[conv_index].attributes['pad_left'] == 0
        assert list(hls_model.get_layers())[conv_index].attributes['pad_right'] == 0


padds_options = [0, 1]


@pytest.mark.parametrize('padds', padds_options)
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'oneAPI', 'XLS'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_conv2d(test_case_id, padds, backend, io_type):
    if backend == 'XLS' and io_type != 'io_parallel':
        pytest.skip(f'XLS backend only supports IOType: io_parallel, but got: {io_type}')
    n_in = 2
    n_out = 2
    kernel_size = 3
    size_in_width = 4
    size_in_height = 4

    model = torch.nn.Sequential(nn.Conv2d(n_in, n_out, kernel_size, padding=padds), nn.ReLU()).to()
    model.eval()

    X_input = np.random.rand(100, n_in, size_in_height, size_in_width)
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    if io_type == 'io_stream':
        X_input = np.ascontiguousarray(X_input.transpose(0, 2, 3, 1))
        config = config_from_pytorch_model(
            model, (n_in, size_in_height, size_in_width), channels_last_conversion='internal', transpose_outputs=False
        )
    else:
        config = config_from_pytorch_model(
            model, (n_in, size_in_height, size_in_width), channels_last_conversion='full', transpose_outputs=True
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

    from torch.fx import symbolic_trace

    traced_model = symbolic_trace(model)
    nNodes = 0
    convNode = None
    reluNode = None
    for _node in traced_model.graph.nodes:
        nNodes += 1
        if nNodes == 2:
            convNode = _node
        if nNodes == 3:
            reluNode = _node
    # if io_type == 'io_stream':
    #    assert nNodes -1 == len(hls_model.get_layers())
    # else:
    #    assert nNodes == len(hls_model.get_layers())

    children = {c[0]: c[1] for c in model.named_children()}
    class_object_conv = children[convNode.target]
    class_object_relu = children[reluNode.target]

    from hls4ml.converters.utils import compute_padding_2d

    padding = 'valid' if padds == 0 else 'same'
    out_dims_hls = compute_padding_2d(
        padding,
        size_in_height,
        size_in_width,
        1,
        1,
        kernel_size,
        kernel_size,
    )

    out_width = int(
        (
            size_in_width
            + 2 * class_object_conv.padding[1]
            - class_object_conv.dilation[1] * (class_object_conv.kernel_size[1] - 1)
            - 1
        )
        / class_object_conv.stride[1]
        + 1
    )  # following https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    assert out_dims_hls[0] == out_width
    out_height = int(
        (
            size_in_height
            + 2 * class_object_conv.padding[0]
            - class_object_conv.dilation[0] * (class_object_conv.kernel_size[0] - 1)
            - 1
        )
        / class_object_conv.stride[0]
        + 1
    )  # following https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    assert out_dims_hls[1] == out_height

    if io_type == 'io_stream':
        hls_prediction = np.transpose(
            np.reshape(hls_model.predict(X_input), (100, out_height, out_width, n_out)), (0, 3, 1, 2)
        )
    else:
        hls_prediction = np.reshape(hls_model.predict(X_input), (100, n_out, out_height, out_width))
    # results are not very good at the moment
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=5e-2)

    if not ((backend == 'Vivado' or backend == 'Vitis') and io_type == 'io_stream' and padds == 1):
        # Vivado inserts and additional layer for 'same' padding in io_stream
        conv_index = 2
        act_index = 3
        if io_type == 'io_stream':
            conv_index = 1
            act_index = 2
        assert list(hls_model.get_layers())[conv_index].attributes['name'] == convNode.name
        assert list(hls_model.get_layers())[conv_index].attributes['class_name'] == 'Conv2D'
        assert (
            list(hls_model.get_layers())[act_index].attributes['activation'] == class_object_relu.__class__.__name__.lower()
        )
        assert list(hls_model.get_layers())[conv_index].attributes['in_width'] == size_in_width
        assert list(hls_model.get_layers())[conv_index].attributes['in_height'] == size_in_height
        assert list(hls_model.get_layers())[conv_index].attributes['filt_width'] == class_object_conv.kernel_size[1]
        assert list(hls_model.get_layers())[conv_index].attributes['filt_height'] == class_object_conv.kernel_size[0]
        assert list(hls_model.get_layers())[conv_index].attributes['n_chan'] == class_object_conv.in_channels
        assert list(hls_model.get_layers())[conv_index].attributes['n_filt'] == class_object_conv.out_channels
        assert list(hls_model.get_layers())[conv_index].attributes['stride_width'] == class_object_conv.stride[1]
        assert list(hls_model.get_layers())[conv_index].attributes['stride_height'] == class_object_conv.stride[0]
        padding = padds
        assert padding == class_object_conv.padding[0]
        assert list(hls_model.get_layers())[conv_index].attributes['data_format'] == 'channels_last'

        pad_along_width = max(
            (out_width - 1) * class_object_conv.stride[1] + class_object_conv.kernel_size[1] - size_in_width, 0
        )
        pad_along_height = max(
            (out_height - 1) * class_object_conv.stride[0] + class_object_conv.kernel_size[0] - size_in_height, 0
        )

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        if padds == 1:
            assert list(hls_model.get_layers())[conv_index].attributes['pad_left'] == pad_left
            assert list(hls_model.get_layers())[conv_index].attributes['pad_right'] == pad_right
            assert list(hls_model.get_layers())[conv_index].attributes['pad_top'] == pad_top
            assert list(hls_model.get_layers())[conv_index].attributes['pad_bottom'] == pad_bottom
        elif padds == 0:
            assert list(hls_model.get_layers())[conv_index].attributes['pad_left'] == 0
            assert list(hls_model.get_layers())[conv_index].attributes['pad_right'] == 0
            assert list(hls_model.get_layers())[conv_index].attributes['pad_top'] == 0
            assert list(hls_model.get_layers())[conv_index].attributes['pad_bottom'] == 0


padds_options = [0, 1]
pooling_layers = [MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d]


@pytest.mark.parametrize('pooling', pooling_layers, ids=['MaxPool1d', 'MaxPool2d', 'AvgPool1d', 'AvgPool2d'])
@pytest.mark.parametrize('padds', padds_options)
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'oneAPI', 'XLS'])
def test_pooling(test_case_id, pooling, padds, backend):
    assert '1d' in pooling.__name__ or '2d' in pooling.__name__

    if '2d' in pooling.__name__:
        n_in = 2
        size_in_height = 15
        size_in_width = 18
    else:
        n_in = 2
        size_in_width = 121
        size_in_height = 0

    input_shape = (1, n_in, size_in_height, size_in_width) if '2d' in pooling.__name__ else (1, n_in, size_in_width)
    input_shape_forHLS = (n_in, size_in_height, size_in_width) if '2d' in pooling.__name__ else (n_in, size_in_width)
    X_input = np.random.rand(*input_shape)

    model = torch.nn.Sequential(pooling(2, padding=padds)).to()
    model.eval()
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model, input_shape_forHLS, transpose_outputs=True)
    output_dir = str(test_root_path / test_case_id)
    hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend)
    hls_model.compile()

    from torch.fx import symbolic_trace

    traced_model = symbolic_trace(model)
    nNodes = 0
    poolNode = None
    for _node in traced_model.graph.nodes:
        nNodes += 1
        if nNodes == 2:
            poolNode = _node
    assert nNodes + 1 == len(hls_model.get_layers())
    children = {c[0]: c[1] for c in model.named_children()}
    class_object_pool = children[poolNode.target]

    if 'Max' in pooling.__name__:
        out_height = int(
            math.floor(
                float(size_in_height + 2 * padds - class_object_pool.dilation * (class_object_pool.kernel_size - 1) - 1)
                / float(class_object_pool.stride)
                + 1
            )
        )
        out_width = int(
            math.floor(
                float(size_in_width + 2 * padds - class_object_pool.dilation * (class_object_pool.kernel_size - 1) - 1)
                / float(class_object_pool.stride)
                + 1
            )
        )
    else:
        if '2d' in pooling.__name__:
            out_height = int(
                math.floor((size_in_height + 2 * padds - class_object_pool.kernel_size) / class_object_pool.stride + 1)
            )
            out_width = int(
                math.floor((size_in_width + 2 * padds - class_object_pool.kernel_size) / class_object_pool.stride + 1)
            )
        else:
            out_height = int(
                math.floor((size_in_height + 2 * padds - class_object_pool.kernel_size[0]) / class_object_pool.stride[0] + 1)
            )
            out_width = int(
                math.floor((size_in_width + 2 * padds - class_object_pool.kernel_size[0]) / class_object_pool.stride[0] + 1)
            )

    if '2d' in pooling.__name__:
        hls_prediction = np.reshape(hls_model.predict(X_input), (1, n_in, out_height, out_width))

    else:
        pred = hls_model.predict(X_input)
        hls_prediction = np.reshape(pred, (1, n_in, out_width))

    # results are not very good at the moment
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=5e-2)

    # Verify correct parsing of layer
    hls_pool = list(hls_model.get_layers())[-2]
    if '2d' in pooling.__name__:
        assert hls_pool.attributes['name'] == '_' + poolNode.name.split('_')[-1]
        assert hls_pool.attributes['class_name'][-2] == str(2)
        assert hls_pool.attributes['stride_height'] == class_object_pool.stride
        assert hls_pool.attributes['stride_width'] == class_object_pool.stride
        assert hls_pool.attributes['pool_height'] == class_object_pool.kernel_size
        assert hls_pool.attributes['pool_width'] == class_object_pool.kernel_size
        assert hls_pool.attributes['padding'] == 'valid' if class_object_pool.padding == 0 else 'same'

    elif '1d' in pooling.__name__:
        if 'Max' in pooling.__name__:
            assert hls_pool.attributes['name'] == '_' + poolNode.name.split('_')[-1]
            assert hls_pool.attributes['class_name'][-2] == str(1)
            assert hls_pool.attributes['pool_width'] == class_object_pool.kernel_size
            assert hls_pool.attributes['stride_width'] == class_object_pool.stride
            assert hls_pool.attributes['padding'] == 'valid' if class_object_pool.padding == 0 else 'same'

        else:
            assert hls_pool.attributes['name'] == '_' + poolNode.name.split('_')[-1]
            assert hls_pool.attributes['class_name'][-2] == str(1)
            assert hls_pool.attributes['pool_width'] == class_object_pool.kernel_size[0]
            assert hls_pool.attributes['stride_width'] == class_object_pool.stride[0]
            assert hls_pool.attributes['padding'] == 'same' if class_object_pool.padding == 0 else 'valid'


class BatchNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 8)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(8)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)  # This is to prevent merging of BN into Linear
        return self.bn(x)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'oneAPI', 'XLS'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_bn(test_case_id, backend, io_type):
    if backend == 'XLS' and io_type != 'io_parallel':
        pytest.skip(f'XLS backend only supports IOType: io_parallel, but got: {io_type}')
    model = BatchNormModel()
    model.eval()

    X_input = np.random.rand(1, 5)

    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy().flatten()

    config = config_from_pytorch_model(model, (5,))
    output_dir = str(test_root_path / test_case_id)

    hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)

    hls_model.compile()

    hls_prediction = hls_model.predict(X_input).flatten()

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)

    assert list(hls_model.get_layers())[3].attributes['class_name'] == 'BatchNormalization'
    assert list(hls_model.get_layers())[3].attributes['n_in'] == 8
    assert list(hls_model.get_layers())[3].attributes['n_out'] == 8


class SqueezeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 3, bias=False)
        self.bn = nn.BatchNorm1d(3)
        nn.init.ones_(self.linear.weight)  # This test is not about precision, so put 1's here

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)  # (1, 5) -> (1, 1, 5)
        x = self.linear(x)  # (1, 1, 3)
        x = torch.squeeze(x)  # (3,)
        x = torch.relu(x)  # (3,)
        return x


# TODO: this test fails for XLS due to PyTorch weights shape mismatch.
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_squeeze(test_case_id, backend, io_type):
    model = SqueezeModel()
    model.eval()

    X_input = np.random.rand(1, 5)

    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy().flatten()

    config = config_from_pytorch_model(model, (5,))
    del config['Model']['ChannelsLastConversion']  # We don't want anything touched for this test
    output_dir = str(test_root_path / test_case_id)

    hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)

    hls_model.compile()

    hls_prediction = hls_model.predict(X_input).flatten()

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)

    # oneAPI doesn't use the Repack class (and for io_stream does not use inplace variables)
    if io_type == 'io_parallel' or backend == 'oneAPI':
        assert list(hls_model.get_layers())[1].attributes['class_name'] == 'Reshape'
        assert list(hls_model.get_layers())[1].attributes['target_shape'] == [1, 5]
        assert list(hls_model.get_layers())[3].attributes['class_name'] == 'Reshape'
        assert list(hls_model.get_layers())[3].attributes['target_shape'] == [3]
    elif io_type == 'io_stream':
        assert list(hls_model.get_layers())[1].class_name == 'Repack'
        assert list(hls_model.get_layers())[1].attributes['target_shape'] == [1, 5]
        assert list(hls_model.get_layers())[3].attributes['class_name'] == 'Reshape'  # Exists as in-place variable
        assert list(hls_model.get_layers())[3].attributes['target_shape'] == [3]


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'oneAPI', 'XLS'])
def test_flatten(test_case_id, backend):
    input = torch.randn(1, 1, 5, 5)
    model = nn.Sequential(nn.Conv2d(1, 32, 5, 1, 1), nn.Flatten(), nn.ReLU())
    pytorch_prediction = model(input).detach().numpy()
    input_shape = (1, 5, 5)

    config = config_from_pytorch_model(model, input_shape)
    output_dir = str(test_root_path / test_case_id)
    hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend)
    hls_model.compile()

    pred = hls_model.predict(input.detach().numpy())
    hls_prediction = np.reshape(pred, (1, 288))

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=5e-2)


class ModelSkippedLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=3, bias=False)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=6, out_channels=5, kernel_size=3, bias=False)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout()  # Should be skipped
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=5 * 4, out_features=6, bias=False)
        self.dropout2 = nn.Dropout()  # Should be skipped
        self.fc2 = nn.Linear(in_features=6, out_features=5, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'oneAPI', 'XLS'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_skipped_layers(test_case_id, backend, io_type):
    if backend == 'XLS' and io_type != 'io_parallel':
        pytest.skip(f'XLS backend only supports IOType: io_parallel, but got: {io_type}')
    model = ModelSkippedLayers()
    model.eval()

    input_shape = (3, 8)
    config = config_from_pytorch_model(
        model,
        input_shape,
        default_precision='ap_fixed<32,16>',
        channels_last_conversion='full',
        transpose_outputs=False,
    )
    output_dir = str(test_root_path / test_case_id)
    hls_model = convert_from_pytorch_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        io_type=io_type,
        backend=backend,
    )

    hls_model.compile()

    input = torch.randn(10, 3, 8)

    pytorch_prediction = model(input).detach().numpy().flatten()
    hls_prediction = hls_model.predict(input.detach().numpy()).flatten()

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=5e-2)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'oneAPI', 'XLS'])
@pytest.mark.parametrize('io_type', ['io_parallel'])  # Only io_parallel for now
@pytest.mark.parametrize('tensor_rank', [2, 3])
def test_remove_transpose(test_case_id, backend, io_type, tensor_rank):
    class TestModel(nn.Module):
        def __init__(self, tensor_rank):
            super().__init__()
            if tensor_rank == 2:
                self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, bias=False)
                self.relu1 = nn.ReLU()
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(in_features=4 * 6, out_features=5, bias=False)
                self.relu2 = nn.ReLU()
            else:
                self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, bias=False)
                self.relu1 = nn.ReLU()
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(in_features=4 * 6 * 6, out_features=5, bias=False)
                self.relu2 = nn.ReLU()

        def forward(self, x):
            # In the hls4ml model, there should be a Transpose node on the input tensor before conv1
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.flatten(x)  # This should result in a Transpose node that we aim to remove
            x = self.fc1(x)
            x = self.relu2(x)
            return x

    model = TestModel(tensor_rank=tensor_rank)
    if tensor_rank == 2:
        input_shape = (1, 8)
        input_tensor = torch.randn(10, 1, 8)
        hls_input = np.ascontiguousarray(torch.permute(input_tensor, (0, 2, 1)).detach().numpy())
    else:
        input_shape = (1, 8, 8)
        input_tensor = torch.randn(10, 1, 8, 8)
        hls_input = np.ascontiguousarray(torch.permute(input_tensor, (0, 2, 3, 1)).detach().numpy())

    config = config_from_pytorch_model(
        model,
        input_shape,
        default_precision='ap_fixed<32,16>',
        channels_last_conversion='full',  # Crucial for testing if the first Transpose was removed
    )
    output_dir = str(test_root_path / test_case_id)
    hls_model = convert_from_pytorch_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        io_type=io_type,
        backend=backend,
    )

    hls_model.compile()

    # Test optimizers removed the two Transpose layers
    transpose_layers = [layer for layer in list(hls_model.get_layers()) if layer.class_name == 'Transpose']
    assert len(transpose_layers) == 0

    # Test predictions match
    pytorch_prediction = model(input_tensor).detach().numpy().flatten()
    hls_prediction = hls_model.predict(hls_input).flatten()

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=5e-2)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'oneAPI', 'XLS'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_view(test_case_id, backend, io_type):
    if backend == 'XLS' and io_type != 'io_parallel':
        pytest.skip(f'XLS backend only supports IOType: io_parallel, but got: {io_type}')

    class TestModel(nn.Module):
        def __init__(self, n_in, n_out, size_in):
            super().__init__()
            self.view_mult = n_out * size_in

            self.conv1 = nn.Conv1d(
                n_in,
                n_out,
                kernel_size=3,
                padding=1,
                bias=False,
            )

        def forward(self, x):
            z = self.conv1(x)
            z = z.view(-1, self.view_mult)
            return z

    n_in = 2
    n_out = 4
    size_in = 128
    n_batch = 100

    model = TestModel(n_in, n_out, size_in)
    model = model.to(memory_format=torch.channels_last)
    model.eval()

    X_input = np.random.rand(n_batch, n_in, size_in)
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    # X_input is channels last
    X_input = np.ascontiguousarray(X_input.transpose(0, 2, 1))
    config = config_from_pytorch_model(model, (n_in, size_in), channels_last_conversion='internal', transpose_outputs=False)

    output_dir = str(test_root_path / test_case_id)
    hls_model = convert_from_pytorch_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
    )

    hls_model.compile()

    hls_prediction = hls_model.predict(X_input)

    rtol = 0
    atol = 5.0e-2
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=rtol, atol=atol)


class EinsumOuterProduct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.einsum('bi,bj->bij', x, y)


class EinsumBatchMatMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.einsum('bij,bjk->bik', x, y)


class EinsumSingleInput(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(self.input_dim, self.input_dim)

    def forward(self, x):
        """using torch einsum to get the dot product"""
        out = self.linear(x)
        return torch.einsum('ij,ij->i', out, out)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_einsum_outer_product(test_case_id, backend, io_type):
    model = EinsumOuterProduct()
    model.eval()

    X_input = np.random.rand(3, 4)
    Y_input = np.random.rand(3, 5)

    pytorch_prediction = model(torch.Tensor(X_input), torch.Tensor(Y_input)).detach().numpy()

    config = config_from_pytorch_model(
        model,
        [(None, 4), (None, 5)],
        default_precision='ap_fixed<16,6>',
        channels_last_conversion='internal',
        transpose_outputs=False,
    )
    output_dir = str(test_root_path / test_case_id)

    hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)

    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict([X_input, Y_input]), pytorch_prediction.shape)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_einsum_batch_matmul(test_case_id, backend, io_type):
    model = EinsumBatchMatMul()
    model.eval()

    X_input = np.random.rand(3, 2, 5)
    Y_input = np.random.rand(3, 5, 4)

    pytorch_prediction = model(torch.Tensor(X_input), torch.Tensor(Y_input)).detach().numpy()

    config = config_from_pytorch_model(
        model,
        [(None, 2, 5), (None, 5, 4)],
        default_precision='ap_fixed<16,6>',
        channels_last_conversion='internal',
        transpose_outputs=False,
    )
    output_dir = str(test_root_path / test_case_id)

    hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)

    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict([X_input, Y_input]), pytorch_prediction.shape)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_einsum_single_input(test_case_id, backend, io_type):
    model = EinsumSingleInput()
    model.eval()

    X_input = np.random.rand(3, 8)

    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(
        model,
        [(None, 8)],
        default_precision='ap_fixed<32,12>',
        channels_last_conversion='internal',
        transpose_outputs=False,
    )
    output_dir = str(test_root_path / test_case_id)

    hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)

    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict(X_input), pytorch_prediction.shape)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)


# ---------------------------------------------------------------------------- #
# Parser behavior for configurations that used to be silently misparsed:
# faithful 'same'/'valid' padding, positional arguments of functional calls,
# clipped ReLU activations and rejects of unsupported options.
# ---------------------------------------------------------------------------- #


def _convert_parse_only(model, test_case_id, input_shape, io_type='io_parallel'):
    """Converts without compiling, to assert parse results and parse-time rejects."""
    config = config_from_pytorch_model(model, input_shape, channels_last_conversion='off', transpose_outputs=False)
    output_dir = str(test_root_path / test_case_id)
    return convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend='Vivado', io_type=io_type)


@pytest.mark.parametrize('kernel_size', [1, 2, 3, 4, 5, 7])
@pytest.mark.parametrize('padds', ['same', 'valid'])
def test_conv1d_string_padding(test_case_id, kernel_size, padds):
    n_chan = 2
    size_in = 8

    model = torch.nn.Sequential(nn.Conv1d(n_chan, n_chan, kernel_size, padding=padds)).to()
    model.eval()

    X_input = np.random.rand(5, n_chan, size_in)
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model, (n_chan, size_in), channels_last_conversion='full', transpose_outputs=True)
    output_dir = str(test_root_path / test_case_id)
    hls_model = convert_from_pytorch_model(
        model, hls_config=config, output_dir=output_dir, backend='Vivado', io_type='io_parallel'
    )
    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict(X_input), pytorch_prediction.shape)
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=5e-2)


@pytest.mark.parametrize('kernel_size', [3, 4])
@pytest.mark.parametrize('padds', ['same', 'valid'])
def test_conv2d_string_padding(test_case_id, kernel_size, padds):
    n_chan = 2
    size_in = 8

    model = torch.nn.Sequential(nn.Conv2d(n_chan, n_chan, kernel_size, padding=padds)).to()
    model.eval()

    X_input = np.random.rand(5, n_chan, size_in, size_in)
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(
        model, (n_chan, size_in, size_in), channels_last_conversion='full', transpose_outputs=True
    )
    output_dir = str(test_root_path / test_case_id)
    hls_model = convert_from_pytorch_model(
        model, hls_config=config, output_dir=output_dir, backend='Vivado', io_type='io_parallel'
    )
    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict(X_input), pytorch_prediction.shape)
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=5e-2)


def test_conv2d_asymmetric_dilation_rejected(test_case_id):
    model = torch.nn.Sequential(nn.Conv2d(2, 2, 3, dilation=(2, 1)))
    model.eval()

    with pytest.raises(NotImplementedError, match='dilation'):
        _convert_parse_only(model, test_case_id, (2, 8, 8))


class FunctionalCallModel(nn.Module):
    """Wraps a function of one tensor, so that it traces as a functional call."""

    def __init__(self, function):
        super().__init__()
        self.function = function
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.function(self.linear(x))


functional_activations = {
    'leaky_relu_positional': lambda x: torch.nn.functional.leaky_relu(x, 0.2),
    'leaky_relu_default': lambda x: torch.nn.functional.leaky_relu(x),
    'elu_positional': lambda x: torch.nn.functional.elu(x, 0.5),
    'threshold_positional': lambda x: torch.nn.functional.threshold(x, 0.5, 0.0),
    'softmax_positional': lambda x: torch.nn.functional.softmax(x, 1),
}


@pytest.mark.parametrize('activation', functional_activations.keys())
def test_functional_activation_arguments(test_case_id, activation):
    """Functional activations with positional (or omitted) arguments parse and match torch."""
    model = FunctionalCallModel(functional_activations[activation])
    model.eval()

    X_input = np.random.rand(10, 4)
    X_input = np.round(X_input * 2**10) * 2**-10
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model, (4,))
    output_dir = str(test_root_path / test_case_id)
    hls_model = convert_from_pytorch_model(
        model, hls_config=config, output_dir=output_dir, backend='Vivado', io_type='io_parallel'
    )
    hls_model.compile()

    hls_prediction = hls_model.predict(X_input)
    # the table-based softmax is coarser than the other activations at the default precision
    atol = 0.05 if 'softmax' in activation else 0.01
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=atol)


@pytest.mark.parametrize('activation', ['relu6_module', 'hardtanh_module', 'relu6_functional', 'hardtanh_functional'])
def test_clipped_relu_parsed(test_case_id, activation):
    # ReLU6 and Hardtanh(0, x) are ingested as the ClippedReLU parametrized activation.
    # No kernel computes it yet; the backends are responsible for rejecting it.
    if activation == 'relu6_module':
        model = torch.nn.Sequential(nn.Linear(4, 4), nn.ReLU6())
        max_value = 6.0
    elif activation == 'hardtanh_module':
        model = torch.nn.Sequential(nn.Linear(4, 4), nn.Hardtanh(0.0, 4.0))
        max_value = 4.0
    elif activation == 'relu6_functional':
        model = FunctionalCallModel(lambda x: torch.nn.functional.relu6(x))
        max_value = 6.0
    else:
        model = FunctionalCallModel(lambda x: torch.nn.functional.hardtanh(x, 0.0, 4.0))
        max_value = 4.0
    model.eval()

    hls_model = _convert_parse_only(model, test_case_id, (4,))

    act_layer = list(hls_model.get_layers())[-1]
    assert act_layer.attributes['activation'] == 'clippedrelu'
    assert act_layer.attributes['activ_param'] == max_value


def test_hardtanh_below_zero_rejected(test_case_id):
    model = torch.nn.Sequential(nn.Linear(4, 4), nn.Hardtanh())  # min_val=-1 by default
    model.eval()

    with pytest.raises(NotImplementedError, match='min_val'):
        _convert_parse_only(model, test_case_id, (4,))


def test_threshold_replacement_value_rejected(test_case_id):
    # nn.Threshold(threshold, value) outputs `value` below the threshold; only 0 maps to ThresholdedReLU
    model = torch.nn.Sequential(nn.Linear(4, 4), nn.Threshold(1.0, 5.0))
    model.eval()

    with pytest.raises(NotImplementedError, match='replacement value'):
        _convert_parse_only(model, test_case_id, (4,))


class FunctionalPoolModel(nn.Module):
    """Wraps a pooling function of one tensor, so that it traces as a functional call."""

    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


functional_pools = {
    'max_pool1d_positional': (lambda x: torch.nn.functional.max_pool1d(x, 2, 2), 1),
    'max_pool2d_default_stride': (lambda x: torch.nn.functional.max_pool2d(x, 2), 2),
    'avg_pool1d_positional': (lambda x: torch.nn.functional.avg_pool1d(x, 2, 2), 1),
    'avg_pool2d_no_count_include_pad': (lambda x: torch.nn.functional.avg_pool2d(x, 2, 2, 1, False, False), 2),
}


@pytest.mark.parametrize('pool', functional_pools.keys())
def test_functional_pooling_arguments(test_case_id, pool):
    """Functional pooling with positional (or omitted) arguments parses and matches torch."""
    function, dims = functional_pools[pool]
    n_chan = 2
    size_in = 8

    model = FunctionalPoolModel(function)
    model.eval()

    input_shape = (n_chan, size_in) if dims == 1 else (n_chan, size_in, size_in)
    X_input = np.random.rand(5, *input_shape)
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model, input_shape, channels_last_conversion='full', transpose_outputs=True)
    output_dir = str(test_root_path / test_case_id)
    hls_model = convert_from_pytorch_model(
        model, hls_config=config, output_dir=output_dir, backend='Vivado', io_type='io_parallel'
    )
    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict(X_input), pytorch_prediction.shape)
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=5e-2)


pooling_rejects = {
    'ceil_mode': (lambda: nn.MaxPool2d(2, ceil_mode=True), 'ceil_mode'),
    'dilation': (lambda: nn.MaxPool1d(3, stride=1, dilation=2), 'dilation'),
    'divisor_override': (lambda: nn.AvgPool2d(2, divisor_override=3), 'divisor_override'),
}


@pytest.mark.parametrize('reject', pooling_rejects.keys())
def test_pooling_option_rejected(test_case_id, reject):
    layer_factory, match = pooling_rejects[reject]
    model = torch.nn.Sequential(layer_factory())
    model.eval()

    input_shape = (2, 8) if '1d' in type(model[0]).__name__.lower() else (2, 8, 8)
    with pytest.raises(NotImplementedError, match=match):
        _convert_parse_only(model, test_case_id, input_shape)


functional_pooling_rejects = {
    # signature positions: max_pool(input, kernel_size, stride, padding, dilation, ceil_mode),
    # avg_pool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    'ceil_mode': (lambda x: torch.nn.functional.max_pool2d(x, 2, 2, 0, 1, True), (2, 8, 8), 'ceil_mode'),
    'dilation': (lambda x: torch.nn.functional.max_pool1d(x, 3, 1, 0, 2), (2, 8), 'dilation'),
    'divisor_override': (
        lambda x: torch.nn.functional.avg_pool2d(x, 2, 2, 0, False, True, 3),
        (2, 8, 8),
        'divisor_override',
    ),
}


@pytest.mark.parametrize('reject', functional_pooling_rejects.keys())
def test_functional_pooling_option_rejected(test_case_id, reject):
    """The unsupported pooling options are also rejected when passed positionally to the functional forms."""
    function, input_shape, match = functional_pooling_rejects[reject]
    model = FunctionalPoolModel(function)
    model.eval()

    with pytest.raises(NotImplementedError, match=match):
        _convert_parse_only(model, test_case_id, input_shape)


def test_view_literal_batch_size(test_case_id):
    """A literal batch size in view() must not corrupt the deduction of a -1 entry."""

    class ViewModel(nn.Module):
        # the linear layer keeps the reshape from being the final layer, which would be optimized away
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(12, 4)

        def forward(self, x):
            return self.linear(x.view(4, -1))

    model = ViewModel()
    model.eval()

    hls_model = _convert_parse_only(model, test_case_id, (2, 6))

    reshape_layer = next(layer for layer in hls_model.get_layers() if layer.attributes['class_name'] == 'Reshape')
    assert reshape_layer.attributes['target_shape'] == [12]


def test_functional_threshold_replacement_value_rejected(test_case_id):
    model = FunctionalCallModel(lambda x: torch.nn.functional.threshold(x, 0.5, 5.0))
    model.eval()

    with pytest.raises(NotImplementedError, match='replacement value'):
        _convert_parse_only(model, test_case_id, (4,))
