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


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_linear(backend, io_type):
    model = LinearModel()
    model.eval()

    X_input = np.random.rand(1)

    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model, (1,))
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_linear_{backend}_{io_type}')

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
    assert list(hls_model.get_layers())[0].attributes['class_name'] == "InputLayer"
    assert list(hls_model.get_layers())[1].attributes["class_name"] == "Dense"
    assert list(hls_model.get_layers())[0].attributes['input_shape'] == [1]
    assert list(hls_model.get_layers())[1].attributes['n_in'] == 1
    assert list(hls_model.get_layers())[1].attributes['n_out'] == 1


# TODO: add ThresholdedReLU test when it can be made to pass
@pytest.mark.parametrize(
    "activation_function",
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
)
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_activations(activation_function, backend, io_type):
    model = torch.nn.Sequential(nn.Linear(1, 1), activation_function).to()
    model.eval()

    X_input = np.random.rand(1)

    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model, (1,))
    output_dir = str(
        test_root_path / f'hls4mlprj_pytorch_api_activations_{activation_function.__class__.__name__}_{backend}_{io_type}'
    )
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
    "activation_function",
    [
        SoftmaxModel(),
        ReLuModel(),
        TanHModel(),
        LeakyReLuModel(),
        EluModel(),
        SigmoidModel(),
        ThresholdModel(),
    ],
)
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_activation_functionals(activation_function, backend, io_type):
    model = activation_function
    model.eval()

    X_input = np.random.rand(1)

    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model, (1,))
    fn_name = activation_function.__class__.__name__
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_activations_functional_{fn_name}_{backend}_{io_type}')
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
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_conv1d(padds, backend, io_type):
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
            model, (n_in, size_in), channels_last_conversion="internal", transpose_outputs=False
        )
    else:
        config = config_from_pytorch_model(model, (n_in, size_in), channels_last_conversion="full", transpose_outputs=True)

    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_conv1d_{padds}_{backend}_{io_type}')
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
        if (backend == "Vivado" or backend == "Vitis") and padds == 1:
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
    if io_type == "io_stream" and not ((backend == "Vivado" or backend == "Vitis") and padds == 1):
        conv_index = 1
        act_index = 2
    assert list(hls_model.get_layers())[conv_index].attributes['name'] == convNode.name
    assert list(hls_model.get_layers())[conv_index].attributes['class_name'] == 'Conv1D'
    assert list(hls_model.get_layers())[act_index].attributes['activation'] == class_object_relu.__class__.__name__.lower()
    if io_type == "io_stream" and (backend == "Vivado" or backend == "Vitis") and padds == 1:
        assert list(hls_model.get_layers())[conv_index].attributes["in_width"] == size_in + 2
    else:
        assert list(hls_model.get_layers())[conv_index].attributes["in_width"] == size_in
    assert list(hls_model.get_layers())[conv_index].attributes['filt_width'] == class_object_conv.kernel_size[0]
    assert list(hls_model.get_layers())[conv_index].attributes['n_chan'] == class_object_conv.in_channels
    assert list(hls_model.get_layers())[conv_index].attributes['n_filt'] == class_object_conv.out_channels
    assert list(hls_model.get_layers())[conv_index].attributes['stride_width'] == class_object_conv.stride[0]
    padding = padds
    if io_type == "io_stream" and (backend == "Vivado" or backend == "Vitis") and padds == 1:
        padding = 1
        padds = 0

    assert padding == class_object_conv.padding[0]
    assert list(hls_model.get_layers())[conv_index].attributes['data_format'] == 'channels_last'
    assert list(hls_model.get_layers())[conv_index].attributes["out_width"] == out_width

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
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_conv2d(padds, backend, io_type):
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
            model, (n_in, size_in_height, size_in_width), channels_last_conversion="internal", transpose_outputs=False
        )
    else:
        config = config_from_pytorch_model(
            model, (n_in, size_in_height, size_in_width), channels_last_conversion="full", transpose_outputs=True
        )

    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_conv2d_{padds}_{backend}_{io_type}')
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
        if io_type == "io_stream":
            conv_index = 1
            act_index = 2
        assert list(hls_model.get_layers())[conv_index].attributes['name'] == convNode.name
        assert list(hls_model.get_layers())[conv_index].attributes['class_name'] == 'Conv2D'
        assert (
            list(hls_model.get_layers())[act_index].attributes['activation'] == class_object_relu.__class__.__name__.lower()
        )
        assert list(hls_model.get_layers())[conv_index].attributes["in_width"] == size_in_width
        assert list(hls_model.get_layers())[conv_index].attributes["in_height"] == size_in_height
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


@pytest.mark.parametrize('pooling', pooling_layers)
@pytest.mark.parametrize('padds', padds_options)
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
def test_pooling(pooling, padds, backend):
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

    config = config_from_pytorch_model(model, input_shape_forHLS)
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_pooling_{pooling.__name__}_padds_{padds}_backend_{backend}')
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

    if "Max" in pooling.__name__:
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
        assert hls_pool.attributes['name'] == "_" + poolNode.name.split("_")[-1]
        assert hls_pool.attributes['class_name'][-2] == str(2)
        assert hls_pool.attributes['stride_height'] == class_object_pool.stride
        assert hls_pool.attributes['stride_width'] == class_object_pool.stride
        assert hls_pool.attributes['pool_height'] == class_object_pool.kernel_size
        assert hls_pool.attributes['pool_width'] == class_object_pool.kernel_size
        assert hls_pool.attributes['padding'] == 'valid' if class_object_pool.padding == 0 else 'same'

    elif '1d' in pooling.__name__:
        if "Max" in pooling.__name__:
            assert hls_pool.attributes['name'] == "_" + poolNode.name.split("_")[-1]
            assert hls_pool.attributes['class_name'][-2] == str(1)
            assert hls_pool.attributes['pool_width'] == class_object_pool.kernel_size
            assert hls_pool.attributes['stride_width'] == class_object_pool.stride
            assert hls_pool.attributes['padding'] == 'valid' if class_object_pool.padding == 0 else 'same'

        else:
            assert hls_pool.attributes['name'] == "_" + poolNode.name.split("_")[-1]
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


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_bn(backend, io_type):
    model = BatchNormModel()
    model.eval()

    X_input = np.random.rand(1, 5)

    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy().flatten()

    config = config_from_pytorch_model(model, (5,))
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_bn_{backend}_{io_type}')

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


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_squeeze(backend, io_type):
    model = SqueezeModel()
    model.eval()

    X_input = np.random.rand(1, 5)

    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy().flatten()

    config = config_from_pytorch_model(model, (5,))
    del config['Model']['ChannelsLastConversion']  # We don't want anything touched for this test
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_squeeze_{backend}_{io_type}')

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


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
def test_flatten(backend):
    input = torch.randn(1, 1, 5, 5)
    model = nn.Sequential(nn.Conv2d(1, 32, 5, 1, 1), nn.Flatten(), nn.ReLU())
    pytorch_prediction = model(input).detach().numpy()
    input_shape = (1, 5, 5)

    config = config_from_pytorch_model(model, input_shape)
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_flatten_backend_{backend}')
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


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_skipped_layers(backend, io_type):
    model = ModelSkippedLayers()
    model.eval()

    input_shape = (3, 8)
    config = config_from_pytorch_model(
        model,
        input_shape,
        default_precision='ap_fixed<32,16>',
        channels_last_conversion="full",
        transpose_outputs=False,
    )
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_skipped_{backend}_{io_type}')
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


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel'])  # Only io_parallel for now
@pytest.mark.parametrize('tensor_rank', [2, 3])
def test_remove_transpose(backend, io_type, tensor_rank):
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
        channels_last_conversion="full",  # Crucial for testing if the first Transpose was removed
    )
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_transpose_nop_{tensor_rank}d_{backend}_{io_type}')
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


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_view(backend, io_type):

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
    config = config_from_pytorch_model(model, (n_in, size_in), channels_last_conversion="internal", transpose_outputs=False)

    output_dir = str(test_root_path / f'hls4mlprj_pytorch_view_{backend}_{io_type}')
    hls_model = convert_from_pytorch_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
    )

    hls_model.compile()

    # reshape hls prediction to channels last, then transpose, then reshape
    # to match .view
    hls_prediction = np.reshape(
        np.transpose(np.reshape(hls_model.predict(X_input), (n_batch, size_in, n_out)), (0, 2, 1)),
        (n_batch, size_in * n_out),
    )

    rtol = 0
    atol = 5.0e-2
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=rtol, atol=atol)
