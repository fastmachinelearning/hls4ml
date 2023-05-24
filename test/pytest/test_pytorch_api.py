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


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_linear(backend, io_type):
    model = LinearModel()
    model.eval()

    X_input = np.random.rand(1)

    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model)
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_linear_{backend}_{io_type}')

    hls_model = convert_from_pytorch_model(
        model, (None, 1), hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )

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
        nn.ReLU(),
        nn.LeakyReLU(negative_slope=1.0),
        nn.ELU(alpha=1.0),
        nn.PReLU(init=0.25),
        nn.Sigmoid(),
        nn.Threshold(threshold=1.0, value=0.0),
    ],
)
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_activations(activation_function, backend, io_type):
    model = torch.nn.Sequential(nn.Linear(1, 1), activation_function).to()
    model.eval()

    X_input = np.random.rand(1)

    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model)
    output_dir = str(
        test_root_path / f'hls4mlprj_pytorch_api_activations_{activation_function.__class__.__name__}_{backend}_{io_type}'
    )
    hls_model = convert_from_pytorch_model(
        model, (None, 1), hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    hls_model.compile()

    hls_prediction = hls_model.predict(X_input)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)

    from torch.fx import symbolic_trace

    traced_model = symbolic_trace(model)

    nNodes = 0
    for _node in traced_model.graph.nodes:
        nNodes += 1

    assert nNodes - 1 == len(hls_model.get_layers())

    if activation_function.__class__.__name__ == 'ReLU' or activation_function.__class__.__name__ == 'Sigmoid':
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
        ReLuModel(),
        LeakyReLuModel(),
        EluModel(),
        SigmoidModel(),
        ThresholdModel(),
    ],
)
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_activation_functionals(activation_function, backend, io_type):
    model = activation_function
    model.eval()

    X_input = np.random.rand(1)

    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model)
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_activations_functional_relu_{backend}_{io_type}')
    hls_model = convert_from_pytorch_model(
        model, (None, 1), hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    hls_model.compile()

    hls_prediction = hls_model.predict(X_input)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)

    from torch.fx import symbolic_trace

    traced_model = symbolic_trace(model)

    nNodes = 0
    for _node in traced_model.graph.nodes:
        nNodes += 1

    assert nNodes - 1 == len(hls_model.get_layers())


padds_options = [0, 1]


@pytest.mark.parametrize('padds', padds_options)
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
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
        config = config_from_pytorch_model(model, inputs_channel_last=True, transpose_outputs=False)
    else:
        config = config_from_pytorch_model(model, inputs_channel_last=False, transpose_outputs=True)

    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_conv1d_{padds}_{backend}_{io_type}')
    hls_model = convert_from_pytorch_model(
        model, (None, n_in, size_in), hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
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

    if io_type == 'io_stream':
        # Vivado inserts and additional layer for 'same' padding in io_stream
        if backend == "Vivado" and padds == 1:
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
    if io_type == "io_stream" and not (backend == "Vivado" and padds == 1):
        conv_index = 1
        act_index = 2
    assert list(hls_model.get_layers())[conv_index].attributes['name'] == 'layer' + convNode.name
    assert list(hls_model.get_layers())[conv_index].attributes['class_name'] == 'Conv1D'
    assert list(hls_model.get_layers())[act_index].attributes['activation'] == class_object_relu.__class__.__name__
    if io_type == "io_stream" and backend == "Vivado" and padds == 1:
        assert list(hls_model.get_layers())[conv_index].attributes["in_width"] == size_in + 2
    else:
        assert list(hls_model.get_layers())[conv_index].attributes["in_width"] == size_in
    assert list(hls_model.get_layers())[conv_index].attributes['filt_width'] == class_object_conv.kernel_size[0]
    assert list(hls_model.get_layers())[conv_index].attributes['n_chan'] == class_object_conv.in_channels
    assert list(hls_model.get_layers())[conv_index].attributes['n_filt'] == class_object_conv.out_channels
    assert list(hls_model.get_layers())[conv_index].attributes['stride_width'] == class_object_conv.stride[0]
    if list(hls_model.get_layers())[conv_index].attributes['padding'] == 'valid':
        padding = 0
    else:
        padding = 1
    if io_type == "io_stream" and backend == "Vivado" and padds == 1:
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
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
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
        config = config_from_pytorch_model(model, inputs_channel_last=True, transpose_outputs=False)
    else:
        config = config_from_pytorch_model(model, inputs_channel_last=False, transpose_outputs=True)

    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_conv2d_{padds}_{backend}_{io_type}')
    hls_model = convert_from_pytorch_model(
        model,
        (None, n_in, size_in_height, size_in_width),
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

    if not (backend == 'Vivado' and io_type == 'io_stream' and padds == 1):
        # Vivado inserts and additional layer for 'same' padding in io_stream
        conv_index = 2
        act_index = 3
        if io_type == "io_stream":
            conv_index = 1
            act_index = 2
        assert list(hls_model.get_layers())[conv_index].attributes['name'] == 'layer' + convNode.name
        assert list(hls_model.get_layers())[conv_index].attributes['class_name'] == 'Conv2D'
        assert list(hls_model.get_layers())[act_index].attributes['activation'] == class_object_relu.__class__.__name__
        assert list(hls_model.get_layers())[conv_index].attributes["in_width"] == size_in_width
        assert list(hls_model.get_layers())[conv_index].attributes["in_height"] == size_in_height
        assert list(hls_model.get_layers())[conv_index].attributes['filt_width'] == class_object_conv.kernel_size[1]
        assert list(hls_model.get_layers())[conv_index].attributes['filt_height'] == class_object_conv.kernel_size[0]
        assert list(hls_model.get_layers())[conv_index].attributes['n_chan'] == class_object_conv.in_channels
        assert list(hls_model.get_layers())[conv_index].attributes['n_filt'] == class_object_conv.out_channels
        assert list(hls_model.get_layers())[conv_index].attributes['stride_width'] == class_object_conv.stride[1]
        assert list(hls_model.get_layers())[conv_index].attributes['stride_height'] == class_object_conv.stride[0]
        if list(hls_model.get_layers())[conv_index].attributes['padding'] == 'valid':
            padding = 0
        else:
            padding = 1
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
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
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
    input_shape_forHLS = (
        (None, n_in, size_in_height, size_in_width) if '2d' in pooling.__name__ else (None, n_in, size_in_width)
    )
    X_input = np.random.rand(*input_shape)

    model = torch.nn.Sequential(pooling(2, padding=padds)).to()
    model.eval()
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model)
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_pooling_{pooling.__name__}_padds_{padds}_backend_{backend}')
    hls_model = convert_from_pytorch_model(
        model, input_shape_forHLS, hls_config=config, output_dir=output_dir, backend=backend
    )
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
        assert hls_pool.attributes['name'] == "layer" + poolNode.name
        assert hls_pool.attributes['class_name'][-2] == str(2)
        assert hls_pool.attributes['stride_height'] == class_object_pool.stride
        assert hls_pool.attributes['stride_width'] == class_object_pool.stride
        assert hls_pool.attributes['pool_height'] == class_object_pool.kernel_size
        assert hls_pool.attributes['pool_width'] == class_object_pool.kernel_size
        assert hls_pool.attributes['padding'] == 'valid' if class_object_pool.padding == 0 else 'same'

    elif '1d' in pooling.__name__:
        if "Max" in pooling.__name__:
            assert hls_pool.attributes['name'] == "layer" + poolNode.name
            assert hls_pool.attributes['class_name'][-2] == str(1)
            assert hls_pool.attributes['pool_width'] == class_object_pool.kernel_size
            assert hls_pool.attributes['stride_width'] == class_object_pool.stride
            assert hls_pool.attributes['padding'] == 'valid' if class_object_pool.padding == 0 else 'same'

        else:
            assert hls_pool.attributes['name'] == "layer" + poolNode.name
            assert hls_pool.attributes['class_name'][-2] == str(1)
            assert hls_pool.attributes['pool_width'] == class_object_pool.kernel_size[0]
            assert hls_pool.attributes['stride_width'] == class_object_pool.stride[0]
            assert hls_pool.attributes['padding'] == 'same' if class_object_pool.padding == 0 else 'valid'
