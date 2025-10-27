from pathlib import Path

import brevitas.nn as qnn
import numpy as np
import pytest
import torch
from brevitas.quant import Int8ActPerTensorFixedPoint, Int8WeightPerTensorFixedPoint, Int8WeightPerTensorFloat
from torch import nn
from torch.nn import Module

import hls4ml
from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model

test_root_path = Path(__file__).parent

quants = {
    'Int8WeightPerTensorFixedPoint': Int8WeightPerTensorFixedPoint,
    'Int8ActPerTensorFixedPoint': Int8ActPerTensorFixedPoint,
    'Int8WeightPerTensorFloat': Int8WeightPerTensorFloat,
}


class QuantModelConv2d(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = qnn.QuantConv2d(3, 6, 5, bias=True, weight_quant=Int8WeightPerTensorFixedPoint)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        return out


class QuantModelConv1d(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = qnn.QuantConv1d(3, 6, 4, bias=True, weight_quant=Int8WeightPerTensorFixedPoint)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        return out


class QuantModelLinear(Module):
    def __init__(self, weight_quant, input_quant):
        super().__init__()
        self.lin1 = qnn.QuantLinear(4, 4, bias=False, weight_quant=quants[weight_quant], input_quant=quants[input_quant])
        self.relu1 = qnn.QuantReLU(act_quant=quants[input_quant])

    def forward(self, x):
        out = self.relu1(self.lin1(x))
        return out


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('weight_quant', ['Int8WeightPerTensorFixedPoint'])
@pytest.mark.parametrize('io_quant', ['Int8ActPerTensorFixedPoint'])
def test_quantlinear(backend, io_type, weight_quant, io_quant):
    model = QuantModelLinear(weight_quant, io_quant)

    x = torch.rand(1, 4)
    pytorch_prediction = model(x).detach().numpy()
    config = config_from_pytorch_model(model, input_shape=(None, 4))
    output_dir = str(test_root_path / f'hls4mlprj_brevitas_linear_{backend}_{io_type}_{weight_quant}_{io_quant}')

    hls_model = convert_from_pytorch_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
    )
    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict(x.detach().numpy()), pytorch_prediction.shape)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0.0, atol=0.05)


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_quantconv1d(backend, io_type):
    model = QuantModelConv1d()

    n_in = 3
    n_out = 6
    size_in = 5

    x = torch.randn(1, n_in, size_in)

    pytorch_prediction = model(x).detach().numpy()
    if io_type == 'io_stream':
        x = np.ascontiguousarray(x.permute(0, 2, 1))
        config = config_from_pytorch_model(
            model, (None, n_in, size_in), channels_last_conversion="internal", transpose_outputs=False
        )
    else:
        config = config_from_pytorch_model(
            model, (None, n_in, size_in), channels_last_conversion="full", transpose_outputs=True
        )

    output_dir = str(test_root_path / f'hls4mlprj_brevitas_conv1d_{backend}_{io_type}')

    from hls4ml.utils.torch import CustomFXTracer

    tracer = CustomFXTracer()
    traced_model = tracer.trace(model)
    nNodes = 0
    convNode = None
    for _node in traced_model.nodes:
        nNodes += 1
        if nNodes == 2:
            convNode = _node

    children = {c[0]: c[1] for c in model.named_children()}
    class_object_conv = children[convNode.target]

    out_width = int(
        (
            size_in
            + 2 * class_object_conv.padding[0]
            - class_object_conv.dilation[0] * (class_object_conv.kernel_size[0] - 1)
            - 1
        )
        / class_object_conv.stride[0]
        + 1
    )  # following https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

    hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)
    hls_model.compile()

    if io_type == 'io_stream':
        hls_prediction = np.transpose(np.reshape(hls_model.predict(x), (1, out_width, n_out)), (0, 2, 1))
    else:
        hls_prediction = np.reshape(hls_model.predict(x.detach().numpy()), pytorch_prediction.shape)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_quantconv2d(backend, io_type):
    model = QuantModelConv2d()

    n_in = 3
    n_out = 6
    size_in_width = 5
    size_in_height = 6

    x = torch.randn(1, n_in, size_in_height, size_in_width)

    pytorch_prediction = model(x).detach().numpy()
    if io_type == 'io_stream':
        x = np.ascontiguousarray(x.permute(0, 2, 3, 1))
        config = config_from_pytorch_model(
            model, (None, n_in, size_in_height, size_in_width), channels_last_conversion="internal", transpose_outputs=False
        )
    else:
        config = config_from_pytorch_model(
            model, (None, n_in, size_in_height, size_in_width), channels_last_conversion="full", transpose_outputs=True
        )

    output_dir = str(test_root_path / f'hls4mlprj_brevitas_conv2d_{backend}_{io_type}')

    from hls4ml.utils.torch import CustomFXTracer

    tracer = CustomFXTracer()
    traced_model = tracer.trace(model)

    nNodes = 0
    convNode = None
    for _node in traced_model.nodes:
        nNodes += 1
        if nNodes == 2:
            convNode = _node

    children = {c[0]: c[1] for c in model.named_children()}
    class_object_conv = children[convNode.target]

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

    hls_model = convert_from_pytorch_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
    )
    hls_model.compile()

    if io_type == 'io_stream':
        hls_prediction = np.transpose(np.reshape(hls_model.predict(x), (1, out_height, out_width, n_out)), (0, 3, 1, 2))
    else:
        hls_prediction = np.reshape(hls_model.predict(x.detach().numpy()), pytorch_prediction.shape)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0.0, atol=0.05)


in_height = 6
in_width = 8
in_feat = 4

size = 2
atol = 5e-3


@pytest.fixture(scope='module')
def data_1d():
    X = np.random.rand(100, in_feat, in_width)
    return X


@pytest.fixture(scope='module')
def data_2d():
    X = np.random.rand(100, in_feat, in_height, in_width)
    return X


class QuantUpsample1DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = qnn.QuantIdentity(act_quant=Int8ActPerTensorFixedPoint, return_quant_tensor=True)
        self.upsample = qnn.QuantUpsample(scale_factor=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.upsample(self.identity(x)))


class QuantUpsample2DModel(nn.Module):
    def __init__(self):
        super().__init__()
        # this scale_factor tests proper output shape calculation with fractional scaling and parsing per-axis scales
        self.identity = qnn.QuantIdentity(act_quant=Int8ActPerTensorFixedPoint, return_quant_tensor=True)
        self.upsample = qnn.QuantUpsamplingNearest2d(scale_factor=(1, 2.4))  # Would also work with Upsample(mode='nearest')
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.upsample(self.identity(x)))


@pytest.mark.parametrize('io_type', ['io_parallel'])  # Quant upsampling layers currently not supported in io_stream
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus'])
def test_pytorch_upsampling1d(data_1d, io_type, backend):
    model = QuantUpsample1DModel()

    config = hls4ml.utils.config_from_pytorch_model(
        model,
        (None, in_feat, in_width),
        default_precision='ap_fixed<16,6>',
        channels_last_conversion="internal",
        transpose_outputs=False,
    )
    odir = str(test_root_path / f'hls4mlprj_pytorch_upsampling_1d_{backend}_{io_type}')
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model, hls_config=config, io_type=io_type, output_dir=odir, backend=backend
    )
    hls_model.compile()

    data_1d_t = np.ascontiguousarray(data_1d.transpose([0, 2, 1]))

    pytorch_prediction = model(torch.Tensor(data_1d)).value.detach().numpy()
    hls_prediction = hls_model.predict(data_1d_t)

    pred_shape = list(pytorch_prediction.shape)
    pred_shape.append(pred_shape.pop(1))  # Transpose shape to channels_last
    hls_prediction = hls_prediction.reshape(pred_shape).transpose([0, 2, 1])  # Transpose back

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)


@pytest.mark.parametrize('io_type', ['io_parallel'])  # Fractional scaling doesn't work with io_stream
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus'])
def test_pytorch_upsampling2d(data_2d, io_type, backend):
    model = QuantUpsample2DModel()

    config = hls4ml.utils.config_from_pytorch_model(
        model,
        (in_feat, in_height, in_width),
        default_precision='ap_fixed<16,6>',
        channels_last_conversion="full",  # With conversion to channels_last
        transpose_outputs=True,
    )
    odir = str(test_root_path / f'hls4mlprj_pytorch_upsampling_2d_{backend}_{io_type}')
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model, hls_config=config, io_type=io_type, output_dir=odir, backend=backend
    )
    hls_model.compile()

    pytorch_prediction = model(torch.Tensor(data_2d)).value.detach().numpy().flatten()
    hls_prediction = hls_model.predict(data_2d).flatten()

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)


class QuantEltwiseAddModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.add = qnn.QuantEltwiseAdd(input_quant=Int8ActPerTensorFixedPoint, output_quant=Int8ActPerTensorFixedPoint)

    def forward(self, x, y):
        return self.add(x, y)


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus'])
def test_brevitas_quanteltwiseadd(io_type, backend):
    model = QuantEltwiseAddModel()

    x = torch.rand(1, 4, 4)
    y = torch.rand(1, 4, 4)

    pytorch_prediction = model(torch.Tensor(x), torch.Tensor(y)).detach().numpy()

    config = hls4ml.utils.config_from_pytorch_model(
        model,
        [(None, 4, 4), (None, 4, 4)],
        default_precision='ap_fixed<16,6>',
        channels_last_conversion="off",
        transpose_outputs=False,
    )
    odir = str(test_root_path / f'hls4mlprj_brevitas_quanteltwiseadd_{backend}_{io_type}')
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model, hls_config=config, io_type=io_type, output_dir=odir, backend=backend
    )
    hls_model.compile()

    hls_prediction = hls_model.predict([x.detach().numpy(), y.detach().numpy()])

    pred_shape = pytorch_prediction.shape
    hls_prediction = hls_prediction.reshape(pred_shape)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=5e-2, atol=0.05)
