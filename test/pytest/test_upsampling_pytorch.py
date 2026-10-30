from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import hls4ml

test_root_path = Path(__file__).parent

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


class Upsample1DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.upsample(x))


class Upsample2DModel(nn.Module):
    def __init__(self):
        super().__init__()
        # this scale_factor tests proper output shape calculation with fractional scaling and parsing per-axis scales
        self.upsample = nn.UpsamplingNearest2d(scale_factor=(1, 2.4))  # Would also work with Upsample(mode='nearest')
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.upsample(x))


@pytest.mark.parametrize('io_type', ['io_stream', 'io_parallel'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_pytorch_upsampling1d(test_case_id, data_1d, io_type, backend):
    model = Upsample1DModel()

    config = hls4ml.utils.config_from_pytorch_model(
        model,
        (None, in_feat, in_width),
        default_precision='ap_fixed<16,6>',
        channels_last_conversion='internal',
        transpose_outputs=False,
    )
    odir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model, hls_config=config, io_type=io_type, output_dir=odir, backend=backend
    )
    hls_model.compile()

    data_1d_t = np.ascontiguousarray(data_1d.transpose([0, 2, 1]))

    pytorch_prediction = model(torch.Tensor(data_1d)).detach().numpy()
    hls_prediction = hls_model.predict(data_1d_t)

    pred_shape = list(pytorch_prediction.shape)
    pred_shape.append(pred_shape.pop(1))  # Transpose shape to channels_last
    hls_prediction = hls_prediction.reshape(pred_shape).transpose([0, 2, 1])  # Transpose back

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)


@pytest.mark.parametrize('io_type', ['io_parallel'])  # Fractional scaling doesn't work with io_stream
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_pytorch_upsampling2d(test_case_id, data_2d, io_type, backend):
    model = Upsample2DModel()

    config = hls4ml.utils.config_from_pytorch_model(
        model,
        (in_feat, in_height, in_width),
        default_precision='ap_fixed<16,6>',
        channels_last_conversion='full',  # With conversion to channels_last
        transpose_outputs=True,
    )
    odir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model, hls_config=config, io_type=io_type, output_dir=odir, backend=backend
    )
    hls_model.compile()

    pytorch_prediction = model(torch.Tensor(data_2d)).detach().numpy().flatten()
    hls_prediction = hls_model.predict(data_2d).flatten()

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)


class UpsampleSizeModel(nn.Module):
    """Upsampling specified through the target size instead of a scale factor."""

    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(size=in_width * 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.upsample(x))


@pytest.mark.parametrize('backend', ['Vivado'])
def test_pytorch_upsampling_size(test_case_id, data_1d, backend):
    model = UpsampleSizeModel()

    config = hls4ml.utils.config_from_pytorch_model(
        model,
        (None, in_feat, in_width),
        default_precision='ap_fixed<16,6>',
        channels_last_conversion='internal',
        transpose_outputs=False,
    )
    odir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model, hls_config=config, io_type='io_parallel', output_dir=odir, backend=backend
    )
    hls_model.compile()

    data_1d_t = np.ascontiguousarray(data_1d.transpose([0, 2, 1]))

    pytorch_prediction = model(torch.Tensor(data_1d)).detach().numpy()
    hls_prediction = hls_model.predict(data_1d_t)

    pred_shape = list(pytorch_prediction.shape)
    pred_shape.append(pred_shape.pop(1))  # Transpose shape to channels_last
    hls_prediction = hls_prediction.reshape(pred_shape).transpose([0, 2, 1])  # Transpose back

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)


def test_pytorch_upsampling_fractional_size_rejected(test_case_id):
    class FractionalSizeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.upsample = nn.Upsample(size=in_width + 2)  # not a multiple of the input size

        def forward(self, x):
            return self.upsample(x)

    model = FractionalSizeModel()

    odir = str(test_root_path / test_case_id)
    with pytest.raises(NotImplementedError, match='integer multiple'):
        # config_from_pytorch_model traces the model too, so the reject can fire already there
        config = hls4ml.utils.config_from_pytorch_model(
            model, (None, in_feat, in_width), channels_last_conversion='internal', transpose_outputs=False
        )
        hls4ml.converters.convert_from_pytorch_model(
            model, hls_config=config, io_type='io_parallel', output_dir=odir, backend='Vivado'
        )


class Upsample2DSizeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.UpsamplingNearest2d(size=(in_height * 2, in_width * 3))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.upsample(x))


@pytest.mark.parametrize('backend', ['Vivado'])
def test_pytorch_upsampling2d_size(test_case_id, data_2d, backend):
    model = Upsample2DSizeModel()

    config = hls4ml.utils.config_from_pytorch_model(
        model,
        (in_feat, in_height, in_width),
        default_precision='ap_fixed<16,6>',
        channels_last_conversion='full',
        transpose_outputs=True,
    )
    odir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model, hls_config=config, io_type='io_parallel', output_dir=odir, backend=backend
    )
    hls_model.compile()

    pytorch_prediction = model(torch.Tensor(data_2d)).detach().numpy().flatten()
    hls_prediction = hls_model.predict(data_2d).flatten()

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)
