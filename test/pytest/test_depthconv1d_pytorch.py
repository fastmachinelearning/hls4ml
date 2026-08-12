from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import hls4ml

test_root_path = Path(__file__).parent

padds_options = [0, 1]
strides_options = [(1), (2)]
kernel_options = [(2), (3)]
bias_options = [False]
rf_options = [1, 4, 6]  # each rf corresponds to one of the three cases of depthwise resource for io_stream
input_size_options = [4]


@pytest.mark.parametrize('padds', padds_options)
@pytest.mark.parametrize('strides', strides_options)
@pytest.mark.parametrize('kernels', kernel_options)
@pytest.mark.parametrize('bias', bias_options)
@pytest.mark.parametrize(
    'backend, io_type, strategy',
    [
        ('Vivado', 'io_parallel', 'latency'),
        ('Vitis', 'io_parallel', 'latency'),
        ('Vivado', 'io_stream', 'latency'),
        ('Vitis', 'io_stream', 'latency'),
        ('Vivado', 'io_stream', 'resource'),
        ('Vitis', 'io_stream', 'resource'),
        ('Catapult', 'io_stream', 'latency'),
    ],
)
@pytest.mark.parametrize('rf', rf_options)
@pytest.mark.parametrize('input_size', input_size_options)
def test_depthconv1d_pytorch(test_case_id, padds, strides, kernels, bias, io_type, backend, strategy, rf, input_size):
    model = nn.Sequential(
        nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
            kernel_size=kernels,
            stride=strides,
            padding=padds,
            groups=input_size,  # depthwise convolution
            bias=bias,
        ),
    )
    model.eval()

    input_shape = (input_size, 16)
    X_input = np.random.rand(100, *input_shape)  # (batch_size, channels, width)
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    if io_type == 'io_stream':
        X_input = np.ascontiguousarray(X_input.transpose(0, 2, 1))
        channels_last_conversion = 'internal'
        transpose_outputs = False
    else:
        channels_last_conversion = 'full'
        transpose_outputs = True

    config = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shape=input_shape,
        default_precision='ap_fixed<32,8>',
        channels_last_conversion=channels_last_conversion,
        transpose_outputs=transpose_outputs,
    )

    config['Model']['Strategy'] = strategy
    config['Model']['ReuseFactor'] = rf

    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model, hls_config=config, output_dir=output_dir, io_type=io_type, backend=backend
    )
    assert all(layer.class_name == 'DepthwiseConv1D' for layer in hls_model.get_layers() if 'Conv' in layer.class_name), (
        'depthwise conv must map to DepthwiseConv1D'
    )
    hls_model.compile()
    hls_prediction = hls_model.predict(X_input)
    if io_type == 'io_stream':
        channels_last_shape = pytorch_prediction.transpose(0, 2, 1).shape
        hls_prediction = hls_prediction.reshape(channels_last_shape).transpose(0, 2, 1)
    else:
        hls_prediction = hls_prediction.reshape(pytorch_prediction.shape)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=0.001)
