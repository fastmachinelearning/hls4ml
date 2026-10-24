from pathlib import Path

import keras
import numpy as np
import pytest
import torch

import hls4ml

test_root_path = Path(__file__).parent

atol = 1e-3

# (input width, kernel size, stride, padding, output_padding)
conv1d_cases = [
    pytest.param(8, 3, 2, 'valid', None, id='k3_s2_valid'),
    pytest.param(8, 4, 2, 'same', None, id='k4_s2_same'),
    pytest.param(8, 2, 3, 'valid', None, id='k2_s3_valid'),
    pytest.param(8, 3, 2, 'same', 1, id='k3_s2_same_op1'),
]

# (input shape, kernel size, strides, padding, output_padding)
conv2d_cases = [
    pytest.param((5, 5), (3, 3), (2, 2), 'valid', None, id='k3x3_s2x2_valid'),
    pytest.param((4, 6), (3, 3), (2, 2), 'same', None, id='in4x6_k3x3_s2x2_same'),
    pytest.param((4, 5), (3, 3), (2, 3), 'same', (1, 2), id='k3x3_s2x3_same_op'),
]


def make_model(layer, input_shape, seed=0, **kwargs):
    model = keras.Sequential([keras.Input(input_shape), layer(name='convtr', **kwargs)])
    rng = np.random.default_rng(seed)
    model.set_weights([rng.uniform(-1, 1, w.shape) for w in model.get_weights()])
    return model


def convert(model, test_case_id, backend, strategy='Latency', layer_config=None):
    config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', default_precision='fixed<32,16>', backend=backend
    )
    config['Model']['Strategy'] = strategy
    if layer_config is not None:
        config['LayerName']['convtr'].update(layer_config)
    return hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(test_root_path / test_case_id),
        backend=backend,
        io_type='io_parallel',
    )


def check_prediction(model, hls_model, seed=1):
    X = np.random.default_rng(seed).uniform(-1, 1, (10, *model.input_shape[1:])).astype(np.float32)
    y_keras = model.predict(X, verbose=0)
    hls_model.compile()
    y_hls = hls_model.predict(X).reshape(y_keras.shape)
    np.testing.assert_allclose(y_hls, y_keras, rtol=0, atol=atol)


@pytest.mark.parametrize('strategy', ['Latency', 'Resource'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('in_width, kernel, stride, padding, output_padding', conv1d_cases)
def test_conv1d_transpose(test_case_id, in_width, kernel, stride, padding, output_padding, backend, strategy):
    model = make_model(
        keras.layers.Conv1DTranspose,
        (in_width, 3),
        filters=4,
        kernel_size=kernel,
        strides=stride,
        padding=padding,
        output_padding=output_padding,
    )
    hls_model = convert(model, test_case_id, backend, strategy=strategy)
    check_prediction(model, hls_model)


@pytest.mark.parametrize('strategy', ['Latency', 'Resource'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('in_shape, kernel, strides, padding, output_padding', conv2d_cases)
def test_conv2d_transpose(test_case_id, in_shape, kernel, strides, padding, output_padding, backend, strategy):
    model = make_model(
        keras.layers.Conv2DTranspose,
        (*in_shape, 2),
        filters=3,
        kernel_size=kernel,
        strides=strides,
        padding=padding,
        output_padding=output_padding,
    )
    hls_model = convert(model, test_case_id, backend, strategy=strategy)
    check_prediction(model, hls_model)


@pytest.mark.parametrize(
    'dim, strategy, layer_config',
    [
        # Valid reuse factors depend on the 12 x 4 (1D) and 24 x 3 (2D) weight matrix of all kernel elements of a
        # block. With 4 (1D) and 3 (2D), some multipliers take inputs from two output groups
        pytest.param('1d', 'Resource', {'ReuseFactor': 4}, id='1d_rf4'),
        pytest.param('2d', 'Resource', {'ReuseFactor': 3}, id='2d_rf3'),
        # The layer has 9 output blocks, so a parallelization factor of 3 splits them into 3 parts
        pytest.param('2d', 'Latency', {'ParallelizationFactor': 3}, id='2d_latency_pf3'),
        pytest.param('2d', 'Resource', {'ParallelizationFactor': 3}, id='2d_resource_pf3'),
    ],
)
def test_conv_transpose_settings(test_case_id, dim, strategy, layer_config):
    if dim == '1d':
        model = make_model(keras.layers.Conv1DTranspose, (8, 3), filters=4, kernel_size=4, strides=2)
    else:
        model = make_model(keras.layers.Conv2DTranspose, (2, 2, 2), filters=3, kernel_size=(4, 3), strides=(2, 2))
    hls_model = convert(model, test_case_id, 'Vivado', strategy=strategy, layer_config=layer_config)

    layer = hls_model.graph['convtr']
    if 'ReuseFactor' in layer_config:
        assert layer.get_attr('reuse_factor') == layer_config['ReuseFactor']
    else:
        assert layer.get_attr('n_partitions') == 3
    check_prediction(model, hls_model)


def test_conv_transpose_layer(test_case_id):
    """The layer has only the real kernel elements and the configured types."""

    model = make_model(keras.layers.Conv2DTranspose, (4, 5, 2), filters=3, kernel_size=(3, 3), strides=(2, 3))
    hls_model = convert(
        model, test_case_id, 'Vivado', layer_config={'Precision': {'weight': 'fixed<12,4>', 'result': 'fixed<20,8>'}}
    )

    layer = hls_model.graph['convtr']
    # Every one of the 3 x 3 kernel elements is in exactly one of the 2 x 3 groups: no zero weights are added
    assert layer.get_attr('n_buf') == 3 * 3 * 2
    assert layer.get_weights('weight').data.shape == (3 * 3 * 2, 3)
    assert layer.get_weights('weight').type.precision.width == 12
    assert layer.get_output_variable().type.precision.width == 20
    # As for Conv, a layer computed in several parts needs the dataflow style
    assert hls_model.config.pipeline_style == 'dataflow'


def conv_transpose_reference(x, layer):
    """Transposed convolution computed one kernel element at a time, for the cases Keras on TensorFlow cannot run."""

    kernel, bias = layer.get_weights()
    n, *in_shape, _ = x.shape
    *kernel_size, n_filt, _ = kernel.shape
    out_shape = layer.output.shape[1:-1]
    full_shape = [
        (i - 1) * s + (k - 1) * d + 1 for i, s, k, d in zip(in_shape, layer.strides, kernel_size, layer.dilation_rate)
    ]
    if layer.padding == 'same':
        crop = [max(f - o, 0) // 2 for f, o in zip(full_shape, out_shape)]
    else:
        crop = [0] * len(out_shape)
    y = np.zeros((n, *[max(f, c + o) for f, c, o in zip(full_shape, crop, out_shape)], n_filt))
    for pos in np.ndindex(*in_shape):
        for element in np.ndindex(*kernel_size):
            out_pos = tuple(p * s + t * d for p, s, t, d in zip(pos, layer.strides, element, layer.dilation_rate))
            y[(slice(None), *out_pos)] += x[(slice(None), *pos)] @ kernel[element].T
    return y[(slice(None), *[slice(c, c + o) for c, o in zip(crop, out_shape)])] + bias


# (input shape, kernel size, dilation rate, padding); Keras allows dilation only with stride 1, and Keras 2 only the same
# dilation rate on both axes
dilation_cases = [
    pytest.param((6, 2), 4, 3, 'same', id='1d_k4_d3_same'),
    pytest.param((4, 5, 2), (3, 2), (2, 2), 'same', id='2d_k3x2_d2_same'),
]


@pytest.mark.parametrize('strategy', ['Latency', 'Resource'])
@pytest.mark.parametrize('in_shape, kernel, dilation, padding', dilation_cases)
def test_conv_transpose_dilation(test_case_id, in_shape, kernel, dilation, padding, strategy):
    layer = keras.layers.Conv1DTranspose if len(in_shape) == 2 else keras.layers.Conv2DTranspose
    model = make_model(layer, in_shape, filters=3, kernel_size=kernel, dilation_rate=dilation, padding=padding)
    hls_model = convert(model, test_case_id, 'Vivado', strategy=strategy)

    X = np.random.default_rng(1).uniform(-1, 1, (10, *in_shape))
    y_ref = conv_transpose_reference(X, model.get_layer('convtr'))
    hls_model.compile()
    y_hls = hls_model.predict(X).reshape(y_ref.shape)
    np.testing.assert_allclose(y_hls, y_ref, rtol=0, atol=atol)


# (input shape, kernel size, stride, padding, output_padding, dilation); PyTorch allows stride and dilation together
pytorch_cases = [
    pytest.param((3, 8), 3, 2, 1, 1, 1, id='1d_k3_s2_p1_op1'),
    pytest.param((3, 8), 4, 3, 0, 0, 2, id='1d_k4_s3_d2'),
    pytest.param((2, 4, 5), (3, 3), (2, 2), (1, 0), (1, 0), (1, 1), id='2d_k3x3_s2_p1x0_op1x0'),
    pytest.param((2, 4, 5), (3, 2), (2, 3), (1, 1), (1, 2), (2, 1), id='2d_k3x2_s2x3_p1_op1x2_d2x1'),
]


@pytest.mark.parametrize('strategy', ['Latency', 'Resource'])
@pytest.mark.parametrize('in_shape, kernel, stride, padding, output_padding, dilation', pytorch_cases)
def test_conv_transpose_pytorch(test_case_id, in_shape, kernel, stride, padding, output_padding, dilation, strategy):
    layer = torch.nn.ConvTranspose1d if len(in_shape) == 2 else torch.nn.ConvTranspose2d
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        layer(in_shape[0], 3, kernel, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation)
    )
    model.eval()

    config = hls4ml.utils.config_from_pytorch_model(
        model, in_shape, default_precision='fixed<32,16>', channels_last_conversion='full', transpose_outputs=True
    )
    config['Model']['Strategy'] = strategy
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model, hls_config=config, output_dir=str(test_root_path / test_case_id), backend='Vivado', io_type='io_parallel'
    )

    X = np.random.default_rng(1).uniform(-1, 1, (10, *in_shape)).astype(np.float32)
    y_torch = model(torch.tensor(X)).detach().numpy()
    hls_model.compile()
    y_hls = hls_model.predict(X).reshape(y_torch.shape)
    np.testing.assert_allclose(y_hls, y_torch, rtol=0, atol=atol)
