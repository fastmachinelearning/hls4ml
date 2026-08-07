"""Test Conv1D/Conv2D with groups == channels is emitted as DepthwiseConv."""

from pathlib import Path

import keras
import numpy as np
import pytest

if keras.__version__ < '3.0':
    pytest.skip('Only applicable to the Keras 3 (keras_v3) converter', allow_module_level=True)

from keras.layers import Conv1D, Conv2D, Input  # noqa: E402

import hls4ml  # noqa: E402

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('padding', ['same', 'valid', 'causal'])
def test_depthwise1d_via_groups(test_case_id, backend, io_type, padding):
    n_chan = 4
    X = np.random.rand(10, 16, n_chan)
    X = np.round(X * 2**10) * 2**-10
    model = keras.Sequential([Input((16, n_chan)), Conv1D(n_chan, 3, padding=padding, groups=n_chan, name='gc')])
    model.compile()

    config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', default_precision='fixed<32,12>', backend=backend
    )
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    assert hls_model.graph['gc'].class_name == 'DepthwiseConv1D'
    hls_model.compile()

    y_keras = model.predict(X, verbose=0)
    y_hls = hls_model.predict(X).reshape(y_keras.shape)
    np.testing.assert_allclose(y_hls, y_keras, rtol=1e-2, atol=0.01)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_depthwise2d_via_groups(test_case_id, backend, io_type):
    n_chan = 4
    X = np.random.rand(10, 8, 8, n_chan)
    X = np.round(X * 2**10) * 2**-10
    model = keras.Sequential([Input((8, 8, n_chan)), Conv2D(n_chan, (3, 3), padding='same', groups=n_chan, name='gc')])
    model.compile()

    config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', default_precision='fixed<32,12>', backend=backend
    )
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    assert hls_model.graph['gc'].class_name == 'DepthwiseConv2D'
    hls_model.compile()

    y_keras = model.predict(X, verbose=0)
    y_hls = hls_model.predict(X).reshape(y_keras.shape)
    np.testing.assert_allclose(y_hls, y_keras, rtol=1e-2, atol=0.01)


@pytest.mark.parametrize(
    'n_chan, filters, groups',
    [(4, 4, 2), (8, 8, 4), (4, 8, 4)],
    ids=['grouped_2of4', 'grouped_4of8', 'depth_multiplier_2'],
)
def test_unsupported_grouped_conv_raises(test_case_id, n_chan, filters, groups):
    """Unsupported grouped convs must raise instead of producing wrong results."""
    model = keras.Sequential([Input((16, n_chan)), Conv1D(filters, 3, padding='same', groups=groups, name='gc')])
    model.compile()
    output_dir = str(test_root_path / test_case_id)
    with pytest.raises(NotImplementedError):
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, output_dir=output_dir, backend='Vitis', io_type='io_parallel'
        )
