"""Depthwise Conv1D / Conv2D expressed as a grouped convolution (groups == channels).

A Keras ``Conv1D`` / ``Conv2D`` with ``groups == in_channels`` is a depthwise
convolution, but the keras_v3 ``ConvHandler`` routed it to the dense Conv path,
loading the (k, 1, n_chan) depthwise kernel into a dense matmul and silently
producing wrong output. It is now reshaped to the depthwise layout and emitted as
a ``DepthwiseConv``. General grouped convolutions (1 < groups < in_channels) and
depth_multiplier > 1 have no correct hls4ml kernel and are rejected with a clear
error rather than silently miscomputed.
"""

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
    """Conv1D(groups == channels) must be emitted as DepthwiseConv1D and match Keras."""
    n_chan = 4
    X = np.random.rand(10, 16, n_chan)
    X = np.round(X * 2**10) * 2**-10  # exact on the fixed-point grid
    model = keras.Sequential([Input((16, n_chan)), Conv1D(n_chan, 3, padding=padding, groups=n_chan, name='gc')])
    model.compile()

    config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', default_precision='fixed<32,12>', backend=backend
    )
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    # Routed to the depthwise implementation, not a dense Conv1D (which produced garbage pre-fix).
    assert hls_model.graph['gc'].class_name == 'DepthwiseConv1D'
    hls_model.compile()

    y_keras = model.predict(X, verbose=0)
    y_hls = hls_model.predict(X).reshape(y_keras.shape)
    np.testing.assert_allclose(y_hls, y_keras, rtol=1e-2, atol=0.01)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_depthwise2d_via_groups(test_case_id, backend, io_type):
    """Conv2D(groups == channels) must be emitted as DepthwiseConv2D and match Keras."""
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
    """General grouped convs and depth_multiplier > 1 have no correct hls4ml kernel and
    must raise at conversion rather than be emitted as a (wrong) dense or depthwise conv."""
    model = keras.Sequential([Input((16, n_chan)), Conv1D(filters, 3, padding='same', groups=groups, name='gc')])
    model.compile()
    output_dir = str(test_root_path / test_case_id)
    # The keras_v3 ConvHandler runs during model parsing (config_from_keras_model already
    # invokes it), so the rejection surfaces as soon as the model is parsed.
    with pytest.raises(NotImplementedError):
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, output_dir=output_dir, backend='Vitis', io_type='io_parallel'
        )
