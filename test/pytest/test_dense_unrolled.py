from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import GRU, LSTM, Conv1D, Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential

from hls4ml.converters import convert_from_keras_model
from hls4ml.utils import config_from_keras_model

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('strategy', ['ResourceUnrolled', 'resource_unrolled', 'Resource_Unrolled'])
def test_resource_unrolled_parsing(strategy):
    model = Sequential()
    model.add(
        Dense(8, input_shape=(16,), kernel_initializer='lecun_uniform', bias_initializer='lecun_uniform', name='dense')
    )
    model.compile('adam', 'mse')

    config = config_from_keras_model(model, default_precision='ac_fixed<32, 16>', backend='Vitis', default_reuse_factor=8)
    config['Model']['Strategy'] = strategy

    output_dir = str(test_root_path / f'hls4mlprj_resource_unrolled_parsing_{strategy}')
    hls_model = convert_from_keras_model(model, hls_config=config, output_dir=output_dir, backend='Vitis')

    # Check if strategy was not overridden
    assert list(hls_model.get_layers())[1].get_attr('strategy') == 'resource_unrolled'


# Tests a wide range of RF to ensure the unrolled resource kernel is correct
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('reuse_factor', [1, 2, 4, 8, 16, 32, 48, 64, 96, 192])
@pytest.mark.parametrize('backend', ['Vitis', 'Vivado'])
def test_resource_unrolled_dense(io_type, reuse_factor, backend):
    input_shape = (16,)
    X = np.random.rand(100, *input_shape)

    model = Sequential()
    model.add(
        Dense(
            12, input_shape=input_shape, kernel_initializer='lecun_uniform', bias_initializer='lecun_uniform', name='dense'
        )
    )
    model.compile('adam', 'mse')
    keras_prediction = model.predict(X)

    config = config_from_keras_model(
        model, default_precision='ac_fixed<32, 16>', backend=backend, default_reuse_factor=reuse_factor
    )
    config['Model']['Strategy'] = 'ResourceUnrolled'

    output_dir = str(test_root_path / f'hls4mlprj_resource_unrolled_dense_{io_type}_{reuse_factor}_{backend}')
    hls_model = convert_from_keras_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)

    # Check if strategy was not overridden
    assert list(hls_model.get_layers())[1].get_attr('strategy') == 'resource_unrolled' if reuse_factor > 1 else 'latency'

    hls_model.compile()

    hls_prediction = hls_model.predict(X)
    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=1e-2)


# Tests a wide range RF on streaming Conv1D/2D to ensure the unrolled resource kernel is correct
@pytest.mark.parametrize('dim', [1, 2])
@pytest.mark.parametrize('io_type', ['io_stream'])
@pytest.mark.parametrize('reuse_factor', [1, 3, 9, 27, 54, 108])
def test_resource_unrolled_streaming_conv(dim, io_type, reuse_factor):
    input_shape = (8,) * dim + (3,)
    X = np.random.rand(100, *input_shape)
    conv_class = Conv1D if dim == 1 else Conv2D

    model = Sequential()
    model.add(
        conv_class(
            4, (3,) * dim, input_shape=input_shape, kernel_initializer='lecun_uniform', bias_initializer='lecun_uniform'
        )
    )
    model.add(Flatten())
    model.add(Dense(1, kernel_initializer='lecun_uniform', bias_initializer='lecun_uniform'))
    model.compile('adam', 'mse')
    keras_prediction = model.predict(X)

    config = config_from_keras_model(model, default_precision='ac_fixed<32, 16>', default_reuse_factor=reuse_factor)
    config['Model']['Strategy'] = 'ResourceUnrolled'

    output_dir = str(test_root_path / f'hls4mlprj_resource_unrolled_conv{dim}d_{io_type}_{reuse_factor}')
    hls_model = convert_from_keras_model(model, hls_config=config, output_dir=output_dir, backend='Vivado', io_type=io_type)

    # Check if strategy was not overridden
    assert list(hls_model.get_layers())[1].get_attr('strategy') == 'resource_unrolled' if reuse_factor > 1 else 'latency'

    hls_model.compile()

    hls_prediction = hls_model.predict(X)
    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=1e-2)


@pytest.mark.parametrize('rnn_layer', [LSTM, GRU])
@pytest.mark.parametrize('backend', ['Vitis', 'Vivado'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('static', [True, False])
@pytest.mark.parametrize('reuse_factor', [1, 4, 32, 128])  # RF=128 also tests if setting closest RF works well
def test_resource_unrolled_rnn(rnn_layer, backend, io_type, static, reuse_factor):
    # Subtract 0.5 to include negative values
    input_shape = (12, 8)
    X = np.random.rand(50, *input_shape) - 0.5

    layer_name = rnn_layer.__name__.lower()
    keras_model = Sequential()
    keras_model.add(
        rnn_layer(
            units=8,
            input_shape=input_shape,
            kernel_initializer='lecun_uniform',
            recurrent_initializer='lecun_uniform',
            bias_initializer='lecun_uniform',
            return_sequences=False,
            name=layer_name,
        )
    )
    keras_model.compile()

    default_precision = 'ap_fixed<32, 16>' if backend in ['Vivado', 'Vitis'] else 'ac_fixed<32, 16, true>'
    hls_config = config_from_keras_model(
        keras_model, granularity='name', default_precision=default_precision, backend=backend
    )
    hls_config['LayerName'][layer_name]['static'] = static
    hls_config['LayerName'][layer_name]['Strategy'] = 'ResourceUnrolled'
    hls_config['LayerName'][layer_name]['ReuseFactor'] = reuse_factor
    prj_name = f'hls4mlprj_resource_unrolled_rnn_{layer_name}_static_{int(static)}_{io_type}_{reuse_factor}_{backend}'
    output_dir = str(test_root_path / prj_name)

    hls_model = convert_from_keras_model(
        keras_model, hls_config=hls_config, output_dir=output_dir, backend=backend, io_type=io_type
    )

    # Check if strategy was not overridden
    assert list(hls_model.get_layers())[1].get_attr('strategy') == 'resource_unrolled' if reuse_factor > 1 else 'latency'

    hls_model.compile()

    keras_prediction = keras_model.predict(X)
    hls_prediction = hls_model.predict(X)
    np.testing.assert_allclose(hls_prediction.flatten(), keras_prediction.flatten(), rtol=0.0, atol=5e-2)
