from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent

in_shape = (10, 8)


@pytest.fixture(scope='module')
def data():
    np.random.seed(0)
    return np.random.rand(100, *in_shape)


@pytest.fixture(scope='module')
def model():
    model = Sequential()
    layer = LayerNormalization(input_shape=in_shape)
    model.add(layer)
    model.compile()

    np.random.seed(0)
    weights = np.random.normal(1.0, 0.1, size=(in_shape[-1],))
    biases = np.random.normal(0.0, 0.1, size=(in_shape[-1],))
    layer.set_weights([weights, biases])
    return model


@pytest.fixture(scope='module')
def custom_epsilon_model():
    model = Sequential()
    layer = LayerNormalization(input_shape=in_shape, epsilon=1e-2)
    model.add(layer)
    model.compile()
    return model


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_layernorm_parsing(custom_epsilon_model, backend):
    custom_config = hls4ml.utils.config_from_keras_model(custom_epsilon_model, granularity='name', backend=backend)
    custom_config['LayerName']['layer_normalization']['Precision']['accum'] = 'ap_fixed<10,4>'
    custom_config['LayerName']['layer_normalization']['table_t'] = 'ap_fixed<12,5>'
    custom_config['LayerName']['layer_normalization']['TableSize'] = 2048
    custom_config['LayerName']['layer_normalization']['TableRangePower2'] = 1
    output_dir = str(test_root_path / f'hls4mlprj_layernorm_config_{backend}_io_parallel')
    hls_model = hls4ml.converters.convert_from_keras_model(
        custom_epsilon_model, backend=backend, hls_config=custom_config, io_type='io_parallel', output_dir=output_dir
    )
    hls_model.write()

    # Check that custom configuration is picked up correctly
    hls_layer = list(hls_model.get_layers())[1]  # 0 is input, 1 is LayerNorm
    assert hls_layer.attributes['accum_t'].precision.definition_cpp() == 'ap_fixed<10,4>'
    assert hls_layer.attributes['table_t'].precision.definition_cpp() == 'ap_fixed<12,5>'
    assert hls_layer.attributes['table_size'] == 2048
    assert hls_layer.attributes['table_range_power2'] == 1
    assert hls_layer.attributes['epsilon_power_of_10'] == 2


# Currently only Vivado/Vitis in io_parallel mode is supported
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_layernorm_accuracy(model, data, backend):
    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    output_dir = str(test_root_path / f'hls4mlprj_layernorm_{backend}_io_parallel')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, backend=backend, hls_config=config, io_type='io_parallel', output_dir=output_dir
    )
    hls_model.compile()

    # Predict
    y_keras = model.predict(data).flatten()
    y_hls = hls_model.predict(data).flatten()
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=5e-2, verbose=True)
