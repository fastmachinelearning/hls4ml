import pytest
import numpy as np
from pathlib import Path

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

from hls4ml.utils import config_from_keras_model
from hls4ml.converters import convert_from_keras_model

test_root_path = Path(__file__).parent

# Tests a wide range of RF to ensure the unrolled Dense is correct
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('reuse_factor', [1, 2, 4, 8, 16, 32, 48, 64, 96, 192])
def test_dense_unrolled(io_type, reuse_factor):
    input_shape = (16, )
    X = np.random.rand(100, *input_shape)

    model = Sequential()
    model.add(Dense(12, input_shape=input_shape, kernel_initializer='lecun_uniform', bias_initializer='lecun_uniform'))
    model.compile('adam', 'mse')
    keras_prediction = model.predict(X)

    config = config_from_keras_model(model, default_precision='ac_fixed<32, 16>', default_reuse_factor=reuse_factor)
    config['Model']['Strategy'] = 'Resource'
    config['Model']['DenseResourceImplementation'] = 'Unrolled'
    
    output_dir = str(test_root_path / f'hls4mlprj_dense_unrolled_{io_type}_{reuse_factor}')
    hls_model = convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend='Vivado', io_type=io_type
    )
    hls_model.compile()

    hls_prediction = hls_model.predict(X)
    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=1e-2)

# Tests a wide range RF on streaming Conv2D to ensure the unrolled Dense is correct
@pytest.mark.parametrize('io_type', ['io_stream'])
@pytest.mark.parametrize('reuse_factor', [1, 3, 9, 27, 54, 108])
def test_dense_unrolled_streaming_conv(io_type, reuse_factor):
    input_shape = (8, 8, 3)
    X = np.random.rand(100, *input_shape)

    model = Sequential()
    model.add(Conv2D(4, (3, 3), input_shape=input_shape, kernel_initializer='lecun_uniform', bias_initializer='lecun_uniform'))
    model.add(Flatten())
    model.add(Dense(1, kernel_initializer='lecun_uniform', bias_initializer='lecun_uniform'))
    model.compile('adam', 'mse')
    keras_prediction = model.predict(X)

    config = config_from_keras_model(model, default_precision='ac_fixed<32, 16>', default_reuse_factor=reuse_factor)
    config['Model']['Strategy'] = 'Resource'
    config['Model']['DenseResourceImplementation'] = 'Unrolled'
    
    output_dir = str(test_root_path / f'hls4mlprj_dense_unrolled_conv2d_{io_type}_{reuse_factor}')
    hls_model = convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend='Vivado', io_type=io_type
    )
    hls_model.compile()

    hls_prediction = hls_model.predict(X)
    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=1e-2)
