from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense

import hls4ml

test_root_path = Path(__file__).parent


@pytest.mark.parametrize(
    'backend, strategy',
    [
        ('Vivado', 'Latency'),
        ('Vivado', 'Resource'),
        ('Vitis', 'Latency'),
        ('Vitis', 'Resource'),
        ('Quartus', 'Resource'),
        ('oneAPI', 'Resource'),
        ('Catapult', 'Latency'),
        ('Catapult', 'Resource'),
    ],
)
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('shape', [(4, 3), (4, 1), (2, 3, 2), (1, 3, 1)])
def test_multi_dense(backend, strategy, io_type, shape):
    model = tf.keras.models.Sequential()
    model.add(Dense(7, input_shape=shape, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.compile(optimizer='adam', loss='mse')

    X_input = np.random.rand(100, *shape)
    X_input = np.round(X_input * 2**10) * 2**-10  # make it an exact ap_fixed<16,6>

    keras_prediction = model.predict(X_input)

    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    config['Model']['Strategy'] = strategy
    shapestr = '_'.join(str(x) for x in shape)
    output_dir = str(test_root_path / f'hls4mlprj_multi_dense_{backend}_{strategy}_{io_type}_{shapestr}')

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )

    hls_model.compile()

    hls_prediction = hls_model.predict(X_input).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=1e-2, atol=0.01)
