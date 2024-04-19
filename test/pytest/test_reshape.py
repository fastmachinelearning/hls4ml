""" Test that reshape is properly handled by optimizers.
"""

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

import hls4ml

test_root_path = Path(__file__).parent


def randX(batch_size, N):
    return np.random.rand(batch_size, N)


@pytest.fixture(scope='module')
def randX_20_10():
    return randX(20, 10)


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus', 'Catapult', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_reshape_parallel(randX_20_10, backend, io_type):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(10 * 3),
            tf.keras.layers.Reshape((10, 3)),
            tf.keras.layers.ReLU(),
        ]
    )
    model.compile(optimizer='adam', loss='mse')
    config = hls4ml.utils.config_from_keras_model(model, default_precision='fixed<32,16>')
    prj_name = f'hls4mlprj_reshape_{backend}_{io_type}'
    output_dir = str(test_root_path / prj_name)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, io_type=io_type, backend=backend
    )
    hls_model.compile()

    X = randX_20_10
    y_qkeras = model.predict(X)
    y_hls4ml = hls_model.predict(X)

    # check that the values are close
    np.testing.assert_allclose(y_qkeras.ravel(), y_hls4ml.ravel(), atol=0.02)
