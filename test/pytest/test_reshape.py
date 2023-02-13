""" Test that reshape is properly handled by optimizers.
"""

import pytest
import tensorflow as tf

import hls4ml


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_reshape_parallel(backend, io_type):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(10),
            tf.keras.layers.Dense(10 * 3),
            tf.keras.layers.Reshape((10, 3)),
            tf.keras.layers.ReLU(),
        ]
    )
    model.compile(optimizer='adam', loss='mse')
    config = hls4ml.utils.config_from_keras_model(model)
    output_dir = f'hls4mlprj_reshape_{backend}_{io_type}'
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, io_type=io_type, backend=backend
    )
    hls_model.compile()
