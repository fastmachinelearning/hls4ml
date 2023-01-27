""" Test that reshape is properly handled by optimizers.
"""

import pytest
import tensorflow as tf

import hls4ml


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
def test_reshape_parallel(backend):
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
    output_dir = f'hls4mlprj_reshape_parallel_{backend}'
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir, backend=backend)
    hls_model.compile()


def test_reshape_stream():
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
    output_dir = 'hls4mlprj_reshape_stream'
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, io_type='io_stream'
    )
    hls_model.compile()
