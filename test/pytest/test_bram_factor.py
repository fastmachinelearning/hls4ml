from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense

import hls4ml

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_bram_factor(backend, io_type):
    '''A copy of the test_dense from test_keras_api.py with BramFactor set to 0'''
    model = tf.keras.models.Sequential()
    model.add(
        Dense(
            2,
            input_shape=(1,),
            name='Dense',
            use_bias=True,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=1, maxval=10),
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )
    )
    model.add(Activation(activation='elu', name='Activation'))
    model.compile(optimizer='adam', loss='mse')

    X_input = np.random.rand(100, 1)

    keras_prediction = model.predict(X_input)

    config = hls4ml.utils.config_from_keras_model(model)
    config["Model"]["BramFactor"] = 0
    output_dir = str(test_root_path / f'hls4mlprj_bram_factor_{backend}_{io_type}')

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, io_type=io_type, backend=backend
    )

    hls_model.compile()

    hls_prediction = hls_model.predict(X_input)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=1e-2, atol=0.01)

    # Check that there weights are actually remote
    model_brams = [var for var in hls_model.get_weight_variables() if var.storage.lower() == 'bram']
    assert len(model_brams) == 2
