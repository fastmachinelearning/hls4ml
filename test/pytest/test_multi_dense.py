from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense

import hls4ml

test_root_path = Path(__file__).parent


@pytest.mark.parametrize(
    'backend, io_type',
    [
        ('Quartus', 'io_parallel'),
        ('Vivado', 'io_parallel'),
        ('Vitis', 'io_parallel'),
        ('Vivado', 'io_stream'),
        ('Vivado', 'io_stream'),
        ('Vitis', 'io_stream'),
    ],
)
def test_multi_dense(backend, io_type):
    model = tf.keras.models.Sequential()
    model.add(
        Dense(
            4,
            input_shape=(
                8,
                8,
            ),
            name='Dense',
            use_bias=True,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=1, maxval=10),
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            activation='relu',
        )
    )
    model.compile(optimizer='adam', loss='mse')

    X_input = np.random.rand(100, 8, 8)

    keras_prediction = model.predict(X_input)

    default_precision = 'ap_fixed<32, 16>' if backend in ['Vivado', 'Vitis'] else 'ac_fixed<32, 16, true>'
    config = hls4ml.utils.config_from_keras_model(model, default_precision=default_precision)
    output_dir = str(test_root_path / f'hls4mlprj_multi_dense_{backend}_{io_type}')

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )

    hls_model.compile()

    hls_prediction = hls_model.predict(X_input).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=1e-2, atol=0.01)

    assert list(hls_model.get_layers())[1].class_name == 'PointwiseConv1D'
