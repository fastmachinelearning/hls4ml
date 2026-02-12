import os
from pathlib import Path

import keras
import numpy as np
import pytest
from conftest import get_pytest_case_id

import hls4ml

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('io_type', ['io_parallel'])
@pytest.mark.parametrize('backend', ['Vitis', 'Vivado'])
@pytest.mark.parametrize('use_h5', [True, False])
def test_time_distributed_layer(request, io_type, backend, use_h5):
    input_shape = (5, 10, 10, 3)

    inputs = keras.layers.Input(shape=input_shape)
    conv_2d_layer = keras.layers.Conv2D(4, (3, 3), kernel_initializer='ones', use_bias=False)
    pool_2d_layer = keras.layers.MaxPooling2D((2, 2))
    conv_outputs = keras.layers.TimeDistributed(conv_2d_layer)(inputs)
    outputs = keras.layers.TimeDistributed(pool_2d_layer)(conv_outputs)
    keras_model = keras.models.Model(inputs, outputs)

    out_dir = str(test_root_path / get_pytest_case_id(request))

    config = hls4ml.utils.config.create_config(output_dir=out_dir)

    if use_h5:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        keras_model.save(out_dir + '/time_distributed.h5')
        config['KerasH5'] = out_dir + '/time_distributed.h5'
    else:
        config['KerasModel'] = keras_model

    config['Backend'] = backend
    config['IOType'] = io_type
    config['HLSConfig'] = hls4ml.utils.config_from_keras_model(keras_model, default_precision='fixed<10,8>')

    hls_model = hls4ml.converters.keras_v2_to_hls(config)
    hls_model.compile()

    x = np.random.randint(0, 5, size=(10, *input_shape)).astype('float')
    keras_prediction = keras_model.predict(x)

    hls_prediction = hls_model.predict(x)
    np.testing.assert_allclose(hls_prediction.flatten(), keras_prediction.flatten(), rtol=0, atol=1e-2)


@pytest.mark.parametrize('io_type', ['io_parallel'])
@pytest.mark.parametrize('backend', ['Vitis', 'Vivado'])
@pytest.mark.parametrize('use_h5', [True, False])
def test_time_distributed_layer_lstm(request, io_type, backend, use_h5):
    input_shape = (8, 8)

    inputs = keras.layers.Input(shape=input_shape)
    lstm_layer = keras.layers.LSTM(4, return_sequences=True)
    dense_layer = keras.layers.Dense(1)
    lstm_outputs = (lstm_layer)(inputs)
    outputs = keras.layers.TimeDistributed(dense_layer)(lstm_outputs)
    keras_model = keras.models.Model(inputs, outputs)

    out_dir = str(test_root_path / get_pytest_case_id(request))

    config = hls4ml.utils.config.create_config(output_dir=out_dir)

    if use_h5:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        keras_model.save(out_dir + '/time_distributed.h5')
        config['KerasH5'] = out_dir + '/time_distributed.h5'
    else:
        config['KerasModel'] = keras_model

    config['Backend'] = backend
    config['IOType'] = io_type
    config['HLSConfig'] = hls4ml.utils.config_from_keras_model(keras_model, default_precision='fixed<32,16>')

    hls_model = hls4ml.converters.keras_v2_to_hls(config)
    hls_model.compile()

    x = np.random.rand(10, *input_shape) - 0.5
    keras_prediction = keras_model.predict(x)

    hls_prediction = hls_model.predict(x)
    np.testing.assert_allclose(hls_prediction.flatten(), keras_prediction.flatten(), rtol=0, atol=5e-2)


@pytest.mark.parametrize('io_type', ['io_parallel'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('use_h5', [True, False])
def test_time_distributed_model(request, io_type, backend, use_h5):
    input_shape = (5, 10, 10, 3)

    model_nested = keras.models.Sequential(
        [
            keras.layers.Conv2D(4, (3, 3), kernel_initializer='ones', use_bias=False),
            keras.layers.MaxPooling2D((2, 2)),
        ]
    )

    keras_model = keras.models.Sequential(
        [
            keras.layers.Input(shape=input_shape, batch_size=32),
            keras.layers.TimeDistributed(model_nested),
        ]
    )

    out_dir = str(test_root_path / get_pytest_case_id(request))

    config = hls4ml.utils.config.create_config(output_dir=out_dir)

    if use_h5:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        keras_model.save(out_dir + '/time_distributed.h5')
        config['KerasH5'] = out_dir + '/time_distributed.h5'
    else:
        config['KerasModel'] = keras_model

    config['Backend'] = backend
    config['IOType'] = io_type
    config['HLSConfig'] = hls4ml.utils.config_from_keras_model(keras_model, default_precision='fixed<10,8>')

    hls_model = hls4ml.converters.keras_v2_to_hls(config)
    hls_model.compile()

    x = np.random.randint(0, 5, size=(10, *input_shape)).astype('float')
    keras_prediction = keras_model.predict(x)

    hls_prediction = hls_model.predict(x)
    np.testing.assert_allclose(hls_prediction.flatten(), keras_prediction.flatten(), rtol=0, atol=1e-2)
