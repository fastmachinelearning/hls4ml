from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import AveragePooling1D, AveragePooling2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent

in_shape = 124
in_filt = 5
atol = 5e-3


@pytest.fixture(scope='module')
def data_1d():
    return np.random.rand(100, in_shape, in_filt)


@pytest.fixture(scope='module')
def keras_model_1d(request):
    model_type = request.param['model_type']
    pads = request.param['padding']
    model = Sequential()
    if model_type == 'avg':
        model.add(AveragePooling1D(pool_size=3, input_shape=(in_shape, in_filt), padding=pads))
    elif model_type == 'max':
        model.add(MaxPooling1D(pool_size=3, input_shape=(in_shape, in_filt), padding=pads))
    model.compile()
    return model, model_type, pads


@pytest.mark.parametrize('backend', ['Quartus', 'Vitis', 'Vivado', 'Catapult', 'oneAPI'])
@pytest.mark.parametrize(
    'keras_model_1d',
    [
        {'model_type': 'max', 'padding': 'valid'},
        {'model_type': 'max', 'padding': 'same'},
        {'model_type': 'avg', 'padding': 'valid'},
        {'model_type': 'avg', 'padding': 'same'},
    ],
    ids=[
        'model_type-max-padding-valid',
        'model_type-max-padding-same',
        'model_type-avg-padding-valid',
        'model_type-avg-padding-same',
    ],
    indirect=True,
)
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_pool1d(backend, keras_model_1d, data_1d, io_type):
    model, model_type, padding = keras_model_1d

    config = hls4ml.utils.config_from_keras_model(
        model, default_precision='ap_fixed<32,9>', granularity='name', backend=backend
    )

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        io_type=io_type,
        output_dir=str(test_root_path / f'hls4mlprj_globalplool1d_{backend}_{io_type}_{model_type}_padding_{padding}'),
        backend=backend,
    )
    hls_model.compile()

    y_keras = model.predict(data_1d)
    y_hls = hls_model.predict(data_1d).reshape(y_keras.shape)
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)


@pytest.mark.parametrize('backend', ['Quartus', 'Vitis', 'Vivado', 'oneAPI'])
@pytest.mark.parametrize(
    'keras_model_1d',
    [
        {'model_type': 'max', 'padding': 'valid'},
        {'model_type': 'avg', 'padding': 'valid'},
    ],
    ids=[
        'model_type-max-padding-valid',
        'model_type-avg-padding-valid',
    ],
    indirect=True,
)
@pytest.mark.parametrize('io_type', ['io_stream'])
def test_pool1d_stream(backend, keras_model_1d, data_1d, io_type):
    model, model_type, padding = keras_model_1d

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,9>', granularity='name')

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        io_type=io_type,
        output_dir=str(test_root_path / f'hls4mlprj_globalplool1d_{backend}_{io_type}_{model_type}_padding_{padding}'),
        backend=backend,
    )
    hls_model.compile()

    y_keras = model.predict(data_1d)
    y_hls = hls_model.predict(data_1d).reshape(y_keras.shape)
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)


@pytest.fixture(scope='module')
def data_2d():
    return np.random.rand(100, in_shape, in_shape, in_filt)


@pytest.fixture(scope='module')
def keras_model_2d(request):
    model_type = request.param['model_type']
    pads = request.param['padding']
    model = Sequential()
    if model_type == 'avg':
        model.add(AveragePooling2D(input_shape=(in_shape, in_shape, in_filt), padding=pads))
    elif model_type == 'max':
        model.add(MaxPooling2D(input_shape=(in_shape, in_shape, in_filt), padding=pads))
    model.compile()
    return model, model_type, pads


@pytest.mark.parametrize('backend', ['Quartus', 'Vitis', 'Vivado', 'Catapult', 'oneAPI'])
@pytest.mark.parametrize(
    'keras_model_2d',
    [
        {'model_type': 'max', 'padding': 'valid'},
        {'model_type': 'max', 'padding': 'same'},
        {'model_type': 'avg', 'padding': 'valid'},
        {'model_type': 'avg', 'padding': 'same'},
    ],
    ids=[
        'model_type-max-padding-valid',
        'model_type-max-padding-same',
        'model_type-avg-padding-valid',
        'model_type-avg-padding-same',
    ],
    indirect=True,
)
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_pool2d(backend, keras_model_2d, data_2d, io_type):
    model, model_type, padding = keras_model_2d

    config = hls4ml.utils.config_from_keras_model(
        model, default_precision='ap_fixed<32,9>', granularity='name', backend=backend
    )

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        io_type=io_type,
        output_dir=str(test_root_path / f'hls4mlprj_globalplool2d_{backend}_{io_type}_{model_type}_padding_{padding}'),
        backend=backend,
    )
    hls_model.compile()

    y_keras = model.predict(data_2d)
    y_hls = hls_model.predict(data_2d).reshape(y_keras.shape)
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)


@pytest.mark.parametrize('backend', ['Quartus', 'Vitis', 'Vivado', 'oneAPI'])
@pytest.mark.parametrize(
    'keras_model_2d',
    [
        {'model_type': 'max', 'padding': 'valid'},
        {'model_type': 'avg', 'padding': 'valid'},
    ],
    ids=[
        'model_type-max-padding-valid',
        'model_type-avg-padding-valid',
    ],
    indirect=True,
)
@pytest.mark.parametrize('io_type', ['io_stream'])
def test_pool2d_stream(backend, keras_model_2d, data_2d, io_type):
    model, model_type, padding = keras_model_2d

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,9>', granularity='name')

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        io_type=io_type,
        output_dir=str(test_root_path / f'hls4mlprj_globalplool2d_{backend}_{io_type}_{model_type}_padding_{padding}'),
        backend=backend,
    )
    hls_model.compile()

    y_keras = model.predict(data_2d)
    y_hls = hls_model.predict(data_2d).reshape(y_keras.shape)
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)
