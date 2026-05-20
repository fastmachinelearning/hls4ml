from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import AveragePooling1D, AveragePooling2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent

atol = 5e-3


# XLS tests are slow due to big IR size, we reduce dimensions to make it faster.
def in_shape(backend):
    if backend == 'XLS':
        return 17
    return 124


def in_filt(backend):
    if backend == 'XLS':
        return 3
    return 5


def input_shape_1d(backend):
    return (in_shape(backend), in_filt(backend))


@pytest.fixture()
def data_1d(backend):
    return np.random.rand(100, *input_shape_1d(backend))


@pytest.fixture()
def keras_model_1d(request, backend):
    model_type = request.param['model_type']
    pads = request.param['padding']
    strides = request.param.get('strides', None)
    input_shape = input_shape_1d(backend)
    model = Sequential()
    if model_type == 'avg':
        model.add(AveragePooling1D(pool_size=3, input_shape=input_shape, padding=pads, strides=strides))
    elif model_type == 'max':
        model.add(MaxPooling1D(pool_size=3, input_shape=input_shape, padding=pads))
    model.compile()
    return model, model_type, pads, strides


@pytest.mark.parametrize('backend', ['Quartus', 'Vitis', 'Vivado', 'Catapult', 'oneAPI', 'XLS'])
@pytest.mark.parametrize(
    'keras_model_1d',
    [
        {'model_type': 'max', 'padding': 'valid', 'strides': None},
        {'model_type': 'max', 'padding': 'same', 'strides': None},
        {'model_type': 'avg', 'padding': 'valid', 'strides': None},
        {'model_type': 'avg', 'padding': 'same', 'strides': None},
        {'model_type': 'max', 'padding': 'same', 'strides': 4},
        {'model_type': 'avg', 'padding': 'valid', 'strides': 1},
    ],
    ids=[
        'model_type-max-padding-valid',
        'model_type-max-padding-same',
        'model_type-avg-padding-valid',
        'model_type-avg-padding-same',
        'model_type-max-padding-same-strides-4',
        'model_type-avg-padding-valid-strides-1',
    ],
    indirect=True,
)
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_pool1d(test_case_id, backend, keras_model_1d, data_1d, io_type):
    model, model_type, padding, strides = keras_model_1d

    config = hls4ml.utils.config_from_keras_model(
        model, default_precision='ap_fixed<32,9>', granularity='name', backend=backend
    )

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        io_type=io_type,
        output_dir=str(test_root_path / test_case_id),
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
def test_pool1d_stream(test_case_id, backend, keras_model_1d, data_1d, io_type):
    model, model_type, padding, _ = keras_model_1d

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,9>', granularity='name')

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        io_type=io_type,
        output_dir=str(test_root_path / test_case_id),
        backend=backend,
    )
    hls_model.compile()

    y_keras = model.predict(data_1d)
    y_hls = hls_model.predict(data_1d).reshape(y_keras.shape)
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)


def input_shape_2d(backend):
    return (in_shape(backend), in_shape(backend), in_filt(backend))


@pytest.fixture()
def data_2d(backend):
    return np.random.rand(100, *input_shape_2d(backend))


@pytest.fixture()
def keras_model_2d(request, backend):
    model_type = request.param['model_type']
    pads = request.param['padding']
    strides = request.param.get('strides', None)
    input_shape = input_shape_2d(backend)
    model = Sequential()
    if model_type == 'avg':
        model.add(AveragePooling2D(input_shape=input_shape, padding=pads, strides=strides))
    elif model_type == 'max':
        model.add(MaxPooling2D(input_shape=input_shape, padding=pads, strides=strides))
    model.compile()
    return model, model_type, pads, strides


@pytest.mark.parametrize('backend', ['Quartus', 'Vitis', 'Vivado', 'Catapult', 'oneAPI', 'XLS'])
@pytest.mark.parametrize(
    'keras_model_2d',
    [
        {'model_type': 'max', 'padding': 'valid', 'strides': None},
        {'model_type': 'max', 'padding': 'same', 'strides': None},
        {'model_type': 'avg', 'padding': 'valid', 'strides': None},
        {'model_type': 'avg', 'padding': 'same', 'strides': None},
        {'model_type': 'max', 'padding': 'same', 'strides': (4, 2)},
        {'model_type': 'avg', 'padding': 'valid', 'strides': (1, 3)},
    ],
    ids=[
        'model_type-max-padding-valid',
        'model_type-max-padding-same',
        'model_type-avg-padding-valid',
        'model_type-avg-padding-same',
        'model_type-max-padding-same-strides-4',
        'model_type-avg-padding-valid-strides-1',
    ],
    indirect=True,
)
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_pool2d(test_case_id, backend, keras_model_2d, data_2d, io_type):
    model, model_type, padding, strides = keras_model_2d

    config = hls4ml.utils.config_from_keras_model(
        model, default_precision='ap_fixed<32,9>', granularity='name', backend=backend
    )

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        io_type=io_type,
        output_dir=str(test_root_path / test_case_id),
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
def test_pool2d_stream(test_case_id, backend, keras_model_2d, data_2d, io_type):
    model, model_type, padding, _ = keras_model_2d

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,9>', granularity='name')

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        io_type=io_type,
        output_dir=str(test_root_path / test_case_id),
        backend=backend,
    )
    hls_model.compile()

    y_keras = model.predict(data_2d)
    y_hls = hls_model.predict(data_2d).reshape(y_keras.shape)
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)
