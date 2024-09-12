from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent

in_shape = 18
in_filt = 6
atol = 5e-3


@pytest.fixture(scope='module')
def data_1d():
    return np.random.rand(100, in_shape, in_filt)


@pytest.fixture(scope='module')
def keras_model_1d(request):
    model_type = request.param['model_type']
    keepdims = request.param['keepdims']
    model = Sequential()
    if model_type == 'avg':
        model.add(GlobalAveragePooling1D(input_shape=(in_shape, in_filt), keepdims=keepdims))
    elif model_type == 'max':
        model.add(GlobalMaxPooling1D(input_shape=(in_shape, in_filt), keepdims=keepdims))
    model.compile()
    return model, model_type, keepdims


@pytest.mark.parametrize('backend', ['Quartus', 'Vitis', 'Vivado', 'Catapult', 'oneAPI'])
@pytest.mark.parametrize(
    'keras_model_1d',
    [
        {'model_type': 'max', 'keepdims': True},
        {'model_type': 'max', 'keepdims': False},
        {'model_type': 'avg', 'keepdims': True},
        {'model_type': 'avg', 'keepdims': False},
    ],
    ids=[
        'model_type-max-keepdims-True',
        'model_type-max-keepdims-False',
        'model_type-avg-keepdims-True',
        'model_type-avg-keepdims-False',
    ],
    indirect=True,
)
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_global_pool1d(backend, keras_model_1d, data_1d, io_type):
    model, model_type, keepdims = keras_model_1d

    config = hls4ml.utils.config_from_keras_model(
        model, default_precision='ap_fixed<32,9>', granularity='name', backend=backend
    )

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        io_type=io_type,
        output_dir=str(test_root_path / f'hls4mlprj_globalplool1d_{backend}_{io_type}_{model_type}_keepdims{keepdims}'),
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
    keepdims = request.param['keepdims']
    model = Sequential()
    if model_type == 'avg':
        model.add(GlobalAveragePooling2D(input_shape=(in_shape, in_shape, in_filt), keepdims=keepdims))
    elif model_type == 'max':
        model.add(GlobalMaxPooling2D(input_shape=(in_shape, in_shape, in_filt), keepdims=keepdims))
    model.compile()
    return model, model_type, keepdims


@pytest.mark.parametrize('backend', ['Quartus', 'Vitis', 'Vivado', 'Catapult', 'oneAPI'])
@pytest.mark.parametrize(
    'keras_model_2d',
    [
        {'model_type': 'max', 'keepdims': True},
        {'model_type': 'max', 'keepdims': False},
        {'model_type': 'avg', 'keepdims': True},
        {'model_type': 'avg', 'keepdims': False},
    ],
    ids=[
        'model_type-max-keepdims-True',
        'model_type-max-keepdims-False',
        'model_type-avg-keepdims-True',
        'model_type-avg-keepdims-False',
    ],
    indirect=True,
)
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_global_pool2d(backend, keras_model_2d, data_2d, io_type):
    model, model_type, keepdims = keras_model_2d

    config = hls4ml.utils.config_from_keras_model(
        model, default_precision='ap_fixed<32,9>', granularity='name', backend=backend
    )

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        io_type=io_type,
        output_dir=str(test_root_path / f'hls4mlprj_globalplool2d_{backend}_{io_type}_{model_type}_keepdims{keepdims}'),
        backend=backend,
    )
    hls_model.compile()

    y_keras = model.predict(data_2d)
    y_hls = hls_model.predict(data_2d).reshape(y_keras.shape)
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)
