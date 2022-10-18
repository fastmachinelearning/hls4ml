import pytest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling2D, GlobalMaxPooling2D
import numpy as np
import hls4ml
from pathlib import Path

test_root_path = Path(__file__).parent

in_shape = 18
in_filt = 6
atol = 5e-3

@pytest.fixture(scope='module')
def data_1d():
    return np.random.rand(100, in_shape, in_filt)

@pytest.fixture(scope='module')
def keras_model_max_1d():
    model = Sequential()
    model.add(GlobalMaxPooling1D(input_shape=(in_shape, in_filt)))
    model.compile()
    return model

@pytest.fixture(scope='module')
def keras_model_avg_1d():
    model = Sequential()
    model.add(GlobalAveragePooling1D(input_shape=(in_shape, in_filt)))
    model.compile()
    return model
 
@pytest.mark.parametrize('backend, io_type', [
                                ('Vivado', 'io_parallel'), 
                                ('Vivado','io_stream'),

                                # TODO - Quartus Streaming Global Pooling
                                ('Quartus', 'io_parallel'),
                        ])
@pytest.mark.parametrize('model_type', ['max', 'avg'])
def test_global_pool1d(backend, keras_model_max_1d, keras_model_avg_1d, data_1d, model_type, io_type):
    if model_type == 'avg':
        model = keras_model_avg_1d
    else:
        model = keras_model_max_1d
    
    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,9>', granularity='name')

    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                           hls_config=config,
                                                           io_type=io_type,
                                                           output_dir=str(test_root_path / f'hls4mlprj_globalplool1d_{backend}_{io_type}_{model_type}'),
                                                           backend=backend)
    hls_model.compile()
    
    y_keras = np.squeeze(model.predict(data_1d))
    y_hls = hls_model.predict(data_1d)
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)

@pytest.fixture(scope='module')
def data_2d():
    return np.random.rand(100, in_shape, in_shape, in_filt)

@pytest.fixture(scope='module')
def keras_model_max_2d():
    model = Sequential()
    model.add(GlobalMaxPooling2D(input_shape=(in_shape, in_shape, in_filt)))
    model.compile()
    return model

@pytest.fixture(scope='module')
def keras_model_avg_2d():
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=(in_shape, in_shape, in_filt)))
    model.compile()
    return model

# TODO - Add Streaming 2D Pooling in Vivado & Quartus
@pytest.mark.parametrize('backend', ['Quartus', 'Vivado'])
@pytest.mark.parametrize('model_type', ['max', 'avg'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_global_pool2d(backend, keras_model_max_2d, keras_model_avg_2d, data_2d, model_type, io_type):
    
    if model_type == 'avg':
        model = keras_model_avg_2d
    else:
        model = keras_model_max_2d
    
    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,9>', granularity='name')

    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                           hls_config=config,
                                                           io_type=io_type,
                                                           output_dir=str(test_root_path / f'hls4mlprj_globalplool2d_{backend}_{io_type}_{model_type}'),
                                                           backend=backend)
    hls_model.compile()
    
    y_keras = np.squeeze(model.predict(data_2d))
    y_hls = hls_model.predict(data_2d)
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)
