import pytest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
import numpy as np
import hls4ml
from pathlib import Path

test_root_path = Path(__file__).parent

in_shape = 8
in_feat = 4
atol = 5e-3

@pytest.fixture(scope='module')
def data():
    X = np.random.rand(100, in_shape, in_feat)
    return X


@pytest.fixture(scope='module')
def keras_model_max():
    model = Sequential()
    model.add(GlobalMaxPooling1D(input_shape=(in_shape, in_feat)))
    model.compile()
    return model

@pytest.fixture(scope='module')
def keras_model_ave():
    model = Sequential()
    model.add(GlobalAveragePooling1D(input_shape=(in_shape, in_feat)))
    model.compile()
    return model

  
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('model_type', ['max', 'ave'])
def test_global_pool1d(keras_model_max, keras_model_ave, data, model_type, io_type):
    if model_type == 'ave':
        model = keras_model_ave
    else:
        model = keras_model_max
    config = hls4ml.utils.config_from_keras_model(model, 
                                                  default_precision='ap_fixed<32,1>',
                                                  granularity='name')
    if model_type == 'ave':
        config['LayerName']['global_average_pooling1d']['accum_t'] = 'ap_fixed<32,6>'

    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                           hls_config=config,
                                                           io_type=io_type,
                                                           output_dir=str(test_root_path / f'hls4mlprj_globalplool1d_{model_type}_{io_type}'),
                                                           part='xcvu9p-flgb2104-2-i')
    hls_model.compile()
    

    # Predict
    y_keras = np.squeeze(model.predict(data))
    y_hls = hls_model.predict(data)
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)
