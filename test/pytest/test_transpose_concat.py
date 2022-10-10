import pytest
import hls4ml
import numpy as np
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.layers import Input, Permute, Concatenate, Activation
import yaml

@pytest.fixture(scope='module')
def data():
    X = np.random.rand(100, 2, 3)
    return X

@pytest.fixture(scope='module')
def keras_model():
    inp = Input(shape=(2, 3), name='input_1')
    x = Permute((2, 1))(inp)
    y = Concatenate(axis=1)([x, x])
    x = Activation('relu', name='relu')(x)
    out = Concatenate(axis=1)([x, y])
    model = Model(inputs=inp, outputs=out)
    return model

@pytest.fixture      
@pytest.mark.parametrize('io_type', ['io_parallel',
                                     'io_stream'])
def hls_model(keras_model, io_type):
    hls_config = hls4ml.utils.config_from_keras_model(keras_model, 
                                                      default_precision='ap_fixed<16,3,AP_RND_CONV,AP_SAT>',
                                                      granularity='name')
    hls_config['LayerName']['relu']['Precision'] = 'ap_ufixed<17,3>'
    hls_model = hls4ml.converters.convert_from_keras_model(keras_model,
                                                           hls_config=hls_config,
                                                           io_type=io_type,
                                                           output_dir='hls4mlprj_transpose_{}'.format(io_type))

    hls_model.compile()
    return hls_model

# TODO - Add Quartus test after merging PR #634
@pytest.mark.parametrize('io_type', ['io_parallel', 
                                     'io_stream'])
def test_accuracy(data, keras_model, hls_model):
    X = data
    model = keras_model
    # model under test predictions and accuracy
    y_keras = model.predict(X)
    y_hls4ml   = hls_model.predict(X).reshape(y_keras.shape)
    # "accuracy" of hls4ml predictions vs keras
    np.testing.assert_allclose(y_keras, y_hls4ml, rtol=0, atol=1e-04, verbose=True)
