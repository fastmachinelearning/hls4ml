from hls4ml.converters.keras_to_hls import keras_to_hls
import pytest
import hls4ml
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import yaml
from pathlib import Path

test_root_path = Path(__file__).parent
example_model_path = (test_root_path / '../../example-models').resolve()

@pytest.fixture(scope='module')
def data():
    X = np.random.rand(100,100,7)
    return X

@pytest.fixture(scope='module')
def keras_model():
    model_path = example_model_path / 'keras/KERAS_conv1d.json'
    with model_path.open('r') as f:
        jsons = f.read()
    model = model_from_json(jsons)
    model.load_weights(example_model_path / 'keras/KERAS_conv1d_weights.h5')
    return model

@pytest.fixture      
@pytest.mark.parametrize('settings', [('io_parallel', 'latency'),
                                      ('io_parallel', 'resource'),
                                      ('io_stream', 'latency'),
                                      ('io_stream', 'resource')])
def hls_model(settings):
    io_type = settings[0]
    strategy = settings[1]
    config = hls4ml.converters.create_config(output_dir = 'hls4mlprj_conv1d_{}_{}'.format(io_type, strategy))
    config['KerasJson'] = str(example_model_path / 'keras/KERAS_conv1d.json')
    config['KerasH5'] = str(example_model_path / 'keras/KERAS_conv1d_weights.h5')
    config['OutputDir'] = str(test_root_path / 'hls4mlprj_conv1d_{}_{}'.format(io_type, strategy))
    config['IOType'] = io_type
    
    hls_config = {'Model' : {'Strategy' : strategy,
                             'ReuseFactor' : 1,
                             'Precision' : 'ap_fixed<16,3,AP_RND_CONV,AP_SAT>'}}
    # Some model specific precision tuning
    config['LayerName'] = {}
    config['LayerName']['fc1_relu'] = {'Precision':{'weight' : 'ap_fixed<16,3>', 'result' : 'ap_fixed<16,6,AP_RND_CONV,AP_SAT>'}}
    config['LayerName']['output_softmax'] = {'Precision':{'weight' : 'ap_fixed<16,6>', 'result' : 'ap_fixed<16,6,AP_RND_CONV,AP_SAT>'}}
    config['LayerName']['output_softmax_softmax'] = {'Strategy':'Stable'}
    config['HLSConfig'] = hls_config
    hls_model = keras_to_hls(config)
    hls_model.compile()
    return hls_model

@pytest.mark.parametrize('settings', [('io_parallel', 'latency'),
                                      ('io_parallel', 'resource'),
                                      ('io_stream', 'latency'),
                                      ('io_stream', 'resource')])
def test_accuracy(data, keras_model, hls_model):
    X = data
    model = keras_model
    # model under test predictions and accuracy
    y_keras = model.predict(X)
    y_hls4ml   = hls_model.predict(X)
    # "accuracy" of hls4ml predictions vs keras
    rel_acc = accuracy_score(np.argmax(y_keras, axis=1), np.argmax(y_hls4ml, axis=1))

    print('hls4ml accuracy relative to keras: {}'.format(rel_acc))

    assert rel_acc > 0.98
