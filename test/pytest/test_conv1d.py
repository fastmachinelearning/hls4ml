import pytest
import hls4ml
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import model_from_json
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
@pytest.mark.parametrize('backend, io_type, strategy', [
                                      ('Quartus', 'io_parallel', 'resource'),
                                      ('Vivado', 'io_parallel', 'resource'),

                                      ('Vivado', 'io_parallel', 'latency'),
                                      
                                      ('Vivado', 'io_stream', 'latency'),
                                      ('Vivado', 'io_stream', 'resource')
                                    ])
def hls_model(keras_model, backend, io_type, strategy):
    default_precision = 'ap_fixed<16,3,AP_RND_CONV,AP_SAT>' if backend=='Vivado' else 'ac_fixed<16,3,true,AC_RND_CONV,AC_SAT>'
    fc1_weight_precision = 'ap_fixed<16,3>' if backend=='Vivado' else 'ac_fixed<16,3,true>'
    fc1_result_precision = 'ap_fixed<16,6,AP_RND_CONV,AP_SAT>' if backend=='Vivado' else 'ac_fixed<16,6,true,AC_RND_CONV,AC_SAT>'
    output_softmax_weight_precision = 'ap_fixed<16,6>' if backend=='Vivado' else 'ac_fixed<16,6,true>'
    output_softmax_result_precision = 'ap_fixed<16,6,AP_RND_CONV,AP_SAT>' if backend=='Vivado' else 'ac_fixed<16,6,true,AP_RND_CONV,AP_SAT>'

    # Default config
    hls_config = hls4ml.utils.config_from_keras_model(keras_model)
    hls_config['Model']['Strategy'] = strategy
    hls_config['Model']['ReuseFactor'] = 1
    hls_config['Model']['Precision'] = default_precision

    # Some model-specific precision tuning
    hls_config['LayerName'] = {}
    hls_config['LayerName']['fc1_relu'] = {'Precision':{'weight' : fc1_weight_precision, 'result' : fc1_result_precision}}
    hls_config['LayerName']['output_softmax'] = {'Precision':{'weight' : output_softmax_weight_precision, 'result' : output_softmax_result_precision}}
    hls_config['LayerName']['output_softmax_softmax'] = {'Strategy':'Stable'}
    
    output_dir = str(test_root_path / 'hls4mlprj_conv1d_{}_{}_{}'.format(backend, io_type, strategy))
    hls_model = hls4ml.converters.convert_from_keras_model(keras_model, hls_config=hls_config, backend=backend, io_type=io_type, output_dir=output_dir)
    hls_model.compile()
    return hls_model

@pytest.mark.parametrize('backend, io_type, strategy', [
                                      ('Quartus', 'io_parallel', 'resource'),
                                      ('Vivado', 'io_parallel', 'resource'),

                                      ('Vivado', 'io_parallel', 'latency'),
                                      
                                      ('Vivado', 'io_stream', 'latency'),
                                      ('Vivado', 'io_stream', 'resource')
                                    ])
def test_accuracy(data, keras_model, hls_model):
    X = data
    model = keras_model

    # Model under test predictions and accuracy
    y_keras = model.predict(X)
    y_hls4ml   = hls_model.predict(X)
    
    # "Accuracy" of hls4ml predictions vs keras
    rel_acc = accuracy_score(np.argmax(y_keras, axis=1), np.argmax(y_hls4ml, axis=1))
    print('hls4ml accuracy relative to keras: {}'.format(rel_acc))
    assert rel_acc > 0.98
