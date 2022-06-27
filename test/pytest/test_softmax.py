import hls4ml
import tensorflow as tf
import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from pathlib import Path

test_root_path = Path(__file__).parent

def flat_distribution(shape):
    return np.random.rand(*shape)


def high_accuracy_distribution(shape):
    '''Start with a flat distribution, then pick a random member of each row to amplify'''
    x = np.random.rand(*shape)
    imax = np.random.randint(0, shape[1], size=shape[0])
    x[:, imax] *= 10
    return x


@pytest.fixture()
def generate_data(function, input_shape):
    return function((1000, *input_shape))


# TODO: Include latency strategy with flat_distribution when it can be made to pass
@pytest.mark.parametrize('backend,strategy,function,input_shape,io_type', [   
                            ('Vivado', 'stable', flat_distribution, (8,), 'io_parallel'),
                            ('Vivado', 'stable', high_accuracy_distribution, (8,), 'io_parallel'),
                            
                            ('Quartus', 'stable', flat_distribution, (8,), 'io_parallel'),
                            ('Quartus', 'stable', high_accuracy_distribution, (8,), 'io_parallel'),

                            # Streaming, single-dimensional implementation (not supported on Quartus yet)
                            ('Vivado', 'stable', flat_distribution, (8,), 'io_stream'),
                            ('Vivado', 'stable', high_accuracy_distribution, (8,), 'io_stream'),
                            
                            # Multi-dimensional tests, only for io_stream for now
                            ('Vivado', 'stable', flat_distribution, (8, 8, 3), 'io_stream'),
                            ('Vivado', 'stable', high_accuracy_distribution, (8, 8, 3), 'io_stream'),

                            # Latency, include when test pass
                            #('Vivado', 'latency', flat_distribution, (8,), 'io_parallel'),
                            #('Vivado', 'latency', flat_distribution, (8, 8, 3), 'io_stream'),
                            #('Quartus', 'latency', flat_distribution, (8,), 'io_parallel'),
                            
                        ])
def test_softmax(backend, strategy, generate_data, input_shape, io_type):
    X = generate_data
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Activation(input_shape=input_shape, activation='softmax', name='softmax'))
    model.compile()
    
    f_type = 'ac_fixed<18,8,true,AC_RND,AC_SAT>' if backend == 'Quartus' else 'ap_fixed<18,8,AP_RND,AP_SAT>'
    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')
    cfg['LayerName']['softmax']['Strategy'] = strategy
    cfg['LayerName']['softmax']['inv_table_t'] = f_type
    cfg['LayerName']['softmax']['exp_table_t'] = f_type
    
    odir = str(test_root_path / 'hls4mlprj_softmax_{}'.format(strategy))
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=cfg, io_type=io_type,
                                                           output_dir=odir, backend=backend)
    hls_model.compile()
   
    y_keras = model.predict(X)
    y_hls4ml = hls_model.predict(X).reshape(y_keras.shape)
    acc_hls4ml = accuracy_score(np.argmax(y_keras, axis=-1).ravel(), np.argmax(y_hls4ml, axis=-1).ravel())

    print('Accuracy hls4ml relative to keras: {}'.format(acc_hls4ml))

    assert acc_hls4ml >= 0.98
