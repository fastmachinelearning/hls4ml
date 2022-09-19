import hls4ml
import tensorflow as tf
import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from pathlib import Path

test_root_path = Path(__file__).parent

@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('input_shape, io_type', [
                            ((8, ), 'io_parallel'),
                            ((8, ), 'io_stream'),
                            ((8, 8, 3), 'io_stream')
                        ])
def test_softsign(backend, input_shape, io_type):
    X = np.random.rand(1000, *input_shape)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Activation(input_shape=input_shape, activation='softsign', name='softsign'))
    model.compile()
    
    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')    
    odir = str(test_root_path / 'hls4mlprj_softsign_{}_{}_{}'.format(backend, io_type, str(input_shape)))
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=cfg, io_type=io_type,
                                                           output_dir=odir, backend=backend)
    hls_model.compile()
   
    y_keras = model.predict(X)
    y_hls4ml = hls_model.predict(X).reshape(y_keras.shape)
    acc_hls4ml = accuracy_score(np.argmax(y_keras, axis=-1).ravel(), np.argmax(y_hls4ml, axis=-1).ravel())

    print('Accuracy hls4ml relative to keras: {}'.format(acc_hls4ml))
    assert acc_hls4ml >= 0.97