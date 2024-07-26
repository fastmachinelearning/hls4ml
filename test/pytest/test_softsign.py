from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from sklearn.metrics import accuracy_score

import hls4ml

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'Catapult'])
@pytest.mark.parametrize('input_shape, io_type', [((8,), 'io_parallel'), ((8,), 'io_stream'), ((8, 8, 3), 'io_stream')])
def test_softsign(backend, input_shape, io_type):
    X = np.random.rand(1000, *input_shape)
    X = np.round(X * 2**10) * 2**-10
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Activation(input_shape=input_shape, activation='softsign', name='softsign'))
    model.compile()

    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name', default_precision='fixed<20,4>', backend=backend)
    # Since softsign implementation is lookup-based increasing the precision and size of the table helps with accuracy
    cfg['LayerName']['softsign']['table_t'] = 'fixed<20,4>'
    cfg['LayerName']['softsign']['table_size'] = 2048
    odir = str(test_root_path / f'hls4mlprj_softsign_{backend}_{io_type}_{str(input_shape)}')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, io_type=io_type, output_dir=odir, backend=backend
    )
    hls_model.compile()

    y_keras = model.predict(X)
    y_hls4ml = hls_model.predict(X).reshape(y_keras.shape)
    acc_hls4ml = accuracy_score(np.argmax(y_keras, axis=-1).ravel(), np.argmax(y_hls4ml, axis=-1).ravel())

    print(f'Accuracy hls4ml relative to keras: {acc_hls4ml}')
    assert acc_hls4ml >= 0.96
