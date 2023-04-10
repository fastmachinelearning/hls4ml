from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from sklearn.metrics import accuracy_score

import hls4ml

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


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus'])
@pytest.mark.parametrize('strategy', ['stable', 'argmax'])
@pytest.mark.parametrize(
    'function,input_shape,io_type',
    [
        (flat_distribution, (8,), 'io_parallel'),
        (high_accuracy_distribution, (8,), 'io_parallel'),
        (flat_distribution, (8,), 'io_stream'),
        (high_accuracy_distribution, (8,), 'io_stream'),
        (flat_distribution, (8, 8, 3), 'io_stream'),
        (high_accuracy_distribution, (8, 8, 3), 'io_stream'),
    ],
)
def test_softmax(backend, strategy, generate_data, input_shape, io_type, function):
    X = generate_data
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Activation(input_shape=input_shape, activation='softmax', name='softmax'))
    model.compile()

    f_type = 'ac_fixed<18,8,true,AC_RND,AC_SAT>' if backend == 'Quartus' else 'ap_fixed<18,8,AP_RND,AP_SAT>'
    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')
    cfg['LayerName']['softmax']['Strategy'] = strategy
    cfg['LayerName']['softmax']['inv_table_t'] = f_type
    cfg['LayerName']['softmax']['exp_table_t'] = f_type

    odir = str(test_root_path / 'hls4mlprj_softmax_{}_{}_{}_{}_{}').format(
        backend, io_type, strategy, function.__name__, str(input_shape)
    )
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, io_type=io_type, output_dir=odir, backend=backend
    )
    hls_model.compile()

    y_keras = model.predict(X)
    y_hls4ml = hls_model.predict(X).reshape(y_keras.shape)
    acc_hls4ml = accuracy_score(np.argmax(y_keras, axis=-1).ravel(), np.argmax(y_hls4ml, axis=-1).ravel())

    print(f'Accuracy hls4ml relative to keras: {acc_hls4ml}')

    assert acc_hls4ml >= 0.98


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_softmax_skipped(backend, io_type):
    X = np.random.rand(100, 10)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(14, input_shape=(10,), name='dense'))
    model.add(tf.keras.layers.Activation(activation='softmax', name='softmax'))
    model.compile()

    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')
    cfg['LayerName']['softmax']['skip'] = True

    odir = str(test_root_path / 'hls4mlprj_softmax_skipped_{}_{}').format(backend, io_type)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, io_type=io_type, output_dir=odir, backend=backend
    )
    hls_model.compile()

    # Verify Softmax was removed
    hls_layers = list(hls_model.get_layers())  # 0 is Input, 1 is Dense, 2 is Softmax (if not removed)
    assert len(hls_layers) == 2

    # Verify hls4ml output is equal to Dense output
    y_keras = model.predict(X)
    y_hls4ml = hls_model.predict(X).reshape(y_keras.shape)
    keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, X)
    np.testing.assert_allclose(y_hls4ml, keras_trace['dense'], rtol=0, atol=2e-2)
