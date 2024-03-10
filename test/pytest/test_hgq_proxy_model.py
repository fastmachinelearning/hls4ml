from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from hls4ml.converters import convert_from_keras_model
from hls4ml.utils.fixed_point_quantizer import FixedPointQuantizer

#################################################################
# Proxy model is implemented as a submodule of HGQ.             #
# As HGQ requires python>=3.10,<3.12, and tensorflow==2.13,     #
# As the current testing environment is based on python==3.8,   #
# HGQ cannot be marked as a dependency at the moment.           #
#################################################################


test_root_path = Path(__file__).parent
example_model_path = test_root_path.parent.parent / 'example-models'


@pytest.fixture(scope='module')
def jet_classifier_model():
    with open(example_model_path / 'keras/proxy_jet_classifier.json') as f:
        model_config = f.read()
    co = {'FixedPointQuantizer': FixedPointQuantizer}
    model: keras.Model = keras.models.model_from_json(model_config, custom_objects=co)  # type: ignore
    model.load_weights(example_model_path / 'keras/proxy_jet_classifier.h5')
    return model


@pytest.fixture(scope='module')
def jet_classifier_data():
    print('Fetching data...')
    data = fetch_openml('hls4ml_lhc_jets_hlf')

    X, y = data['data'], data['target']
    codecs = {'g': 0, 'q': 1, 't': 4, 'w': 2, 'z': 3}
    y = np.array([codecs[i] for i in y])

    X_train_val, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_val, X_test = X_train_val.astype(np.float32), X_test.astype(np.float32)

    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)

    X_test = np.ascontiguousarray(X_test)
    return X_test, y_test


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
@pytest.mark.parametrize('overflow', [True, False])
def test_proxy_jet_classifier(jet_classifier_model, jet_classifier_data, backend: str, io_type: str, overflow: bool):
    X, y = jet_classifier_data
    if overflow:
        X *= 2  # This will cause overflow

    output_dir = str(test_root_path / f'hls4mlprj_proxy_jet_classifier_{backend}_{io_type}_{overflow}')
    hls_config = {'Model': {'Precision': 'fixed<1,0>', 'ReuseFactor': 1}}
    model_hls = convert_from_keras_model(
        jet_classifier_model, backend=backend, output_dir=output_dir, hls_config=hls_config, io_type=io_type
    )
    model_hls.compile()

    r_hls = model_hls.predict(X)
    r_keras = jet_classifier_model(X).numpy()
    acc = np.mean(np.argmax(r_hls, axis=1) == y)

    if overflow:
        assert acc < 0.7
    if not overflow and io_type == 'io_parallel':
        assert 0.750 < acc < 0.751
    assert np.all(r_hls == r_keras)


def get_mnist_model_stream():
    with open(example_model_path / 'keras/proxy_mnist_homogeneous_act.json') as f:
        model_config = f.read()
    co = {'FixedPointQuantizer': FixedPointQuantizer}
    model: keras.Model = keras.models.model_from_json(model_config, custom_objects=co)  # type: ignore
    model.load_weights(example_model_path / 'keras/proxy_mnist_homogeneous_act.h5')
    return model


def get_mnist_model_parallel():
    with open(example_model_path / 'keras/proxy_mnist_heterogeneous_act.json') as f:
        model_config = f.read()
    co = {'FixedPointQuantizer': FixedPointQuantizer}
    model: keras.Model = keras.models.model_from_json(model_config, custom_objects=co)  # type: ignore
    model.load_weights(example_model_path / 'keras/proxy_mnist_heterogeneous_act.h5')
    return model


@pytest.fixture(scope='module')
def mnist_data():
    mnist = tf.keras.datasets.mnist
    _, (X_test, y_test) = mnist.load_data()
    X_test = (X_test / 255.0).astype(np.float32)
    X_test = np.ascontiguousarray(X_test)
    return X_test, y_test


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('overflow', [True, False])
def test_proxy_mnist(mnist_data, backend: str, io_type: str, overflow: bool):
    X, y = mnist_data
    if overflow:
        X *= 2  # This will cause overflow

    print(X[0].mean())
    if backend.lower() != 'quartus':
        model = get_mnist_model_stream() if io_type == 'io_stream' else get_mnist_model_parallel()
    else:
        # Codegen is not working for Quartus backend, intra-layer heterogeneous activation quantization not possible.
        # Only use stream-compatible model, in which all quantizer layers are fusible (homogeneous + layer has no sibling)
        model = get_mnist_model_stream()

    output_dir = str(test_root_path / f'hls4mlprj_proxy_mnist_{backend}_{io_type}_{overflow}')
    hls_config = {
        'Strategy': 'Latency',
        'Model': {'Precision': 'fixed<1,0>', 'ReuseFactor': 1},
    }  # Accum for io_stream is not fixed. Set a large number as placeholder.

    model_hls = convert_from_keras_model(
        model, backend=backend, output_dir=output_dir, hls_config=hls_config, io_type=io_type
    )

    if backend.lower() != 'quartus':
        if io_type == 'io_parallel':
            # Check parallel factor is propagated to the hls model
            assert model_hls.graph['h_conv2d'].attributes.attributes['n_partitions'] == 1
            assert model_hls.graph['h_conv2d_1'].attributes.attributes['n_partitions'] == 1
        else:
            assert model_hls.graph['h_conv2d_2'].attributes.attributes['n_partitions'] == 26**2
            assert model_hls.graph['h_conv2d_3'].attributes.attributes['n_partitions'] == 11**2
    else:
        # n_partitions is not used in Quartus backend
        assert model_hls.graph['h_conv2d_2'].attributes.attributes['parallelization'] == 1
        assert model_hls.graph['h_conv2d_3'].attributes.attributes['parallelization'] == 1

    model_hls.compile()
    r_keras = model(X).numpy()  # type: ignore
    acc = np.mean(np.argmax(r_keras, axis=1) == y)

    if overflow:
        assert acc < 0.9
    else:
        if io_type == 'io_parallel' and backend.lower() != 'quartus':
            assert 0.927 < acc < 0.928
        else:
            assert 0.957 < acc < 0.958

    r_hls = model_hls.predict(X)
    mismatch_ph = r_hls != r_keras
    assert np.all(
        r_hls == r_keras
    ), f"Proxy-HLS4ML mismatch for out: {np.sum(np.any(mismatch_ph,axis=1))} out of {len(X)} samples are different. Sample: {r_keras[mismatch_ph].ravel()[:5]} vs {r_hls[mismatch_ph].ravel()[:5]}"  # noqa:
