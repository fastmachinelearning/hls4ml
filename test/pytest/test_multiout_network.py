from pathlib import Path

import numpy as np
import pytest
from keras.layers import Dense
from tensorflow import keras

from hls4ml.converters import convert_from_keras_model

test_root_path = Path(__file__).parent


@pytest.fixture(scope='module')
def model():
    inp = keras.Input(shape=(10,))
    x = Dense(10, name='dense1')(inp)
    y = Dense(10, name='dense2')(inp)
    model = keras.Model(inp, [x, y])
    return model


@pytest.fixture(scope='module')
def model_corner_cases():
    in1 = keras.layers.Input(shape=(24, 8))
    in2 = keras.layers.Input(shape=(16))
    out1 = keras.layers.Conv1D(1, 3)(in1)
    out1 = keras.layers.Flatten()(out1)
    out2 = keras.layers.Dense(16, activation='relu')(out1)
    out2 = keras.layers.Add()([out2, in2])
    out3 = keras.layers.Dense(2)(out1)
    out4 = keras.layers.Dense(2)(out2)
    out4 = keras.layers.Flatten()(out4)
    model = keras.models.Model(inputs=[in1, in2], outputs=[out1, out2, out3, out4])
    return model


@pytest.fixture(scope='module')
def data():
    X = np.random.normal(0, 1, (1000, 10))
    X = np.clip(X, -16, 15)
    return X


@pytest.fixture(scope='module')
def data_corner_cases():
    X1 = np.random.normal(0, 1, (1000, 24, 8))
    X2 = np.random.normal(0, 1, (1000, 16))
    X1 = np.clip(X1, -16, 15)
    X2 = np.clip(X2, -16, 15)
    return X1, X2


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus', 'Vitis'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_multi_output_nn(model, data, backend: str, io_type: str):
    output_dir = str(test_root_path / f'hls4mlprj_multiout_network_{backend}_{io_type}')
    hls_config = {'Model': {'Precision': 'fixed<32,5>', 'ReuseFactor': 1}}
    model_hls = convert_from_keras_model(
        model, backend=backend, output_dir=output_dir, hls_config=hls_config, io_type=io_type
    )

    assert model_hls.graph['dense1'].attributes['result_t'] != model_hls.graph['dense2'].attributes['result_t']

    model_hls.compile()
    r_hls = model_hls.predict(data)
    r_keras = [x.numpy() for x in model(data)]

    assert np.allclose(r_hls[0], r_keras[0], atol=1e-5, rtol=0)
    assert np.allclose(r_hls[1], r_keras[1], atol=1e-5, rtol=0)


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus', 'Vitis', 'Catapult', 'OneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('strategy', ['latency', 'resource'])
def test_multi_output_nn_corner_cases(model_corner_cases, data_corner_cases, backend: str, io_type: str, strategy: str):
    """Cover corner cases, when:
    - a layer outputs both to the next layer(s) and to the model output
       - when an node removal/insertion is triggered internally
    - a reshape in io_parallel, or flatten in io_stream layer's output is used multiple times
       - and as layer output
       - and by layer taking multiple inputs
    - a Flatten layer outputs to the model output in io_stream
    """
    output_dir = str(test_root_path / f'hls4mlprj_multiout_network_2_{backend}_{io_type}_{strategy}')
    hls_config = {'Model': {'Precision': 'fixed<32,5>', 'ReuseFactor': 1}, 'Strategy': strategy}

    model_hls = convert_from_keras_model(
        model_corner_cases, backend=backend, output_dir=output_dir, hls_config=hls_config, io_type=io_type
    )

    model_hls.compile()
    r_hls = model_hls.predict(data_corner_cases)
    r_keras = model_corner_cases.predict(data_corner_cases, verbose=0, batch_size=1000)

    assert np.allclose(r_hls[0], r_keras[0], atol=1e-5, rtol=0)
    assert np.allclose(r_hls[1], r_keras[1], atol=1e-5, rtol=0)
    assert np.allclose(r_hls[2], r_keras[2], atol=1e-5, rtol=0)
    assert np.allclose(r_hls[3], r_keras[3], atol=1e-5, rtol=0)
