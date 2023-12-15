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
def data():
    X = np.random.normal(0, 1, (1000, 10))
    X = np.clip(X, -16, 15)
    return X


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus', 'Vitis'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_multi_clone(model, data, backend: str, io_type: str):
    output_dir = str(test_root_path / f'hls4mlprj_multiout_network_{backend}_{io_type}')
    hls_config = {'Model': {'Precision': 'fixed<32,5>', 'ReuseFactor': 1}}
    layer_config = {
        'dense1': {'Precision': {'result': 'fixed<35,5>'}},
        'dense2': {'Precision': {'result': 'fixed<40,5>'}},
        'dense1_linear': {'Precision': {'result': 'fixed<35,5>'}},
        'dense2_linear': {'Precision': {'result': 'fixed<40,5>'}},
    }
    hls_config['LayerName'] = layer_config
    model_hls = convert_from_keras_model(
        model, backend=backend, output_dir=output_dir, hls_config=hls_config, io_type=io_type
    )

    assert model_hls.graph['dense1'].attributes['result_t'] != model_hls.graph['dense2'].attributes['result_t']

    model_hls.compile()
    r_hls = model_hls.predict(data)
    r_keras = [x.numpy() for x in model(data)]

    assert np.allclose(r_hls[0], r_keras[0], atol=1e-5, rtol=0)
    assert np.allclose(r_hls[1], r_keras[1], atol=1e-5, rtol=0)
