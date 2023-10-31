import os
import random
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from keras.layers import Dense
from tensorflow import keras

from hls4ml.converters import convert_from_keras_model

test_root_path = Path(__file__).parent


@pytest.fixture(scope='module')
def model():
    seed = 42
    os.environ['RANDOM_SEED'] = f'{seed}'
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.get_logger().setLevel('ERROR')
    random.seed(seed)

    inp = keras.Input(shape=(10,))
    x = Dense(10)(inp)
    y = Dense(10)(inp)
    model = keras.Model(inp, [x, y])
    return model


@pytest.fixture(scope='module')
def data():
    rng = np.random.RandomState(42)
    X = rng.normal(0, 1, (1000, 10))
    X = np.clip(X, -16, 15)
    return X


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus', 'Vitis'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_multi_clone(model, data, backend: str, io_type: str):
    output_dir = str(test_root_path / f'hls4mlprj_multiout_network_{backend}_{io_type}')
    hls_config = {'Model': {'Precision': 'fixed<32,5>', 'ReuseFactor': 1}}
    model_hls = convert_from_keras_model(
        model, backend=backend, output_dir=output_dir, hls_config=hls_config, io_type=io_type
    )
    model_hls.compile()
    r_hls = model_hls.predict(data)
    r_keras = [x.numpy() for x in model(data)]

    assert np.allclose(r_hls[0], r_keras[0], atol=1e-5, rtol=0)
    assert np.allclose(r_hls[1], r_keras[1], atol=1e-5, rtol=0)
