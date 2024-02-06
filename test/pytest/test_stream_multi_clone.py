from pathlib import Path

import numpy as np
import pytest
from keras.layers import Add, Dense
from tensorflow import keras

from hls4ml.converters import convert_from_keras_model

test_root_path = Path(__file__).parent


@pytest.fixture(scope='module')
def model():
    inp = keras.Input(shape=(10,))
    x = Dense(10)(inp)
    y = Dense(10)(inp)
    z = Dense(10)(inp)
    xy = Add()([x, y])  # 5
    xy = Add()([xy, y])  # 5
    out = Add()([xy, z])  # 5
    model = keras.Model(inp, out)
    return model


@pytest.fixture(scope='module')
def data():
    X = np.random.normal(0, 1, (1000, 10))
    X = np.clip(X, -16, 15)
    return X


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus', 'Vitis'])
def test_multi_clone(model, data, backend: str):
    output_dir = str(test_root_path / f'hls4mlprj_stream_multi_clone_{backend}')
    hls_config = {'Model': {'Precision': 'fixed<32,5>', 'ReuseFactor': 1}}
    model_hls = convert_from_keras_model(
        model,
        backend=backend,
        output_dir=output_dir,
        hls_config=hls_config,
        io_type='io_stream',  # clone only happens with stream io.
    )
    model_hls.compile()
    r_hls = model_hls.predict(data)
    r_keras = model(data).numpy()

    assert np.allclose(r_hls, r_keras, atol=1e-5, rtol=0)
