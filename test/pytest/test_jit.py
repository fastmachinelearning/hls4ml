from pathlib import Path

import numpy as np
import pytest
from keras.layers import Dense
from tensorflow import keras

from hls4ml.converters import convert_from_keras_model

test_root_path = Path(__file__).parent


@pytest.fixture(scope='module')
def model():
    in1 = keras.Input(shape=(10,))
    in2 = keras.Input(shape=(9,))
    x = Dense(8, name='dense1')(in1)
    y = Dense(7, name='dense2')(in2)
    model = keras.Model([in1, in2], [x, y])
    return model


@pytest.fixture(scope='module')
def data():
    IN1 = np.random.normal(0, 1, (1000, 10)).astype(np.float32)
    IN2 = np.random.normal(0, 1, (1000, 9)).astype(np.float32)
    return IN1, IN2


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus', 'Vitis'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('strategy', ['Resource', 'Latency'])
def test_jit(model, data, backend: str, io_type: str, strategy: str):
    output_dir = str(test_root_path / f'hls4mlprj_jit_{backend}_{io_type}_{strategy}')

    model_hls = convert_from_keras_model(
        model, backend=backend, output_dir=output_dir, io_type=io_type, hls_config={'Model': {'Strategy': strategy, 'ReuseFactor': 1, 'Precision': 'ap_fixed<16,6>'}}
    )

    model_hls.write()
    model_hls.compile_shared_lib()
    model_hls.jit_compile()

    ctypes_pred = model_hls.predict(data, jit=False)
    jit_pred = model_hls.predict(data, jit=True)

    assert np.all(ctypes_pred[0] == jit_pred[0])
    assert np.all(ctypes_pred[1] == jit_pred[1])
