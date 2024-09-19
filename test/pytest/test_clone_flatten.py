from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import Concatenate, Flatten, Input
from tensorflow.keras.models import Model

import hls4ml

test_root_path = Path(__file__).parent


@pytest.fixture(scope='module')
def data():
    X = np.random.randint(-5, 5, (1, 2, 3), dtype='int32')
    return X


@pytest.fixture(scope='module')
def keras_model():
    inp1 = Input(shape=(2, 3), name='input_1')
    x = Flatten()(inp1)
    y = Flatten()(inp1)
    out = Concatenate(axis=1)([x, y])
    model = Model(inputs=inp1, outputs=out)
    return model


@pytest.fixture
@pytest.mark.parametrize('io_type', ['io_stream'])
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus', 'Catapult'])
def hls_model(keras_model, backend, io_type):
    hls_config = hls4ml.utils.config_from_keras_model(
        keras_model, default_precision='ap_int<6>', granularity='name', backend=backend
    )
    output_dir = str(test_root_path / f'hls4mlprj_clone_flatten_{backend}_{io_type}')
    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model,
        hls_config=hls_config,
        io_type=io_type,
        backend=backend,
        output_dir=output_dir,
    )

    hls_model.compile()
    return hls_model


@pytest.mark.parametrize('io_type', ['io_stream'])
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
def test_accuracy(data, keras_model, hls_model):
    X = data
    model = keras_model
    # model under test predictions and accuracy
    y_keras = model.predict(X)
    y_hls4ml = hls_model.predict(X.astype('float32')).reshape(y_keras.shape)
    # "accuracy" of hls4ml predictions vs keras
    np.testing.assert_array_equal(y_keras, y_hls4ml, verbose=True)
