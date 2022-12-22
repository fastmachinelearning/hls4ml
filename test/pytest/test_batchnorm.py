from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent

in_shape = 16
atol = 5e-3


@pytest.fixture(scope='module')
def data():
    np.random.seed(0)
    X = np.random.rand(100, in_shape)
    return X


@pytest.fixture(scope='module')
def model():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(in_shape,)))
    model.compile()
    return model


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
def test_batchnorm(model, data, backend, io_type):

    default_precision = 'ac_fixed<32, 1, true>' if backend == 'Quartus' else 'ac_fixed<32, 1>'

    config = hls4ml.utils.config_from_keras_model(model, default_precision=default_precision, granularity='name')
    output_dir = str(test_root_path / f'hls4mlprj_batchnorm_{backend}_{io_type}')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, backend=backend, hls_config=config, io_type=io_type, output_dir=output_dir
    )
    hls_model.compile()

    # Predict
    y_keras = np.squeeze(model.predict(data))
    y_hls = hls_model.predict(data)
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)
