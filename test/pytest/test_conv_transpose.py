from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import Conv1DTranspose, Conv2DTranspose
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent


@pytest.fixture(scope='module')
def data2D():
    X = np.random.rand(10, 5, 5, 3)
    return X


@pytest.fixture(scope='module')
def data1D():
    X = np.random.rand(10, 5, 3)
    return X


@pytest.fixture(scope='module')
def model2D():
    model = Sequential()
    model.add(Conv2DTranspose(4, (3, 3), input_shape=(5, 5, 3)))
    model.compile()
    return model


@pytest.fixture(scope='module')
def model1D():
    model = Sequential()
    model.add(Conv1DTranspose(4, 3, input_shape=(5, 3)))
    model.compile()
    return model


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('strategy', ['Resource'])
@pytest.mark.filterwarnings("error")
def test_conv1dtranspose(data1D, model1D, io_type, backend, strategy):
    '''
    Check that the implementation does not have leftover data.
    '''

    X = data1D
    model = model1D

    output_dir = str(test_root_path / f'hls4mlprj_conv1Dtranspose_{backend}_{io_type}_{strategy}')

    config = hls4ml.utils.config_from_keras_model(model)
    config['Model']['Strategy'] = strategy

    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, io_type=io_type, output_dir=output_dir)
    hls_model.compile()

    # model under test predictions and accuracy
    y_keras = model.predict(X)
    y_hls4ml = hls_model.predict(X)

    np.testing.assert_allclose(y_keras.ravel(), y_hls4ml.ravel(), atol=0.05)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('strategy', ['Resource'])
@pytest.mark.filterwarnings("error")
def test_conv2dtranspose(data2D, model2D, io_type, backend, strategy):
    '''
    Check that the implementation does not have leftover data.
    '''

    X = data2D
    model = model2D

    output_dir = str(test_root_path / f'hls4mlprj_conv2Dtranspose_{backend}_{io_type}_{strategy}')

    config = hls4ml.utils.config_from_keras_model(model)
    config['Model']['Strategy'] = strategy

    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, io_type=io_type, output_dir=output_dir)
    hls_model.compile()

    # model under test predictions and accuracy
    y_keras = model.predict(X)
    y_hls4ml = hls_model.predict(X)

    np.testing.assert_allclose(y_keras.ravel(), y_hls4ml.ravel(), atol=0.05)
