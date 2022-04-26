import pytest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import UpSampling2D
import numpy as np
import hls4ml


in_height = 6
in_width = 8
in_feat = 4

size = 2
atol = 5e-3


@pytest.fixture(scope='module')
def data_2d():
    X = np.random.rand(100, in_height, in_width, in_feat)
    return X



@pytest.fixture(scope='module')
def keras_model_2d():
    model = Sequential()
    model.add(UpSampling2D(input_shape=(in_height, in_width, in_feat), size=(size, size)))
    model.compile()
    return model


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_upsampling2d(keras_model_2d, data_2d, io_type):
    
    model = keras_model_2d
    data = data_2d

    config = hls4ml.utils.config_from_keras_model(model,
                                                  default_precision='ap_fixed<32,1>',
                                                  granularity='name')
    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                           hls_config=config,
                                                           io_type=io_type,
                                                           output_dir=f'hls4mlprj_upsampling_2d_{io_type}',
                                                           part='xcvu9p-flgb2104-2-i')
    hls_model.compile()

    # Predict
    y_keras = model.predict(data).flatten()
    y_hls = hls_model.predict(data).flatten()
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)