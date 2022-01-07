import pytest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import hls4ml


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
def test_global_pool1d(model, data, io_type):

    config = hls4ml.utils.config_from_keras_model(model, 
                                                  default_precision='ap_fixed<32,1>',
                                                  granularity='name')

    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                           hls_config=config,
                                                           io_type=io_type,
                                                           output_dir=f'hls4mlprj_batchnorm_{io_type}',
                                                           part='xcvu9p-flgb2104-2-i')
    hls_model.compile()
    

    # Predict
    y_keras = np.squeeze(model.predict(data))
    y_hls = hls_model.predict(data)
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)
