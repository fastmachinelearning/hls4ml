import pytest
import hls4ml
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import Activation, BatchNormalization
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu, ternary, binary

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

@pytest.fixture(scope='module')
def get_jettagging_data():
  '''
  Download the jet tagging dataset
  '''
  print("Fetching data from openml")
  data = fetch_openml('hls4ml_lhc_jets_hlf')
  X, y = data['data'], data['target']
  le = LabelEncoder()
  y = le.fit_transform(y)
  y = to_categorical(y, 5)
  X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  scaler = StandardScaler()
  X_train_val = scaler.fit_transform(X_train_val)
  X_test = scaler.transform(X_test)
  return X_train_val, X_test, y_train_val, y_test

@pytest.fixture(scope='module')
def train_jettagging_model(get_jettagging_data):
  ''' 
  Train a 3 hidden layer QKeras model on the jet tagging dataset
  '''
  X_train_val, X_test, y_train_val, y_test = get_jettagging_data
  model = Sequential()
  model.add(QDense(64, input_shape=(16,), name='fc1',
                  kernel_quantizer=quantized_bits(6,0,alpha=1), bias_quantizer=quantized_bits(6,0,alpha=1),
                  kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
  model.add(QActivation(activation=quantized_relu(6), name='relu1'))
  model.add(QDense(32, name='fc2',
                  kernel_quantizer=quantized_bits(6,0,alpha=1), bias_quantizer=quantized_bits(6,0,alpha=1),
                  kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
  model.add(QActivation(activation=quantized_relu(6), name='relu2'))
  model.add(QDense(32, name='fc3',
                  kernel_quantizer=quantized_bits(6,0,alpha=1), bias_quantizer=quantized_bits(6,0,alpha=1),
                  kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
  model.add(QActivation(activation=quantized_relu(6), name='relu3'))
  model.add(QDense(5, name='output',
                  kernel_quantizer=quantized_bits(6,0,alpha=1), bias_quantizer=quantized_bits(6,0,alpha=1),
                  kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
  model.add(Activation(activation='softmax', name='softmax'))
  adam = Adam(lr=0.0001)
  model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
  model.fit(X_train_val, y_train_val, batch_size=1024,
          epochs=10, validation_split=0.25, shuffle=True)
  model.save('qkeras-jettagging.h5')
  return model

@pytest.fixture
@pytest.mark.parametrize('strategy', ['latency', 'resource'])
def convert(train_jettagging_model, strategy):
  '''
  Convert a QKeras model trained on the jet tagging dataset
  '''
  model = train_jettagging_model
  hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
  hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
  hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

  config = hls4ml.utils.config_from_keras_model(model, granularity='name')
  config['Model']['Strategy'] = strategy
  config['LayerName']['softmax']['exp_table_t'] = 'ap_fixed<18,8>'
  config['LayerName']['softmax']['inv_table_t'] = 'ap_fixed<18,4>'
  hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       hls_config=config,
                                                       output_dir='qkeras-jettagging-hls4ml-prj',
                                                       fpga_part='xcu250-figd2104-2L-e')
  hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = []                                                     
  hls_model.compile()
  return hls_model

@pytest.mark.parametrize('strategy', ['latency', 'resource'])
def test_accuracy(convert, train_jettagging_model, get_jettagging_data, strategy):
  '''
  Test the hls4ml-evaluated accuracy of a 3 hidden layer QKeras model trained on
  the jet tagging dataset. QKeras model accuracy is required to be over 70%, and
  hls4ml accuracy required to be within 1% of the QKeras model accuracy.
  '''
  print("Test accuracy")
  from sklearn.metrics import accuracy_score

  X_train_val, X_test, y_train_val, y_test = get_jettagging_data

  hls_model = convert
  model = train_jettagging_model

  y_qkeras = model.predict(np.ascontiguousarray(X_test))
  y_hls4ml = hls_model.predict(np.ascontiguousarray(X_test))

  acc_qkeras = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_qkeras, axis=1))
  acc_hls4ml = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls4ml, axis=1))
  rel_diff = abs(acc_qkeras - acc_hls4ml) / acc_qkeras

  print('Accuracy qkeras:     {}'.format(acc_qkeras))
  print('Accuracy hls4ml:     {}'.format(acc_hls4ml))
  print('Relative difference: {}'.format(rel_diff))

  assert acc_qkeras > 0.7 and rel_diff < 0.01

def randX(batch_size, N):
  return np.random.rand(batch_size,N)

@pytest.fixture(scope='module')
def randX_100_16():
  return randX(100, 16)

@pytest.mark.parametrize('bits', [4, 6, 8])
def test_single_dense_activation_exact(randX_100_16, bits):
  '''
  Test a single Dense -> Activation layer topology for
  bit exactness with number of bits parameter
  '''
  X = randX_100_16
  model = Sequential()
  model.add(QDense(16, input_shape=(16,), name='fc1',
                  kernel_quantizer=quantized_bits(bits,0,alpha=1), bias_quantizer=quantized_bits(bits,0,alpha=1),
                  kernel_initializer='lecun_uniform'))
  model.add(QActivation(activation=quantized_relu(bits,0), name='relu1'))
  model.compile()

  hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
  hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
  hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
  config = hls4ml.utils.config_from_keras_model(model, granularity='name')
  hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       hls_config=config,
                                                       output_dir='qkeras-simple-hls4ml-prj-{}'.format(bits),
                                                       fpga_part='xcu250-figd2104-2L-e')
  hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = []                                                   
  hls_model.compile()

  y_qkeras = model.predict(X)
  y_hls4ml = hls_model.predict(X)
  # Goal is to get it passing with all equal
  #np.testing.assert_array_equal(y_qkeras, y_hls4ml)
  # For now allow matching within 1 bit
  np.testing.assert_allclose(y_qkeras.ravel(), y_hls4ml.ravel(), atol=2**-bits, rtol=1.0)

@pytest.fixture
def make_btnn(N, kernel_quantizer, bias_quantizer, activation_quantizer, use_batchnorm, is_xnor):
  shape = (N,)
  model = Sequential()
  model.add(QDense(10, input_shape=shape, kernel_quantizer=kernel_quantizer,
                   bias_quantizer=bias_quantizer, name='dense'))
  if use_batchnorm:
    model.add(BatchNormalization(name='bn'))
  model.add(QActivation(activation=activation_quantizer))
  model.compile()
  is_xnor = activation_quantizer == 'binary'
  return model, is_xnor

@pytest.fixture(scope='module')
def randX_100_10():
  return randX(100, 10)

@pytest.mark.parametrize('N,kernel_quantizer,bias_quantizer,activation_quantizer,use_batchnorm,is_xnor',
                          [(10, ternary(alpha=1), quantized_bits(5,2), 'binary_tanh', False, True),
                           (10, binary(), quantized_bits(5,2), 'binary_tanh', False, True),
                           (10, ternary(alpha='auto'), quantized_bits(5,2), binary(), True, False),
                           (10, ternary(alpha='auto'), quantized_bits(5,2), 'ternary', True, True),
                           (10, ternary(alpha='auto'), quantized_bits(5,2), ternary(threshold=0.2), True, False),
                           (10, ternary(alpha='auto'), quantized_bits(5,2), ternary(threshold=0.8), True, False),
                           (10, binary(), quantized_bits(5,2), binary(), False, True)])
def test_btnn(make_btnn, randX_100_10):
  model, is_xnor = make_btnn
  X = randX_100_10
  cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')
  hls_model = hls4ml.converters.convert_from_keras_model(model, output_dir='btnn', hls_config=cfg)
  hls_model.compile()
  y_hls = hls_model.predict(X)
  # hls4ml may return XNOR binary
  if is_xnor:
    y_hls = np.where(y_hls == 0, -1, 1)
  y_ker = model.predict(X)
  wrong = (y_hls != y_ker).ravel()
  assert sum(wrong) / len(wrong) < 0.005