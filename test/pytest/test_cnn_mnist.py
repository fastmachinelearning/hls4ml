from hls4ml.converters.keras_to_hls import keras_to_hls
import pytest
import hls4ml
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from qkeras.utils import _add_supported_quantized_objects; co = {}; _add_supported_quantized_objects(co)
import yaml
from pathlib import Path

test_root_path = Path(__file__).parent
example_model_path = (test_root_path / '../../example-models').resolve()

@pytest.fixture(scope='module')
def mnist_data():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  # Scale images to the [0, 1] range
  x_train = x_train.astype("float32") / 255
  x_test = x_test.astype("float32") / 255
  # Make sure images have shape (28, 28, 1)
  x_train = np.expand_dims(x_train, -1)
  x_test = np.expand_dims(x_test, -1)
  y_train = tf.keras.utils.to_categorical(y_train, 10)
  y_test = tf.keras.utils.to_categorical(y_test, 10)
  return x_train, y_train, x_test, y_test

@pytest.fixture(scope='module')
def mnist_model():
  model_path = example_model_path / 'keras/qkeras_mnist_cnn.json'
  with model_path.open('r') as f:
    jsons = f.read()
  model = model_from_json(jsons, custom_objects=co)
  model.load_weights(example_model_path / 'keras/qkeras_mnist_cnn_weights.h5')
  return model

@pytest.fixture      
@pytest.mark.parametrize('settings', [('io_parallel', 'latency'),
                                      ('io_parallel', 'resource'),
                                      ('io_stream', 'latency'),
                                      ('io_stream', 'resource')])
def hls_model(settings):
  io_type = settings[0]
  strategy = settings[1]
  yml_path = example_model_path / 'config-files/qkeras_mnist_cnn_config.yml'
  with yml_path.open('r') as f:
    config = yaml.safe_load(f.read())
  config['KerasJson'] = str(example_model_path / 'keras/qkeras_mnist_cnn.json')
  config['KerasH5'] = str(example_model_path / 'keras/qkeras_mnist_cnn_weights.h5')
  config['OutputDir'] = str(test_root_path / 'hls4mlprj_cnn_mnist_{}_{}'.format(io_type, strategy))
  config['IOType'] = io_type
  config['HLSConfig']['Model']['Strategy'] = strategy
  config['HLSConfig']['LayerName']['softmax']['Strategy'] = 'Stable'
  hls_model = keras_to_hls(config)
  hls_model.compile()
  return hls_model

@pytest.mark.parametrize('settings', [('io_parallel', 'latency'),
                                      ('io_parallel', 'resource'),
                                      ('io_stream', 'latency'),
                                      ('io_stream', 'resource')])
def test_accuracy(mnist_data, mnist_model, hls_model):
  x_train, y_train, x_test, y_test = mnist_data
  x_test, y_test = x_test[:5000], y_test[:5000]
  model = mnist_model
  # model under test predictions and accuracy
  y_keras = model.predict(x_test)
  y_hls4ml   = hls_model.predict(x_test)

  acc_keras = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))
  acc_hls4ml = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls4ml, axis=1))
  rel_diff = abs(acc_keras - acc_hls4ml) / acc_keras

  print('Accuracy keras:      {}'.format(acc_keras))
  print('Accuracy hls4ml:     {}'.format(acc_hls4ml))
  print('Relative difference: {}'.format(rel_diff))

  assert acc_keras > 0.95 and rel_diff < 0.01
