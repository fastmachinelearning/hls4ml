import pytest
import hls4ml
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from qkeras.utils import _add_supported_quantized_objects; co = {}; _add_supported_quantized_objects(co)
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
@pytest.mark.parametrize('backend,io_type,strategy', [
                                      ('Quartus', 'io_parallel', 'resource'),
                                      ('Vivado', 'io_parallel', 'resource'),

                                      ('Vivado', 'io_parallel', 'latency'),
                                      
                                      ('Vivado', 'io_stream', 'latency'),
                                      ('Vivado', 'io_stream', 'resource')
                                    ])
def hls_model(mnist_model, backend, io_type, strategy):
  keras_model = mnist_model
  hls_config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name')     
  hls_config['Model']['Strategy'] = 'Resource'
  hls_config['LayerName']['softmax']['Strategy'] = 'Stable'
  output_dir = str(test_root_path / 'hls4mlprj_cnn_mnist_qkeras_{}_{}_{}'.format(backend, io_type, strategy))

  hls_model = hls4ml.converters.convert_from_keras_model(
                        keras_model, 
                        hls_config=hls_config, 
                        output_dir=output_dir, 
                        backend=backend,
                        io_type=io_type)
 
  hls_model.compile()
  return hls_model

@pytest.mark.parametrize('backend,io_type,strategy', [
                                      ('Quartus', 'io_parallel', 'resource'),
                                      ('Vivado', 'io_parallel', 'resource'),

                                      ('Vivado', 'io_parallel', 'latency'),
                                      
                                      ('Vivado', 'io_stream', 'latency'),
                                      ('Vivado', 'io_stream', 'resource')
                                    ])
def test_accuracy(mnist_data, mnist_model, hls_model):
  x_train, y_train, x_test, y_test = mnist_data
  x_test, y_test = x_test[:5000], y_test[:5000]
  # model under test predictions and accuracy
  y_keras = mnist_model.predict(x_test)
  y_hls4ml = hls_model.predict(x_test)

  acc_keras = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))
  acc_hls4ml = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls4ml, axis=1))
  rel_diff = abs(acc_keras - acc_hls4ml) / acc_keras

  print('Accuracy keras:      {}'.format(acc_keras))
  print('Accuracy hls4ml:     {}'.format(acc_hls4ml))
  print('Relative difference: {}'.format(rel_diff))

  assert acc_keras > 0.92 and rel_diff < 0.01
