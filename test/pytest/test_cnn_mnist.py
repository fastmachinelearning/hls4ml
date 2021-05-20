import pytest
import hls4ml
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Dense, Activation, MaxPooling2D, Flatten, Dropout
from qkeras import QDense, QConv2D, quantized_bits

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
def mnist_model(mnist_data):
  x_train, y_train, x_test, y_test = mnist_data
  model = tf.keras.Sequential(
    [
        Input(shape=(28,28,1)),
        QConv2D(16, kernel_size=(3, 3), activation="relu",
                kernel_quantizer=quantized_bits(6,0,alpha=1),
                bias_quantizer=quantized_bits(6,0,alpha=1)),
        MaxPooling2D(pool_size=(2, 2)),
        QConv2D(16, kernel_size=(3, 3), activation="relu",
                kernel_quantizer=quantized_bits(6,0,alpha=1),
                bias_quantizer=quantized_bits(6,0,alpha=1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(0.5),
        QDense(10, kernel_quantizer=quantized_bits(6,0,alpha=1),
                   bias_quantizer=quantized_bits(6,0,alpha=1)),
        Activation(activation="softmax", name="softmax"),
    ]
  )
  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
  model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.1)
  return model

@pytest.fixture      
@pytest.mark.parametrize('io_type',['io_parallel', 'io_stream'])
@pytest.mark.parametrize('strategy', ['latency', 'resource'])                                 
def hls_model(mnist_model, io_type, strategy):
  model = mnist_model
  config = hls4ml.utils.config.config_from_keras_model(model, granularity='name')
  config['Model']['Strategy'] = strategy
  config['LayerName']['softmax']['Strategy'] = 'Stable'
  hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                         hls_config=config,
                                                         output_dir='mnist-hls4ml-prj',
                                                         io_type=io_type)
  hls_model.compile()
  return hls_model

@pytest.mark.parametrize('io_type',['io_parallel', 'io_stream'])
@pytest.mark.parametrize('strategy', ['latency', 'resource'])     
def test_accuracy(mnist_data, mnist_model, hls_model):
  x_train, y_train, x_test, y_test = mnist_data
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