import hls4ml
import tensorflow as tf
import numpy as np
import pytest
from sklearn.metrics import accuracy_score

def flat_distribution(N, M):
  return np.random.rand(N, M)

def high_accuracy_distribution(N, M):
  '''Start with a flat distribution, then pick a random member of each row to amplify'''
  x = np.random.rand(N, M)
  imax = np.random.randint(0,M,size=N)
  x[:,imax] *= 10
  return x

@pytest.fixture()
def generate_data(function):
  return function(1000,8)

@pytest.mark.parametrize('strategy,function', [('latency', flat_distribution),
                                               ('stable', flat_distribution),
                                               ('stable', high_accuracy_distribution)])
def test_softmax(strategy, generate_data):
  X = generate_data
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Activation(input_shape=(8,), activation='softmax', name='softmax'))
  model.compile()
  cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')
  cfg['LayerName']['softmax']['Strategy'] = strategy
  cfg['LayerName']['softmax']['inv_table_t'] = 'ap_fixed<18,8>'
  hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=cfg, output_dir='softmax_prj')
  hls_model.compile()
  y_keras = model.predict(X)
  y_hls4ml = hls_model.predict(X)

  acc_hls4ml = accuracy_score(np.argmax(y_keras, axis=1), np.argmax(y_hls4ml, axis=1))

  print('Accuracy hls4ml relative to keras: {}'.format(acc_hls4ml))

  assert acc_hls4ml >= 0.98
