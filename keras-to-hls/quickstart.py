# Create the HLS model

import keras_to_hls as kth

yamlConfig = kth.parse_config('keras-config.yml')
hls_model = kth.keras_to_hls_model(yamlConfig)

# Load the keras model
import keras
import json
import sys
import os
sys.path.append('../../keras-training/layers')
sys.path.append('../../keras-training/models')
sys.path.append('../../keras-training/train')
from constraints import ZeroSomeWeights
from keras.utils.generic_utils import get_custom_objects
#import tensorflow as tf
#tf.keras.backend.set_floatx('float64')
get_custom_objects().update({"ZeroSomeWeights": ZeroSomeWeights})
import yaml
#from train import parse_config, get_features
from quantized_layers import Clip, BinaryDense, TernaryDense, QuantizedDense
from models import binary_tanh, ternary_tanh, quantized_relu, relu1

json_file = open(yamlConfig['KerasJson'], 'r')
loaded_model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json(loaded_model_json, custom_objects={'ZeroSomeWeights':ZeroSomeWeights,
                                                           'BinaryDense': BinaryDense,
                                                           'binary_tanh': binary_tanh,
                                                           'quantized_relu': quantized_relu,
                                                           'relu1' : relu1,
                                                           'Clip': Clip})
model.load_weights(yamlConfig['KerasH5'])

# Test
#bn = model.layers[1]
import numpy as np
from numpy import array
x = np.zeros(shape=(1, 16))
#print(model.predict(x))

def model_up_to(model, n):
  m = keras.models.Sequential()
  for layer in model.layers[:n]:
    m.add(layer)
  return m

#for n in range(0, len(model.layers)):
#  print(n, model_up_to(model, n).predict(x))
