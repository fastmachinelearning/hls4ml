import os
import hls4ml
import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import numpy as np
from qkeras import *

print(hls4ml.__version__)

# Load test data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Creating a smaller dataset
train_labels = np.tile(train_labels, 3)
train_labels = np.clip(train_labels, a_min=0, a_max=7)
train_labels = train_labels[:125440].reshape([10, 112, 112])
test_labels = np.tile(test_labels, 13)
test_labels = np.clip(test_labels, a_min=0, a_max=7)
test_labels = test_labels[:125440].reshape([10, 112, 112])

# Normalizing the dataset (data appears to be 8bit ints anyway so no normalization needed?)
train_images = train_images[:1920].astype('float32') / 1
test_images = test_images[:1920].astype('float32') / 1

# Reshaping the data for inputing into the model
train_images = train_images.reshape((10,  224, 224,3))
test_images = test_images.reshape((10,  224, 224,3))

# Write testbench data (as integers)
np.savetxt('tb_input_features.dat', np.array(test_images.reshape(test_images.shape[0], -1), dtype='int32'), fmt='%d')
np.savetxt('tb_output_predictions.dat', np.array(test_labels.reshape(test_labels.shape[0], -1), dtype='int32'), fmt='%d')

# Defining and compiling the keras model
def create_model():
    model = tf.keras.Sequential()
    model.add(QConv2DBatchnorm(filters=8, kernel_size=3, padding='same', strides=2, activation='relu', input_shape=(224,224,3), 
        kernel_quantizer=quantizers.quantized_bits(bits=8, integer=0), bias_quantizer=quantizers.quantized_bits(bits=8, integer=0)))
    #Compiling the model
    model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
    return model

# Create a basic model instance
model = create_model()
model.summary()

model.fit(train_images, train_labels, epochs=1, validation_data=(test_images,test_labels))

# Serialize model to json
json_model = model.to_json()

# Save the model architecture to JSON file
with open('fashionmnist_model.json', 'w') as json_file:
    json_file.write(json_model)

# Saving the weights of the model
model.save_weights('FashionMNIST_weights.h5')

# Model loss and accuracy
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
model.save('FashionMNIST.h5')

import json
# Write Json
with open('fashionmnist_model.json', 'r') as f:
  x = json.loads(f.read())
# Rewrite pretty-printed
with open('fashionmnist_model2.json', 'w') as f:
  f.write(json.dumps(x, indent=2))

config = {}
config['Backend'] = 'Vivado'
config['ClockPeriod'] = 100
config['HLSConfig'] = {'Model': {'Precision': 'ac_fixed<16,8>', 'ReuseFactor': 1}}
config['IOType'] = 'io_stream'
config['KerasH5'] = 'FashionMNIST_weights.h5'
config['KerasJson'] = 'fashionmnist_model2.json'
config['ProjectName'] = 'myproject'
config['Part'] = 'xcku115-flvb2104-2-i'
config['XilinxPart'] = 'xcku115-flvb2104-2-i'
config['OutputDir'] = 'my-Vivado-test'
config['InputData'] = 'tb_input_features.dat'
config['OutputPredictions'] = 'tb_output_predictions.dat'

hls_model = hls4ml.converters.keras_to_hls(config)
hls_model.build()

