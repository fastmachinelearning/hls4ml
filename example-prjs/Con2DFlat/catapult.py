import os
import hls4ml
import yaml
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from keras.utils import np_utils
import numpy as np
from qkeras import *
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Define the model architecture
def mnist_model():
    model = Sequential()
    model.add(Conv2D(20, (5, 5), use_bias=True, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(50, (3, 3), use_bias=True, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10, use_bias=True, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create an instance of the model
model = mnist_model()

# Train the model
model.fit(X_train, y_train, batch_size=128, epochs=5, validation_data=(X_test, y_test))

# Save the input feature set and output predictions in .dat format
np.savetxt('tb_data/tb_input_features.dat', X_test.reshape(X_test.shape[0], -1), fmt='%f')
np.savetxt('tb_data/tb_output_predictions.dat', np.argmax(model.predict(X_test), axis=1), fmt='%d')

# Serialize model to json
json_model = model.to_json()

# Save the model architecture to JSON file
with open('densemnist_model.json', 'w') as json_file:
    json_file.write(json_model)

# Saving the weights of the model
print("\n============================================================================================")
print("Writing weights file")
model.save_weights('DenseMNIST_weights.h5')
model.save('DenseMNIST.h5')

# Write Json
print("\n============================================================================================")
print("Writing json file")
with open('densemnist_model.json', 'r') as f:
    x = json.loads(f.read())
# Rewrite pretty-printed
with open('densemnist_model2.json', 'w') as f:
    f.write(json.dumps(x, indent=2))

print("\n============================================================================================")
print("Configuring HLS4ML")
config = {}
config['Backend'] = 'Catapult'
config['ClockPeriod'] = 100
config['HLSConfig'] = {'Model': {'Precision': 'ac_fixed<16,8>', 'ReuseFactor': 1}}
config['IOType'] = 'io_stream'
config['KerasH5'] = 'DenseMNIST_weights.h5'
config['KerasJson'] = 'densemnist_model2.json'
config['ProjectName'] = 'myproject'
config['Part'] = 'xcku115-flvb2104-2-i'
config['XilinxPart'] = 'xcku115-flvb2104-2-i'
config['OutputDir'] = 'my-Catapult-test'
config['InputData'] = 'tb_data/tb_input_features.dat'
config['OutputPredictions'] = 'tb_data/tb_output_predictions.dat'

print("\n============================================================================================")
print("HLS4ML converting keras model to HLS C++")
hls_model = hls4ml.converters.keras_to_hls(config)

print("\n============================================================================================")
print("Building HLS C++ model")
hls_model.build()

