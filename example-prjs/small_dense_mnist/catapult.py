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
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import load_img

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Resize the MNIST images to 8x8 and reshape to have a single channel
new_size = (4, 4)
x_train_resized = tf.image.resize(x_train[..., tf.newaxis], new_size)
x_test_resized = tf.image.resize(x_test[..., tf.newaxis], new_size)

# Preprocess the data
x_train = x_train_resized / 255.0
x_test = x_test_resized / 255.0

# Define the model
model = Sequential()
model.add(Flatten(input_shape=(4, 4, 1)))  # Flattens the 8x8 input into a 64-dimensional vector
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

#==========================================================#
x_test_flat = tf.reshape(x_test, (-1, 16))
predictions = model.predict(x_test)

# Save input features and output predictions
np.savetxt('tb_data/tb_input_features.dat', x_test_flat.numpy(), fmt='%f')
np.savetxt('tb_data/tb_output_predictions.dat', np.argmax(model.predict(x_test), axis=1), fmt='%d')

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

# print("\n============================================================================================")
print("Configuring HLS4ML")
with open('densemnist_config.yml', 'r') as ymlfile:
  config = yaml.safe_load(ymlfile)
config['Backend'] = 'Catapult'
config['OutputDir'] = 'my-Catapult-test'

print("\n============================================================================================")
print("HLS4ML converting keras model to HLS C++")
hls_model = hls4ml.converters.keras_to_hls(config)

print("\n============================================================================================")
print("Building HLS C++ model")
hls_model.build()

