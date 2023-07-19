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
from tensorflow.keras import layers

# Load and preprocess the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Resize the input data to 4x4
x_train_resized = tf.image.resize(x_train, size=(4, 4)).numpy()
x_test_resized = tf.image.resize(x_test, size=(4, 4)).numpy()

num_classes = 10
input_shape = (4, 4, 1)

# Build the model
model = tf.keras.Sequential()

model.add(layers.Input(shape=input_shape))
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(num_classes, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(x_train_resized, y_train, batch_size=128, epochs=10, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test_resized, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")


# Save the input feature set and output predictions in .dat format
np.savetxt('tb_data/tb_input_features.dat', x_test_resized.reshape(x_test_resized.shape[0], -1), fmt='%f')
np.savetxt('tb_data/tb_output_predictions.dat', np.argmax(model.predict(x_test_resized), axis=1), fmt='%d')

# Serialize model to json
json_model = model.to_json()

# Save the model architecture to JSON file
with open('BatchConMnist_model.json', 'w') as json_file:
    json_file.write(json_model)

# Saving the weights of the model
print("\n============================================================================================")
print("Writing weights file")
model.save_weights('BatchConMnist_weights.h5')
model.save('BatchConMnist.h5')

# Write Json
print("\n============================================================================================")
print("Writing json file")
with open('BatchConMnist_model.json', 'r') as f:
    x = json.loads(f.read())
# Rewrite pretty-printed
with open('BatchConMnist_model2.json', 'w') as f:
    f.write(json.dumps(x, indent=2))

print("\n============================================================================================")
print("Configuring HLS4ML")
with open('BatchConMnist_config.yml', 'r') as ymlfile:
  config = yaml.safe_load(ymlfile)
config['Backend'] = 'Vivado'
config['OutputDir'] = 'my-Vivado-test'

print("\n============================================================================================")
print("HLS4ML converting keras model to HLS C++")
hls_model = hls4ml.converters.keras_to_hls(config)

print("\n============================================================================================")
print("Building HLS C++ model")
hls_model.build()

