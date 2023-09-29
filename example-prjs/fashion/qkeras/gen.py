#Importing required libararies
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import numpy as np
from qkeras import *

# Load and preprocess the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values

# Reshape the data to fit the Conv2D layer
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#Defining and compiling the keras model
def create_model():
    model = tf.keras.Sequential()
    model.add(QConv2DBatchnorm(filters=8, kernel_size=3, padding='same', strides=2, activation='relu', input_shape=(28,28,1),kernel_quantizer=quantizers.quantized_bits(bits=8, integer=0), bias_quantizer=quantizers.quantized_bits(bits=8, integer=0)))
    model.add(Activation("softmax"))

    #Compiling the model
    model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

    return model
# Create a basic model instance
model = create_model()
model.summary()

model.fit(x_train,
          y_train,

          epochs=1,
          validation_data=(x_test,y_test))


# serialize model to json
json_model = model.to_json()
#save the model architecture to JSON file
with open('fashionmnist_model.json', 'w') as json_file:
    json_file.write(json_model)
#saving the weights of the model
model.save_weights('FashionMNIST_weights.h5')
#Model loss and accuracy
loss,acc = model.evaluate(x_test, y_test, verbose=2)
model.save('FashionMNIST.h5')

import json
with open('fashionmnist_model.json', 'r') as f:
  x = json.loads(f.read())
with open('fashionmnist_model2.json', 'w') as f:
  f.write(json.dumps(x, indent=2))
