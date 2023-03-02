#Importing required libararies
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import numpy as np
from qkeras import *



#Loading Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
#creating a smaller dataset
train_labels = np.tile(train_labels, 3)
train_labels = np.clip(train_labels, a_min=0, a_max=7)
train_labels = train_labels[:125440].reshape([10, 112, 112])
test_labels = np.tile(test_labels, 13)
test_labels = np.clip(test_labels, a_min=0, a_max=7)
test_labels = test_labels[:125440].reshape([10, 112, 112])
#Normalizing the dataset
train_images = train_images[:1920].astype('float32') / 255
test_images = test_images[:1920].astype('float32') / 255
# Reshaping the data for inputing into the model
train_images = train_images.reshape((10,  224, 224,3))
test_images = test_images.reshape((10,  224, 224,3))
#Defining and compiling the keras model
def create_model():
    model = tf.keras.Sequential()
    # Must define the input shape in the first layer of the neural network
   # model.add(QConv2D(filters=8, kernel_size=3, padding='same', strides=2, activation='relu', input_shape=(224,224,3), 
   #     kernel_quantizer=quantizers.quantized_bits(bits=8, integer=0), bias_quantizer=quantizers.quantized_bits(bits=8, integer=0)))
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

model.fit(train_images,
          train_labels,

          epochs=1,
          validation_data=(test_images,test_labels))


# serialize model to json
json_model = model.to_json()
#save the model architecture to JSON file
with open('fashionmnist_model.json', 'w') as json_file:
    json_file.write(json_model)
#saving the weights of the model
model.save_weights('FashionMNIST_weights.h5')
#Model loss and accuracy
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
model.save('FashionMNIST.h5')

import json
with open('fashionmnist_model.json', 'r') as f:
  x = json.loads(f.read())
with open('fashionmnist_model2.json', 'w') as f:
  f.write(json.dumps(x, indent=2))
