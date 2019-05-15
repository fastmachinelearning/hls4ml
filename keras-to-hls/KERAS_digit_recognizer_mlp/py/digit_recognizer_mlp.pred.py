from __future__ import print_function

import keras
from keras.models import model_from_json
from keras.datasets import mnist
from keras.utils import np_utils

import numpy as np

import matplotlib.pyplot as plt
import numpy as np

# Model reconstruction from JSON file
with open('../../example-keras-model-files/KERAS_digit_recognizer_mlp.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('../../example-keras-model-files/KERAS_digit_recognizer_mlp_weights.h5')


# Load the MNIST data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test /= 255

# Convert class vectors to binary class matrices
y_test = keras.utils.to_categorical(y_test, 10)

# Choose a test image
index_image = 1
image = x_test[index_image]
#print("INFO: x_test size", x_test.shape())

# Run prediction
pred = model.predict(np.array([image]))

# Some information
print('INFO: input shape: ', image.shape)
print('INFO: image shape: ', image.reshape(28, 28).shape)
print('INFO: predictions: ', pred[0])
print('INFO: top prediction: ', pred.argmax())

def print_array_to_h(data):
    f = open("inputs.h","w")

    f.write("input_t  data_str[N_INPUTS] = {")

    i=0
    for d in data:
        if i==0:
            f.write("%.12f" % d)
        else:
            f.write(", %.12f" % d)
        i=i+1
    f.write("};\n")
    f.close()


print_array_to_h(image.reshape(28*28, 1))

# Show image
plt.imshow(np.array([image]).reshape(28, 28), cmap='Greys')
plt.show()

