import os
import hls4ml
import yaml
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from qkeras import *
import json
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1

## Function to find the index of the maximum value in a list of numbers
def find_max_index(numbers):
    max_index = 0
    max_value = float(numbers[0])

    for i in range(1, len(numbers)):
        if float(numbers[i]) > max_value:
            max_index = i
            max_value = float(numbers[i])

    return max_index

## Function to compare two text files line by line
def compare_files(file1_path, file2_path):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        lines1 = file1.readlines()
        lines2 = file2.readlines()

    total_lines = max(len(lines1), len(lines2))
    mismatches = 0

    for line_num in range(total_lines):
        if line_num < len(lines1) and line_num < len(lines2):
            if lines1[line_num] != lines2[line_num]:
                mismatches += 1
        else:
            mismatches += 1

    return mismatches, total_lines

## Function to create a simple Convolutional Neural Network model
def create_model():
    # Create a model
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(4, 4)))  # Flattens the 28x28 input into a 784-dimensional vector
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model

## Function to save model architecture, weights, and configuration
def save_model(model, name=None):
    if name is None:
        name = model.name
    model.save(name + '.h5')
    model.save_weights(name + '_weights.h5')
    with open(name + '.json', 'w') as outfile:
        outfile.write(model.to_json())
    return

if __name__ == '__main__':
    
    ## Load and preprocess the Fashion MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # # # Resize the images to 4x4
    x_train_resized = tf.image.resize(x_train[..., np.newaxis], size=(4, 4)).numpy()
    x_test_resized = tf.image.resize(x_test[..., np.newaxis], size=(4, 4)).numpy()

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    ## Create and compile the model
    model = create_model()
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    ## Train the model
    model.fit(x_train_resized, y_train, epochs=10)

    ## Evaluate the model on test data
    test_loss, test_acc = model.evaluate(x_test_resized, y_test)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_acc}")

    ## Save input features and model predictions
    np.savetxt('tb_data/tb_input_features.dat', x_test_resized.reshape(x_test_resized.shape[0], -1), fmt='%f')
    np.savetxt('tb_data/tb_output_predictions.dat', np.argmax(model.predict(x_test_resized), axis=1), fmt='%d')
    np.savetxt('tb_data/y_test_labels.dat', y_test, fmt='%d')  ## Save y_test labels as well
    save_model(model, name='dense')
    print(hls4ml.__version__)

 
    ## Configure and convert the model for Catapult HLS
    config_ccs = {
    'KerasJson': 'dense.json',
    'KerasH5': 'dense_weights.h5',
    'OutputDir': 'my-Catapult-test',
    'ProjectName': 'myproject',
    'Part': 'xcku115-flvb2104-2-i',
    'XilinxPart': 'xcku115-flvb2104-2-i',
    'InputData': 'tb_data/tb_input_features.dat',
    'OutputPredictions': 'tb_data/tb_output_predictions.dat',
    'ClockPeriod': 5,
    'Backend': 'Catapult',
    'IOType': 'io_stream',
    'HLSConfig': {
        'Model': {
            'Precision': 'ac_fixed<16,6,true>',
            'ReuseFactor': 1,
            'Strategy': 'Latency',
        },
        'LayerName': {
            'softmax': {
                'Precision': 'ac_fixed<16,6,false>',
                'Strategy': 'Stable',
                'exp_table_t': 'ac_fixed<18,8,true>',
                'inv_table_t': 'ac_fixed<18,4,true>',
                'table_size': 1024,
            },
            'Dense1_input': {
                'Precision': {
                    'result': 'ac_fixed<16,6,true>',
                },
            },
            'relu1': {
                'Precision': {
                    'result': 'ac_fixed<7,1,true>',
                },
            },
            'Dense2_input': {
                'Precision': {
                    'result': 'ac_fixed<16,6,true>',
                },
            },
            'Dense1': {
                'Precision': {
                    'bias': 'ac_fixed<6,1,true>',
                    'weight': 'ac_fixed<6,1,true>',
                },
            },
            'Dense2': {
                'Precision': {
                    'bias': 'ac_fixed<6,1,true>',
                    'weight': 'ac_fixed<6,1,true>',
                },
            },
        },
    },
}

    print("\n============================================================================================")
    print("HLS4ML converting keras model/Catapult to HLS C++")
    hls_model_ccs = hls4ml.converters.keras_to_hls(config_ccs)
    hls_model_ccs.build()
 
