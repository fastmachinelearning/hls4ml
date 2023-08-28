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
    model.add(layers.Flatten(input_shape=(8, 8)))  # Flattens the 28x28 input into a 784-dimensional vector
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
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
    x_train_resized = tf.image.resize(x_train[..., np.newaxis], size=(8, 8)).numpy()
    x_test_resized = tf.image.resize(x_test[..., np.newaxis], size=(8, 8)).numpy()

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

    ## Configure and convert the model for Vivado HLS
    print("\n============================================================================================")
    print("Configuring HLS4ML for Vivado")
    with open('Vivado_dense_config.yml', 'r') as ymlfile:
        config_vivado = yaml.safe_load(ymlfile)
    config_vivado['Backend'] = 'Vivado'
    config_vivado['OutputDir'] = 'my-Vivado-test'
    print("\n============================================================================================")
    print("HLS4ML converting keras model/Vivado to HLS C++")
    hls_model_vivado = hls4ml.converters.keras_to_hls(config_vivado)
    hls_model_vivado.compile()
    vivado_hls_model_predictions = hls_model_vivado.predict(x_test_resized)
 
    ## Compare Vivado HLS predictions with original model predictions
    print("HLS4ML Vivado Comparison")
    np.savetxt('tb_data/vivado_hls_model_predictions.dat', np.argmax(vivado_hls_model_predictions, axis=1), fmt='%d')
    mismatches, total_lines = compare_files('tb_data/vivado_hls_model_predictions.dat', 'tb_data/tb_output_predictions.dat')
    print(f"Number of mismatches: {mismatches}")
    print(f"Total number of lines: {total_lines}")
    
    ## Configure and convert the model for Catapult HLS
    print("\n============================================================================================")
    print("Configuring HLS4ML for Catapult")
    with open('Catapult_dense_config.yml', 'r') as ymlfile:
        config_ccs = yaml.safe_load(ymlfile)
    config_ccs['Backend'] = 'Catapult'
    config_ccs['OutputDir'] = 'my-Catapult-test'
    print("\n============================================================================================")
    print("HLS4ML converting keras model/Catapult to HLS C++")
    hls_model_ccs = hls4ml.converters.keras_to_hls(config_ccs)
    hls_model_ccs.compile()
    ccs_hls_model_predictions = hls_model_ccs.predict(x_test_resized)
    print("HLS4ML Vivado Comparison")
    np.savetxt('tb_data/ccs_hls_model_predictions.dat', np.argmax(ccs_hls_model_predictions, axis=1), fmt='%d')
    
    ## Compare Catapult HLS predictions with original model predictions
    mismatches, total_lines = compare_files('tb_data/ccs_hls_model_predictions.dat', 'tb_data/tb_output_predictions.dat')
    print(f"Number of mismatches: {mismatches}")
    print(f"Total number of lines: {total_lines}")

