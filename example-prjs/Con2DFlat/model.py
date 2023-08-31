import os
import hls4ml
import yaml
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from qkeras import *
import json
import subprocess
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from qkeras import QConv2DBatchnorm

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
    model = tf.keras.Sequential()
    # Create the QConv2DBatchnorm layer
    quantizer = quantizers.quantized_bits(bits=8, integer=0)
    qconv_batchnorm = QConv2DBatchnorm(
        filters=5,
        kernel_size=5,
        strides=3,
        activation='relu',
        kernel_quantizer=quantizer,
        bias_quantizer=quantizer
    )
    
    # Create the rest of the model
    model = Sequential()
    model.add(qconv_batchnorm)
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation="softmax"))
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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
    input_shape = (28, 28, 1)
    
    ## Load and preprocess the Fashion MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    ## Create and compile the model
    model = create_model()

    ## Train the model
    model.fit(x_train, y_train, epochs=10)

    ## Evaluate the model on test data
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_acc}")

    ## Save input features and model predictions
    np.savetxt('tb_data/tb_input_features.dat', x_test.reshape(x_test.shape[0], -1), fmt='%f')
    np.savetxt('tb_data/tb_output_predictions.dat', np.argmax(model.predict(x_test), axis=1), fmt='%d')
    np.savetxt('tb_data/y_test_labels.dat', y_test, fmt='%d')  ## Save y_test labels as well
    save_model(model, name='conv2d')
    print(hls4ml.__version__)

    # ## Configure and convert the model for Vivado HLS
    # print("\n============================================================================================")
    # print("Configuring HLS4ML for Vivado")
    # with open('Vivado_con2d_config.yml', 'r') as ymlfile:
        # config_vivado = yaml.safe_load(ymlfile)
    # config_vivado['Backend'] = 'Vivado'
    # config_vivado['OutputDir'] = 'my-Vivado-test'
    # print("\n============================================================================================")
    # print("HLS4ML converting keras model/Vivado to HLS C++")
    # hls_model_vivado = hls4ml.converters.keras_to_hls(config_vivado)
    # hls_model_vivado.compile()
    # vivado_hls_model_predictions = hls_model_vivado.predict(x_test)
 
    # ## Compare Vivado HLS predictions with original model predictions
    # print("HLS4ML Vivado Comparison")
    # np.savetxt('tb_data/vivado_hls_model_predictions.dat', np.argmax(vivado_hls_model_predictions, axis=1), fmt='%d')
    # mismatches, total_lines = compare_files('tb_data/vivado_hls_model_predictions.dat', 'tb_data/tb_output_predictions.dat')
    # print(f"Number of mismatches: {mismatches}")
    # print(f"Total number of lines: {total_lines}")
    
    # ## Configure and convert the model for Catapult HLS
    # print("\n============================================================================================")
    # print("Configuring HLS4ML for Catapult")
    # with open('Catapult_con2d_config.yml', 'r') as ymlfile:
        # config_ccs = yaml.safe_load(ymlfile)
    # config_ccs['Backend'] = 'Catapult'
    # config_ccs['OutputDir'] = 'my-Catapult-test'
    config_ccs = {
    'Backend': 'Catapult',
    'ClockPeriod': 5,
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
            'Conv2DQ_input': {
                'Precision': {
                    'result': 'ac_fixed<16,6,true>',
                },
            },
            'Dense_input': {
                'Precision': {
                    'result': 'ac_fixed<16,6,true>',
                },
            },
            'relu1': {
                'Precision': {
                    'result': 'ac_fixed<7,1,true>',
                },
            },
            'BatchN_input': {
                'Precision': {
                    'result': 'ac_fixed<16,6,true>',
                },
            },
            'BatchN': {
                'Precision': {
                    'bias': 'ac_fixed<6,1,true>',
                    'weight': 'ac_fixed<6,1,true>',
                },
            },
            'Conv2DQ': {
                'Precision': {
                    'bias': 'ac_fixed<6,1,true>',
                    'weight': 'ac_fixed<6,1,true>',
                },
            },
            'Dense': {
                'Precision': {
                    'bias': 'ac_fixed<6,1,true>',
                    'weight': 'ac_fixed<6,1,true>',
                },
            },
        },
    },
    'IOType': 'io_stream',
    'KerasH5': 'conv2d_weights.h5',
    'KerasJson': 'conv2d.json',
    'OutputDir': 'my-Catapult-test',
    'ProjectName': 'myproject',
    'Part': 'xcku115-flvb2104-2-i',
    'XilinxPart': 'xcku115-flvb2104-2-i',
    'InputData': 'tb_data/tb_input_features.dat',
    'OutputPredictions': 'tb_data/tb_output_predictions.dat',}

    
    print("\n============================================================================================")
    print("HLS4ML converting keras model/Catapult to HLS C++")
    hls_model_ccs = hls4ml.converters.keras_to_hls(config_ccs)
    hls_model_ccs.compile()
    ccs_hls_model_predictions = hls_model_ccs.predict(x_test)
    print("HLS4ML Vivado Comparison")
    np.savetxt('tb_data/ccs_hls_model_predictions.dat', np.argmax(ccs_hls_model_predictions, axis=1), fmt='%d')
    
    ## Compare Catapult HLS predictions with original model predictions
    mismatches, total_lines = compare_files('tb_data/ccs_hls_model_predictions.dat', 'tb_data/tb_output_predictions.dat')
    print(f"Number of mismatches: {mismatches}")
    print(f"Total number of lines: {total_lines}")

