import os
import shutil
import hls4ml
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from qkeras import QConv2DBatchnorm, quantizers  
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score

## Function to create a simple Convolutional Neural Network model
def create_model():
    model = Sequential()
    quantizer = quantizers.quantized_bits(bits=8, integer=0)
    qconv_batchnorm = QConv2DBatchnorm(
        filters=5,
        kernel_size=5,
        strides=3,
        activation='relu',
        kernel_quantizer=quantizer,
        bias_quantizer=quantizer
    )
    model.add(qconv_batchnorm)
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation="softmax"))
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

    import os
    
    # Remove files and directories
    file_list = ['tb_data', 'my-*', 'conv2d*', 'a.out']
    for item in file_list:
        if '*' in item:
            matching_items = [f for f in os.listdir() if f.startswith(item.replace('*', ''))]
            for matching_item in matching_items:
                if os.path.isfile(matching_item):
                    os.remove(matching_item)
                elif os.path.isdir(matching_item):
                    shutil.rmtree(matching_item)
        else:
            if os.path.isfile(item):
                os.remove(item)
            elif os.path.isdir(item):
                shutil.rmtree(item)
    
    # Create directory
    os.makedirs('tb_data', exist_ok=True)

    input_shape = (28, 28, 1)
    
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
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
    np.savetxt('tb_data/y_test_labels.dat', y_test, fmt='%d')
    save_model(model, name='conv2d')
    print(hls4ml.__version__)

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
    np.savetxt('tb_data/ccs_hls_model_predictions.dat', ccs_hls_model_predictions, fmt='%d')
    print('QKeras Accuracy: {}'.format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test), axis=1))))
    print('hls4ml Accuracy: {}'.format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(ccs_hls_model_predictions, axis=1))))

    print("\n============================================================================================")
    print("Building HLS C++ model")
    hls_model_ccs.build(csim=True, synth=True, cosim=True, validation=True, vsynth=True)
    # hls_model_ccs.build()
