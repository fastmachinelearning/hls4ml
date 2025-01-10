# Skip Keras Surgeon tests for now, due to conflicting PyTest versions
import keras
import pytest
from packaging import version
from qkeras import QActivation, QConv2D, QDense, quantized_bits
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, Conv2D, Dense, Flatten, MaxPooling2D, ReLU, Softmax
from tensorflow.keras.models import Sequential

from hls4ml.optimization.dsp_aware_pruning.keras.reduction import reduce_model
from hls4ml.optimization.dsp_aware_pruning.keras.utils import get_model_sparsity

pytest.skip(allow_module_level=True)


'''
Set some neurons / filters to zero and verify that these are removed
Even is some neurons (columns) in the output layer are zero, these should not be removed (to match data set labels)
Test verify the above property, by setting some zeros in the last layer and verifying these remain in place
'''


@pytest.mark.skipif(
    version.parse(keras.__version__) > version.parse('2.12.0'), reason='Keras Surgeon only works until Keras 2.12'
)
def test_keras_model_reduction():
    model = Sequential()
    model.add(Conv2D(8, (3, 3), input_shape=(64, 64, 1), name='conv2d_1', padding='same'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2D(32, (5, 5), padding='same', name='conv2d_2'))
    model.add(AveragePooling2D())
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Flatten())
    model.add(Dense(32, input_shape=(16,), name='dense_1', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(14, name='dense_2', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(5, name='dense_3'))
    model.add(Softmax())

    indices = {
        'conv2d_1': [2, 4, 7],
        'conv2d_2': [0, 1, 2, 3, 4, 5],
        'dense_1': [0, 5, 17, 28],
        'dense_2': [1, 9, 4],
        'dense_3': [3],
    }
    for layer in model.layers:
        if isinstance(layer, Dense):
            weights = layer.get_weights()
            weights[0][:, indices[layer.name]] = 0
            layer.set_weights(weights)
        if isinstance(layer, Conv2D):
            weights = layer.get_weights()
            weights[0][:, :, :, indices[layer.name]] = 0
            layer.set_weights(weights)

    sparsity, _ = get_model_sparsity(model)
    assert sparsity > 0

    reduced = reduce_model(model)
    assert reduced.get_layer('conv2d_1').get_weights()[0].shape == (3, 3, 1, 5)
    assert reduced.get_layer('conv2d_2').get_weights()[0].shape == (5, 5, 5, 26)
    assert reduced.get_layer('dense_1').get_weights()[0].shape == (6656, 28)
    assert reduced.get_layer('dense_2').get_weights()[0].shape == (28, 11)
    assert reduced.get_layer('dense_3').get_weights()[0].shape == (11, 5)

    _, layer_sparsity = get_model_sparsity(reduced)
    assert layer_sparsity['conv2d_1'] == 0
    assert layer_sparsity['conv2d_2'] == 0
    assert layer_sparsity['dense_1'] == 0
    assert layer_sparsity['dense_2'] == 0
    assert layer_sparsity['dense_3'] > 0


@pytest.mark.skipif(
    version.parse(keras.__version__) > version.parse('2.12.0'), reason='Keras Surgeon only works until Keras 2.12'
)
def test_qkeras_model_reduction():
    bits = 8
    activation = 'quantized_relu(4)'
    quantizer = quantized_bits(bits, 0)

    model = Sequential()
    model.add(QConv2D(8, (3, 3), input_shape=(64, 64, 1), name='qconv2d_1', padding='same', kernel_quantizer=quantizer))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(QActivation(activation, name='qrelu_1'))
    model.add(QConv2D(32, (5, 5), padding='same', name='qconv2d_2', kernel_quantizer=quantizer))
    model.add(AveragePooling2D())
    model.add(BatchNormalization())
    model.add(QActivation(activation, name='qrelu_2'))
    model.add(Flatten())
    model.add(QDense(32, input_shape=(16,), name='qdense_1', kernel_quantizer=quantizer))
    model.add(QActivation(activation, name='qrelu_3'))
    model.add(BatchNormalization())
    model.add(QDense(14, name='qdense_2', kernel_quantizer=quantizer))
    model.add(QActivation(activation, name='qrelu_4'))
    model.add(BatchNormalization())
    model.add(QDense(5, name='qdense_3', kernel_quantizer=quantizer))
    model.add(Softmax())

    indices = {
        'qconv2d_1': [2, 4, 7],
        'qconv2d_2': [0, 1, 2, 3, 4, 5],
        'qdense_1': [0, 5, 17, 28],
        'qdense_2': [1, 9, 4],
        'qdense_3': [3],
    }
    for layer in model.layers:
        if isinstance(layer, QDense):
            weights = layer.get_weights()
            weights[0][:, indices[layer.name]] = 0
            layer.set_weights(weights)
        if isinstance(layer, QConv2D):
            weights = layer.get_weights()
            weights[0][:, :, :, indices[layer.name]] = 0
            layer.set_weights(weights)

    sparsity, _ = get_model_sparsity(model)
    assert sparsity > 0

    reduced = reduce_model(model)
    assert reduced.get_layer('qconv2d_1').get_weights()[0].shape == (3, 3, 1, 5)
    assert reduced.get_layer('qconv2d_2').get_weights()[0].shape == (5, 5, 5, 26)
    assert reduced.get_layer('qdense_1').get_weights()[0].shape == (6656, 28)
    assert reduced.get_layer('qdense_2').get_weights()[0].shape == (28, 11)
    assert reduced.get_layer('qdense_3').get_weights()[0].shape == (11, 5)

    _, layer_sparsity = get_model_sparsity(reduced)
    assert layer_sparsity['qconv2d_1'] == 0
    assert layer_sparsity['qconv2d_2'] == 0
    assert layer_sparsity['qdense_1'] == 0
    assert layer_sparsity['qdense_2'] == 0
    assert layer_sparsity['qdense_3'] > 0

    # Verify network surgery has no impact on quantization
    assert isinstance(reduced.get_layer('qrelu_1'), QActivation)
    assert isinstance(reduced.get_layer('qrelu_2'), QActivation)
    assert isinstance(reduced.get_layer('qrelu_3'), QActivation)
    assert isinstance(reduced.get_layer('qrelu_4'), QActivation)
    assert reduced.get_layer('qconv2d_1').kernel_quantizer['config']['bits'] == bits
    assert reduced.get_layer('qconv2d_2').kernel_quantizer['config']['bits'] == bits
    assert reduced.get_layer('qdense_1').kernel_quantizer['config']['bits'] == bits
    assert reduced.get_layer('qdense_2').kernel_quantizer['config']['bits'] == bits
    assert reduced.get_layer('qdense_3').kernel_quantizer['config']['bits'] == bits
