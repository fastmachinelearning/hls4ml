import pytest
import hls4ml
import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Activation, Conv1D, Conv2D, \
                                    Reshape, ELU, LeakyReLU, ThresholdedReLU, \
                                    PReLU, BatchNormalization, Add, Subtract, \
                                    Multiply, Average, Maximum, Minimum, Concatenate, \
                                    MaxPooling1D, MaxPooling2D, AveragePooling1D, \
                                    AveragePooling2D
import math
from tensorflow.keras import backend as K

def test_dense():
    model = tf.keras.models.Sequential()
    model.add(Dense(2,
              input_shape=(1,),
              name='Dense',
              use_bias=True,
              kernel_initializer= tf.keras.initializers.RandomUniform(minval=1, maxval=10),
              bias_initializer='zeros',
              kernel_regularizer=None,
              bias_regularizer=None,
              activity_regularizer=None,
              kernel_constraint=None,
              bias_constraint=None))
    model.add(Activation(activation='elu', name='Activation'))
    model.compile(optimizer='adam', loss='mse')

    X_input = np.random.rand(100,1)

    keras_prediction = model.predict(X_input)

    config = hls4ml.utils.config_from_keras_model(model)

    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)

    hls_model.compile()

    hls_prediction = hls_model.predict(X_input)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=1e-2, atol=0.01)

    assert len(model.layers) + 1 == len(hls_model.get_layers())
    assert list(hls_model.get_layers())[0].attributes['class_name'] == "InputLayer"
    assert list(hls_model.get_layers())[1].attributes["class_name"] == model.layers[0]._name
    assert list(hls_model.get_layers())[2].attributes['class_name'] == model.layers[1]._name
    assert list(hls_model.get_layers())[0].attributes['input_shape'] == list(model.layers[0].input_shape[1:])
    assert list(hls_model.get_layers())[1].attributes['n_in'] == model.layers[0].input_shape[1:][0]
    assert list(hls_model.get_layers())[1].attributes['n_out'] == model.layers[0].output_shape[1:][0]
    assert list(hls_model.get_layers())[2].attributes['activation'] == str(model.layers[1].activation).split()[1]
    assert list(hls_model.get_layers())[1].attributes['activation'] == str(model.layers[0].activation).split()[1]

@pytest.mark.parametrize("activation_function", [Activation(activation='relu', name='Activation'),
                                                 LeakyReLU(alpha=1.0),
                                                 ELU(alpha=1.0),
                                                 PReLU(alpha_initializer="zeros",),])
                                                 # TODO: add ThresholdedReLU test when it can be made to pass
                                                 #ThresholdedReLU(theta=1.0)])
def test_activations(activation_function):
    model = tf.keras.models.Sequential()
    model.add(Dense(64,
              input_shape=(1,),
              name='Dense',
              kernel_initializer='lecun_uniform',
              kernel_regularizer=None))
    model.add(activation_function)

    model.compile(optimizer='adam', loss='mse')
    X_input = np.random.rand(100,1)
    keras_prediction = model.predict(X_input)
    config = hls4ml.utils.config_from_keras_model(model)
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)
    hls_model.compile()
    hls_prediction = hls_model.predict(X_input)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=1e-2, atol=0.01)

    assert len(model.layers) + 1 == len(hls_model.get_layers())

    assert list(hls_model.get_layers())[2].attributes['class_name'] == activation_function.__class__.__name__


keras_conv1d = [Conv1D]
padds_options = ['same', 'valid']
@pytest.mark.parametrize("conv1d", keras_conv1d)
@pytest.mark.parametrize("padds", padds_options)
def test_conv1d(conv1d, padds):
    model = tf.keras.models.Sequential()
    input_shape = (10, 128, 4)
    model.add(conv1d(filters=32,
                     kernel_size=3,
                     strides=1,
                     padding=padds,
                     activation='relu',
                     input_shape=input_shape[1:],
                     kernel_initializer='normal',
                     use_bias=False,
                     data_format='channels_last'))
    model.add(Activation(activation='relu'))

    model.compile(optimizer='adam', loss='mse')
    X_input = np.random.rand(10,128,4)
    keras_prediction = model.predict(X_input)
    config = hls4ml.utils.config_from_keras_model(model)
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)
    hls_model.compile()
    hls_prediction = hls_model.predict(X_input)

    assert len(model.layers) + 2 == len(hls_model.get_layers())
    assert list(hls_model.get_layers())[1].attributes['name'] == model.layers[0]._name

    print(list(hls_model.get_layers())[1].attributes)
    if conv1d == 'Conv1D':
      assert list(hls_model.get_layers())[1].attributes['class_name'] == 'Conv1D'
    assert list(hls_model.get_layers())[1].attributes['activation'] == str(model.layers[0].activation).split()[1]
    assert list(hls_model.get_layers())[1].attributes["in_width"] == model.layers[0]._batch_input_shape[1]
    assert list(hls_model.get_layers())[1].attributes['filt_width'] == model.layers[0].kernel_size[0]
    assert list(hls_model.get_layers())[1].attributes['n_chan'] == model.layers[0].input_shape[2]
    assert list(hls_model.get_layers())[1].attributes['n_filt'] == model.layers[0].filters
    assert list(hls_model.get_layers())[1].attributes['stride_width'] == model.layers[0].strides[0]
    assert list(hls_model.get_layers())[1].attributes['padding'] == model.layers[0].padding
    assert list(hls_model.get_layers())[1].attributes['data_format'] == model.layers[0].data_format
    assert list(hls_model.get_layers())[1].attributes["out_width"] == list(model.layers[0].output_shape)[1]

    out_width	= math.ceil(float(model.layers[0]._batch_input_shape[2]) / float(model.layers[0].strides[0]))
    pad_along_width	= max((out_width - 1) * model.layers[0].strides[0] + model.layers[0].kernel_size[0] - model.layers[0]._batch_input_shape[2], 0)
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    out_valid	= math.ceil(float(model.layers[0]._batch_input_shape[1] - model.layers[0].kernel_size[0] + 1) / float(model.layers[0].strides[0]))

    if model.layers[0].padding == 'same':
        assert list(hls_model.get_layers())[1].attributes['pad_left'] == pad_left
        assert list(hls_model.get_layers())[1].attributes['pad_right'] == pad_right

    elif model.layers[0].padding == 'valid':
        assert list(hls_model.get_layers())[1].attributes['pad_left'] == 0
        assert list(hls_model.get_layers())[1].attributes['pad_right'] == 0

# DONE
keras_conv2d = [Conv2D]
padds_options = ['same', 'valid']
chans_options = ['channels_last']
@pytest.mark.parametrize("conv2d", keras_conv2d)
@pytest.mark.parametrize("chans", chans_options)
@pytest.mark.parametrize("padds", padds_options)
def test_conv2d(conv2d, chans, padds):
    model = tf.keras.models.Sequential()
    input_shape = (4, 4, 28, 30)
    model.add(conv2d(filters=32,
                     kernel_size=(4,4),
                     strides=(4,4),
                     padding=padds,
                     activation='relu',
                     input_shape=input_shape[1:],
                     kernel_initializer='normal',
                     use_bias=False,
                     data_format=chans
                     ))
    model.add(Activation(activation='relu'))

    model.compile(optimizer='adam', loss='mse')
    X_input = np.random.rand(4, 4, 28, 30)
    keras_prediction = model.predict(X_input)
    config = hls4ml.utils.config_from_keras_model(model)
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)
    hls_model.compile()
    hls_prediction = hls_model.predict(X_input)

    assert len(model.layers) + 2 == len(hls_model.get_layers())
    assert list(hls_model.get_layers())[1].attributes['name'] == model.layers[0]._name

    if conv2d == 'Conv2D':
      assert list(hls_model.get_layers())[1].attributes['class_name'] == 'Conv2D'
    assert list(hls_model.get_layers())[1].attributes['activation'] == str(model.layers[0].activation).split()[1]
    assert list(hls_model.get_layers())[1].attributes['filt_width'] == model.layers[0].kernel_size[1]
    assert list(hls_model.get_layers())[1].attributes['filt_height'] == model.layers[0].kernel_size[0]
    assert list(hls_model.get_layers())[1].attributes['n_filt'] == model.layers[0].filters
    assert list(hls_model.get_layers())[1].attributes['stride_width'] == model.layers[0].strides[1]
    assert list(hls_model.get_layers())[1].attributes['stride_height'] == model.layers[0].strides[0]
    assert list(hls_model.get_layers())[1].attributes['padding'] == model.layers[0].padding
    assert list(hls_model.get_layers())[1].attributes['data_format'] == model.layers[0].data_format

    if model.layers[0].data_format == 'channels_first':
      assert list(hls_model.get_layers())[1].attributes['n_chan'] == model.layers[0]._batch_input_shape[1]
      assert list(hls_model.get_layers())[1].attributes['in_height'] == model.layers[0]._batch_input_shape[2]
      assert list(hls_model.get_layers())[1].attributes['in_width'] == model.layers[0]._batch_input_shape[3]
      assert list(hls_model.get_layers())[1].attributes['out_height'] == model.layers[0].output_shape[2]
      assert list(hls_model.get_layers())[1].attributes['out_width'] == model.layers[0].output_shape[3]
    elif model.layers[0].data_format == 'channels_last':
      assert list(hls_model.get_layers())[1].attributes['n_chan'] == model.layers[0]._batch_input_shape[3]
      assert list(hls_model.get_layers())[1].attributes['in_height'] == model.layers[0]._batch_input_shape[1]
      assert list(hls_model.get_layers())[1].attributes['in_width'] == model.layers[0]._batch_input_shape[2]
      assert list(hls_model.get_layers())[1].attributes['out_height'] == model.layers[0].output_shape[1]
      assert list(hls_model.get_layers())[1].attributes['out_width'] == model.layers[0].output_shape[2]

    if model.layers[0].padding =='same':
      if model.layers[0].data_format == 'channels_first':
        out_height	= model.layers[0].output_shape[2]
        out_width	= model.layers[0].output_shape[3]
        pad_along_height	= max((out_height - 1) * model.layers[0].strides[0] + model.layers[0].kernel_size[0] - model.layers[0]._batch_input_shape[2], 0)
        pad_along_width	= max((out_width - 1) * model.layers[0].strides[1] + model.layers[0].kernel_size[1] - model.layers[0]._batch_input_shape[3], 0)

      elif model.layers[0].data_format == 'channels_last':
        out_height	= model.layers[0].output_shape[1]
        out_width	= model.layers[0].output_shape[2]
        pad_along_height	= max((out_height - 1) * model.layers[0].strides[0] + model.layers[0].kernel_size[0] - model.layers[0]._batch_input_shape[1], 0)
        pad_along_width	= max((out_width - 1) * model.layers[0].strides[1] + model.layers[0].kernel_size[1] - model.layers[0]._batch_input_shape[2], 0)

      pad_top	= pad_along_height // 2
      pad_bottom	= pad_along_height - pad_top
      pad_left	= pad_along_width // 2
      pad_right	= pad_along_width - pad_left
      assert list(hls_model.get_layers())[1].attributes['pad_top'] == pad_top
      assert list(hls_model.get_layers())[1].attributes['pad_bottom'] == pad_bottom
      assert list(hls_model.get_layers())[1].attributes['pad_left'] == pad_left
      assert list(hls_model.get_layers())[1].attributes['pad_right'] == pad_right

    elif model.layers[0].padding =='valid':
      assert list(hls_model.get_layers())[1].attributes['pad_top'] == 0
      assert list(hls_model.get_layers())[1].attributes['pad_bottom'] == 0
      assert list(hls_model.get_layers())[1].attributes['pad_left'] == 0
      assert list(hls_model.get_layers())[1].attributes['pad_right'] == 0

pooling_layers = [MaxPooling1D, MaxPooling2D, AveragePooling1D, AveragePooling2D]
padds_options = ['same', 'valid']
chans_options = ['channels_first', 'channels_last']
@pytest.mark.parametrize("poolings", pooling_layers)
@pytest.mark.parametrize("padds", padds_options)
@pytest.mark.parametrize("chans", chans_options)
def test_pooling(poolings, padds, chans):
    model = tf.keras.models.Sequential()
    if poolings == 'MaxPooling2D' or poolings == 'AveragePooling2D':
        model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

        model.compile(optimizer='adam', loss='mse')
        X_input = np.random.rand(10, 128,4)
        keras_prediction = model.predict(X_input)
        config = hls4ml.utils.config_from_keras_model(model)
        hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)
        hls_model.compile()
        hls_prediction = hls_model.predict(X_input)

        for i in range(2):
          assert list(hls_model.get_layers())[i + 3].attributes['name'] == model.layers[i + 1]._name
          assert list(hls_model.get_layers())[i + 3].attributes['class_name'][-2] == str(2)
          assert list(hls_model.get_layers())[i + 3].attributes['stride_height'] == model.layers[i + 1].strides[0]
          assert list(hls_model.get_layers())[i + 3].attributes['stride_width'] == model.layers[i + 1].strides[1]
          assert list(hls_model.get_layers())[i + 3].attributes['pool_height'] == model.layers[i + 1].pool_size[1]
          assert list(hls_model.get_layers())[i + 3].attributes['pool_width'] == model.layers[i + 1].pool_size[0]
          assert list(hls_model.get_layers())[i + 3].attributes['padding'] == model.layers[i + 1].padding

          if list(hls_model.get_layers())[i + 3].attributes['data_format'] == 'channels_last':
            assert list(hls_model.get_layers())[i + 3].attributes['in_height'] == model.layers[i + 1].input_shape[1]
            assert list(hls_model.get_layers())[i + 3].attributes['in_width'] == model.layers[i + 1].input_shape[2]
            assert list(hls_model.get_layers())[i + 3].attributes['n_filt'] == model.layers[i + 1].input_shape[3]
          elif list(hls_model.get_layers())[i + 3].attributes['data_format'] == 'channels_first':
            assert list(hls_model.get_layers())[i + 3].attributes['in_height'] == model.layers[i + 1].input_shape[2]
            assert list(hls_model.get_layers())[i + 3].attributes['in_width'] == model.layers[i + 1].input_shape[3]
            assert list(hls_model.get_layers())[i + 3].attributes['n_filt'] == model.layers[i + 1].input_shape[1]

          if list(hls_model.get_layers())[i + 3].attributes['padding'] == 'same':
            # Height
            in_height = model.layers[i + 1].input_shape[1]
            if model.layers[i + 1].data_format == 'channels_first':
              in_height = model.layers[i + 1].input_shape[2]
            out_height = int(math.ceil(float(in_height) / float(model.layers[i + 1].strides[0])))
            assert out_height == list(hls_model.get_layers())[i + 3].attributes['out_height']
            if in_height % model.layers[i + 1].strides[0] == 0:
              pad_along_height = max(model.layers[i + 1].pool_size[1] - model.layers[i + 1].strides[0], 0)
            else:
              pad_along_height = max(model.layers[i + 1].pool_size[1] - (in_height % model.layers[i + 1].strides[0]), 0)
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            assert pad_bottom == list(hls_model.get_layers())[i + 3].attributes['pad_bottom']
            assert pad_top == list(hls_model.get_layers())[i + 3].attributes['pad_top']

            # Width
            in_width = model.layers[i + 1].input_shape[2]
            if model.layers[i + 1].data_format == 'channels_first':
              in_height = model.layers[1].input_shape[i + 3]
            out_width = int(math.ceil(float(in_width) / float(model.layers[i + 1].strides[1])))
            assert out_width == list(hls_model.get_layers())[i + 3].attributes['out_width']
            if in_width % model.layers[i + 1].strides[1] == 0:
              pad_along_width = max(model.layers[i + 1].pool_size[0] - model.layers[i + 1].strides[1], 0)
            else:
              pad_along_width = max(model.layers[i + 1].pool_size[0] - (in_width % model.layers[i + 1].strides[1]), 0)
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left
            assert pad_left == list(hls_model.get_layers())[i + 3].attributes['pad_left']
            assert pad_right == list(hls_model.get_layers())[i + 3].attributes['pad_right']

          elif list(hls_model.get_layers())[i + 3].attributes['padding'] == 'valid':
            if list(hls_model.get_layers())[i + 3].attributes['data_format'] == 'channels_first':
              in_height = model.layers[i + 1].input_shape[2]
              in_width = model.layers[i + 1].input_shape[3]
            elif list(hls_model.get_layers())[i + 3].attributes['data_format'] == 'channels_last':
              in_height = model.layers[i + 1].input_shape[1]
              in_width = model.layers[i + 1].input_shape[2]

            out_width = int(math.ceil(float(in_width - model.layers[i + 1].pool_size[0] + 1) / float(model.layers[i + 1].strides[1])))
            out_height = int(math.ceil(float(in_height - model.layers[i + 1].pool_size[1] + 1) / float(model.layers[i + 1].strides[0])))

            assert list(hls_model.get_layers())[i + 3].attributes['out_height'] == out_height
            assert list(hls_model.get_layers())[i + 3].attributes['out_width'] == out_width
            assert list(hls_model.get_layers())[i + 3].attributes['pad_top'] == 0
            assert list(hls_model.get_layers())[i + 3].attributes['pad_bottom'] == 0
            assert list(hls_model.get_layers())[i + 3].attributes['pad_left'] == 0
            assert list(hls_model.get_layers())[i + 3].attributes['pad_right'] == 0

    elif poolings == 'MaxPooling1D' or poolings == 'AveragePooling2D':
        input_shape = (10, 128, 4)
        model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu',
                            input_shape=input_shape[1:], kernel_initializer='normal', use_bias=False,
                            data_format='channels_last'))
        model.add(MaxPooling1D(pool_size=2, strides=None, padding=padds, data_format=chans))
        model.add(AveragePooling1D(pool_size=2, strides=None, padding=padds, data_format=chans))

        model.compile(optimizer='adam', loss='mse')
        X_input = np.random.rand(10, 128,4)
        keras_prediction = model.predict(X_input)
        config = hls4ml.utils.config_from_keras_model(model)
        hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)
        hls_model.compile()
        hls_prediction = hls_model.predict(X_input)

        for i in range(2):
          assert list(hls_model.get_layers())[i + 3].attributes['name'] == model.layers[i + 1]._name
          assert list(hls_model.get_layers())[i + 3].attributes['class_name'][-2] == str(1)
          assert list(hls_model.get_layers())[i + 3].attributes['n_in'] == model.layers[i + 1].input_shape[1]
          assert list(hls_model.get_layers())[i + 3].attributes['n_filt'] == model.layers[i + 1].input_shape[2]
          assert list(hls_model.get_layers())[i + 3].attributes['pool_size'] == model.layers[i + 1].pool_size[0]
          assert list(hls_model.get_layers())[i + 3].attributes['stride'] == model.layers[i + 1].strides[0]
          assert list(hls_model.get_layers())[i + 3].attributes['padding'] == model.layers[i + 1].padding

          out_same	= math.ceil(float(model.layers[i + 1].input_shape[1]) / float(model.layers[i + 1].strides[0]))
          out_valid	= math.ceil(float(model.layers[i + 1].input_shape[1] - model.layers[i + 1].pool_size[0] + 1) / model.layers[i + 1].strides[0])

          if list(hls_model.get_layers())[i + 3].attributes['padding'] == 'same':
            assert list(hls_model.get_layers())[i + 3].attributes['n_out'] == out_same
            if model.layers[i + 1].input_shape[1] % model.layers[i + 1].strides[0] == 0:
              pad_along_width = max(model.layers[i + 1].pool_size[0] - model.layers[i + 1].strides[0], 0)
            else:
              pad_along_width = max(model.layers[i + 1].pool_size[0] - (model.layers[i + 1].input_shape[1] % model.layers[i + 1].strides[0]), 0)
            assert list(hls_model.get_layers())[i + 3].attributes['pad_left'] == pad_along_width // 2
            assert list(hls_model.get_layers())[i + 3].attributes['pad_right'] == pad_along_width - pad_along_width // 2

          elif list(hls_model.get_layers())[i + 3].attributes['padding'] == 'valid':
            assert list(hls_model.get_layers())[i + 3].attributes['n_out'] == out_valid
            assert list(hls_model.get_layers())[i + 3].attributes['pad_left'] == 0
            assert list(hls_model.get_layers())[i + 3].attributes['pad_right'] == 0