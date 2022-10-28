import pytest
import hls4ml
import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Activation, Conv1D, Conv2D, \
                                    Reshape, ELU, LeakyReLU, ThresholdedReLU, \
                                    PReLU, BatchNormalization, Add, Subtract, \
                                    Multiply, Average, Maximum, Minimum, Concatenate, \
                                    MaxPooling1D, MaxPooling2D, AveragePooling1D, \
                                    AveragePooling2D
import math
from tensorflow.keras import backend as K

test_root_path = Path(__file__).parent

@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_dense(backend, io_type):
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
    output_dir = str(test_root_path / f'hls4mlprj_keras_api_dense_{backend}_{io_type}')

    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)

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

# TODO: add ThresholdedReLU test when it can be made to pass
# https://github.com/fastmachinelearning/hls4ml/issues/376
@pytest.mark.parametrize("activation_function", [Activation(activation='relu', name='Activation'),
                                                 LeakyReLU(alpha=1.0),
                                                 ELU(alpha=1.0),
                                                 PReLU(alpha_initializer="zeros",),
                                                 Activation(activation='sigmoid', name='Activation')])
                                                 #ThresholdedReLU(theta=1.0)])
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_activations(activation_function, backend, io_type):
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
    output_dir = str(test_root_path / 'hls4mlprj_keras_api_activations_{}_{}_{}'.format(activation_function.__class__.__name__, backend, io_type))
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)
    hls_model.compile()
    hls_prediction = hls_model.predict(X_input)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=1e-2, atol=0.01)

    assert len(model.layers) + 1 == len(hls_model.get_layers())

    assert list(hls_model.get_layers())[2].attributes['class_name'] == activation_function.__class__.__name__

padds_options = ['same', 'valid']
@pytest.mark.parametrize('padds', padds_options)
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
def test_conv1d(padds, backend):
    model = tf.keras.models.Sequential()
    input_shape = (10, 128, 4)
    model.add(Conv1D(filters=32,
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
    output_dir = str(test_root_path / 'hls4mlprj_keras_api_conv1d_{}_{}'.format(padds, backend))
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir, backend=backend)
    hls_model.compile()
    hls_prediction = hls_model.predict(X_input).reshape(keras_prediction.shape)

     # 5e-2 might be too high
    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=5e-2)

    assert len(model.layers) + 2 == len(hls_model.get_layers())
    assert list(hls_model.get_layers())[1].attributes['name'] == model.layers[0]._name
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

    if model.layers[0].padding == 'same':
        assert list(hls_model.get_layers())[1].attributes['pad_left'] == pad_left
        assert list(hls_model.get_layers())[1].attributes['pad_right'] == pad_right
    elif model.layers[0].padding == 'valid':
        assert list(hls_model.get_layers())[1].attributes['pad_left'] == 0
        assert list(hls_model.get_layers())[1].attributes['pad_right'] == 0

chans_options=['channels_last']
padds_options=['same', 'valid']
@pytest.mark.parametrize('chans', chans_options)
@pytest.mark.parametrize('padds',  padds_options)
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
def test_conv2d(chans, padds, backend):
    model = tf.keras.models.Sequential()
    input_shape = (28,28,3)
    model.add(Conv2D(filters=32,
                     kernel_size=(4,4),
                     strides=(4,4),
                     padding=padds,
                     input_shape=input_shape,
                     kernel_initializer='normal',
                     use_bias=False,
                     data_format=chans))
    model.compile(optimizer='adam', loss='mse')
    
    X_input = np.random.rand(100, *input_shape)
    keras_prediction = model.predict(X_input)
    
    config = hls4ml.utils.config_from_keras_model(model)
    output_dir = str(test_root_path / 'hls4mlprj_keras_api_conv2d_{}_{}_{}'.format(backend, chans, padds))
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir, backend=backend)
    hls_model.compile()
    hls_prediction = hls_model.predict(X_input).reshape(keras_prediction.shape)

    # A high tolerance, simply to verify correct functionality
    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=5e-2)

    assert len(model.layers) + 1 == len(hls_model.get_layers())
    assert list(hls_model.get_layers())[1].attributes['name'] == model.layers[0]._name
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
@pytest.mark.parametrize('pooling', pooling_layers)
@pytest.mark.parametrize('padds', padds_options)
@pytest.mark.parametrize('chans', chans_options)
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
def test_pooling(pooling, padds, chans, backend):
    assert '1D' in pooling.__name__ or '2D' in pooling.__name__
    
    input_shape = (18, 15, 3) if '2D' in pooling.__name__ else (121, 3)
    X_input = np.random.rand(100, *input_shape)
    
    keras_model = tf.keras.models.Sequential()
    keras_model.add(pooling(padding=padds, input_shape=input_shape))
    keras_model.compile()

    hls_cfg = hls4ml.utils.config_from_keras_model(keras_model)
    output_dir = str(test_root_path / 'hls4mlprj_keras_api_pooling_{}_channels_{}_padds_{}_backend_{}'.format(pooling.__name__, chans, padds, backend))
    hls_model = hls4ml.converters.convert_from_keras_model(keras_model, hls_config=hls_cfg, output_dir=output_dir, backend=backend)
    hls_model.compile()

    # Verify accuracy
    keras_prediction = keras_model.predict(X_input)
    hls_prediction = hls_model.predict(X_input).reshape(keras_prediction.shape)
    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=3e-2)

    # Verify correct parsing of layer
    hls_pool = list(hls_model.get_layers())[-1]
    ker_pool = keras_model.layers[-1]
    if '2D' in pooling.__name__:
      assert hls_pool.attributes['name'] == ker_pool._name
      assert hls_pool.attributes['class_name'][-2] == str(2)
      assert hls_pool.attributes['stride_height'] == ker_pool.strides[0]
      assert hls_pool.attributes['stride_width'] == ker_pool.strides[1]
      assert hls_pool.attributes['pool_height'] == ker_pool.pool_size[1]
      assert hls_pool.attributes['pool_width'] == ker_pool.pool_size[0]
      assert hls_pool.attributes['padding'] == ker_pool.padding

      if hls_pool.attributes['data_format'] == 'channels_last':
        assert hls_pool.attributes['in_height'] == ker_pool.input_shape[1]
        assert hls_pool.attributes['in_width'] == ker_pool.input_shape[2]
        assert hls_pool.attributes['n_filt'] == ker_pool.input_shape[3]
      elif hls_pool.attributes['data_format'] == 'channels_first':
        assert hls_pool.attributes['in_height'] == ker_pool.input_shape[2]
        assert hls_pool.attributes['in_width'] == ker_pool.input_shape[3]
        assert hls_pool.attributes['n_filt'] == ker_pool.input_shape[1]

      if hls_pool.attributes['padding'] == 'same':
        # Height
        in_height = ker_pool.input_shape[1]
        if ker_pool.data_format == 'channels_first':
          in_height = ker_pool.input_shape[2]
        out_height = int(math.ceil(float(in_height) / float(ker_pool.strides[0])))
        assert out_height == hls_pool.attributes['out_height']
        if in_height % ker_pool.strides[0] == 0:
          pad_along_height = max(ker_pool.pool_size[1] - ker_pool.strides[0], 0)
        else:
          pad_along_height = max(ker_pool.pool_size[1] - (in_height % ker_pool.strides[0]), 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        assert pad_bottom == hls_pool.attributes['pad_bottom']
        assert pad_top == hls_pool.attributes['pad_top']

        # Width
        in_width = ker_pool.input_shape[2]
        if ker_pool.data_format == 'channels_first':
          in_height = keras_model.layers[1].input_shape[-1]
        out_width = int(math.ceil(float(in_width) / float(ker_pool.strides[1])))
        assert out_width == hls_pool.attributes['out_width']
        if in_width % ker_pool.strides[1] == 0:
          pad_along_width = max(ker_pool.pool_size[0] - ker_pool.strides[1], 0)
        else:
          pad_along_width = max(ker_pool.pool_size[0] - (in_width % ker_pool.strides[1]), 0)
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        assert pad_left == hls_pool.attributes['pad_left']
        assert pad_right == hls_pool.attributes['pad_right']

      elif hls_pool.attributes['padding'] == 'valid':
        if hls_pool.attributes['data_format'] == 'channels_first':
          in_height = ker_pool.input_shape[2]
          in_width = ker_pool.input_shape[3]
        elif hls_pool.attributes['data_format'] == 'channels_last':
          in_height = ker_pool.input_shape[1]
          in_width = ker_pool.input_shape[2]

        out_width = int(math.ceil(float(in_width - ker_pool.pool_size[0] + 1) / float(ker_pool.strides[1])))
        out_height = int(math.ceil(float(in_height - ker_pool.pool_size[1] + 1) / float(ker_pool.strides[0])))

        assert hls_pool.attributes['out_height'] == out_height
        assert hls_pool.attributes['out_width'] == out_width
        assert hls_pool.attributes['pad_top'] == 0
        assert hls_pool.attributes['pad_bottom'] == 0
        assert hls_pool.attributes['pad_left'] == 0
        assert hls_pool.attributes['pad_right'] == 0

    elif '1D' in pooling.__name__:
      assert hls_pool.attributes['name'] == ker_pool._name
      assert hls_pool.attributes['class_name'][-2] == str(1)
      assert hls_pool.attributes['n_in'] == ker_pool.input_shape[1]
      assert hls_pool.attributes['n_filt'] == ker_pool.input_shape[2]
      assert hls_pool.attributes['pool_width'] == ker_pool.pool_size[0]
      assert hls_pool.attributes['stride_width'] == ker_pool.strides[0]
      assert hls_pool.attributes['padding'] == ker_pool.padding

      out_same	= math.ceil(float(ker_pool.input_shape[1]) / float(ker_pool.strides[0]))
      out_valid	= math.ceil(float(ker_pool.input_shape[1] - ker_pool.pool_size[0] + 1) / ker_pool.strides[0])

      if hls_pool.attributes['padding'] == 'same':
        assert hls_pool.attributes['n_out'] == out_same
        if ker_pool.input_shape[1] % ker_pool.strides[0] == 0:
          pad_along_width = max(ker_pool.pool_size[0] - ker_pool.strides[0], 0)
        else:
          pad_along_width = max(ker_pool.pool_size[0] - (ker_pool.input_shape[1] % ker_pool.strides[0]), 0)
        assert hls_pool.attributes['pad_left'] == pad_along_width // 2
        assert hls_pool.attributes['pad_right'] == pad_along_width - pad_along_width // 2

      elif hls_pool.attributes['padding'] == 'valid':
        assert hls_pool.attributes['n_out'] == out_valid
        assert hls_pool.attributes['pad_left'] == 0
        assert hls_pool.attributes['pad_right'] == 0
