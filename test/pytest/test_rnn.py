import pytest
import hls4ml
import numpy as np
from pathlib import Path
import math
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, GRU

test_root_path = Path(__file__).parent

rnn_layers = [SimpleRNN, LSTM, GRU]
@pytest.mark.parametrize('rnn_layer', rnn_layers)
@pytest.mark.parametrize('return_sequences', [True, False])
def test_rnn_parsing(rnn_layer, return_sequences):
    time_steps = 3
    input_size = 8
    input_shape = (time_steps, input_size)

    model_input = Input(shape=input_shape)
    model_output = rnn_layer(64, return_sequences=return_sequences)(model_input)

    model = Model(model_input, model_output)
    model.compile(optimizer='adam', loss='mse')

    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    prj_name = 'hls4mlprj_rnn_{}_seq_{}'.format(
        rnn_layer.__class__.__name__.lower(),
        int(return_sequences)
    )
    output_dir = str(test_root_path / prj_name)
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir)

    hls_layer = list(hls_model.get_layers())[1] # 0 is input, 1 is the RNN layer
    keras_layer = model.layers[1]

    # Basic sanity check, I/O, activations
    assert hls_layer.class_name == rnn_layer.__name__
    assert hls_layer.attributes['n_out'] == keras_layer.units
    assert hls_layer.attributes['activation'] == keras_layer.activation.__name__
    if 'recurrent_activation' in hls_layer.attributes: # SimpleRNN doesn't have this
        assert hls_layer.attributes['recurrent_activation'] == keras_layer.recurrent_activation.__name__
    assert hls_layer.get_input_variable().shape == list(input_shape)
    assert hls_layer.get_output_variable().shape == model_output.shape.as_list()[1:] # Ignore the batch size

    # Compare weights
    hls_weights = list(hls_layer.get_weights()) # [weights, recurrent_weights, bias, recurrent_bias]
    rnn_weights = keras_layer.get_weights() # [weights, recurrent_weights, bias]

    assert hls_weights[0].data.shape == rnn_weights[0].shape
    assert hls_weights[1].data.shape == rnn_weights[1].shape
    if 'gru' in rnn_layer.__name__.lower():
        # GRU has both bias and recurrent bias
        assert hls_weights[2].data.shape == rnn_weights[2][0].shape
        assert hls_weights[3].data.shape == rnn_weights[2][1].shape
    else:
        # LSTM and SimpleRNN only have bias
        assert hls_weights[2].data.shape == rnn_weights[2].shape
        
    np.testing.assert_array_equal(hls_weights[0].data, rnn_weights[0])
    np.testing.assert_array_equal(hls_weights[1].data, rnn_weights[1])
    if 'gru' in rnn_layer.__name__.lower():
        np.testing.assert_array_equal(hls_weights[2].data, rnn_weights[2][0])
        np.testing.assert_array_equal(hls_weights[3].data, rnn_weights[2][1])
    else:
        np.testing.assert_array_equal(hls_weights[2].data, rnn_weights[2])

@pytest.mark.parametrize('rnn_layer', [LSTM, GRU])
@pytest.mark.parametrize('return_sequences', [True, False])
@pytest.mark.parametrize('backend', ['Vivado'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('static', [True, False])
def test_rnn_accuracy(rnn_layer, return_sequences, backend, io_type, static):
    # Subtract 0.5 to include negative values
    input_shape = (5, 8)
    X = np.random.rand(50, *input_shape) - 0.5      
    
    layer_name = rnn_layer.__class__.__name__.lower()
    keras_model = Sequential()
    keras_model.add(rnn_layer(units=32, input_shape=input_shape, kernel_initializer='lecun_uniform', bias_initializer='lecun_uniform', return_sequences=return_sequences, name=layer_name))
    keras_model.compile()

    default_precision = 'ap_fixed<32, 16>' if backend == 'Vivado' else 'ac_fixed<32, 16, true>'
    hls_config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name', default_precision=default_precision)
    hls_config['LayerName'][layer_name]['static'] = static
    output_dir = 'hls4mlprj_rnn_accuracy_{}_static_{}_ret_seq_{}_{}_{}'.format(
        rnn_layer.__class__.__name__.lower(),
        int(static),
        int(return_sequences),
        backend,
        io_type
    )

    hls_model = hls4ml.converters.convert_from_keras_model(
                            keras_model, 
                            hls_config=hls_config,
                            output_dir=output_dir, 
                            backend=backend,
                            io_type=io_type
    )
    hls_model.compile()

    keras_prediction = keras_model.predict(X)
    hls_prediction = hls_model.predict(X)
    np.testing.assert_allclose(hls_prediction.flatten(), keras_prediction.flatten(), rtol=0.0, atol=3e-2)
