from pathlib import Path

import numpy as np
import pytest
from keras.layers import GRU, LSTM, Bidirectional, Input, SimpleRNN
from keras.models import Model, Sequential

import hls4ml

test_root_path = Path(__file__).parent

rnn_layers = [SimpleRNN, LSTM, GRU, Bidirectional]


def create_model_parsing(rnn_layer, return_sequences):
    time_steps = 3
    input_size = 8
    input_shape = (time_steps, input_size)

    model_input = Input(shape=input_shape)
    if rnn_layer.__name__ != 'Bidirectional':
        model_output = rnn_layer(64, return_sequences=return_sequences)(model_input)
    else:
        forward_layer = LSTM(37, return_sequences=return_sequences)
        bacwkard_layer = GRU(27, return_sequences=return_sequences, go_backwards=True)
        model_output = rnn_layer(forward_layer, backward_layer=bacwkard_layer)(model_input)

    model = Model(model_input, model_output)
    model.compile(optimizer='adam', loss='mse')

    return model


def compare_attributes(hls_layer, keras_layer):
    import keras

    assert hls_layer.class_name == keras_layer.__class__.__name__
    if keras.__version__ >= '3.0':
        input = list(keras_layer.input.shape)[1:]  # Ignore the batch size
        output = list(keras_layer(np.random.rand(1, *input)).shape)[1:]  # Ignore the batch size
        assert hls_layer.get_input_variable().shape == input
        assert hls_layer.get_output_variable().shape == output
    else:
        assert hls_layer.get_input_variable().shape == list(keras_layer.input_shape)[1:]  # Ignore the batch size
        assert hls_layer.get_output_variable().shape == list(keras_layer.output_shape)[1:]  # Ignore the batch size
    if keras_layer.__class__.__name__ != 'Bidirectional':
        assert hls_layer.attributes['n_out'] == keras_layer.units
        assert hls_layer.attributes['activation'] == keras_layer.activation.__name__
        if 'recurrent_activation' in hls_layer.attributes:  # SimpleRNN doesn't have this
            assert hls_layer.attributes['recurrent_activation'] == keras_layer.recurrent_activation.__name__
    else:
        assert hls_layer.attributes['merge_mode'] == keras_layer.merge_mode
        n_out = 0
        for inner_layer, direction in [(keras_layer.forward_layer, 'forward'), (keras_layer.backward_layer, 'backward')]:
            assert hls_layer.attributes[f'{direction}_n_states'] == inner_layer.units
            if hls_layer.attributes['merge_mode'] == 'concat':
                n_out += inner_layer.units
            else:
                n_out = inner_layer.units
            assert hls_layer.attributes[f'{direction}_activation'] == inner_layer.activation.__name__
            if f'{direction}_recurrent_activation' in hls_layer.attributes:  # SimpleRNN doesn't have this
                assert hls_layer.attributes[f'{direction}_recurrent_activation'] == inner_layer.recurrent_activation.__name__
        assert hls_layer.attributes['n_out'] == n_out


def compare_weights(hls_weights, keras_weights, keras_layer):
    def comparison(hls_weights, keras_weights, class_name):
        assert hls_weights[0].data.shape == keras_weights[0].shape
        assert hls_weights[1].data.shape == keras_weights[1].shape
        if class_name == 'GRU':
            # GRU has both bias and recurrent bias
            assert hls_weights[2].data.shape == keras_weights[2][0].shape
            assert hls_weights[3].data.shape == keras_weights[2][1].shape
        else:
            # LSTM and SimpleRNN only have bias
            assert hls_weights[2].data.shape == keras_weights[2].shape

        np.testing.assert_array_equal(hls_weights[0].data, keras_weights[0])
        np.testing.assert_array_equal(hls_weights[1].data, keras_weights[1])
        if class_name == 'GRU':
            np.testing.assert_array_equal(hls_weights[2].data, keras_weights[2][0])
            np.testing.assert_array_equal(hls_weights[3].data, keras_weights[2][1])
        else:
            np.testing.assert_array_equal(hls_weights[2].data, keras_weights[2])

    if keras_layer.__class__.__name__ != 'Bidirectional':
        comparison(hls_weights, keras_weights, keras_layer.__class__.__name__)
    else:
        for i, inner_layer in enumerate([keras_layer.forward_layer, keras_layer.backward_layer]):
            comparison(hls_weights[4 * i : 4 * (i + 1)], keras_weights[3 * i : 3 * (i + 1)], inner_layer.__class__.__name__)


@pytest.mark.parametrize('rnn_layer', rnn_layers)
@pytest.mark.parametrize('return_sequences', [True, False])
def test_rnn_parsing(rnn_layer, return_sequences):

    model = create_model_parsing(rnn_layer, return_sequences)

    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Vivado')
    prj_name = f'hls4mlprj_rnn_{rnn_layer.__class__.__name__.lower()}_seq_{int(return_sequences)}'
    output_dir = str(test_root_path / prj_name)
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir)

    hls_layer = list(hls_model.get_layers())[1]  # 0 is input, 1 is the RNN layer
    keras_layer = model.layers[1]

    # Basic sanity check, I/O, activations
    compare_attributes(hls_layer, keras_layer)

    # Compare weights
    hls_weights = list(hls_layer.get_weights())  # [weights, recurrent_weights, bias, recurrent_bias]
    keras_weights = keras_layer.get_weights()  # [weights, recurrent_weights, bias]
    compare_weights(hls_weights, keras_weights, keras_layer)


def create_model_accuracy(rnn_layer, return_sequences):
    # Subtract 0.5 to include negative values
    input_shape = (12, 8)
    X = np.random.rand(50, *input_shape) - 0.5

    layer_name = rnn_layer.__name__
    model = Sequential()
    model.add(Input(shape=input_shape))
    if layer_name != 'Bidirectional':
        test_layer = rnn_layer(
            units=32,
            input_shape=input_shape,
            kernel_initializer='lecun_uniform',
            recurrent_initializer='lecun_uniform',
            bias_initializer='lecun_uniform',
            return_sequences=return_sequences,
            name=layer_name,
        )
    else:
        test_layer = Bidirectional(
            LSTM(
                units=15,
                input_shape=input_shape,
                kernel_initializer='lecun_uniform',
                recurrent_initializer='lecun_uniform',
                bias_initializer='lecun_uniform',
                return_sequences=return_sequences,
            ),
            backward_layer=GRU(
                units=17,
                input_shape=input_shape,
                kernel_initializer='lecun_uniform',
                recurrent_initializer='lecun_uniform',
                bias_initializer='lecun_uniform',
                return_sequences=return_sequences,
                go_backwards=True,
            ),
            name=layer_name,
        )
    model.add(test_layer)
    model.compile()
    return model, X


@pytest.mark.parametrize(
    'rnn_layer, backend, io_type, strategy',
    [
        (SimpleRNN, 'Quartus', 'io_parallel', 'resource'),
        (SimpleRNN, 'oneAPI', 'io_parallel', 'resource'),
        (LSTM, 'Vivado', 'io_parallel', 'resource'),
        (LSTM, 'Vivado', 'io_parallel', 'latency'),
        (LSTM, 'Vitis', 'io_parallel', 'resource'),
        (LSTM, 'Vitis', 'io_parallel', 'latency'),
        (LSTM, 'Quartus', 'io_parallel', 'resource'),
        (LSTM, 'oneAPI', 'io_parallel', 'resource'),
        (LSTM, 'Vivado', 'io_stream', 'resource'),
        (LSTM, 'Vivado', 'io_stream', 'latency'),
        (LSTM, 'Vitis', 'io_stream', 'resource'),
        (LSTM, 'Vitis', 'io_stream', 'latency'),
        (GRU, 'Vivado', 'io_parallel', 'resource'),
        (GRU, 'Vivado', 'io_parallel', 'latency'),
        (GRU, 'Vitis', 'io_parallel', 'resource'),
        (GRU, 'Vitis', 'io_parallel', 'latency'),
        (GRU, 'Quartus', 'io_parallel', 'resource'),
        (GRU, 'oneAPI', 'io_parallel', 'resource'),
        (GRU, 'Vivado', 'io_stream', 'resource'),
        (GRU, 'Vivado', 'io_stream', 'latency'),
        (GRU, 'Vitis', 'io_stream', 'resource'),
        (GRU, 'Vitis', 'io_stream', 'latency'),
        (GRU, 'Quartus', 'io_stream', 'resource'),
        (GRU, 'oneAPI', 'io_stream', 'resource'),
        (Bidirectional, 'Vivado', 'io_parallel', 'resource'),
        (Bidirectional, 'Vivado', 'io_parallel', 'latency'),
        (Bidirectional, 'Vitis', 'io_parallel', 'resource'),
        (Bidirectional, 'Vitis', 'io_parallel', 'latency'),
    ],
)
@pytest.mark.parametrize('return_sequences', [True, False])
@pytest.mark.parametrize('static', [True, False])
def test_rnn_accuracy(rnn_layer, return_sequences, backend, io_type, strategy, static):
    layer_name = rnn_layer.__name__

    model, X = create_model_accuracy(rnn_layer, return_sequences)

    default_precision = 'ap_fixed<32, 16>' if backend in ['Vivado', 'Vitis'] else 'ac_fixed<32, 16, true>'
    hls_config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', default_precision=default_precision, backend=backend
    )
    hls_config['LayerName'][layer_name]['static'] = static
    hls_config['LayerName'][layer_name]['Strategy'] = strategy
    prj_name = (
        'hls4mlprj_rnn_accuracy_'
        + f'{layer_name}_static_{int(static)}_ret_seq_{int(return_sequences)}_'
        + f'{backend}_{io_type}_{strategy}'
    )
    output_dir = str(test_root_path / prj_name)

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=hls_config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    hls_model.compile()

    keras_prediction = model.predict(X)
    hls_prediction = hls_model.predict(X)
    np.testing.assert_allclose(hls_prediction.flatten(), keras_prediction.flatten(), rtol=0.0, atol=5e-2)
