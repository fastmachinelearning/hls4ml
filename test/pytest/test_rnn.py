import warnings
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


@pytest.mark.parametrize('rnn_layer', rnn_layers, ids=['SimpleRNN', 'LSTM', 'GRU', 'Bidirectional'])
@pytest.mark.parametrize('return_sequences', [True, False])
def test_rnn_parsing(test_case_id, rnn_layer, return_sequences):
    model = create_model_parsing(rnn_layer, return_sequences)

    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Vivado')
    output_dir = str(test_root_path / test_case_id)
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
        (SimpleRNN, 'Altera', 'io_parallel', 'resource'),
        (SimpleRNN, 'Vivado', 'io_parallel', 'resource'),
        (SimpleRNN, 'Vivado', 'io_parallel', 'latency'),
        (SimpleRNN, 'Vitis', 'io_parallel', 'resource'),
        (SimpleRNN, 'Vitis', 'io_parallel', 'latency'),
        (SimpleRNN, 'Vivado', 'io_stream', 'resource'),
        (SimpleRNN, 'Vivado', 'io_stream', 'latency'),
        (SimpleRNN, 'Vitis', 'io_stream', 'resource'),
        (SimpleRNN, 'Vitis', 'io_stream', 'latency'),
        (LSTM, 'Vivado', 'io_parallel', 'resource'),
        (LSTM, 'Vivado', 'io_parallel', 'latency'),
        (LSTM, 'Vitis', 'io_parallel', 'resource'),
        (LSTM, 'Vitis', 'io_parallel', 'latency'),
        (LSTM, 'Quartus', 'io_parallel', 'resource'),
        (LSTM, 'Altera', 'io_parallel', 'resource'),
        (LSTM, 'Vivado', 'io_stream', 'resource'),
        (LSTM, 'Vivado', 'io_stream', 'latency'),
        (LSTM, 'Vitis', 'io_stream', 'resource'),
        (LSTM, 'Vitis', 'io_stream', 'latency'),
        (GRU, 'Vivado', 'io_parallel', 'resource'),
        (GRU, 'Vivado', 'io_parallel', 'latency'),
        (GRU, 'Vitis', 'io_parallel', 'resource'),
        (GRU, 'Vitis', 'io_parallel', 'latency'),
        (GRU, 'Quartus', 'io_parallel', 'resource'),
        (GRU, 'Altera', 'io_parallel', 'resource'),
        (GRU, 'Vivado', 'io_stream', 'resource'),
        (GRU, 'Vivado', 'io_stream', 'latency'),
        (GRU, 'Vitis', 'io_stream', 'resource'),
        (GRU, 'Vitis', 'io_stream', 'latency'),
        (GRU, 'Quartus', 'io_stream', 'resource'),
        (GRU, 'Altera', 'io_stream', 'resource'),
        (Bidirectional, 'Vivado', 'io_parallel', 'resource'),
        (Bidirectional, 'Vivado', 'io_parallel', 'latency'),
        (Bidirectional, 'Vitis', 'io_parallel', 'resource'),
        (Bidirectional, 'Vitis', 'io_parallel', 'latency'),
    ],
    ids=[
        'SimpleRNN-Quartus-io_parallel-resource',
        'SimpleRNN-Altera-io_parallel-resource',
        'SimpleRNN-Vivado-io_parallel-resource',
        'SimpleRNN-Vivado-io_parallel-latency',
        'SimpleRNN-Vitis-io_parallel-resource',
        'SimpleRNN-Vitis-io_parallel-latency',
        'SimpleRNN-Vivado-io_stream-resource',
        'SimpleRNN-Vivado-io_stream-latency',
        'SimpleRNN-Vitis-io_stream-resource',
        'SimpleRNN-Vitis-io_stream-latency',
        'LSTM-Vivado-io_parallel-resource',
        'LSTM-Vivado-io_parallel-latency',
        'LSTM-Vitis-io_parallel-resource',
        'LSTM-Vitis-io_parallel-latency',
        'LSTM-Quartus-io_parallel-resource',
        'LSTM-Altera-io_parallel-resource',
        'LSTM-Vivado-io_stream-resource',
        'LSTM-Vivado-io_stream-latency',
        'LSTM-Vitis-io_stream-resource',
        'LSTM-Vitis-io_stream-latency',
        'GRU-Vivado-io_parallel-resource',
        'GRU-Vivado-io_parallel-latency',
        'GRU-Vitis-io_parallel-resource',
        'GRU-Vitis-io_parallel-latency',
        'GRU-Quartus-io_parallel-resource',
        'GRU-Altera-io_parallel-resource',
        'GRU-Vivado-io_stream-resource',
        'GRU-Vivado-io_stream-latency',
        'GRU-Vitis-io_stream-resource',
        'GRU-Vitis-io_stream-latency',
        'GRU-Quartus-io_stream-resource',
        'GRU-Altera-io_stream-resource',
        'Bidirectional-Vivado-io_parallel-resource',
        'Bidirectional-Vivado-io_parallel-latency',
        'Bidirectional-Vitis-io_parallel-resource',
        'Bidirectional-Vitis-io_parallel-latency',
    ],
)
@pytest.mark.parametrize('return_sequences', [True, False])
@pytest.mark.parametrize('static', [True, False])
def test_rnn_accuracy(test_case_id, rnn_layer, return_sequences, backend, io_type, strategy, static):
    layer_name = rnn_layer.__name__

    model, X = create_model_accuracy(rnn_layer, return_sequences)

    default_precision = 'ap_fixed<32, 16>' if backend in ['Vivado', 'Vitis'] else 'ac_fixed<32, 16, true>'
    hls_config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', default_precision=default_precision, backend=backend
    )
    hls_config['LayerName'][layer_name]['static'] = static
    hls_config['LayerName'][layer_name]['Strategy'] = strategy
    if rnn_layer is SimpleRNN and backend in ['Vivado', 'Vitis']:
        # The SimpleRNN state is the output of the tanh lookup table, so the table's error adds up over the time steps
        hls_config['LayerName'][layer_name]['TableSize'] = 4096
    output_dir = str(test_root_path / test_case_id)

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=hls_config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    hls_model.compile()

    keras_prediction = model.predict(X)
    hls_prediction = hls_model.predict(X)
    np.testing.assert_allclose(hls_prediction.flatten(), keras_prediction.flatten(), rtol=0.0, atol=5e-2)


@pytest.mark.parametrize('rnn_layer', [SimpleRNN, LSTM, GRU], ids=['SimpleRNN', 'LSTM', 'GRU'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('static', [True, False])
def test_rnn_mixed_precision(test_case_id, rnn_layer, backend, io_type, static):
    """A layer whose accumulator and result types differ from its input type builds and matches Keras."""

    layer_name = rnn_layer.__name__
    model, X = create_model_accuracy(rnn_layer, return_sequences=False)

    hls_config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', default_precision='ap_fixed<16, 6>', backend=backend
    )
    hls_config['LayerName'][layer_name]['Precision']['accum'] = 'ap_fixed<24, 10>'
    hls_config['LayerName'][layer_name]['Precision']['result'] = 'ap_fixed<18, 8>'
    hls_config['LayerName'][layer_name]['static'] = static
    if rnn_layer is SimpleRNN:
        # The SimpleRNN state is the output of the tanh lookup table, so the table's error adds up over the time steps
        hls_config['LayerName'][layer_name]['TableSize'] = 4096
    output_dir = str(test_root_path / test_case_id)

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=hls_config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    hls_model.compile()

    keras_prediction = model.predict(X)
    hls_prediction = hls_model.predict(X)
    np.testing.assert_allclose(hls_prediction.flatten(), keras_prediction.flatten(), rtol=0.0, atol=5e-2)


def create_model_reuse(rnn_layer):
    """A small model of 16 units, or 8 in each direction of the Bidirectional layer, on 6 features."""

    model = Sequential()
    model.add(Input(shape=(8, 6)))
    if rnn_layer.__name__ != 'Bidirectional':
        model.add(rnn_layer(16, name='rnn'))
    else:
        model.add(Bidirectional(LSTM(8), backward_layer=GRU(8, go_backwards=True), name='rnn'))
    return model


@pytest.mark.parametrize(
    'rnn_layer, reuse_factor',
    [(SimpleRNN, 20), (LSTM, 20), (GRU, 20), (Bidirectional, 4), (Bidirectional, 12)],
    ids=['SimpleRNN-20', 'LSTM-20', 'GRU-20', 'Bidirectional-4', 'Bidirectional-12'],
)
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_rnn_resource_reuse_factor(test_case_id, rnn_layer, reuse_factor, backend):
    """The resource strategy gives the same results as the latency strategy when the reuse factor is neither a divisor
    nor a multiple of the number of units, so the units are computed in groups over several cycles: four units for five
    cycles at 20, and two units for three cycles at 12 with 8 units in each direction.
    """

    model = create_model_reuse(rnn_layer)
    X = np.random.default_rng(0).uniform(-0.5, 0.5, (20, 8, 6))
    precision, table_size = ('ap_fixed<24, 10>', 4096) if rnn_layer is SimpleRNN else ('ap_fixed<16, 6>', 1024)

    predictions = []
    for strategy in ('Latency', 'Resource'):
        hls_config = hls4ml.utils.config_from_keras_model(
            model, granularity='name', default_precision=precision, backend=backend
        )
        hls_config['LayerName']['rnn']['TableSize'] = table_size
        hls_config['LayerName']['rnn']['Strategy'] = strategy
        hls_config['LayerName']['rnn']['ReuseFactor'] = reuse_factor
        output_dir = str(test_root_path / f'{test_case_id}_{strategy.lower()}')
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=hls_config, output_dir=output_dir, backend=backend
        )
        hls_model.compile()
        predictions.append(hls_model.predict(X))

    np.testing.assert_array_equal(predictions[1], predictions[0])
    np.testing.assert_allclose(predictions[1].flatten(), model.predict(X).flatten(), rtol=0.0, atol=5e-2)


@pytest.mark.parametrize(
    'rnn_layer, reuse_factor, expected',
    [
        (LSTM, 1, (1, 16, 1, 6, 16)),
        (LSTM, 3, (3, 16, 3, 2, 6)),
        (LSTM, 8, (8, 2, 1, 6, 16)),
        (LSTM, 20, (20, 4, 5, 2, 4)),
        (LSTM, 64, (64, 1, 4, 2, 4)),
        (LSTM, 1000, (256, 1, 16, 1, 1)),
        (GRU, 18, (18, 8, 9, 1, 2)),
    ],
)
def test_rnn_resource_schedule(test_case_id, rnn_layer, reuse_factor, expected):
    """The reuse factor sets how many units the resource strategy computes at a time and for how many cycles: a step
    takes 16 units / units at a time * cycles. Of the choices giving the reuse factor, the one with the fewest units
    at a time is used. A reuse factor no choice gives, such as one above 16 units times 16 cycles, is replaced by the
    closest one that does.
    """

    model = create_model_reuse(rnn_layer)
    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Vivado')
    hls_config['LayerName']['rnn']['Strategy'] = 'Resource'
    hls_config['LayerName']['rnn']['ReuseFactor'] = reuse_factor
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=hls_config, output_dir=output_dir)

    layer = hls_model.graph['rnn']
    attributes = ('reuse_factor', 'n_unit_par', 'n_cycles', 'n_lanes_in', 'n_lanes_state')
    assert tuple(layer.get_attr(a) for a in attributes) == expected


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_bidirectional_resource_schedule(test_case_id, backend):
    """Each direction of a Bidirectional layer gets the schedule for its own number of units."""

    model = Sequential()
    model.add(Input(shape=(8, 6)))
    model.add(Bidirectional(LSTM(8), backward_layer=GRU(4, go_backwards=True), name='rnn'))
    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    hls_config['LayerName']['rnn']['Strategy'] = 'Resource'
    hls_config['LayerName']['rnn']['ReuseFactor'] = 8
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=hls_config, output_dir=output_dir, backend=backend
    )

    layer = hls_model.graph['rnn']
    attributes = ('reuse_factor', 'n_unit_par', 'n_cycles')
    assert tuple(layer.get_attr(f'forward_{a}') for a in attributes) == (8, 1, 1)
    assert tuple(layer.get_attr(f'backward_{a}') for a in attributes) == (8, 1, 2)


def create_model_resource_cell(kind, return_sequences):
    """A model with one recurrent layer of 16 units, or 8 in each direction, on 6 features."""

    layers = {
        'SimpleRNN': lambda: SimpleRNN(16, return_sequences=return_sequences, name='rnn'),
        'LSTM': lambda: LSTM(16, return_sequences=return_sequences, name='rnn'),
        'GRU': lambda: GRU(16, return_sequences=return_sequences, name='rnn'),
        'GRU_reset_before': lambda: GRU(16, return_sequences=return_sequences, reset_after=False, name='rnn'),
        'Bidirectional': lambda: Bidirectional(
            LSTM(8, return_sequences=return_sequences),
            backward_layer=GRU(8, return_sequences=return_sequences, go_backwards=True),
            name='rnn',
        ),
    }
    model = Sequential()
    model.add(Input(shape=(8, 6)))
    model.add(layers[kind]())
    return model


def compare_resource_with_latency(test_case_id, model, reuse_factor, io_type='io_parallel', precision=None):
    """Convert the model with the latency and with the resource strategy, and check they give the same results."""

    X = np.random.default_rng(0).uniform(-1, 1, (20, 8, 6))
    predictions = []
    for strategy in ('Latency', 'Resource'):
        hls_config = hls4ml.utils.config_from_keras_model(
            model, granularity='name', default_precision='ap_fixed<16, 6>', backend='Vivado'
        )
        hls_config['LayerName']['rnn']['Strategy'] = strategy
        hls_config['LayerName']['rnn']['ReuseFactor'] = reuse_factor
        if precision is not None:
            hls_config['LayerName']['rnn']['Precision'].update(precision)
        output_dir = str(test_root_path / f'{test_case_id}_{strategy.lower()}')
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=hls_config, output_dir=output_dir, io_type=io_type
        )
        hls_model.compile()
        predictions.append(hls_model.predict(X))

    np.testing.assert_array_equal(predictions[1], predictions[0])


@pytest.mark.parametrize('kind', ['SimpleRNN', 'LSTM', 'GRU', 'GRU_reset_before', 'Bidirectional'])
@pytest.mark.parametrize('reuse_factor', [1, 3, 24, 64])
def test_rnn_resource_cell(test_case_id, kind, reuse_factor):
    """The cell of the resource strategy gives the same results as the latency strategy: all units at once (1), all
    units over three cycles (3), two units at a time over three cycles (24), and one unit at a time over four cycles
    (64), or over eight for the 8 units in each direction of the Bidirectional layer."""

    compare_resource_with_latency(test_case_id, create_model_resource_cell(kind, False), reuse_factor)


@pytest.mark.parametrize('kind', ['SimpleRNN', 'LSTM', 'GRU', 'GRU_reset_before'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('return_sequences', [True, False])
def test_rnn_resource_cell_io(test_case_id, kind, io_type, return_sequences):
    """The cell of the resource strategy gives the same results as the latency strategy with both io types, and when
    every time step is returned."""

    if io_type == 'io_parallel' and not return_sequences:
        pytest.skip('Covered by test_rnn_resource_cell')
    compare_resource_with_latency(test_case_id, create_model_resource_cell(kind, return_sequences), 24, io_type)


@pytest.mark.parametrize('kind', ['SimpleRNN', 'LSTM', 'GRU', 'Bidirectional'])
def test_rnn_resource_cell_mixed_precision(test_case_id, kind):
    """The cell of the resource strategy gives the same results as the latency strategy when the input, the state,
    the accumulator and the two weights have different types."""

    precision = {
        'accum': 'ap_fixed<24, 10>',
        'result': 'ap_fixed<18, 8>',
    }
    if kind == 'Bidirectional':
        precision.update(forward_recurrent_weight='ap_fixed<12, 4>', backward_recurrent_weight='ap_fixed<12, 4>')
    else:
        precision.update(recurrent_weight='ap_fixed<12, 4>')
    compare_resource_with_latency(test_case_id, create_model_resource_cell(kind, False), 24, precision=precision)


@pytest.mark.parametrize(
    'make_layer',
    [
        lambda: LSTM(8, activation='relu', recurrent_activation='linear', name='rnn'),
        lambda: GRU(8, activation='linear', recurrent_activation='relu', name='rnn'),
        lambda: SimpleRNN(8, activation='relu', name='rnn'),
        lambda: LSTM(8, use_bias=False, name='rnn'),
        lambda: GRU(8, use_bias=False, reset_after=False, name='rnn'),
        lambda: LSTM(7, name='rnn'),
        lambda: Bidirectional(GRU(8, reset_after=False), name='rnn'),
        lambda: Bidirectional(
            LSTM(8, return_sequences=True), backward_layer=GRU(4, return_sequences=True, go_backwards=True), name='rnn'
        ),
    ],
    ids=[
        'LSTM-activations',
        'GRU-activations',
        'SimpleRNN-relu',
        'LSTM-no_bias',
        'GRU_reset_before-no_bias',
        'LSTM-7_units',
        'Bidirectional-GRU_reset_before',
        'Bidirectional-8_4_units-sequences',
    ],
)
def test_rnn_resource_cell_options(test_case_id, make_layer):
    """The cell of the resource strategy gives the same results as the latency strategy for other activations, without
    biases, for a number of units that cannot be split into groups, and for Bidirectional layers of GRU layers that apply
    the reset gate before the recurrent multiplication or of two layers of different sizes."""

    model = Sequential()
    model.add(Input(shape=(8, 6)))
    model.add(make_layer())
    compare_resource_with_latency(test_case_id, model, 24)


def test_rnn_initial_state_rejected(test_case_id):
    """Initial states given to a Keras recurrent layer are not supported and stop the conversion."""

    x, h, c = Input((8, 6)), Input((8,)), Input((8,))
    model = Model([x, h, c], LSTM(8, name='rnn')(x, initial_state=[h, c]))
    output_dir = str(test_root_path / test_case_id)
    with pytest.raises(NotImplementedError, match='initial_state'):
        hls4ml.converters.convert_from_keras_model(model, output_dir=output_dir, backend='Vivado')


def test_rnn_return_state_rejected(test_case_id):
    """return_state=True is not supported and stops the conversion."""

    x = Input((8, 6))
    model = Model(x, LSTM(8, return_state=True, name='rnn')(x)[0])
    output_dir = str(test_root_path / test_case_id)
    with pytest.raises(Exception, match='return_state'):
        hls4ml.converters.convert_from_keras_model(model, output_dir=output_dir, backend='Vivado')


def test_bidirectional_backward_forward_layer_rejected(test_case_id):
    """A Bidirectional layer whose forward layer reads backwards puts that layer's outputs first, which is not
    supported and stops the conversion."""

    model = Sequential()
    model.add(Input(shape=(8, 6)))
    model.add(Bidirectional(GRU(8, go_backwards=True), backward_layer=LSTM(8), name='rnn'))
    output_dir = str(test_root_path / test_case_id)
    with pytest.raises(NotImplementedError, match='go_backwards'):
        hls4ml.converters.convert_from_keras_model(model, output_dir=output_dir, backend='Vivado')


@pytest.mark.parametrize('rnn_layer', [SimpleRNN, LSTM, GRU], ids=['SimpleRNN', 'LSTM', 'GRU'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('static', [True, False])
def test_rnn_static_io_stream(test_case_id, rnn_layer, backend, static):
    """With io_stream one cell is always shared across time steps, so only Static set to False gives a warning."""

    model = Sequential()
    model.add(Input(shape=(8, 6)))
    model.add(rnn_layer(16, name='rnn'))

    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    hls_config['LayerName']['rnn']['Static'] = static
    output_dir = str(test_root_path / test_case_id)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=hls_config, output_dir=output_dir, backend=backend, io_type='io_stream'
        )

    warned = any('Static set to False' in str(w.message) for w in caught)
    assert warned != static
    assert hls_model.graph['rnn'].get_attr('static')


@pytest.mark.parametrize(
    'rnn_layer', [SimpleRNN, LSTM, GRU, Bidirectional], ids=['SimpleRNN', 'LSTM', 'GRU', 'Bidirectional']
)
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('static', [True, False])
def test_rnn_static_resource(test_case_id, rnn_layer, backend, static):
    """The cell of the resource strategy is always shared across time steps, so only Static set to False gives a
    warning."""

    model = create_model_reuse(rnn_layer)
    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    hls_config['LayerName']['rnn']['Strategy'] = 'Resource'
    hls_config['LayerName']['rnn']['Static'] = static
    output_dir = str(test_root_path / test_case_id)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=hls_config, output_dir=output_dir, backend=backend
        )

    warned = any('Static set to False' in str(w.message) for w in caught)
    assert warned != static
    assert hls_model.graph['rnn'].get_attr('static')


@pytest.mark.parametrize('rnn_layer', [SimpleRNN, LSTM, GRU], ids=['SimpleRNN', 'LSTM', 'GRU'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_rnn_linear_activation(test_case_id, rnn_layer, backend):
    """A recurrent layer with the linear activation builds and matches Keras."""

    model = Sequential()
    model.add(Input(shape=(8, 6)))
    model.add(rnn_layer(16, activation='linear', name='rnn'))
    X = np.random.default_rng(0).uniform(-0.5, 0.5, (20, 8, 6))

    hls_config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', default_precision='ap_fixed<24, 10>', backend=backend
    )
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=hls_config, output_dir=output_dir, backend=backend
    )
    hls_model.compile()

    np.testing.assert_allclose(hls_model.predict(X).flatten(), model.predict(X).flatten(), rtol=0.0, atol=5e-2)


@pytest.mark.parametrize(
    'make_layer, attribute, activation',
    [
        (lambda: SimpleRNN(4, activation='elu', name='rnn'), 'Activation', 'elu'),
        (lambda: LSTM(4, recurrent_activation='hard_sigmoid', name='rnn'), 'Recurrent activation', 'hard_sigmoid'),
        (lambda: GRU(4, activation='softsign', name='rnn'), 'Activation', 'softsign'),
        (lambda: Bidirectional(LSTM(4, activation='elu'), name='rnn'), 'Forward activation', 'elu'),
    ],
    ids=['SimpleRNN', 'LSTM', 'GRU', 'Bidirectional'],
)
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_rnn_unsupported_activation(test_case_id, make_layer, attribute, activation, backend):
    """Recurrent layers stop the conversion for an activation they do not implement."""

    model = Sequential()
    model.add(Input(shape=(5, 8)))
    model.add(make_layer())

    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    output_dir = str(test_root_path / test_case_id)
    with pytest.raises(NotImplementedError, match=f'{attribute} "{activation}"'):
        hls4ml.converters.convert_from_keras_model(model, hls_config=hls_config, output_dir=output_dir, backend=backend)


@pytest.mark.parametrize('rnn_layer', [SimpleRNN, LSTM, GRU], ids=['SimpleRNN', 'LSTM', 'GRU'])
def test_rnn_no_bias(test_case_id, rnn_layer):
    """use_bias=False previously crashed the parser instead of substituting zero biases."""
    model = Sequential()
    model.add(Input(shape=(5, 8)))
    model.add(rnn_layer(4, use_bias=False))

    hls_config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', default_precision='ap_fixed<32, 16>', backend='Vivado'
    )
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=hls_config, output_dir=output_dir, backend='Vivado'
    )
    hls_model.compile()

    X = np.random.rand(20, 5, 8).astype('float32')
    keras_prediction = model.predict(X)
    hls_prediction = hls_model.predict(X)
    np.testing.assert_allclose(hls_prediction.flatten(), keras_prediction.flatten(), rtol=0.0, atol=5e-2)


def test_bidirectional_gru_accuracy(test_case_id):
    """Bidirectional GRU checked numerically against Keras (covers the GRU bias handling)."""
    model = Sequential()
    model.add(Input(shape=(5, 8)))
    model.add(Bidirectional(GRU(4)))

    hls_config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', default_precision='ap_fixed<32, 16>', backend='Vivado'
    )
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=hls_config, output_dir=output_dir, backend='Vivado'
    )
    hls_model.compile()

    X = np.random.rand(20, 5, 8).astype('float32')
    keras_prediction = model.predict(X)
    hls_prediction = hls_model.predict(X)
    np.testing.assert_allclose(hls_prediction.flatten(), keras_prediction.flatten(), rtol=0.0, atol=5e-2)


@pytest.mark.parametrize('cell_type', ['gru', 'lstm'])
def test_bidirectional_no_bias(test_case_id, cell_type):
    """Bidirectional cells with use_bias=False parse (zeros are substituted) and match Keras."""
    model = Sequential()
    model.add(Input(shape=(5, 8)))
    if cell_type == 'gru':
        model.add(Bidirectional(GRU(4, use_bias=False)))
    else:
        model.add(Bidirectional(LSTM(4, use_bias=False)))

    hls_config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', default_precision='ap_fixed<32, 16>', backend='Vivado'
    )
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=hls_config, output_dir=output_dir, backend='Vivado'
    )
    hls_model.compile()

    X = np.random.rand(20, 5, 8).astype('float32')
    keras_prediction = model.predict(X)
    hls_prediction = hls_model.predict(X)
    np.testing.assert_allclose(hls_prediction.flatten(), keras_prediction.flatten(), rtol=0.0, atol=5e-2)


@pytest.mark.parametrize(
    'io_type, merge_mode',
    [('io_stream', 'concat'), ('io_parallel', 'sum'), ('io_parallel', 'mul'), ('io_parallel', 'ave')],
    ids=['io_stream', 'sum', 'mul', 'ave'],
)
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_bidirectional_unsupported(test_case_id, io_type, merge_mode, backend):
    """Bidirectional layers stop the conversion with io_stream and with merge modes other than concat."""

    model = Sequential()
    model.add(Input(shape=(5, 8)))
    model.add(Bidirectional(LSTM(4), merge_mode=merge_mode, name='rnn'))

    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    output_dir = str(test_root_path / test_case_id)
    with pytest.raises(NotImplementedError, match=io_type if io_type == 'io_stream' else merge_mode):
        hls4ml.converters.convert_from_keras_model(
            model, hls_config=hls_config, output_dir=output_dir, backend=backend, io_type=io_type
        )


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_bidirectional_resource_unrolled(test_case_id, backend):
    """Bidirectional layers have no unrolled implementation, so they use the resource strategy and give a warning."""

    model = Sequential()
    model.add(Input(shape=(5, 8)))
    model.add(Bidirectional(LSTM(4), name='rnn'))

    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    hls_config['LayerName']['rnn']['Strategy'] = 'ResourceUnrolled'
    hls_config['LayerName']['rnn']['ReuseFactor'] = 4
    output_dir = str(test_root_path / test_case_id)
    with pytest.warns(UserWarning, match='Unrolled resource strategy is not supported'):
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=hls_config, output_dir=output_dir, backend=backend
        )

    assert hls_model.graph['rnn'].get_attr('strategy') == 'resource'


def create_gru_reset_model(reset_after, bidirectional=False):
    """Small GRU with fixed non-zero weights and biases."""
    n_steps, n_features, n_units = 5, 3, 4
    gru = GRU(n_units, reset_after=reset_after, name='gru')
    model = Sequential()
    model.add(Input(shape=(n_steps, n_features)))
    model.add(Bidirectional(gru, name='gru_bidir') if bidirectional else gru)

    rng = np.random.default_rng(0)
    weights = [rng.uniform(-1, 1, w.shape) for w in model.layers[0].get_weights()]
    model.layers[0].set_weights([(np.round(w * 2**16) * 2**-16).astype('float32') for w in weights])
    X = rng.uniform(-1, 1, (50, n_steps, n_features))
    X = (np.round(X * 2**16) * 2**-16).astype('float32')
    return model, X


def gru_reference(X, kernel, recurrent_kernel, bias, reset_before):
    """NumPy GRU, Keras gate order (z, r, h), last state only.

    reset_before=True:  h~ = tanh(W_h x + b_h + U_h (r * h))
    reset_before=False: h~ = tanh(W_h x + b_h + r * (U_h h + b_rh))
    """
    n = recurrent_kernel.shape[0]
    bias_in, bias_rec = (bias[0], bias[1]) if bias.ndim == 2 else (bias, np.zeros_like(bias))

    def sigmoid(v):
        return 1 / (1 + np.exp(-v))

    h = np.zeros((X.shape[0], n))
    for t in range(X.shape[1]):
        x_part = X[:, t] @ kernel + bias_in
        h_part = h @ recurrent_kernel + bias_rec
        z = sigmoid(x_part[:, :n] + h_part[:, :n])
        r = sigmoid(x_part[:, n : 2 * n] + h_part[:, n : 2 * n])
        if reset_before:
            h_cand = np.tanh(x_part[:, 2 * n :] + (r * h) @ recurrent_kernel[:, 2 * n :])
        else:
            h_cand = np.tanh(x_part[:, 2 * n :] + r * h_part[:, 2 * n :])
        h = z * h + (1 - z) * h_cand
    return h


@pytest.mark.parametrize('reset_after', [True, False])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize(
    'io_type, strategy, static, reuse_factor',
    [
        ('io_parallel', 'latency', True, 1),
        ('io_parallel', 'latency', False, 1),
        ('io_parallel', 'resource', True, 4),
        ('io_parallel', 'resource_unrolled', True, 4),
        ('io_stream', 'latency', True, 1),
        ('io_stream', 'resource', False, 4),
    ],
)
def test_gru_reset_after_accuracy(test_case_id, reset_after, backend, io_type, strategy, static, reuse_factor):
    """GRU matches Keras for both reset_after settings."""
    model, X = create_gru_reset_model(reset_after)

    hls_config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', default_precision='ap_fixed<32, 16>', backend=backend
    )
    hls_config['LayerName']['gru']['static'] = static
    hls_config['LayerName']['gru']['Strategy'] = strategy
    hls_config['LayerName']['gru']['ReuseFactor'] = reuse_factor
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=hls_config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    gru_layer = [layer for layer in hls_model.get_layers() if layer.class_name == 'GRU'][0]
    assert gru_layer.get_attr('apply_reset_gate') == ('after' if reset_after else 'before')
    assert gru_layer.get_attr('recurrent_reuse_factor') == reuse_factor
    hls_model.compile()

    keras_prediction = model.predict(X)
    hls_prediction = hls_model.predict(X).reshape(keras_prediction.shape)
    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0.0, atol=5e-2)


@pytest.mark.parametrize('reset_after', [True, False])
def test_gru_reset_gate_placement(test_case_id, reset_after):
    """HLS GRU follows the candidate equation selected by reset_after, not the other one."""
    model, X = create_gru_reset_model(reset_after)
    kernel, recurrent_kernel, bias = model.layers[0].get_weights()
    expected = gru_reference(X, kernel, recurrent_kernel, bias, reset_before=not reset_after)
    other = gru_reference(X, kernel, recurrent_kernel, bias, reset_before=reset_after)

    keras_prediction = model.predict(X)
    np.testing.assert_allclose(expected, keras_prediction, rtol=0.0, atol=1e-5)
    assert np.abs(expected - other).max() > 0.2  # The two equations give different results

    hls_config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', default_precision='ap_fixed<32, 16>', backend='Vivado'
    )
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=hls_config, output_dir=output_dir, backend='Vivado'
    )
    hls_model.compile()
    hls_prediction = hls_model.predict(X).reshape(expected.shape)

    np.testing.assert_allclose(hls_prediction, expected, rtol=0.0, atol=5e-2)
    assert np.abs(hls_prediction - other).max() > 0.15


@pytest.mark.parametrize('reset_after', [True, False])
@pytest.mark.parametrize('static', [True, False])
def test_bidirectional_gru_reset_after_accuracy(test_case_id, reset_after, static):
    """Bidirectional GRU matches Keras for both reset_after settings."""
    model, X = create_gru_reset_model(reset_after, bidirectional=True)

    hls_config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', default_precision='ap_fixed<32, 16>', backend='Vivado'
    )
    hls_config['LayerName']['gru_bidir']['static'] = static
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=hls_config, output_dir=output_dir, backend='Vivado'
    )
    hls_model.compile()

    keras_prediction = model.predict(X)
    hls_prediction = hls_model.predict(X).reshape(keras_prediction.shape)
    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0.0, atol=5e-2)


@pytest.mark.parametrize('rnn_layer', [SimpleRNN, LSTM, GRU], ids=['SimpleRNN', 'LSTM', 'GRU'])
def test_altera_recurrent_reuse_factor_removed(test_case_id, rnn_layer):
    """Altera ignores RecurrentReuseFactor and warns; the recurrent multiplication uses ReuseFactor."""
    model = Sequential()
    model.add(Input(shape=(8, 6)))
    model.add(rnn_layer(16, name='rnn'))

    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Altera')
    assert 'RecurrentReuseFactor' not in config['LayerName']['rnn']

    config['LayerName']['rnn']['ReuseFactor'] = 8
    config['LayerName']['rnn']['RecurrentReuseFactor'] = 4
    output_dir = str(test_root_path / test_case_id)
    with pytest.warns(UserWarning, match='RecurrentReuseFactor'):
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, output_dir=output_dir, backend='Altera'
        )

    assert hls_model.graph['rnn'].get_attr('recurrent_reuse_factor') == 8

    del config['LayerName']['rnn']['RecurrentReuseFactor']
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=config,
            output_dir=str(test_root_path / f'{test_case_id}_nowarn'),
            backend='Altera',
        )
    assert not any('RecurrentReuseFactor' in str(w.message) for w in caught)
