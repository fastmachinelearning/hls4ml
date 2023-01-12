from hls4ml.converters.keras_to_hls import keras_handler, parse_default_keras_layer

rnn_layers = ['SimpleRNN', 'LSTM', 'GRU']


@keras_handler(*rnn_layers)
def parse_rnn_layer(keras_layer, input_names, input_shapes, data_reader):
    assert keras_layer['class_name'] in rnn_layers

    layer = parse_default_keras_layer(keras_layer, input_names)

    layer['return_sequences'] = keras_layer['config']['return_sequences']
    layer['return_state'] = keras_layer['config']['return_state']

    if layer['class_name'] != 'SimpleRNN':
        layer['recurrent_activation'] = keras_layer['config']['recurrent_activation']

    layer['time_major'] = keras_layer['config']['time_major'] if 'time_major' in keras_layer['config'] else False

    # TODO Should we handle time_major?
    if layer['time_major']:
        raise Exception('Time-major format is not supported by hls4ml')

    layer['n_timesteps'] = input_shapes[0][1]
    layer['n_in'] = input_shapes[0][2]

    layer['n_out'] = keras_layer['config']['units']

    if layer['class_name'] == 'GRU':
        layer['apply_reset_gate'] = 'after' if keras_layer['config']['reset_after'] else 'before'

    if layer['return_sequences']:
        output_shape = [input_shapes[0][0], layer['n_timesteps'], layer['n_out']]
    else:
        output_shape = [input_shapes[0][0], layer['n_out']]

    if layer['return_state']:
        raise Exception('"return_state" of {} layer is not yet supported.')

    return layer, output_shape
