import numpy as np

from hls4ml.converters.keras_v2_to_hls import (
    KerasModelReader,
    KerasNestedFileReader,
    KerasWrappedLayerFileReader,
    KerasWrappedLayerReader,
    get_layer_handlers,
    get_weights_data,
    keras_handler,
    parse_default_keras_layer,
    parse_keras_model,
)

rnn_layers = ['SimpleRNN', 'LSTM', 'GRU']


@keras_handler(*rnn_layers)
def parse_rnn_layer(keras_layer, input_names, input_shapes, data_reader):
    assert keras_layer['class_name'] in rnn_layers or keras_layer['class_name'][1:] in rnn_layers

    layer = parse_default_keras_layer(keras_layer, input_names)
    layer['direction'] = 'forward'

    layer['return_sequences'] = keras_layer['config']['return_sequences']
    layer['return_state'] = keras_layer['config']['return_state']

    if 'SimpleRNN' not in layer['class_name']:
        layer['recurrent_activation'] = keras_layer['config']['recurrent_activation']

    layer['time_major'] = keras_layer['config']['time_major'] if 'time_major' in keras_layer['config'] else False

    # TODO Should we handle time_major?
    if layer['time_major']:
        raise Exception('Time-major format is not supported by hls4ml')

    layer['n_timesteps'] = input_shapes[0][1]
    layer['n_in'] = input_shapes[0][2]

    layer['n_out'] = keras_layer['config']['units']

    layer['weight_data'], layer['recurrent_weight_data'], layer['bias_data'] = get_weights_data(
        data_reader, layer['name'], ['kernel', 'recurrent_kernel', 'bias']
    )

    if layer['bias_data'] is None:
        d_out = layer['bias_data'].shape[-1]
        if 'GRU' in layer['class_name']:
            layer['bias_data'] = np.zeros((2, d_out), dtype=np.float32)
        else:
            layer['bias_data'] = np.zeros((d_out,), dtype=np.float32)

    if 'GRU' in layer['class_name']:
        layer['apply_reset_gate'] = 'after' if keras_layer['config']['reset_after'] else 'before'

        # biases array is actually a 2-dim array of arrays (bias + recurrent bias)
        # both arrays have shape: n_units * 3 (z, r, h_cand)
        biases = layer['bias_data']
        layer['bias_data'] = biases[0]
        layer['recurrent_bias_data'] = biases[1]

    if layer['return_sequences']:
        output_shape = [input_shapes[0][0], layer['n_timesteps'], layer['n_out']]
    else:
        output_shape = [input_shapes[0][0], layer['n_out']]

    if layer['return_state']:
        raise Exception('"return_state" of {} layer is not yet supported.')

    return layer, output_shape


@keras_handler('TimeDistributed')
def parse_time_distributed_layer(keras_layer, input_names, input_shapes, data_reader):
    assert keras_layer['class_name'] == 'TimeDistributed'

    layer = parse_default_keras_layer(keras_layer, input_names)

    wrapped_keras_layer = keras_layer['config']['layer']
    handler = get_layer_handlers()[wrapped_keras_layer['class_name']]
    if wrapped_keras_layer['class_name'] in ['Sequential', 'Model', 'Functional']:
        if isinstance(data_reader, KerasModelReader):
            nested_data_reader = KerasModelReader(data_reader.model.get_layer(layer['name']).layer)
        else:
            nested_data_reader = KerasNestedFileReader(data_reader, layer['name'])
        layer_list, input_layers, output_layers, output_shapes = parse_keras_model(wrapped_keras_layer, nested_data_reader)

        wrapped_layer = layer.copy()
        wrapped_layer['name'] = wrapped_keras_layer['config']['name']
        wrapped_layer['class_name'] = 'LayerGroup'

        if output_layers is None:
            last_layer = layer_list[-1]['name']
        else:
            last_layer = output_layers[0]
        layer_output_shape = output_shapes[last_layer]

        wrapped_layer['layer_list'] = layer_list
        wrapped_layer['input_layers'] = input_layers if input_layers is not None else []
        wrapped_layer['output_layers'] = output_layers if output_layers is not None else []
        wrapped_layer['data_reader'] = nested_data_reader
        wrapped_layer['output_shape'] = layer_output_shape

        layer['wrapped_layer'] = wrapped_layer
    else:
        if isinstance(data_reader, KerasModelReader):
            nested_data_reader = KerasWrappedLayerReader(data_reader.model.get_layer(layer['name']).layer)
        else:
            nested_data_reader = KerasWrappedLayerFileReader(data_reader, f"{layer['name']}/{layer['name']}")

        wrapped_layer, layer_output_shape = handler(wrapped_keras_layer, [layer['name']], input_shapes, nested_data_reader)
        wrapped_layer['output_shape'] = layer_output_shape
        layer['wrapped_layer'] = wrapped_layer

    if layer_output_shape[0] is None:
        layer_output_shape = layer_output_shape[1:]
    output_shape = input_shapes[0]
    output_shape[len(layer_output_shape) - 1 :] = layer_output_shape
    layer['output_shape'] = output_shape[1:]  # Remove the batch dimension
    layer['n_time_steps'] = output_shape[1]

    return layer, output_shape


@keras_handler('Bidirectional')
def parse_bidirectional_layer(keras_layer, input_names, input_shapes, data_reader):
    assert keras_layer['class_name'] == 'Bidirectional'

    rnn_forward_layer = keras_layer['config']['layer']
    swapped_order = False
    if keras_layer['config'].get('backward_layer'):
        rnn_backward_layer = keras_layer['config']['backward_layer']
        if rnn_forward_layer['config']['go_backwards']:
            temp_layer = rnn_forward_layer.copy()
            rnn_forward_layer = rnn_backward_layer.copy()
            rnn_backward_layer = temp_layer
            swapped_order = True
            print(
                f'WARNING: The selected order for forward and backward layers in \"{keras_layer["config"]["name"]}\" '
                f'({keras_layer["class_name"]}) is not supported. Switching to forward layer first, backward layer last.'
            )
    else:
        rnn_backward_layer = rnn_forward_layer

    assert (rnn_forward_layer['class_name'] in rnn_layers or rnn_forward_layer['class_name'][1:] in rnn_layers) and (
        rnn_backward_layer['class_name'] in rnn_layers or rnn_backward_layer['class_name'][1:] in rnn_layers
    )

    layer = parse_default_keras_layer(keras_layer, input_names)

    layer['direction'] = 'bidirectional'
    layer['return_sequences'] = rnn_forward_layer['config']['return_sequences']
    layer['return_state'] = rnn_forward_layer['config']['return_state']
    layer['time_major'] = rnn_forward_layer['config']['time_major'] if 'time_major' in rnn_forward_layer['config'] else False
    # TODO Should we handle time_major?
    if layer['time_major']:
        raise Exception('Time-major format is not supported by hls4ml')
    layer['n_timesteps'] = input_shapes[0][1]
    layer['n_in'] = input_shapes[0][2]
    layer['merge_mode'] = keras_layer['config']['merge_mode']

    for direction, rnn_layer in [('forward', rnn_forward_layer), ('backward', rnn_backward_layer)]:

        layer[f'{direction}_name'] = rnn_layer['config']['name']
        layer[f'{direction}_class_name'] = rnn_layer['class_name']
        if 'activation' in rnn_layer['config']:
            layer[f'{direction}_activation'] = rnn_layer['config']['activation']
        if 'SimpleRNN' not in rnn_layer['class_name']:
            layer[f'{direction}_recurrent_activation'] = rnn_layer['config']['recurrent_activation']

        layer[f'{direction}_data_format'] = rnn_layer['config'].get('data_format', 'channels_last')
        if 'epsilon' in rnn_layer['config']:
            layer[f'{direction}_epsilon'] = rnn_layer['config']['epsilon']
        if 'use_bias' in rnn_layer['config']:
            layer[f'{direction}_use_bias'] = rnn_layer['config']['use_bias']

        rnn_layer_name = rnn_layer['config']['name']
        if 'SimpleRNN' in layer['class_name']:
            cell_name = 'simple_rnn'
        else:
            cell_name = rnn_layer['class_name'].lower()
        temp_dir = direction
        if swapped_order:
            temp_dir = 'backward' if direction == 'forward' else 'forward'
        layer[f'{direction}_weight_data'], layer[f'{direction}_recurrent_weight_data'], layer[f'{direction}_bias_data'] = (
            get_weights_data(
                data_reader,
                layer['name'],
                [
                    f'{temp_dir}_{rnn_layer_name}/{cell_name}_cell/kernel',
                    f'{temp_dir}_{rnn_layer_name}/{cell_name}_cell/recurrent_kernel',
                    f'{temp_dir}_{rnn_layer_name}/{cell_name}_cell/bias',
                ],
            )
        )

        if 'GRU' in rnn_layer['class_name']:
            layer[f'{direction}_apply_reset_gate'] = 'after' if rnn_layer['config']['reset_after'] else 'before'

            # biases array is actually a 2-dim array of arrays (bias + recurrent bias)
            # both arrays have shape: n_units * 3 (z, r, h_cand)
            biases = layer[f'{direction}_bias_data']
            layer[f'{direction}_bias_data'] = biases[0]
            layer[f'{direction}_recurrent_bias_data'] = biases[1]

        layer[f'{direction}_n_states'] = rnn_layer['config']['units']

    if layer['merge_mode'] == 'concat':
        layer['n_out'] = layer['forward_n_states'] + layer['backward_n_states']
    else:
        layer['n_out'] = layer['forward_n_states']

    if layer['return_sequences']:
        output_shape = [input_shapes[0][0], layer['n_timesteps'], layer['n_out']]
    else:
        output_shape = [input_shapes[0][0], layer['n_out']]

    if layer['return_state']:
        raise Exception('"return_state" of {} layer is not yet supported.')

    return layer, output_shape
