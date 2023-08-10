import warnings

from hls4ml.converters.pytorch_to_hls import get_weights_data, pytorch_handler

rnn_layers = ['SimpleRNN', 'LSTM', 'GRU']


@pytorch_handler(*rnn_layers)
def parse_rnn_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation in rnn_layers or operation == "RNN"

    layer = {}

    layer["name"] = layer_name

    layer['inputs'] = [input_names[0]]
    if len(input_names) > 1:
        warnings.warn(
            'hls4ml disregards the initial value of the hidden state passed to the model, assuming that it is all zeros',
            stacklevel=2,
        )
    layer['class_name'] = operation
    if operation == "RNN":
        layer['class_name'] = 'SimpleRNN'

    layer['return_sequences'] = False  # parameter does not exist in pytorch
    layer['return_state'] = False  # parameter does not exist in pytorch

    if layer['class_name'] == 'SimpleRNN':
        layer['activation'] = class_object.nonlinearity  # GRU and LSTM are hard-coded to use tanh in pytorch
    else:
        layer['activation'] = "tanh"  # GRU and LSTM are hard-coded to use tanh in pytorch

    layer['recurrent_activation'] = layer['activation']  # pytorch does not seem to differentiate between the two
    if layer['class_name'] == 'GRU':
        layer['recurrent_activation'] = 'sigmoid'  # seems to be hard-coded in pytorch?

    layer['time_major'] = not class_object.batch_first
    # TODO Should we handle time_major?
    if layer['time_major']:
        raise Exception('hls4ml only supports "batch-first == True"')

    layer['n_timesteps'] = input_shapes[0][1]
    layer['n_in'] = input_shapes[0][2]

    layer['n_out'] = class_object.hidden_size

    if class_object.num_layers > 1:
        raise Exception('hls4ml does not support num_layers > 1')

    if class_object.bidirectional:
        raise Exception('hls4ml does not support birectional RNNs')

    if class_object.dropout > 0:
        raise Exception('hls4ml does not support RNNs with dropout')

    (
        layer['weight_data'],
        layer['recurrent_weight_data'],
        layer['bias_data'],
        layer['recurrent_bias_data'],
    ) = get_weights_data(data_reader, layer['name'], ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0'])

    if layer['class_name'] == 'GRU':
        layer['apply_reset_gate'] = 'after'  # Might be true for pytorch? It's not a free parameter

    output_shape = [[input_shapes[0][0], layer['n_timesteps'], layer['n_out']], [1, layer['n_out']]]

    return layer, output_shape
