import numpy as np

from hls4ml.converters.pytorch_to_hls import pytorch_handler

rnn_layers = ['RNN', 'LSTM', 'GRU']


def _collect_output_index_paths(node, path=(), paths=None):
    """Collects the chains of getitem indices through which the RNN output tuple is consumed.

    Each entry in the returned list is a tuple of indices leading from the RNN output to an actual consumer,
    e.g. ``(0,)`` for the output sequence and ``(1, 0)`` for the hidden state of an LSTM. Getitem nodes with
    no consumers (from unpacking into unused variables) are ignored.
    """
    if paths is None:
        paths = []
    for user in node.users:
        if 'getitem' in user.name:
            if len(user.users) > 0:
                _collect_output_index_paths(user, path + (user.args[1],), paths)
        else:
            paths.append(path)
    return paths


@pytorch_handler(*rnn_layers)
def parse_rnn_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation in rnn_layers

    layer = {}

    layer['name'] = layer_name

    layer['inputs'] = input_names
    if 'IOType' in config.keys():
        if len(input_names) > 1 and config['IOType'] == 'io_stream':
            raise Exception('Passing initial values for the hidden state is not support for io_stream input type.')

    layer['class_name'] = operation
    if operation == 'RNN':
        layer['class_name'] = 'SimpleRNN'

    if getattr(class_object, 'proj_size', 0) > 0:
        raise NotImplementedError(f'Layer {layer_name}: LSTM with proj_size > 0 is not supported.')

    # PyTorch RNN layers return a tuple (output_sequence, final_state); which element the model consumes
    # decides whether the layer must produce the full sequence or only the final state
    index_paths = _collect_output_index_paths(node)
    for p in index_paths:
        if not all(isinstance(i, int) for i in p):
            raise NotImplementedError(
                f'Layer {layer_name}: slicing of the outputs of an RNN layer is not supported '
                '(only selecting the output sequence or the final hidden state).'
            )
    sequence_used = any(p[:1] == (0,) for p in index_paths)
    state_used = any(p[:1] == (1,) for p in index_paths) or any(p == () for p in index_paths)
    if operation == 'LSTM' and any(p[:2] == (1, 1) for p in index_paths):
        raise NotImplementedError(f'Layer {layer_name}: use of the cell state of an LSTM layer is not supported.')
    if sequence_used and state_used:
        raise NotImplementedError(
            f'Layer {layer_name}: use of both the output sequence and the final state of an RNN layer is not supported.'
        )

    layer['return_sequences'] = sequence_used
    layer['return_state'] = False  # parameter does not exist in pytorch

    if layer['class_name'] == 'SimpleRNN':
        layer['activation'] = class_object.nonlinearity  # Default is tanh, can also be ReLU in pytorch
    else:
        layer['activation'] = 'tanh'  # GRU and LSTM are hard-coded to use tanh in pytorch

    if layer['class_name'] == 'GRU' or layer['class_name'] == 'LSTM':
        layer['recurrent_activation'] = 'sigmoid'  # GRU and LSTM are hard-coded to use sigmoid in pytorch

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
        raise Exception('hls4ml does not support bidirectional RNNs')
    if class_object.dropout > 0:
        raise Exception('hls4ml does not support RNNs with dropout')

    # transpose weight and recurrent weight to match keras order used in the HLS code
    layer['weight_data'] = class_object.weight_ih_l0.data.numpy().transpose()
    layer['recurrent_weight_data'] = class_object.weight_hh_l0.data.numpy().transpose()

    if class_object.bias:
        layer['bias_data'] = class_object.bias_ih_l0.data.numpy()
        layer['recurrent_bias_data'] = class_object.bias_hh_l0.data.numpy()
    else:
        # the bias parameters do not exist in the module; substitute zeros of the gate width
        layer['bias_data'] = np.zeros(layer['weight_data'].shape[-1])
        layer['recurrent_bias_data'] = np.zeros(layer['recurrent_weight_data'].shape[-1])

    if layer['class_name'] == 'GRU':
        layer['apply_reset_gate'] = 'after'  # Might be true for pytorch? It's not a free parameter

    if layer['return_sequences']:
        output_shape = [input_shapes[0][0], layer['n_timesteps'], layer['n_out']]
    else:
        output_shape = [input_shapes[0][0], layer['n_out']]

    layer['pytorch'] = True  # need to switch some behaviors to match pytorch implementations
    if len(input_names) == 1:
        layer['pass_initial_states'] = False
    else:
        layer['pass_initial_states'] = True

    return layer, output_shape
