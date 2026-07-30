import numpy as np

from hls4ml.converters.pytorch_to_hls import addQuantizationParameters, convert_uaq_to_apfixed, pytorch_handler
from hls4ml.model.quantizers import BrevitasQuantizer
from hls4ml.model.types import FixedPrecisionType

rnn_layers = ['RNN', 'LSTM', 'GRU']


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

    layer['return_sequences'] = False  # parameter does not exist in pytorch
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
        raise Exception('hls4ml does not support birectional RNNs')
    if class_object.dropout > 0:
        raise Exception('hls4ml does not support RNNs with dropout')

    # transpose weight and recurrent weight to match keras order used in the HLS code
    layer['weight_data'] = class_object.weight_ih_l0.data.numpy().transpose()
    layer['recurrent_weight_data'] = class_object.weight_hh_l0.data.numpy().transpose()
    layer['bias_data'] = class_object.bias_ih_l0.data.numpy()
    layer['recurrent_bias_data'] = class_object.bias_hh_l0.data.numpy()

    if class_object.bias is False:
        layer['bias_data'] = np.zeros(layer['weight_data'].shape[0])
        layer['recurrent_bias_data'] = np.zeros(layer['recurrent_weight_data'].shape[0])

    if layer['class_name'] == 'GRU':
        layer['apply_reset_gate'] = 'after'  # Might be true for pytorch? It's not a free parameter

    output_shape = [input_shapes[0][0], layer['n_out']]

    layer['pytorch'] = True  # need to switch some behaviors to match pytorch implementations
    if len(input_names) == 1:
        layer['pass_initial_states'] = False
    else:
        layer['pass_initial_states'] = True

    return layer, output_shape


# brevitas has no QuantGRU (it defines only QuantRNN and QuantLSTM in brevitas.nn.quant_rnn), so hls4ml's
# GRU layer is reachable only through the unquantized torch.nn.GRU path.
quant_rnn_layers = ['QuantRNN', 'QuantLSTM']

# pytorch concatenates the LSTM gates as input, forget, cell, output, and the HLS code follows suit
lstm_gate_order = ('input_gate_params', 'forget_gate_params', 'cell_gate_params', 'output_gate_params')


def _quant_tensor_precision(quant_tensor):
    """Number of integer and fractional bits of the ap_fixed type that exactly holds a brevitas tensor.

    The integer count includes the sign bit, matching hls4ml's FixedPrecisionType.
    """
    width = int(quant_tensor.bit_width)
    scale = float(quant_tensor.scale.detach())
    mantissa, _ = np.frexp(scale)
    if mantissa != 0.5:
        raise Exception(
            """Non-power of 2 quantization of weights not supported when injecting brevitas models.
            Please used QONNX instead."""
        )
    fract_bitwidth, int_bitwidth = convert_uaq_to_apfixed(width, scale)
    return int(int_bitwidth), int(fract_bitwidth), bool(quant_tensor.signed)


def _brevitas_quant_to_hls(quant_tensors, transpose=False):
    """Turn one or more brevitas quant tensors into a weight array plus the matching hls4ml quantizer.

    Several tensors are concatenated along axis 0, which is how the per-gate LSTM weights combine into
    the single tensor the HLS code expects. Each gate carries its own quantizer, so the returned type is
    widened to whatever exactly represents all of them; it collapses to the common type when the scales
    agree. Transposing afterwards matches the keras order used in the HLS code, as in parse_rnn_layer.
    """
    precisions = [_quant_tensor_precision(t) for t in quant_tensors]
    integer = max(p[0] for p in precisions)
    fractional = max(p[1] for p in precisions)
    signed = any(p[2] for p in precisions)
    width = integer + fractional

    data = np.concatenate([t.detach().value.numpy() for t in quant_tensors], axis=0)
    if transpose:
        data = data.transpose()

    quantizer = BrevitasQuantizer(width, FixedPrecisionType(width=width, integer=integer, signed=signed))
    return data, quantizer


@pytorch_handler(*quant_rnn_layers)
def parse_quant_rnn_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation in quant_rnn_layers
    operation = operation.split('Quant')[-1]

    if len(class_object._modules['layers']) > 1:
        raise Exception('hls4ml does not support num_layers > 1')

    if class_object.num_directions > 1:
        raise Exception('hls4ml does not support birectional RNNs')

    layer = {}

    layer['name'] = layer_name

    layer['inputs'] = input_names
    if 'IOType' in config.keys():
        if len(input_names) > 1 and config['IOType'] == 'io_stream':
            raise Exception('Passing initial values for the hidden state is not supported for io_stream input type.')

    layer['class_name'] = operation
    if operation == 'RNN':
        layer['class_name'] = 'SimpleRNN'

    layer['return_sequences'] = False  # parameter does not exist in pytorch
    layer['return_state'] = False  # parameter does not exist in pytorch

    if layer['class_name'] == 'SimpleRNN':
        layer['activation'] = 'tanh' if 'Tanh' in str(class_object._modules['layers'][0][0].cell.act_fn) else 'ReLU'
    else:
        layer['activation'] = 'tanh'  # GRU and LSTM are hard-coded to use tanh in pytorch

    if layer['class_name'] == 'GRU' or layer['class_name'] == 'LSTM':
        layer['recurrent_activation'] = 'sigmoid'  # GRU and LSTM are hard-coded to use sigmoid in pytorch

    layer['time_major'] = not class_object._modules['layers'][0][0].cell.batch_first
    # TODO Should we handle time_major?
    if layer['time_major']:
        raise Exception('hls4ml only supports "batch-first == True"')

    layer['n_timesteps'] = input_shapes[0][1]
    layer['n_in'] = input_shapes[0][2]

    layer['n_out'] = class_object._modules['layers'][0][0].hidden_size

    RNNObject = class_object._modules['layers'][0][0]

    if layer['class_name'] == 'LSTM':
        # An LSTM keeps one GateParams per gate; the HLS code wants them concatenated in pytorch's order
        gates = [getattr(RNNObject, gate) for gate in lstm_gate_order]
        n_gates = len(gates)
        input_weights = [gate.input_weight.quant_weight() for gate in gates]
        hidden_weights = [gate.hidden_weight.quant_weight() for gate in gates]
        biases = [gate.quant_bias() for gate in gates]
    else:
        gates = [RNNObject.gate_params]
        n_gates = 1
        input_weights = [RNNObject.gate_params.input_weight.quant_weight()]
        hidden_weights = [RNNObject.gate_params.hidden_weight.quant_weight()]
        biases = [RNNObject.gate_params.quant_bias()]

    if all(gate.input_weight.weight_quant.is_quant_enabled for gate in gates):
        layer['weight_data'], layer['weight_quantizer'] = _brevitas_quant_to_hls(input_weights, transpose=True)

    if all(gate.hidden_weight.weight_quant.is_quant_enabled for gate in gates):
        layer['recurrent_weight_data'], layer['recurrent_weight_quantizer'] = _brevitas_quant_to_hls(
            hidden_weights, transpose=True
        )

    if all(bias is not None for bias in biases):
        layer['bias_data'], layer['bias_quantizer'] = _brevitas_quant_to_hls(biases)
    else:
        layer['bias_data'] = np.zeros(n_gates * layer['n_out'])
        layer['bias_quantizer'] = layer['weight_quantizer']

    # brevitas folds both pytorch biases into a single one, so the recurrent bias is always zero
    layer['recurrent_bias_data'] = np.zeros(n_gates * layer['n_out'])
    layer['recurrent_bias_quantizer'] = layer['weight_quantizer']

    # NOTE: brevitas' gate_acc_quant quantizes the gate accumulator *result*, it is not a statement about
    # how wide the accumulator itself has to be. Mapping it onto hls4ml's accum_t makes the accumulator far
    # too narrow (an 8 bit type with the activation's scale saturates at +-1), so the dot products wrap and
    # the layer produces garbage. accum_t is therefore left to the normal hls4ml precision inference, as in
    # parse_rnn_layer for unquantized RNNs.

    if RNNObject.cell.output_quant.is_quant_enabled:
        layer = addQuantizationParameters(layer, RNNObject.cell.output_quant, 'output', act=True)
        layer = addQuantizationParameters(layer, RNNObject.cell.output_quant, 'input', act=True)

    if layer['class_name'] == 'GRU':
        layer['apply_reset_gate'] = 'after'  # Might be true for pytorch? It's not a free parameter

    output_shape = [input_shapes[0][0], layer['n_out']]

    layer['pytorch'] = True  # need to switch some behaviors to match pytorch implementations
    if len(input_names) == 1:
        layer['pass_initial_states'] = False
    else:
        layer['pass_initial_states'] = True

    return layer, output_shape
