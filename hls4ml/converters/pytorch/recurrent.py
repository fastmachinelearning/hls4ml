import numpy as np

from hls4ml.converters.pytorch_to_hls import addQuantizationParameters, convert_uaq_to_apfixed, pytorch_handler
from hls4ml.model.quantizers import BrevitasQuantizer
from hls4ml.model.types import FixedPrecisionType, NamedType

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


quant_rnn_layers = ['QuantRNN']  # QuantLSTM very complex, might come later. No QuantGRU in brevitas at this point


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

    if RNNObject.gate_params.input_weight.weight_quant.is_quant_enabled:
        width = int(RNNObject.gate_params.input_weight.quant_weight().bit_width)
        scale = RNNObject.gate_params.input_weight.quant_weight().scale.detach().numpy()
        signed = RNNObject.gate_params.input_weight.quant_weight().signed
        mantissa, _ = np.frexp(scale)
        # if scale is power of 2 we can simply use hls4ml FixedPrecisionType and directly
        # use the already quantized tensor from brevitas
        if mantissa == 0.5:
            ap_fixed_params = convert_uaq_to_apfixed(width, float(RNNObject.gate_params.input_weight.quant_weight().scale))
            layer['weight_data'] = RNNObject.gate_params.input_weight.quant_weight().detach().value.numpy()
            layer['weight_quantizer'] = BrevitasQuantizer(
                width, FixedPrecisionType(width=width, integer=int(ap_fixed_params[1]), signed=signed)
            )
        else:
            raise Exception(
                """Non-power of 2 quantization of weights not supported when injecting brevitas models.
                Please used QONNX instead."""
            )

    if RNNObject.gate_params.hidden_weight.weight_quant.is_quant_enabled:
        width = int(RNNObject.gate_params.hidden_weight.quant_weight().bit_width)
        scale = RNNObject.gate_params.hidden_weight.quant_weight().scale.detach().numpy()
        signed = RNNObject.gate_params.input_weight.quant_weight().signed
        mantissa, _ = np.frexp(scale)
        # if scale is power of 2 we can simply use hls4ml FixedPrecisionType and directly
        # use the already quantized tensor from brevitas
        if mantissa == 0.5:
            ap_fixed_params = convert_uaq_to_apfixed(width, float(RNNObject.gate_params.hidden_weight.quant_weight().scale))
            layer['recurrent_weight_data'] = RNNObject.gate_params.hidden_weight.quant_weight().detach().value.numpy()
            layer['recurrent_weight_quantizer'] = BrevitasQuantizer(
                width, FixedPrecisionType(width=width, integer=int(ap_fixed_params[1]), signed=signed)
            )
        else:
            raise Exception(
                """Non-power of 2 quantization of weights not supported when injecting brevitas models.
                Please used QONNX instead."""
            )

    input_bias = RNNObject.gate_params.quant_bias()
    if input_bias is not None:
        width = int(input_bias.bit_width)
        scale = input_bias.scale.detach().numpy()
        mantissa, _ = np.frexp(scale)
        # if scale is power of 2 we can simply use hls4ml FixedPrecisionType and directly
        # use the already quantized tensor from brevitas
        if mantissa == 0.5:
            ap_fixed_params = convert_uaq_to_apfixed(width, scale)

            layer['bias_data'] = input_bias.detach().value.numpy()
            layer['bias_quantizer'] = BrevitasQuantizer(
                width, FixedPrecisionType(width=width, integer=int(ap_fixed_params[1]), signed=True)
            )
        else:
            raise Exception(
                """Non-power of 2 quantization of weights not supported when injecting brevitas models.
                Please used QONNX instead."""
            )
    else:
        layer['bias_data'] = np.zeros(layer['weight_data'].shape[0])
        layer['bias_quantizer'] = layer['weight_quantizer']

    layer['recurrent_bias_data'] = np.zeros(layer['recurrent_weight_data'].shape[0])
    layer['recurrent_bias_quantizer'] = layer['weight_quantizer']

    acc_scale = RNNObject.cell.gate_acc_quant.scale()
    acc_bitwdith = int(RNNObject.cell.gate_acc_quant.bit_width())
    mantissa, _ = np.frexp(acc_scale)
    # if scale is power of 2 we can simply use hls4ml FixedPrecisionType and directly
    # use the already quantized tensor from brevitas
    if mantissa == 0.5:
        ap_fixed_params = convert_uaq_to_apfixed(acc_bitwdith, acc_scale)
        precision = FixedPrecisionType(width=width, integer=int(ap_fixed_params[1]), signed=True)
        layer['accum_t'] = NamedType(layer['name'] + '_accum_t', precision)

    else:
        raise Exception(
            """Non-power of 2 quantization of weights not supported when injecting brevitas models.
            Please used QONNX instead."""
        )

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
