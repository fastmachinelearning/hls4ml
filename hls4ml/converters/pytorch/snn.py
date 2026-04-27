import numpy as np

from hls4ml.converters.pytorch_to_hls import pytorch_handler

BETA_TO_IF_TOL = 1e-6


def _to_scalar(value, name):
    if value is None:
        raise Exception(f'Missing SNN parameter: {name}')

    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise Exception(f'Only scalar "{name}" is supported for SNN conversion, got shape {value.shape}')
        return float(value.reshape(-1)[0])

    try:
        return float(value)
    except (TypeError, ValueError) as err:
        raise Exception(f'Could not parse "{name}" as scalar: {value}') from err


def _parse_reset_mechanism(class_object):
    reset = getattr(class_object, 'reset_mechanism', 'subtract')
    reset = str(reset).lower()
    if reset not in ['subtract', 'zero']:
        raise Exception(f'Unsupported reset mechanism "{reset}". Supported: "subtract", "zero".')
    return reset


@pytorch_handler('Leaky')
def parse_lif_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation == 'Leaky'

    beta = _to_scalar(getattr(class_object, 'beta', None), 'beta')

    layer = {}
    layer['class_name'] = 'IFNeuron' if np.isclose(beta, 1.0, rtol=0.0, atol=BETA_TO_IF_TOL) else 'LIFNeuron'
    layer['name'] = layer_name
    layer['inputs'] = input_names
    layer['n_in'] = input_shapes[0][-1]
    layer['n_out'] = input_shapes[0][-1]
    layer['threshold'] = _to_scalar(getattr(class_object, 'threshold', None), 'threshold')
    if layer['class_name'] == 'LIFNeuron':
        layer['beta'] = beta
    layer['reset_mechanism'] = _parse_reset_mechanism(class_object)

    return layer, [shape for shape in input_shapes[0]]


@pytorch_handler('SNNReadout')
def parse_snn_readout_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation == 'SNNReadout'

    layer = {}
    layer['class_name'] = 'SNNReadout'
    layer['name'] = layer_name
    layer['inputs'] = input_names

    n_classes = getattr(class_object, 'n_classes', None)
    if n_classes is None:
        n_classes = input_shapes[0][-1]
    layer['n_classes'] = int(n_classes)
    layer['window_size'] = int(getattr(class_object, 'window_size', 1))
    layer['class_threshold'] = int(getattr(class_object, 'class_threshold', 1))
    layer['decision_rule'] = str(getattr(class_object, 'decision_rule', 'argmax_spike_count'))
    if layer['decision_rule'] not in ['argmax_spike_count', 'first_to_threshold', 'threshold_then_argmax', 'binary_logit']:
        raise Exception(
            f'Unsupported SNN decision rule "{layer["decision_rule"]}". '
            'Supported: argmax_spike_count, first_to_threshold, threshold_then_argmax, binary_logit.'
        )
    if layer['decision_rule'] == 'binary_logit' and layer['n_classes'] != 2:
        raise Exception('binary_logit decision rule requires n_classes == 2')

    output_shape = input_shapes[0][:]
    output_shape[-1] = 1
    return layer, output_shape
