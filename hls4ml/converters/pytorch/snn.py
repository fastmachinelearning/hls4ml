import numpy as np

from hls4ml.converters.pytorch_to_hls import pytorch_handler

# Treat numerically unit leak as IF behavior.
BETA_TO_IF_TOL = 1e-6


def _to_numpy(value, name):
    if value is None:
        raise Exception(f'Missing SNN parameter: {name}')

    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()

    try:
        return np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as err:
        raise Exception(f'Could not parse "{name}" as numpy array: {value}') from err


def _parse_scalar_or_vector(class_object, name, n_out):
    value = getattr(class_object, name, None)
    arr = _to_numpy(value, name)
    flat = arr.reshape(-1)

    if flat.size == 1:
        scalar = float(flat[0])
        return {'mode': 'scalar', 'scalar': scalar, 'vector': None}

    if flat.size == n_out:
        if np.allclose(flat, flat[0], rtol=0.0, atol=BETA_TO_IF_TOL):
            scalar = float(flat[0])
            return {'mode': 'scalar', 'scalar': scalar, 'vector': None}
        return {'mode': 'vector', 'scalar': None, 'vector': flat.astype(np.float32)}

    raise Exception(f'Only scalar or length-{n_out} "{name}" is supported for SNN conversion, got shape {arr.shape}')


def _parse_reset_mechanism(class_object):
    reset = getattr(class_object, 'reset_mechanism', 'subtract')
    reset = str(reset).lower()
    if reset not in ['subtract', 'zero']:
        raise Exception(f'Unsupported reset mechanism "{reset}". Supported: "subtract", "zero".')
    return reset


def _parse_state_reset_policy(class_object):
    policy = getattr(class_object, 'state_reset_policy', None)
    if policy is None:
        policy = getattr(class_object, 'reset_policy', 'fixed_window')
    policy = str(policy).lower()
    if policy not in ['fixed_window', 'tlast', 'host_pulse', 'never']:
        raise Exception(
            f'Unsupported state reset policy "{policy}". Supported: "fixed_window", "tlast", "host_pulse", "never".'
        )
    return policy


def _parse_readout_beta(class_object):
    beta = getattr(class_object, 'beta', 1.0)
    arr = _to_numpy(beta, 'beta').reshape(-1)
    if arr.size != 1:
        raise Exception(f'Only scalar "beta" is supported for SNNReadout membrane mode, got shape {arr.shape}')
    return float(arr[0])


@pytorch_handler('Leaky')
def parse_lif_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation == 'Leaky'

    n_out = input_shapes[0][-1]
    beta = _parse_scalar_or_vector(class_object, 'beta', n_out)
    threshold = _parse_scalar_or_vector(class_object, 'threshold', n_out)

    layer = {}
    use_if = beta['mode'] == 'scalar' and np.isclose(beta['scalar'], 1.0, rtol=0.0, atol=BETA_TO_IF_TOL)
    layer['class_name'] = 'IFNeuron' if use_if else 'LIFNeuron'
    layer['name'] = layer_name
    layer['inputs'] = input_names
    layer['n_in'] = n_out
    layer['n_out'] = n_out
    layer['threshold_mode'] = threshold['mode']
    if threshold['mode'] == 'scalar':
        layer['threshold'] = threshold['scalar']
    else:
        layer['threshold'] = 0.0
        layer['threshold_data'] = threshold['vector']

    if layer['class_name'] == 'LIFNeuron':
        layer['beta_mode'] = beta['mode']
        if beta['mode'] == 'scalar':
            layer['beta'] = beta['scalar']
        else:
            layer['beta'] = 0.0
            layer['beta_data'] = beta['vector']
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
    if hasattr(class_object, 'stream_length'):
        layer['window_size'] = int(class_object.stream_length)
    else:
        layer['window_size'] = int(getattr(class_object, 'window_size', 1))
    layer['class_threshold'] = int(getattr(class_object, 'class_threshold', 1))
    layer['output_mode'] = str(getattr(class_object, 'output_mode', 'spike')).lower()
    if layer['output_mode'] not in ['spike', 'membrane']:
        raise Exception(f'Unsupported SNNReadout output mode "{layer["output_mode"]}". Supported: spike, membrane.')
    layer['beta'] = _parse_readout_beta(class_object)
    default_decision_rule = 'argmax_membrane' if layer['output_mode'] == 'membrane' else 'argmax_spike_count'
    layer['decision_rule'] = str(getattr(class_object, 'decision_rule', default_decision_rule))
    layer['state_reset_policy'] = _parse_state_reset_policy(class_object)
    if layer['decision_rule'] not in [
        'argmax_spike_count',
        'first_to_threshold',
        'threshold_then_argmax',
        'binary_logit',
        'argmax_membrane',
    ]:
        raise Exception(
            f'Unsupported SNN decision rule "{layer["decision_rule"]}". '
            'Supported: argmax_spike_count, first_to_threshold, threshold_then_argmax, binary_logit, argmax_membrane.'
        )
    if layer['decision_rule'] == 'binary_logit' and layer['n_classes'] != 2:
        raise Exception('binary_logit decision rule requires n_classes == 2')
    if layer['output_mode'] == 'membrane' and layer['decision_rule'] not in ['argmax_membrane', 'binary_logit']:
        raise Exception('SNNReadout membrane mode supports decision_rule "argmax_membrane" or "binary_logit".')
    if layer['output_mode'] == 'spike' and layer['decision_rule'] == 'argmax_membrane':
        raise Exception('SNNReadout decision_rule "argmax_membrane" requires output_mode "membrane".')

    output_shape = input_shapes[0][:]
    output_shape[-1] = 1
    return layer, output_shape
