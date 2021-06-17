from __future__ import print_function
import numpy as np
import h5py
import json
import math
import tensorflow.keras as keras
from hls4ml.model.profiling import activations_keras, weights_keras
from hls4ml.model.hls_layers import layer_map
from hls4ml.templates import VivadoBackend
from collections import OrderedDict

QKERAS_DATA_TYPE_PREFIX = '***'


def create_vivado_config(output_dir='my-hls-test', project_name='myproject',
    fpga_part='xcku115-flvb2104-2-i', clock_period=5, io_type='io_parallel'):
    
    config = {}
    
    config['OutputDir'] = output_dir
    config['ProjectName'] = project_name
    config['XilinxPart'] = fpga_part
    config['ClockPeriod'] = clock_period
    config['Backend'] = 'Vivado'
    config['IOType'] = io_type
    config['HLSConfig'] = {}

    return config

def _get_precision_from_quantizer(quantizer, auto_precision_on=False):
    import qkeras
    if isinstance(quantizer, str):
        quantizer_obj = qkeras.get_quantizer(quantizer)
        quantizer = {}
        # Some activations are classes with get_config method
        if hasattr(quantizer_obj, 'get_config'):
            quantizer['class_name'] = quantizer_obj.__class__.__name__
            quantizer['config'] = quantizer_obj.get_config()
        # Some activations are just functions
        else: 
            quantizer['class_name'] = quantizer_obj.__name__

    supported_quantizers = ['quantized_bits', 'quantized_relu', 'quantized_tanh', 'quantized_po2', 'quantized_relu_po2']
    if quantizer['class_name'] in supported_quantizers:
        bits = int(quantizer['config']['bits'])
        # if integer isn't specified, it should be the same as bits
        integer = int(quantizer['config'].get('integer', bits-1)) + 1
        
    elif quantizer['class_name'] in ['binary', 'stochastic_binary', 'binary_tanh']:
        bits = 2
        integer = 2
    
    elif quantizer['class_name'] in ['ternary', 'stochastic_ternary', 'ternary_tanh']:
        bits = 2
        integer = 2
    else:
        raise Exception('ERROR: Unsupported quantizer: {}'.format(quantizer['class_name']))

    decimal = bits - integer

    # If precision is to be set automatically (i.e. auto_precision_on is True), return the data types with the special
    # prefix so that set_data_types_from_keras_model() can flag them as coming from QKeras and leave them unchanged.
    if auto_precision_on:
        prefix = QKERAS_DATA_TYPE_PREFIX
    else:
        prefix = ''

    if decimal > 0:
        return prefix + 'ap_fixed<{},{}>'.format(bits, integer)
    else:
        return prefix + 'ap_int<{}>'.format(bits)


def set_accum_from_keras_model(config, model):
    if 'LayerName' not in config:
        raise RuntimeError("The granularity of the supplied config is not 'name'.")

    def find_optimal_a_b(max_val, min_val):
        max_bits = 64

        for a in range(1, max_bits + 1):
            for b in range(0, a + 1):
                max_possible = 2 ** b - 2 ** (b - a)
                min_possible = 2 ** (b - a)

                if min_possible <= min_val and max_possible >= max_val:
                    return a + 1, b + 1

        return None, None

    for i, layer in enumerate(model.layers):
        name = layer.name

        if name not in config['LayerName']:
            print(f"accum_t profiling: {name} not present in config['LayerName'], ignoring.")
            continue

        if 'accum' not in config['LayerName'][name]['Precision']:
            continue

        if not isinstance(layer, keras.layers.Dense):
            print(f"accum_t profiling: {name} is not a Dense layer, ignoring as only Dense layers are currently "
                  "supported. You can use set_data_types_from_keras_model() instead.")
            continue

        type_w = VivadoBackend.convert_precision_string(None, config['LayerName'][name]['Precision']['weight'])
        type_b = VivadoBackend.convert_precision_string(None, config['LayerName'][name]['Precision']['bias'])

        if i == 0:
            previous_layer_config = config['LayerName'][name + '_input']
        else:
            previous_layer_config = config['LayerName'][model.layers[i - 1].name]

        if isinstance(previous_layer_config['Precision'], dict):
            type_i = VivadoBackend.convert_precision_string(None, previous_layer_config['Precision']['result'])
        else:
            type_i = VivadoBackend.convert_precision_string(None, previous_layer_config['Precision'])

        # Assuming that all of type_w, type_b and type_i are FixedPrecisionType objects.
        n = layer.output_shape[1]
        max_w = 2 ** (type_w.integer - 1) - 2 ** (type_w.integer - type_w.width)
        max_i = 2 ** (type_i.integer - 1) - 2 ** (type_i.integer - type_i.width)
        max_b = 2 ** (type_b.integer - 1) - 2 ** (type_b.integer - type_b.width)
        min_b = 2 ** (type_b.integer - type_b.width)

        max_val = n * max_w * max_i + max_b
        min_val = min_b

        a, b = find_optimal_a_b(max_val, min_val)

        if a is None or b is None:
            raise RuntimeError(f"Could not find an optimal accum_t type for {name}.")

        config['LayerName'][name]['Precision']['accum'] = f'ap_fixed<{a},{b}>'


def set_data_types_from_keras_model(config, model, max_bits, test_inputs=None, best_type_algorithm=None):
    """Adjust data types in a given HLSModel configuration based on a Keras model and test inputs (if supplied).

    The function aims for setting precision of the layers in the configuration to match the distribution of both
    weights in the model and outputs of the model resulting from the test inputs (if supplied).

    set_data_types_from_keras_model() works in a heuristic way and does not account for optimizations that can be
    subsequently made by hls4ml. Therefore, the optimal result is not guaranteed and it might be necessary to do
    post-tuning of the data types in order to achieve the best outcome. A user-defined algorithm can be passed to
    this function as best_type_algorithm to help obtain precision types closer to the optimal ones.

    Args:
        config (dict): HLSModel configuration dictionary to be updated. Its granularity must be 'name'.
        model: Keras model to be used for adjusting the data types.
        max_bits (int): The maximum bit width (excluding the sign bit) all data types in the config should have.
        test_inputs (array-like, optional): Inputs to be used for producing the distribution of model outputs.
            The type of test_inputs is the same as the type of X in hls4ml.model.profiling.numerical(). If not provided,
            precision of the layer outputs/activations will not be updated.
        best_type_algorithm (function (layer_type, max_val, min_val, median, q1, q3, max_bits) -> (A [int], B [int]),
                             optional):
            Algorithm to be used for determining the best data type ap_fixed<A, B> for a specific layer, given the
            following profiling information: corresponding hls4ml layer class name (layer_type), max value (max_val),
            min value (min_val), median (median), 1st quartile (q1), 3rd quartile (q3) and the max bit width without
            the sign bit (max_bits). Because the bit width doesn't include the sign bit, A may exceed max_bits by 1.

            If the algorithm does not find any suitable data type, it should return (None, None).

            If best_type_algorithm is not provided, the default algorithm will be used for all layers (using only
            max_val and min_val to find a data type both covering max_val and with the minimum absolute distance
            between min_val and the minimum representable number).

    Returns:
        None. The function makes changes directly to the supplied config.
    """
    if 'LayerName' not in config:
        raise RuntimeError("The granularity of the supplied config is not 'name'.")

    def find_optimal_a_b(layer_type, max_val, min_val, median, q1, q3, max_bits):
        a_final = None
        b_final = None

        distance_to_min = math.inf

        for a in range(1, max_bits + 1):
            for b in range(0, a + 1):
                max_possible = 2 ** b - 2 ** (b - a)
                min_possible = 2 ** (b - a)

                if max_possible >= max_val and abs(min_possible - min_val) < distance_to_min:
                    a_final = a
                    b_final = b

                    distance_to_min = abs(min_possible - min_val)

        if a_final is not None and b_final is not None:
            # An extra integer bit must be added for the number sign
            return a_final + 1, b_final + 1
        else:
            return None, None

    def get_precision(layer_dict, key):
        if key is None:
            return layer_dict['Precision']
        else:
            return layer_dict['Precision'][key]

    def set_precision(layer_dict, key, value):
        if key is None:
            layer_dict['Precision'] = value
        else:
            layer_dict['Precision'][key] = value

    def process_precision_in_dict(layer_dict, stats, precision_type=None, data_type=None):
        if precision_type is not None and precision_type not in layer_dict['Precision']:
            return None

        current_data_type = get_precision(layer_dict, precision_type)

        if current_data_type.startswith(QKERAS_DATA_TYPE_PREFIX):
            # This data type comes from QKeras, so don't change it (just remove the flag)
            set_precision(layer_dict, precision_type, current_data_type[len(QKERAS_DATA_TYPE_PREFIX):])
            return None

        if data_type is None:
            min_value = stats['whislo']
            max_value = stats['whishi']
            median = stats['med']
            q1 = stats['q1']
            q3 = stats['q3']

            if 'LayerType' in config['LayerName'][layer_name]:
                layer_type = config['LayerName'][layer_name]['LayerType']
            else:
                raise RuntimeError(f"config['LayerName']['{layer_name}'] doesn't have the LayerType key. "
                                    "Make sure that the config has been made by config_from_keras_model().")

            a, b = best_type_algorithm(layer_type, max_value, min_value, median, q1, q3, max_bits)

            if a is None or b is None:
                if precision_type is not None:
                    raise RuntimeError(f"Could not find an optimal data type for {layer_name} ({precision_type}).")
                else:
                    raise RuntimeError(f"Could not find an optimal data type for {layer_name}.")

            data_type = f'ap_fixed<{a},{b}>'

        set_precision(layer_dict, precision_type, data_type)
        return data_type

    if best_type_algorithm is None:
        best_type_algorithm = find_optimal_a_b

    weight_data = weights_keras(model, fmt='summary', plot='boxplot')

    suffix_map = {
        'w': 'weight',
        's': 'scale',
        'b': 'bias'
    }

    for weight_info in weight_data:
        layer_name = weight_info['layer']

        if layer_name not in config['LayerName']:
            print(f"Weight profiling: {layer_name} not present in config['LayerName'], ignoring.")
            continue

        suffix = weight_info['weight'].split('/')[1]

        if suffix not in suffix_map or suffix_map[suffix] not in config['LayerName'][layer_name]['Precision']:
            continue

        process_precision_in_dict(config['LayerName'][layer_name], weight_info, suffix_map[suffix])

    if test_inputs is not None:
        activation_data = activations_keras(model, test_inputs, fmt='summary', plot='boxplot')

        for activation_info in activation_data:
            layer_name = activation_info['weight']

            if layer_name not in config['LayerName']:
                print(f"Activation profiling: {layer_name} not present in config['LayerName'], ignoring.")
                continue

            data_type = None

            if isinstance(config['LayerName'][layer_name]['Precision'], dict):
                data_type = process_precision_in_dict(config['LayerName'][layer_name], activation_info,
                                                      'result', data_type)
                data_type = process_precision_in_dict(config['LayerName'][layer_name], activation_info,
                                                      'accum', data_type)
            else:
                data_type = process_precision_in_dict(config['LayerName'][layer_name], activation_info,
                                                      data_type=data_type)

            if data_type is not None and layer_name + '_linear' in config['LayerName']:
                config['LayerName'][layer_name + '_linear']['Precision'] = data_type


def config_from_keras_model(model, granularity='model', default_precision='ap_fixed<16,6>', default_reuse_factor=1):
    """Create an HLS conversion config given the Keras model.

    This function serves as the initial step in creating the custom conversion configuration.
    Users are advised to inspect the returned object to tweak the conversion configuration.
    The return object can be passed as `hls_config` parameter to `convert_from_keras_model`.

    Args:
        model: Keras model
        granularity (str, optional): Granularity of the created config. Defaults to 'model'.
            Can be set to 'model', 'type' and 'layer'.

            Granularity can be used to generate a more verbose config that can be fine-tuned.
            The default granulrity ('model') will generate config keys that apply to the whole
            model, so changes to the keys will affect the entire model. 'type' granularity will
            generate config keys that affect all layers of a given type, while the 'name' granularity
            will generate config keys for every layer separately, allowing for highly specific
            configuration tweaks.
        default_precision (str, optional): Default precision to use. Defaults to 'ap_fixed<16,6>'.
        default_reuse_factor (int, optional): Default reuse factor. Defaults to 1.

    Raises:
        Exception: If Keras model has layers not supported by hls4ml.

    Returns:
        [dict]: The created config.
    """
    if granularity.lower() not in ['model', 'type', 'name']:
        raise Exception('Invalid configuration granularity specified, expected "model", "type" or "name" got "{}"'.format(granularity))

    #This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    if isinstance(model, dict):
        model_arch = model
    else:
        model_arch = json.loads(model.to_json())

    #Define supported laers
    core_layers = ['InputLayer', 'Dropout', 'Flatten', 'Reshape']
    dense_layers = ['Dense', 'BinaryDense', 'TernaryDense']
    conv_layers = ['Conv1D', 'Conv2D', 'BinaryConv2D']
    pooling_layers = ['MaxPooling1D', 'MaxPooling2D', 'GlobalMaxPooling1D', 'GlobalMaxPooling2D', 'AveragePooling1D', 'AveragePooling2D', 'GlobalAveragePooling1D', 'GlobalAveragePooling2D']
    norm_layers = ['BatchNormalization']
    activation_layers = ['Activation', 'LeakyReLU', 'ThresholdedReLU', 'ELU', 'PReLU', 'Softmax', 'ReLU']
    merge_layers = ['Add', 'Subtract', 'Multiply', 'Average', 'Maximum', 'Minimum', 'Concatenate', 'Dot']
    qkeras_layers = ['QDense', 'QActivation', 'QConv1D', 'QConv2D', 'QBatchNormalization', 'QConv2DBatchnorm']
    #Define layers to skip for conversion to HLS
    skip_layers = ['Dropout', 'Flatten']
    #All supported layers
    supported_layers = core_layers + dense_layers + conv_layers + pooling_layers + norm_layers + activation_layers + merge_layers + qkeras_layers + skip_layers

    keras_layer_config = None
    if model_arch['class_name'] == 'Sequential':
        print('Interpreting Sequential')
        keras_layer_config = model_arch['config']
        if 'layers' in keras_layer_config: # Newer Keras versions have 'layers' in 'config' key
            keras_layer_config = keras_layer_config['layers']
        # Sequential doesn't have InputLayer in TF < 2.3 (Keras 2.4.0)
        if keras_layer_config[0]['class_name'] != 'InputLayer':
            input_layer = {}
            input_layer['name'] = 'input1'
            input_layer['class_name'] = 'Input'
            layer_list.append(input_layer)
    elif model_arch['class_name'] in ['Model', 'Functional']:
        print('Interpreting Model')
        keras_layer_config = model_arch['config']['layers']

    print('Topology:')
    for keras_layer in keras_layer_config:
        if keras_layer['class_name'] not in supported_layers:
            raise Exception('ERROR: Unsupported layer type: {}'.format(keras_layer['class_name']))
        if keras_layer['class_name'] in skip_layers:
            continue

        #Dictionary to fill in and append to layer_list
        layer = {}

        #Extract name for finding weights and biases
        layer['name'] = keras_layer['config']['name']
        layer['class_name'] = keras_layer['class_name']
        layer['config'] = keras_layer['config']

        if layer['class_name'] == 'InputLayer':
            layer['class_name'] = 'Input'

        if layer['class_name'] in qkeras_layers:
            layer['precision'] = {}
            for qname, qclass in layer['config'].items():
                if 'quantizer' in qname.lower():
                    pname = qname.split('_quantizer')[0]
                    if pname == 'kernel': pname = 'weight'
                    if qclass is not None:
                        precision = _get_precision_from_quantizer(qclass)
                        layer['precision'][pname] = precision
                elif qname == 'activation' and layer['class_name'] == 'QActivation':
                    precision = _get_precision_from_quantizer(qclass)
                    layer['precision']['result'] = precision

        print('Layer name: {}, layer type: {}'.format(layer['name'], layer['class_name']))
        layer_list.append( layer )
        if 'activation' in layer['config'] and layer['class_name'] not in activation_layers + qkeras_layers:
            act_layer = {}
            act_layer['name'] = layer['name'] + '_' + layer['config']['activation']
            act_layer['class_name'] = 'Activation'
            print('  -> Activation ({}), layer name: {}'.format(layer['config']['activation'], layer['name']))
            layer_list.append(act_layer)


    def make_layer_config(layer):
        layer_config = {}
        layer_type = layer_map[layer['class_name']].__name__

        hls_layers_accum = ['Dense', 'Conv1D', 'Conv2D']

        layer_config['LayerType'] = layer_map[layer['class_name']].__name__

        if layer['class_name'] in dense_layers + conv_layers:
            layer_config['Precision'] = {}
            layer_config['Precision']['weight'] = default_precision
            layer_config['Precision']['bias'] = default_precision
            layer_config['Precision']['result'] = default_precision
            layer_config['Precision']['accum'] = default_precision
            layer_config['ReuseFactor'] = default_reuse_factor

        elif layer['class_name'] in activation_layers:
            layer_config['Precision'] = default_precision
            layer_config['ReuseFactor'] = default_reuse_factor
            layer_config['table_size'] = 1024
            is_softmax = layer['class_name'] == 'Softmax'
            if 'config' in layer.keys():
                if 'activation' in layer['config'].keys():
                    is_softmax = is_softmax or (layer['config']['activation'] == 'softmax')
            if is_softmax:
               layer_config['LayerType'] = 'Softmax'
               layer_config['exp_table_t'] = 'ap_fixed<18,8,AP_RND,AP_SAT>'
               layer_config['inv_table_t'] = 'ap_fixed<18,8,AP_RND,AP_SAT>'
            else:
                layer_config['table_t'] = 'ap_fixed<18,8>'
        
        elif layer['class_name'] in norm_layers:
            layer_config['Precision'] = {}
            layer_config['Precision']['scale'] = default_precision
            layer_config['Precision']['bias'] = default_precision
            layer_config['ReuseFactor'] = default_reuse_factor
        
        elif layer['class_name'] in qkeras_layers:
            if 'precision' in layer:
                layer_config['Precision'] = {}
                for name, precision in layer['precision'].items():
                    layer_config['Precision'][name] = precision

                if layer_type in hls_layers_accum:
                    layer_config['Precision']['accum'] = default_precision
            else:
                print('WARNING: Found no precision information in QKeras layer {} ({})'.format(layer['name'], layer['class_name']))
                layer_config['Precision'] = default_precision
            layer_config['ReuseFactor'] = default_reuse_factor

        elif layer['class_name'] == 'Input':
            layer_config['Precision'] = {}
            layer_config['Precision']['result'] = default_precision

        else:
            layer_config['Precision'] = default_precision

        return layer_config

    config = {}

    model_config = {}
    model_config['Precision'] = default_precision
    model_config['ReuseFactor'] = default_reuse_factor
    model_config['Strategy'] = 'Latency'
    #model_config['Compression'] = False
    #model_config['Trace'] = False

    config['Model'] = model_config
    
    if granularity.lower() == 'type':
        type_config = {}
        for layer in layer_list:
            if layer['class_name'] in type_config:
                continue
            layer_config = make_layer_config(layer)
            type_config[layer['class_name']] = layer_config
        
        config['LayerType'] = type_config

    elif granularity.lower() == 'name':
        name_config = {}
        for layer in layer_list:
            layer_config = make_layer_config(layer)
            name_config[layer['name']] = layer_config
        
        config['LayerName'] = name_config

    return config
