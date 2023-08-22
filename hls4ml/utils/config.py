import json

import qkeras

import hls4ml


def create_config(output_dir='my-hls-test', project_name='myproject', backend='Vivado', version='1.0.0', **kwargs):
    """Create an initial configuration to guide the conversion process.

    The resulting configuration will contain general information about the project (like project name and output directory)
    as well as the backend-specific configuration (part numbers, clocks etc). Extra arguments of this function will be
    passed to the backend's ``create_initial_config``. For the possible list of arguments, check the documentation of each
    backend.

    Args:
        output_dir (str, optional): The output directory to which the generated project will be written.
            Defaults to 'my-hls-test'.
        project_name (str, optional): The name of the project, that will be used as a top-level function in HLS designs.
            Defaults to 'myproject'.
        backend (str, optional): The backend to use. Defaults to 'Vivado'.
        version (str, optional): Optional string to version the generated project for backends that support it.
            Defaults to '1.0.0'.

    Raises:
        Exception: Raised if unknown backend is specified.

    Returns:
        dict: The conversion configuration.
    """
    backend_list = hls4ml.backends.get_available_backends()
    if backend.lower() not in backend_list:
        raise Exception(f'Unknown backend: {backend}')

    backend = hls4ml.backends.get_backend(backend)

    backend_config = backend.create_initial_config(**kwargs)

    config = {}
    config['OutputDir'] = output_dir
    config['ProjectName'] = project_name
    config['Backend'] = backend.name
    config['Version'] = version
    config.update(backend_config)

    return config


def _get_precision_from_quantizer(quantizer):
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

    supported_quantizers = [
        'quantized_bits',
        'quantized_relu',
        'quantized_tanh',
        'quantized_sigmoid',
        'quantized_po2',
        'quantized_relu_po2',
        'linear',
    ]
    signed = True
    rnd = "AP_TRN"
    overflow = "AP_WRAP"

    if quantizer['class_name'] in supported_quantizers:
        bits = int(quantizer['config']['bits'])
        # if integer isn't specified, it should be the same as bits
        integer = int(quantizer['config'].get('integer', bits - 1)) + 1
        # for quantizers use the following default rounding and overflow
        rnd = "AP_RND_CONV"
        overflow = "AP_SAT"
        if quantizer['class_name'] in ('quantized_relu', 'quantized_relu_po2'):
            signed = False
            integer -= 1
        elif quantizer['class_name'] == 'quantized_tanh':
            overflow = "AP_SAT_SYM" if quantizer['config']['symmetric'] else "AP_SAT"
            integer = 1
        elif quantizer['class_name'] == 'quantized_sigmoid':
            integer = 0
            signed = False

    elif quantizer['class_name'] in ['binary', 'stochastic_binary', 'binary_tanh']:
        bits = 2
        integer = 2

    elif quantizer['class_name'] in ['ternary', 'stochastic_ternary', 'ternary_tanh']:
        bits = 2
        integer = 2
    else:
        raise Exception('ERROR: Unsupported quantizer: {}'.format(quantizer['class_name']))

    decimal = bits - integer

    if decimal > 0:
        return hls4ml.model.types.FixedPrecisionType(
            width=bits, integer=integer, signed=signed, rounding_mode=rnd, saturation_mode=overflow
        )
    else:
        return hls4ml.model.types.IntegerPrecisionType(width=integer, signed=signed)


def config_from_keras_model(
    model, granularity='model', backend=None, default_precision='fixed<16,6>', default_reuse_factor=1
):
    """Create an HLS conversion config given the Keras model.

    This function serves as the initial step in creating the custom conversion configuration.
    Users are advised to inspect the returned object to tweak the conversion configuration.
    The return object can be passed as `hls_config` parameter to `convert_from_keras_model`.

    Args:
        model: Keras model
        granularity (str, optional): Granularity of the created config. Defaults to 'model'.
            Can be set to 'model', 'type' and 'name'.

            Granularity can be used to generate a more verbose config that can be fine-tuned.
            The default granularity ('model') will generate config keys that apply to the whole
            model, so changes to the keys will affect the entire model. 'type' granularity will
            generate config keys that affect all layers of a given type, while the 'name' granularity
            will generate config keys for every layer separately, allowing for highly specific
            configuration tweaks.
        backend(str, optional): Name of the backend to use
        default_precision (str, optional): Default precision to use. Defaults to 'fixed<16,6>'.
        default_reuse_factor (int, optional): Default reuse factor. Defaults to 1.

    Raises:
        Exception: If Keras model has layers not supported by hls4ml.

    Returns:
        [dict]: The created config.
    """
    if granularity.lower() not in ['model', 'type', 'name']:
        raise Exception(
            f'Invalid configuration granularity specified, expected "model", "type" or "name" got "{granularity}"'
        )

    if backend is not None:
        backend = hls4ml.backends.get_backend(backend)

    # This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    if isinstance(model, dict):
        model_arch = model
    else:
        model_arch = json.loads(model.to_json())

    reader = hls4ml.converters.KerasModelReader(model)

    layer_list, _, _, _ = hls4ml.converters.parse_keras_model(model_arch, reader)

    def make_layer_config(layer):
        cls_name = layer['class_name']
        if 'config' in layer.keys():
            if 'activation' in layer['config'].keys():
                if layer['config']['activation'] == 'softmax':
                    cls_name = 'Softmax'

        layer_cls = hls4ml.model.layers.layer_map[cls_name]
        if backend is not None:
            layer_cls = backend.create_layer_class(layer_cls)

        layer_config = {}

        config_attrs = [a for a in layer_cls.expected_attributes if a.configurable]
        for attr in config_attrs:
            if isinstance(attr, hls4ml.model.attributes.TypeAttribute):
                precision_cfg = layer_config.setdefault('Precision', {})
                name = attr.name
                if name.endswith('_t'):
                    name = name[:-2]
                if attr.default is None:
                    precision_cfg[name] = default_precision
                else:
                    precision_cfg[name] = str(attr.default)
            else:
                if attr.default is not None:
                    layer_config[attr.config_name] = attr.default

        quantizers = {qname: qclass for qname, qclass in layer.items() if 'quantizer' in qname and qclass is not None}
        for qname, qclass in quantizers.items():
            pname = qname.lower().split('_quantizer')[0]
            if pname == 'activation':
                pname = 'result'
            if isinstance(qclass, dict):
                precision = _get_precision_from_quantizer(qclass)
            else:
                precision = qclass.hls_type
            # TODO In the next version of this function, these should not be exposed to user to tweak
            layer_config['Precision'][pname] = str(precision)

        if layer['class_name'] in ['GarNet', 'GarNetStack']:
            # Define default precisions for various internal arrays (can be overridden from the config file)
            import math

            log2_reuse = int(math.log(default_reuse_factor, 2.0))
            n_vertices_width = int(math.log(layer['n_vertices'], 2.0))

            # We always give 10 digits for the subintegral part
            fwidth = 10
            # Integral precision for aggr_t depends on how large the temporary sum for weighed feature mean will be
            aggr_intw = max(log2_reuse, n_vertices_width - log2_reuse) + 3  # safety factor 2**3
            aggr_w = aggr_intw + fwidth
            # edge_weight_aggr_t does not need the safety factor
            ew_aggr_intw = aggr_intw - 3
            ew_aggr_w = ew_aggr_intw + fwidth

            layer_config['Precision'] = {}
            layer_config['Precision']['edge_weight'] = 'ap_ufixed<10,0,AP_TRN,AP_SAT>'
            layer_config['Precision']['edge_weight_aggr'] = f'ap_ufixed<{ew_aggr_w},{ew_aggr_intw},AP_TRN,AP_SAT>'
            layer_config['Precision']['aggr'] = f'ap_fixed<{aggr_w},{aggr_intw},AP_TRN,AP_SAT>'
            layer_config['Precision']['norm'] = 'ap_ufixed<14,4,AP_TRN,AP_SAT>'

            layer_config['ReuseFactor'] = default_reuse_factor

        elif layer['class_name'] == 'Input':
            dtype = layer['config']['dtype']
            if dtype.startswith('int') or dtype.startswith('uint'):
                typename = dtype[: dtype.index('int') + 3]
                width = int(dtype[dtype.index('int') + 3 :])
                layer_config['Precision']['result'] = f'ap_{typename}<{width}>'
            # elif bool, q[u]int, ...

        return layer_config

    config = {}

    model_config = {}
    model_config['Precision'] = default_precision
    model_config['ReuseFactor'] = default_reuse_factor
    model_config['Strategy'] = 'Latency'
    model_config['BramFactor'] = 1_000_000_000
    model_config['TraceOutput'] = False

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


def config_from_pytorch_model(
    model,
    granularity='model',
    backend=None,
    default_precision='ap_fixed<16,6>',
    default_reuse_factor=1,
    inputs_channel_last=False,
    transpose_outputs=True,
):
    """Create an HLS conversion config given the PyTorch model.

    This function serves as the initial step in creating the custom conversion configuration.
    Users are advised to inspect the returned object to tweak the conversion configuration.
    The return object can be passed as `hls_config` parameter to `convert_from_pytorch_model`.

    Args:
        model: PyTorch model
        granularity (str, optional): Granularity of the created config. Defaults to 'model'.
            Can be set to 'model', 'type' and 'layer'.

            Granularity can be used to generate a more verbose config that can be fine-tuned.
            The default granularity ('model') will generate config keys that apply to the whole
            model, so changes to the keys will affect the entire model. 'type' granularity will
            generate config keys that affect all layers of a given type, while the 'name' granularity
            will generate config keys for every layer separately, allowing for highly specific
            configuration tweaks.
        backend(str, optional): Name of the backend to use
        default_precision (str, optional): Default precision to use. Defaults to 'fixed<16,6>'.
        default_reuse_factor (int, optional): Default reuse factor. Defaults to 1.
        inputs_channel_last (bool, optional): Set to 'True' if input to the model comes in format
            'channels_last'. Defaults to 'False'. If False, inputs will be transposed internally.
        transpose_outputs (bool, optional): Set to 'False' if the output should not be transposed from
            channels_last into channels_first data format. Defaults to 'False'. If False, outputs needs
            to be transposed manually.

    Raises:
        Exception: If PyTorch model has layers not supported by hls4ml.

    Returns:
        [dict]: The created config.
    """

    config = {}

    model_config = {}
    model_config['Precision'] = default_precision
    model_config['ReuseFactor'] = default_reuse_factor
    model_config['InputsChannelLast'] = inputs_channel_last
    model_config['TransposeOutputs'] = transpose_outputs
    model_config['Strategy'] = 'Latency'

    config['Model'] = model_config

    return config


def config_from_onnx_model(
    model, granularity='model', backend=None, default_precision='ap_fixed<16,6>', default_reuse_factor=1
):
    """Create an HLS conversion config given the ONNX model.

    This function serves as the initial step in creating the custom conversion configuration.
    Users are advised to inspect the returned object to tweak the conversion configuration.
    The return object can be passed as `hls_config` parameter to `convert_from_onnx_model`.

    Args:
        model: ONNX model
        granularity (str, optional): Granularity of the created config. Defaults to 'model'.
            Can be set to 'model', 'type' and 'layer'.

            Granularity can be used to generate a more verbose config that can be fine-tuned.
            The default granularity ('model') will generate config keys that apply to the whole
            model, so changes to the keys will affect the entire model. 'type' granularity will
            generate config keys that affect all layers of a given type, while the 'name' granularity
            will generate config keys for every layer separately, allowing for highly specific
            configuration tweaks.
        backend(str, optional): Name of the backend to use
        default_precision (str, optional): Default precision to use. Defaults to 'fixed<16,6>'.
        default_reuse_factor (int, optional): Default reuse factor. Defaults to 1.

    Raises:
        Exception: If ONNX model has layers not supported by hls4ml.

    Returns:
        [dict]: The created config.
    """

    config = {}

    model_config = {}
    model_config['Precision'] = default_precision
    model_config['ReuseFactor'] = default_reuse_factor
    model_config['Strategy'] = 'Latency'

    config['Model'] = model_config

    return config
