from hls4ml.converters.keras_to_hls import keras_handler, parse_default_keras_layer


@keras_handler('FixedPointQuantizer')
def fixedpoint_quantizer_handler(keras_layer, input_names, input_shapes, data_reader):
    config = parse_default_keras_layer(keras_layer, input_names)

    name = config['name']
    fusible = keras_layer['config']['fusible']
    config['RND'] = keras_layer['config']['RND']
    config['SAT'] = keras_layer['config']['SAT']
    config['fusible'] = fusible
    if not fusible:
        k = data_reader.get_weights_data(name, 'keep_negative')
        b = data_reader.get_weights_data(name, 'bits')
        i = data_reader.get_weights_data(name, 'integers')
        config['mask_kbi'] = k, b, i
    config['overrides'] = keras_layer['config']['overrides']

    layer = config
    return layer, input_shapes[0]
