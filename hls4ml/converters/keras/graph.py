from hls4ml.converters.keras.core import TernaryQuantizer
from hls4ml.converters.keras_to_hls import get_weights_data, keras_handler, parse_default_keras_layer


@keras_handler('GarNet', 'GarNetStack')
def parse_garnet_layer(keras_layer, input_names, input_shapes, data_reader):
    assert keras_layer['class_name'] in ['GarNet', 'GarNetStack']

    if not keras_layer['config']['simplified']:
        raise Exception('HLS GarNet is compatible only with keras GarNet with simplified=True')
    if keras_layer['config']['output_activation'] not in [None, 'linear']:
        raise Exception('HLS GarNet cannot have nonlinear output activation')

    layer = parse_default_keras_layer(keras_layer, input_names)

    layer['input_format'] = keras_layer['config']['input_format']
    if layer['input_format'] != 'xn':
        raise NotImplementedError('HLS GarNet currently only implements signed inputs (input_format="xn")')

    layer['n_vertices'] = input_shapes[0][1]
    layer['collapse'] = keras_layer['config']['collapse']
    layer['mean_by_nvert'] = keras_layer['config']['mean_by_nvert']
    if keras_layer['config']['quantize_transforms']:
        layer['quantizer'] = TernaryQuantizer()

    layer['n_aggregators'] = keras_layer['config']['n_aggregators']
    layer['n_out_features'] = keras_layer['config']['n_filters']  # number of output features
    layer['n_propagate'] = keras_layer['config']['n_propagate']  # number of latent features

    if layer['class_name'] == 'GarNet':
        layer['n_in_features'] = input_shapes[0][2]
        n_out_features = layer['n_out_features']

        weights_source = [
            'FLR_kernel',
            'FLR_bias',
            'S_kernel',
            'S_bias',
            'Fout_kernel',
            'Fout_bias',
        ]
        for weight in weights_source:
            layer[weight + '_data'] = get_weights_data(data_reader, layer['name'], weight)

    elif layer['class_name'] == 'GarNetStack':
        layer['n_sublayers'] = keras_layer['config']['n_sublayers']
        layer['n_in_features'] = [input_shapes[0][2]]

        for il in range(layer['n_sublayers']):
            if il > 0:
                layer['n_in_features'].append(layer['n_out_features'][il - 1])

            weights_source = [
                f'FLR{il}_kernel',
                f'FLR{il}_bias',
                f'S{il}_kernel',
                f'S{il}_bias',
                f'Fout{il}_kernel',
                f'Fout{il}_bias',
            ]
            for weight in weights_source:
                layer[weight + '_data'] = get_weights_data(data_reader, layer['name'], weight)

        n_out_features = layer['n_out_features'][-1]

    if layer['collapse'] in ['mean', 'sum', 'max']:
        output_shape = [input_shapes[0][0], n_out_features]
    else:
        output_shape = input_shapes[0][:2] + [n_out_features]

    return layer, output_shape
