from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler

@keras_handler('KLLoss')
def parse_klloss_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('KLLoss' in keras_layer['class_name'])

    layer = parse_default_keras_layer(keras_layer, input_names)
    
    output_shape = [input_shapes[0][0], 1]

    return layer, output_shape

@keras_handler('CustomMSE')
def parse_klloss_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('CustomMSE' in keras_layer['class_name'])

    layer = parse_default_keras_layer(keras_layer, input_names)

    output_shape = [input_shapes[0][0], 1]

    return layer, output_shape
