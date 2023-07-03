from hls4ml.converters.keras_to_hls import (
    KerasFileReader,
    KerasModelReader,
    KerasNestedFileReader,
    keras_handler,
    parse_default_keras_layer,
    parse_keras_model,
)

model_layers = ['Sequential', 'Functional']


@keras_handler(*model_layers)
def parse_model_layer(keras_layer, input_names, input_shapes, data_reader):
    assert keras_layer['class_name'] in model_layers

    layer = parse_default_keras_layer(keras_layer, input_names)
    layer['class_name'] = 'LayerGroup'

    if isinstance(data_reader, KerasNestedFileReader):
        # In the .h5 file, the paths don't go more than one level deep
        nested_path = data_reader.nested_path
    else:
        nested_path = layer['name']

    if isinstance(data_reader, KerasFileReader):
        nested_reader = KerasNestedFileReader(data_reader, nested_path)
    else:
        nested_reader = KerasModelReader(data_reader.model.get_layer(layer['name']))

    layer_list, input_layers, output_layers, output_shapes = parse_keras_model(keras_layer, nested_reader)

    if output_layers is None:
        last_layer = layer_list[-1]['name']
    else:
        last_layer = output_layers[0]
    output_shape = output_shapes[last_layer]

    layer['layer_list'] = layer_list
    layer['input_layers'] = input_layers if input_layers is not None else []
    layer['output_layers'] = output_layers if output_layers is not None else []
    layer['data_reader'] = nested_reader
    layer['output_shape'] = output_shape

    return layer, output_shape
