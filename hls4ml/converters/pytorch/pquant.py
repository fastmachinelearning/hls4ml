from hls4ml.converters.pytorch.core import parse_activation_layer
from hls4ml.converters.pytorch_to_hls import pytorch_handler


@pytorch_handler('QuantizedActivationTorchWrapper')
def parse_qactivation_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):

    layer, output_shape = parse_activation_layer(
        class_object.activation.name,
        layer_name,
        input_names,
        input_shapes,
        node,
        class_object.activation,
        data_reader,
        config,
    )
    layer['quantization_parameters'] = class_object.activation.quantization_parameters
    layer['class_name'] = 'QActivation'

    return layer, output_shape
