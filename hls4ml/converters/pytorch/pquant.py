from hls4ml.converters.pytorch.core import parse_activation_layer
from hls4ml.converters.pytorch.pooling import parse_pooling_layer
from hls4ml.converters.pytorch_to_hls import pytorch_handler
from hls4ml.model.types import FixedPrecisionType


@pytorch_handler('QuantizedActivationTorchWrapper')
def parse_pquant_activation_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):

    layer, output_shape = parse_activation_layer(
        class_object.activation.__class__.__name__,
        layer_name,
        input_names,
        input_shapes,
        node,
        class_object.activation,
        data_reader,
        config,
    )
    layer['quantization_parameters'] = class_object.activation.quantization_parameters

    if (
        layer['activation'] == 'quantizedtanh'
        and not class_object.activation.config['quantization_parameters']['use_real_tanh']
    ):
        layer['class_name'] = 'HardActivation'
        layer['slope'] = 0.5  # the default values in QKeras
        layer['shift'] = 0.5
        # Quartus seems to have trouble if the width is 1.
        layer['slope_prec'] = FixedPrecisionType(width=2, integer=0, signed=False)
        layer['shift_prec'] = FixedPrecisionType(width=2, integer=0, signed=False)
        layer['activation'] = 'hard_tanh'

    elif (
        layer['activation'] == 'quantizedrelu'
        and not layer['quantization_parameters']["use_high_granularity_quantization"]
        and class_object.activation.config['quantization_parameters']['use_relu_multiplier']
    ):
        layer['class_name'] = 'MultiplierReLU'
        layer['param_data'] = class_object.activation.multiplier.numpy()
        layer['activation'] = 'multiplier_relu'

    else:
        layer['class_name'] = 'QActivation'
        activation_map = {
            'quantizedrelu': 'relu',
            'quantizedtanh': 'tanh',
        }
        layer['activation'] = activation_map.get(layer['activation'], layer['activation'])

    return layer, output_shape


@pytorch_handler('QuantizedPooling')
def parse_pquant_pooling_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):

    layer, output_shape = parse_pooling_layer(
        class_object.pooling.__class__.__name__,
        layer_name,
        input_names,
        input_shapes,
        node,
        class_object.pooling,
        data_reader,
        config,
    )
    layer['quantization_parameters'] = class_object.quantization_parameters

    return layer, output_shape
