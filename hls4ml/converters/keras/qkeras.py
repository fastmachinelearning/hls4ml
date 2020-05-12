from hls4ml.model.hls_model import Quantizer
from hls4ml.model.hls_model import IntegerPrecisionType
from hls4ml.model.hls_model import FixedPrecisionType

from qkeras.quantizers import get_quantizer
import tensorflow as tf

class QKerasQuantizer(Quantizer):
    def __init__(self, config):
        self.quantizer_fn = get_quantizer(config)
        self.bits = config['config']['bits']
        self.hls_type = get_type(config)
    
    def __call__(self, data):
        tf_data = tf.convert_to_tensor(data)
        return self.quantizer_fn(tf_data).numpy()
        #return self.quantizer_fn(data)

def get_type(quantizer_config):
    width = quantizer_config['config']['bits']
    integer = quantizer_config['config'].get('integer', 0)
    if width == integer:
        if width == 1:
            return IntegerPrecisionType(width=1, signed=False)
        else:
            return IntegerPrecisionType(width=width, signed=True)
    else:
        return FixedPrecisionType(width=width+1, integer=integer+1, signed=True)

def get_quantizer_from_config(keras_layer, quantizer_var):
    quantizer_config = keras_layer['config']['{}_quantizer'.format(quantizer_var)]

    return QKerasQuantizer(quantizer_config)

@keras_handler('QDense')
def parse_qdense_layer(keras_layer, input_names, input_shapes, data_reader, config):
    
    
    layer, output_shape = parse_dense_layer(keras_layer, input_names, input_shapes, data_reader, config)

    layer['weight_quantizer'] = get_quantizer_from_config(keras_layer, 'kernel')
    layer['bias_quantizer'] = get_quantizer_from_config(keras_layer, 'bias')

    return layer, output_shape


@keras_handler('QConv1D', 'QConv2D')
def parse_qconv_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('QConv' in keras_layer['class_name'])
    
    if int(keras_layer['class_name'][-2]) == 1:
        layer, output_shape = parse_conv1d_layer(keras_layer, input_names, input_shapes, data_reader, config)
    elif int(keras_layer['class_name'][-2]) == 2:
        layer, output_shape = parse_conv2d_layer(keras_layer, input_names, input_shapes, data_reader, config)

    layer['weight_quantizer'] = get_quantizer_from_config(keras_layer, 'kernel')
    layer['bias_quantizer'] = get_quantizer_from_config(keras_layer, 'bias')

    return layer, output_shape


@keras_handler('QActivation')
def parse_qactivation_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer['class_name'] == 'QActivation')
    supported_activations = ['quantized_relu', 'quantized_tanh']
    
    layer = parse_default_keras_layer(keras_layer, input_names)

    activation_config = keras_layer['config']['activation']
    
    act_class = activation_config['class_name']
    if act_class not in supported_activations:
        raise Exception('Unsupported QKeras activation: {}'.format(act_class))

    layer['class_name'] = 'Activation'
    layer['activation'] = act_class.replace('quantized_', '')
    layer['bits'] = activation_config['config']['bits'] + 1
    layer['integer'] = activation_config['config']['integer'] + 1
    #TODO this needs extra work in HLS model and HLS templates

    return layer, [shape for shape in input_shapes[0]]

