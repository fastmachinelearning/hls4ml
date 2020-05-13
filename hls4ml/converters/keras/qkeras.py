from hls4ml.model.hls_model import Quantizer
from hls4ml.model.hls_model import IntegerPrecisionType
from hls4ml.model.hls_model import FixedPrecisionType

from qkeras.quantizers import get_quantizer
import tensorflow as tf

class QKerasQuantizer(Quantizer):
    def __init__(self, config):
        self.quantizer_fn = get_quantizer(config)
        self.alpha = config['config']['alpha']
        if config['class_name'] == 'quantized_bits':
            self.bits = config['config']['bits']
            self.hls_type = get_type(config)
        # ! includes stochastic_ternary
        elif 'ternary' in config['class_name']:
            self.bits = 2
            self.hls_type = IntegerPrecisionType(width=2, signed=False)
        # ! includes stochastic_binary
        elif 'binary' in config['class_name']:
            self.bits = 1
            self.hls_type = IntegerPrecisionType(width=1, signed=False)
        elif config['class_name'] == 'quantized_po2':
            self.bits = config['config']['bits']
            self.hls_type = Po2Type(width=self.bits, signed=True)
        else:
            print("Unsupported quantizer: " + config['class_name'])
            self.bits = 16
            self.hls_type = FixedPrecisionType(width=16, integer=6, signed=True)
    
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

