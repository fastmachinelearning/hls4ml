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
