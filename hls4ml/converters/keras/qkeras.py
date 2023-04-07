from hls4ml.model.types import QKerasBinaryQuantizer, QKerasPO2Quantizer, QKerasQuantizer


def get_quantizer_from_config(keras_layer, quantizer_var):
    quantizer_config = keras_layer['config'][f'{quantizer_var}_quantizer']
    if keras_layer['class_name'] == 'QBatchNormalization':
        return QKerasQuantizer(quantizer_config)
    elif 'binary' in quantizer_config['class_name']:
        return QKerasBinaryQuantizer(quantizer_config, xnor=(quantizer_var == 'kernel'))
    elif quantizer_config['class_name'] == 'quantized_po2':
        return QKerasPO2Quantizer(quantizer_config)
    else:
        return QKerasQuantizer(quantizer_config)
