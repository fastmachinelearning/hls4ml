from __future__ import absolute_import

from .keras_to_hls import keras_to_hls
from ..utils.config import create_vivado_config

try:
    from .pytorch_to_hls import pytorch_to_hls
    __pytorch_enabled__ = True
except ImportError:
    __pytorch_enabled__ = False

try:
    from .onnx_to_hls import onnx_to_hls
    __onnx_enabled__ = True
except ImportError:
    __onnx_enabled__ = False

try:
    from .tf_to_hls import tf_to_hls
    __tensorflow_enabled__ = True
except ImportError:
    __tensorflow_enabled__ = False


def convert_from_yaml_config(yamlConfig):
    model = None
    if 'OnnxModel' in yamlConfig:
        if __onnx_enabled__:
            model = onnx_to_hls(yamlConfig)
        else:
            raise Exception("ONNX not found. Please install ONNX.")
    elif 'PytorchModel' in yamlConfig:
        if __pytorch_enabled__:
            model = pytorch_to_hls(yamlConfig)
        else:
            raise Exception("PyTorch not found. Please install PyTorch.")
    elif 'TensorFlowModel' in yamlConfig:
        if __tensorflow_enabled__:
            model = tf_to_hls(yamlConfig)
        else:
            raise Exception("TensorFlow not found. Please install TensorFlow.")
    else:
        model = keras_to_hls(yamlConfig)
    
    return model

def convert_from_keras_model(model, output_dir='my-hls-test', project_name='myproject',
    fpga_part='xcku115-flvb2104-2-i', clock_period=5, hls_config={}):
    
    config = create_vivado_config(output_dir=output_dir,
        project_name=project_name, fpga_part=fpga_part, clock_period=clock_period)
    config['KerasModel'] = model

    model_config = hls_config.get('Model', None)
    if model_config is not None:
        if not all(k in model_config for k in ('Precision', 'ReuseFactor')):
            raise Exception('Precision and ReuseFactor must be provided in the hls_config')
    else:
        model_config = {}
        model_config['Precision'] = 'ap_fixed<16,6>'
        model_config['ReuseFactor'] = '1'
    config['HLSConfig']['Model'] = model_config
    
    if 'LayerName' in hls_config:
        config['HLSConfig']['LayerName'] = hls_config['LayerName']
    
    if 'LayerType' in hls_config:
        config['HLSConfig']['LayerType'] = hls_config['LayerType']
    
    return keras_to_hls(config)
