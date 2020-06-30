from urllib.request import urlretrieve
from .config import create_vivado_config

def fetch_example_model(model_name):
    """
    Download an example model from github repo to working directory, and return the corresponding configuration:

    https://github.com/hls-fpga-machine-learning/example-models

    Args:
        - model_name: string, name of the example model in the repo. Example: 'keras_3_layer.h5'
    
    """

    #Initilize the download link and model type
    download_link = 'https://raw.githubusercontent.com/hls-fpga-machine-learning/example-models/master/'
    model_type = None
    model_config = None

    #Check for model's type to update link
    if '.h5' in model_name:
        model_type = 'keras'
        model_config = 'KerasH5'
    elif '.pt' in model_name:
        model_type = 'pytorch'
        model_config = 'PytorchModel'
    elif '.onnx' in model_name:
        model_type = 'onnx'
        model_config ='OnnxModel'
    elif '.pb' in model_name:
        model_type = 'tensorflow'
        model_config = 'TensorFlowModel'
    else:
        raise TypeError('Model type is not supported in hls4ml.')
    

    download_link += model_type + '/' + model_name

    #Initiate the configuration file
    config = create_vivado_config()
        
    #Download the example model
    urlretrieve(download_link, model_name)

    #Additional configuration parameters
    config[model_config] = model_name
    config['HLSConfig']['Model'] = {}
    config['HLSConfig']['Model']['Precision'] = 'ap_fixed<16,6>'
    config['HLSConfig']['Model']['ReuseFactor'] = '1'
    
    return config