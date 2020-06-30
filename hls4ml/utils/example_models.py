import os
from urllib.request import urlretrieve
from .config import create_vivado_config

def fetch_example_model(model_name):
    """
    Download an example model from github repo to working directory, and return the corresponding configuration:

    https://github.com/hls-fpga-machine-learning/example-models

    Args:
        - model_name: string, name of the example model, one of the followings:
            * 'keras_3layer'
    
    """

    #Initiate the configuration file
    config = create_vivado_config()
        
    #Download the example model
    if model_name == 'keras_3layer':
        urlretrieve('https://raw.githubusercontent.com/hls-fpga-machine-learning/example-models/master/keras/keras_3layer.h5', 'keras_3layer.h5')

        #Additional configuration parameters
        config['KerasH5'] = 'keras_3layer.h5'
        config['HLSConfig']['Model'] = {}
        config['HLSConfig']['Model']['Precision'] = 'ap_fixed<16,6>'
        config['HLSConfig']['Model']['ReuseFactor'] = '1'

    #Add more models to the if statement if you want to
    
    return config