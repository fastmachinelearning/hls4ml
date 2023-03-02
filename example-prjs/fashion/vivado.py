import hls4ml
import yaml
import numpy as np

print(hls4ml.__version__)
with open('qkeras/config.yml', 'r') as ymlfile:
  config = yaml.safe_load(ymlfile)

print('NETWORK')
print(config)
config['OutputDir'] = 'my-Vivado-test'
config['Backend'] = 'Vivado'
config['IOType'] = 'io_stream'
hls_model = hls4ml.converters.keras_to_hls(config)
hls_model.build()

