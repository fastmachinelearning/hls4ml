import hls4ml
# import pprint
import yaml
import numpy as np

print(hls4ml.__version__)

with open('keras/qkeras_3layer_config_serial_ccs.yml', 'r') as ymlfile:
  config = yaml.safe_load(ymlfile)

# try tweaking the reuse_factor on one layer to get different pipelining
# config['HLSConfig']['LayerName']['fc1']['ReuseFactor'] = 4

print('NETWORK')
print(config)

config['OutputDir'] = 'my-Vivado-test'
config['Backend'] = 'Vivado'
config['IOType'] = 'io_stream'

# default threshold is infinity
config['HLSConfig']['Model']['BramFactor'] = np.inf
# set to zero to force all weights onto (external function) interface
config['HLSConfig']['Model']['BramFactor'] = 0

print('CURRENT CONFIGURATION')
print('Backend='+config['Backend'])
print('IOType='+config['IOType'])
print('BramFactor={bf}'.format(bf=config['HLSConfig']['Model']['BramFactor']))

# pprint.pprint(config)

#Convert it to a hls project
hls_model = hls4ml.converters.keras_to_hls(config)

hls_model.build(vsynth=True)

# URL for this info: https://fastmachinelearning.org/hls4ml/setup/QUICKSTART.html
