import hls4ml
# import pprint
import yaml
import numpy as np

print(hls4ml.__version__)

print("\n============================================================================================")
print("Configuring HLS4ML")
with open('Vivado_con2d_config.yml', 'r') as ymlfile:
  config = yaml.safe_load(ymlfile)

print("\n============================================================================================")
print("HLS4ML converting keras model to HLS C++")
hls_model = hls4ml.converters.keras_to_hls(config)

print("\n============================================================================================")
print("Building HLS C++ model")
# hls_model.build(csim=True, synth=True, cosim=True, validation=True, vsynth=True)
hls_model.build()

