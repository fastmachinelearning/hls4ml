import hls4ml
from hls4ml.backends import VitisAcceleratorConfig

# Fetch a keras model from our example repository
# This will download our example model to your working directory and return an example configuration file
config = hls4ml.utils.fetch_example_model('KERAS_3layer.json')

# You can print it to see some default parameters
#print(config)

# Default config:
#{
#    'OutputDir': 'my-hls-test',
#    'ProjectName': 'myproject',
#    'Backend': 'Vivado',
#    'Part': 'xcku115-flvb2104-2-i',
#    'ClockPeriod': 5,
#    'IOType': 'io_parallel',
#    'HLSConfig': {
#        'Model': {
#            'Precision': 'ap_fixed<16,6>',
#            'ReuseFactor': 1
#        }
#    },
#    'KerasJson': 'KERAS_3layer.json',
#    'KerasH5': 'KERAS_3layer_weights.h5'
#}

config['OutputDir'] = 'prj_tuto_vitis_stream'
config['ProjectName'] = 'PrjTutoVitisStream'
config['Backend'] = 'VitisAccelerator'
config['Part'] = 'xcu50-fsvh2104-2-e'
config['IOType'] = 'io_stream'
config['AcceleratorConfig'] = {}  # Initialize AcceleratorConfig as a dictionary
config['AcceleratorConfig']['Platform'] = 'xilinx_u50_gen3x16_xdma_5_202210_1'

print("[K] Print config after setting up backend:")
print(config)

# Convert it to a hls project
hls_model = hls4ml.converters.keras_to_hls(config)

## Print full list of example model if you want to explore more
#hls4ml.utils.fetch_example_list()

# Use Vivado HLS to synthesize the model
# This might take several minutes
hls_model.build()

# Print out the report if you want
hls4ml.report.read_vivado_report(' prj_tuto_vitis_stream')
