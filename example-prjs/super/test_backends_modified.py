import os
import sys
import numpy as np

from tensorflow import keras
from sklearn.datasets import fetch_openml
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PATH'] = '/tools/Xilinx/Vivado/2019.1/bin:' + os.environ['PATH']

def print_dict(d, indent=0):
    align=20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent+1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))


# Choose a target board/chip

#BOARD_NAME = 'pynq-z1'
#FPGA_PART = 'xc7z020clg400-1'
#CLOCK_PERIOD = 10
##
#BOARD_NAME = 'arty-a7-100t'
#FPGA_PART = 'xc7a100tcsg324-1'
#CLOCK_PERIOD = 10
#
BOARD_NAME = 'ultra96v2'
FPGA_PART = 'xczu3eg-sbva484-1-e'
CLOCK_PERIOD = 5

# Load and scale dataset
DATA_DIR = 'npy'
data = fetch_openml('hls4ml_lhc_jets_hlf')
X, y = data['data'], data['target']
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y, 5)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test = scaler.transform(X_test)
np.save(DATA_DIR + '/y_test.npy', y_test)
np.save(DATA_DIR + '/X_test.npy', X_test)
np.save(DATA_DIR + '/classes.npy', le.classes_, allow_pickle=True)

# Load pre-trained quantized model
MODEL_DIR = 'model'
WEIGHTS = MODEL_DIR + '/qkeras_3layer_weights.h5'
DESC = MODEL_DIR + '/qkeras_3layer.json'
co = {}
from qkeras.utils import _add_supported_quantized_objects
_add_supported_quantized_objects(co)

with open(DESC) as fl:
    desc = "\n".join(fl.readlines())
model = keras.models.model_from_json(desc, custom_objects=co)
model.load_weights(WEIGHTS)
#model.summary()

# Run QKeras model prediction
y_qkeras = model.predict(X_test)
np.save(DATA_DIR + '/y_qkeras.npy', y_qkeras)

# Setup rounding and saturation modes on activation layers
import hls4ml
hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(
        layers=['Activation'],
        rounding_mode='AP_RND_CONV',
        saturation_mode='AP_SAT')

# Get hls4ml configuration
config = hls4ml.utils.config_from_keras_model(model, granularity='name')
#print("-----------------------------------")
#print("Configuration")
#print_dict(config)
#print("-----------------------------------")

# Setup hls4ml configuration
DEF_RF = 64
config["Model"]["Strategy"] = "Resource"
config["Model"]["ReuseFactor"] = DEF_RF
config["Model"]["BramFactor"] = 0
for layer in config["LayerName"]:
    config["LayerName"][layer]["ReuseFactor"] = DEF_RF
#print("-----------------------------------")
#print("Configuration")
#print_dict(config)
#print("-----------------------------------")

BACKEND='VivadoAccelerator'

# Get hls4ml model

ENABLE_AXI_WRAPPER=False
if BACKEND == 'VivadoAccelerator':
    ENABLE_AXI_WRAPPER=True

if ENABLE_AXI_WRAPPER:
    OUTPUT_DIR = 'wrapped_qresource64-'+BACKEND
    hls_model =  hls4ml.converters.convert_from_keras_model(
            model=model,
            clock_period=CLOCK_PERIOD,
            backend=BACKEND,
            board=BOARD_NAME,
            part=FPGA_PART,
            io_type='io_stream',
            interface='axi_master',
            driver='c',
            input_data_tb=DATA_DIR+'/X_test.npy',
            output_data_tb=DATA_DIR+'/y_test.npy',
            hls_config=config,
            output_dir=OUTPUT_DIR)
else:
    OUTPUT_DIR = 'qresource64-'+BACKEND
    hls_model = hls4ml.converters.convert_from_keras_model(
            model=model,
            clock_period=CLOCK_PERIOD,
            backend=BACKEND,
            hls_config=config,
            part=FPGA_PART,
            io_type='io_stream',
            output_dir=OUTPUT_DIR)

hls_model.compile()

# Run hls4ml model prediction
y_hls = hls_model.predict(np.ascontiguousarray(X_test[:128]))

# Print some predictions
#for i in range(0,1):
#    print("[", i, "]")
#    print("   - Reference: ", y_test[i])
#    print("   - QKeras   : ", y_qkeras[i])
#    print("   - hls4ml   : ", y_hls[i])

if len(sys.argv) == 2 and sys.argv[1] == 'profile':
    print('Number of arguments:', len(sys.argv), 'arguments.')

    from sklearn.metrics import accuracy_score
    print('-----------------------------------')
    print('QKeras Accuracy: {}'.format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_qkeras, axis=1))))
    print('hls4ml Accuracy: {}'.format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))))
    print('-----------------------------------')
else:

    if BACKEND == 'VivadoAccelerator':
        results = hls_model.build(csim=False, synth=True, vsynth=False, export=True, bitfile=True)
    else:
        results = hls_model.build(csim=False, synth=True, vsynth=False, export=True)

    # Show reports
    #hls4ml.report.read_vivado_report(OUTPUT_DIR)

    # Write header files with hardcoded data set
    if BACKEND == 'VivadoAccelerator':
        hls4ml.writer.vivado_accelerator_writer.VivadoAcceleratorWriter.write_header_file(hls_model, X_test, y_test, y_qkeras, y_hls, 64, OUTPUT_DIR + '/sdk/common/data.h')

