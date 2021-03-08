import numpy as np
import random as rnd

import tensorflow
from qkeras.utils import load_qmodel
from qkeras import QConv2DBatchnorm, QActivation, quantized_relu, QDense, quantized_bits, to_categorical, \
    QBatchNormalization, QConv2D
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1

import hls4ml
import yaml

from notebooks.callbacks import all_callbacks

from scipy.io import loadmat

rnd.seed(42)
height = 32
width = 32
chan = 3
input_shape = (height, width, chan)
num_classes = 5
sparcity = 0.
int_bits = 6

model = Sequential()
model.add(Input(shape=input_shape))
model.add(QConv2D(8, (3, 3), kernel_quantizer=quantized_bits(6, 0, alpha=1),
                  bias_quantizer=quantized_bits(6, 0, alpha=1), name='qconv2d'))
model.add((QBatchNormalization(name='qbatchnorm', beta_quantizer=quantized_bits(6, 0, alpha=1),
                               gamma_quantizer=quantized_bits(6, 0, alpha=1),
                               mean_quantizer=quantized_bits(6, 0, alpha=1),
                               variance_quantizer=quantized_bits(6, 0, alpha=1))))
model.add(QActivation(activation=quantized_relu(6), name='relu1'))
model.add(QConv2DBatchnorm(16, (3, 3), kernel_quantizer=quantized_bits(6, 0, alpha=1),
                           bias_quantizer=quantized_bits(6, 0, alpha=1), name='qconv2dbatchnorm'))
model.add(QActivation(activation=quantized_relu(6), name='relu2'))
model.add(Flatten())
model.add(QDense(10, name='output',
                 kernel_quantizer=quantized_bits(6, 0, alpha=1), bias_quantizer=quantized_bits(6, 0, alpha=1),
                 kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
model.add(Activation(activation='softmax', name='softmax'))
print(model.summary())

'''
# Generate some weights with some sparcity
for layer in model.layers:
     old_weights = layer.get_weights()
     if len(old_weights) > 0:
          new_weights = []
          for w in old_weights:
               print(layer.name, w.shape)
               n_zeros = 0
               if sparcity > 0:
                    n_zeros = int(sparcity * np.prod(w.shape))
               if n_zeros > 0:
                    zero_indices = rnd.sample(range(1, np.prod(w.shape)), n_zeros)
               else:
                    zero_indices = []
               new_w = []
               for i in range(np.prod(w.shape)):
                    if i in zero_indices:
                         new_w.append(0)
                    else:
                         #new_w.append(rnd.randint(1, 2**(int_bits - 1)))
                         #new_w.append(rnd.randint(1, 10))
                         new_w.append(rnd.uniform(1, 3))
               new_w = np.asarray(new_w).reshape(w.shape)
               new_weights.append(new_w)
          layer.set_weights(new_weights)
'''

train = loadmat('svhndataset/train_32x32.mat')
train_img = np.array(train['X'])
train_label = train['y']
train_img = np.moveaxis(train_img, -1, 0)
train_label[train_label == 10] = 0
train_img = train_img / 255.0

train_label = to_categorical(train_label)

train = False
if train:
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    callbacks = all_callbacks(stop_patience=1000,
                              lr_factor=0.5,
                              lr_patience=10,
                              lr_epsilon=0.000001,
                              lr_cooldown=2,
                              lr_minimum=0.0000001,
                              outputDir='model')
    model.fit(train_img, train_label, batch_size=1024,
              epochs=1, validation_split=0.25, shuffle=True,
              callbacks=callbacks.callbacks)
    model.save('model/KERAS_check_best_model.h5')
else:
    model = load_qmodel('model/KERAS_check_best_model.h5')

# Let's test it out
# np.random.seed(42)

# predictions = model.predict(x)
# print(predictions.flatten())
# Save it
model.save('dummy_conv2dbatchnorm.h5')
# model_json = model.to_json()
# with open("dummy_cnn.json", "w") as json_file:
#    json_file.write(model_json)
# model.save_weights("dummy_cnn_weights.h5")
# Now hls4ml-ify it
yaml_config = {}

# yaml_config['KerasH5'] = '/home/vloncar/work/CERN/FPGA/hls4ml/example-models/dummy_cnn/dummy_cnn.h5'
yaml_config['KerasModel'] = model
yaml_config['OutputDir'] = '/home/nicolo/CERN-working-dir/hls4ml/qconv2dbatchnorm_test/hls'
yaml_config['ProjectName'] = 'myproject'
yaml_config['XilinxPart'] = 'xcvu9p-flgb2104-2-e'
yaml_config['ClockPeriod'] = 5
# yaml_config['IOType'] = 'io_parallel'
yaml_config['IOType'] = 'io_stream'
yaml_config['HLSConfig'] = {
    'Model': {
        'Precision': 'ap_fixed<16,6>',
        'ReuseFactor': 1,
        'Strategy': 'Resource'
    },
    # 'LayerName': {
    #    'conv2d_3': {'Strategy': 'Resource', 'ReuseFactor': 100},
    #    'conv2d_4': {'Strategy': 'Resource', 'ReuseFactor': 100},
    #    'conv2d_6': {'Strategy': 'Resource', 'ReuseFactor': 100},
    #    'conv2d_7': {'Strategy': 'Resource', 'ReuseFactor': 100},
    #    'conv2d_8': {'Strategy': 'Resource', 'ReuseFactor': 100},
    # }
}
# Convert it
config = hls4ml.utils.config.config_from_keras_model(model, granularity='name')
config['SkipOptimizers'] = ['FuseBatchNormalization']
hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)
# hls_model = hls4ml.converters.keras_to_hls(yaml_config)
# Test it
np.random.seed(42)  # We need this again
# x = np.random.rand(np.prod(model.input.shape[1:])).reshape(model.input.shape[1:])
hls_model.write()
hls_model.compile()

# x = np.random.rand(np.prod(input_shape)).reshape(input_shape)
x = np.ones(np.prod(input_shape)).reshape(input_shape)
x = np.expand_dims(x, axis=0)

y = hls_model.predict(x)
y_ = model.predict(x)

hls_weights = hls_model.graph.get('qconv2dbatchnorm').weights.get('weight').data_unquantized
mod_weights = model.layers[3].get_folded_weights()[0]

hls_bias = hls_model.graph.get('qconv2dbatchnorm').weights.get('bias').data_unquantized
mod_bias = model.layers[3].get_folded_weights()[1]

if tensorflow.equal(hls_weights, mod_weights).numpy().all():
    print('Weights OK!')

if tensorflow.equal(hls_bias, mod_bias).numpy().all():
    print('Bias OK!')

print(y)
print(y_)
# Build it
# report = hls_model.build(csim=True, synth=True, cosim=False, reset=True)
# hls4ml.report.read_vivado_report(yaml_config['OutputDir'])
# report = hls4ml.report.parse_vivado_report(yaml_config['OutputDir'])
# print(report)