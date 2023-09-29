import hls4ml
import yaml
import numpy as np
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
np.savetxt('tb_data/tb_input_features.dat', np.array(test_images.reshape(test_images.shape[0], -1), dtype='int32'), fmt='%d')
np.savetxt('tb_data/tb_output_predictions.dat', np.array(test_labels.reshape(test_labels.shape[0], -1), dtype='int32'), fmt='%d')

print(hls4ml.__version__)
with open('qkeras/config.yml', 'r') as ymlfile:
  config = yaml.safe_load(ymlfile)

print('NETWORK')
print(config)
config['OutputDir'] = 'my-Vivado-test'
config['InputData'] = 'tb_data/tb_input_features.dat'
config['OutputPredictions'] = 'tb_data/tb_output_predictions.dat'
config['Backend'] = 'Vivado'
config['IOType'] = 'io_stream'
hls_model = hls4ml.converters.keras_to_hls(config)
hls_model.build()

