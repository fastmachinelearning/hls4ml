import argparse
import pickle
import hls4ml
import tensorflow as tf
from tensorflow.python.keras.layers.merge import _Merge as Merge
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
import numpy as np
import yaml
from pathlib import Path
from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.model.types import NamedType, FixedPrecisionType, ExponentPrecisionType

test_root_path = Path(__file__).parent

#tf.compat.v1.disable_eager_execution()

# Keras implementation of a custom layer

class Distance(Merge):
    def _check_inputs(self, inputs):
        if len(inputs) not in  [2,3]:
            raise ValueError('A `{}` layer should be called '
                             'on exactly 2 or 3 inputs'.format(self.__class__.__name__))

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        super(Distance, self).build(input_shape)
        self._check_inputs(input_shape)

class KLLoss(Distance):
    ''' Keras implementation of a KL loss custom layer '''
    # def __init__(self):
    #     super(KLLoss, self).__init__()

    def _merge_function(self, inputs):
        self._check_inputs(inputs)

        mean = inputs[0]
        log_var = inputs[1]

        kl = 1. + log_var - math_ops.square(mean) - math_ops.exp(log_var)
        kl = -0.5 * math_ops.reduce_mean(kl, axis=-1, keepdims=True)

        return kl

# hls4ml implementations

class HKLLoss(hls4ml.model.layers.Layer):
    ''' hls4ml implementation of a KL loss custom layer '''

    def initialize(self):
        assert(len(self.inputs) == 2)
        self.add_output_variable(shape=[1], dim_names=['KL_LOSS_{}'.format(self.index)])

        print(self.attributes)
        if 'sum_t' not in self.attributes:
            self.set_attr('sum_t', self.get_attr('accum_t'))
        if 'exp_table_t' not in self.attributes:
            self.set_attr('exp_table_t', NamedType(name=self.name + '_exp_table_t', precision=FixedPrecisionType(width=18, integer=8)))
        if 'table_size' not in self.attributes:
            self.set_attr('table_size', 1024)
        if 'exp_range' not in self.attributes:
            self.set_attr('exp_range', 8)

# Templates
distance_config_template = """struct config{index} : nnet::distance_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = 1;
    typedef {accum_t.name} accum_t;
    typedef {sum_t.name} sum_t;
    typedef {exp_table_t.name} exp_table_t;
    static const unsigned table_size = {table_size};
    static constexpr float exp_range = {exp_range};
}};\n"""
distance_function_template = 'nnet::{distance}<{input1_t}, {input2_t}, {output_t}, {config}>({input1}, {input2}, {output});'
distance_include_list = ['nnet_utils/nnet_distance.h']

class HKLLossConfigTemplate(hls4ml.backends.template.LayerConfigTemplate):
    def __init__(self):
        super().__init__(HKLLoss)
        self.template = distance_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['n_in'] = node.get_input_variable(node.inputs[0]).shape[0]
        params['n_out'] = 1
        return self.template.format(**params)

class HKLLossFunctionTemplate(hls4ml.backends.template.FunctionCallTemplate):
    def __init__(self):
        super().__init__(HKLLoss, include_header=distance_include_list)
        self.template = distance_function_template

    def format(self, node):
        params = {}
        params['distance'] = 'klloss'
        params['config'] = 'config{}'.format(node.index)
        params['input1_t'] = node.get_input_variable(node.inputs[0]).type.name
        params['input2_t'] = node.get_input_variable(node.inputs[1]).type.name
        params['output_t'] = node.get_output_variable().type.name
        params['input1'] = node.get_input_variable(node.inputs[0]).name
        params['input2'] = node.get_input_variable(node.inputs[1]).name
        params['output'] = node.get_output_variable().name

        return self.template.format(**params)


# Parser for converter
def parse_klloss_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('KLLoss' in keras_layer['class_name'])

    layer = parse_default_keras_layer(keras_layer, input_names)

    output_shape = [input_shapes[0][0], 1]

    return layer, output_shape

def test_extensions(tmp_path, dnn):
    # Register the converter for custom Keras layer
    hls4ml.converters.register_keras_layer_handler('KLLoss', parse_klloss_layer)

    # Register the hls4ml's IR layer
    hls4ml.model.layers.register_layer('KLLoss', HKLLoss)

    # Register the optimization passes (if any)
    backend = hls4ml.backends.get_backend('Vivado')

    # Register template passes for the given backend
    backend.register_template(HKLLossConfigTemplate)
    backend.register_template(HKLLossFunctionTemplate)

    # Register HLS implementation
    backend.register_source('nnet_distance.h')

    if dnn:
        model_file = 'output/custom-dnn_vae.h5'
        config_file = 'output/custom-dnn_vae.pickle'
    else:
        model_file = 'output/custom-ptq-conv_vae-8-b0.8-q0-pruned.h5'
        config_file = 'hls/ptq-conv_vae-8-b0.8-q0-pruned/config.pickle'

    # Test if it works
    kmodel = tf.keras.models.load_model(model_file,
          custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
          'KLLoss': KLLoss})

    kmodel.summary()
    print(f'if dnn {dnn}')

    x = np.random.randint(-5, 5, (1, 19,3,1), dtype='int32')
    if dnn: x=x.reshape((1,-1))
    kres = kmodel(x)

    # load config
    with open(config_file, 'rb') as handle:
        config = pickle.load(handle)
    #for layer in config['LayerName'].keys():
    #     config['LayerName'][layer]['Trace'] = True
    config = {}

    config['Model'] = {
        'Precision': 'ap_fixed<16,6>',
        'ReuseFactor': 1,
        'ParallelizationFactor': 1,
        'Strategy': 'Resource',
    }
    config['LayerName'] = {
        'conv2d': {
            'ParallelizationFactor': 9,
            'ReuseFactor': 3,
            'Strategy': 'Latency',
        },
        'conv2d_1': {
            'ParallelizationFactor': 2,
            'ReuseFactor': 4,
            'Strategy': 'Latency',
        }
    }
    print(yaml.dump(config, default_flow_style=False))

    hmodel = hls4ml.converters.convert_from_keras_model(
        kmodel,
        output_dir=str(test_root_path / 'hls4mlprj_extensions'),
        backend='Vivado',
        io_type='io_parallel',
        part='xcvu9p-flga2577-2-e',
        hls_config=config)

    hmodel.compile()
    hres = hmodel.predict(x.astype('float32'))

    #np.testing.assert_array_equal(kres, hres)

    print('Building model')
    report = hmodel.build(reset=True, csim=False, cosim=True, synth=True, vsynth=True)
    print(report)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn', action='store_true')
    args = parser.parse_args()
    test_extensions(test_root_path, **vars(args))