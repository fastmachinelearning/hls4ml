from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

import hls4ml

test_root_path = Path(__file__).parent


# Keras implementation of a custom layer
class KReverse(tf.keras.layers.Layer):
    '''Keras implementation of a hypothetical custom layer'''

    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return tf.reverse(inputs, axis=[-1])

    def get_config(self):
        # Breaks serialization and parsing in hls4ml if not defined
        return super().get_config()


# hls4ml layer implementation
class HReverse(hls4ml.model.layers.Layer):
    '''hls4ml implementation of a hypothetical custom layer'''

    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims)


# hls4ml optimizer to remove duplicate optimizer
class RemoveDuplicateReverse(hls4ml.model.optimizer.OptimizerPass):
    '''OptimizerPass to remove consecutive HReverse layers.'''

    def match(self, node):
        return isinstance(node, HReverse) and isinstance(node.get_input_node(), HReverse)

    def transform(self, model, node):
        first = node.get_input_node()
        second = node

        model.remove_node(first, rewire=True)
        model.remove_node(second, rewire=True)
        return True


# Parser for converter
def parse_reverse_layer(keras_layer, input_names, input_shapes, data_reader):
    layer = {}
    layer['class_name'] = 'HReverse'
    layer['name'] = keras_layer['config']['name']
    layer['n_in'] = input_shapes[0][1]

    if input_names is not None:
        layer['inputs'] = input_names

    return layer, [shape for shape in input_shapes[0]]


# HLS Templates - No specific pragmas used; generic enough for both Intel and Vivado

rev_config_template = """struct config{index} : nnet::reverse_config {{
    static const unsigned n_in = {n_in};
}};\n"""

rev_function_template = 'nnet::reverse<{input_t}, {config}>({input}, {output});'
rev_include_list = ['nnet_utils/nnet_reverse.h']


class HReverseConfigTemplate(hls4ml.backends.template.LayerConfigTemplate):
    def __init__(self):
        super().__init__(HReverse)
        self.template = rev_config_template

    def format(self, node):
        params = self._default_config_params(node)
        return self.template.format(**params)


class HReverseFunctionTemplate(hls4ml.backends.template.FunctionCallTemplate):
    def __init__(self):
        super().__init__(HReverse, include_header=rev_include_list)
        self.template = rev_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)


rev_hls = """#ifndef NNET_REVERSE_H_
#define NNET_REVERSE_H_

#include "nnet_common.h"

namespace nnet {

struct reverse_config {
    static const unsigned n_in = 10;
};

template<class data_T, typename CONFIG_T>
void reverse(
    data_T input[CONFIG_T::n_in],
    data_T reversed[CONFIG_T::n_in]
) {
    for (int i = 0; i < CONFIG_T::n_in; i++) {
        reversed[CONFIG_T::n_in - 1 - i] = input[i];
    }
}

}

#endif
"""


@pytest.fixture(scope='session', autouse=True)
def register_custom_layer():
    # Register the converter for custom Keras layer
    hls4ml.converters.register_keras_layer_handler('KReverse', parse_reverse_layer)

    # Register the hls4ml's IR layer
    hls4ml.model.layers.register_layer('HReverse', HReverse)


@pytest.mark.parametrize('backend_id', ['Vivado', 'Vitis', 'Quartus'])
def test_extensions(tmp_path, backend_id):
    # Register the optimization passes (if any)
    backend = hls4ml.backends.get_backend(backend_id)
    ip_flow = hls4ml.model.flow.get_flow(backend.get_default_flow())
    # Add the pass into the main optimization flow
    optimize_flow = [flow for flow in ip_flow.requires if ':optimize' in flow][0]
    optmizer_name = f'{backend_id.lower()}:remove_duplicate_reverse'
    backend.register_pass(optmizer_name, RemoveDuplicateReverse, flow=optimize_flow)

    # Register template passes for the given backend
    backend.register_template(HReverseConfigTemplate)
    backend.register_template(HReverseFunctionTemplate)

    # Register HLS implementation
    p = tmp_path / 'nnet_reverse.h'
    p.write_text(rev_hls)
    backend.register_source(p)

    # Test if it works
    kmodel = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(8,)),
            KReverse(),
            tf.keras.layers.ReLU(),
            # These two should be removed by the optimizer
            KReverse(),
            KReverse(),
        ]
    )

    x = np.random.randint(-5, 5, (8,), dtype='int32')
    kres = kmodel(x)

    hmodel = hls4ml.converters.convert_from_keras_model(
        kmodel,
        output_dir=str(test_root_path / f'hls4mlprj_extensions_{backend_id}'),
        backend=backend_id,
        io_type='io_parallel',
        hls_config={'Model': {'Precision': 'ap_int<6>', 'ReuseFactor': 1}},
    )

    hmodel.compile()
    hres = hmodel.predict(x.astype('float32'))

    # Check if the optimizer pass was applied
    assert optmizer_name in hmodel._applied_flows[0][optimize_flow]

    # Remove flow from
    hls4ml.model.flow.update_flow(optimize_flow, remove_optimizers=[optmizer_name])

    np.testing.assert_array_equal(kres, hres)
