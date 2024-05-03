========================
Extension API
========================

``hls4ml`` natively supports a large number of neural network layers.
But what if a desired layer is not supported?
If it is standard enough and its implementation would benefit the community as a whole, we would welcome a contribution to add it to the standard set of supported layers.
However, if it is a somewhat niche custom layer, there is another approach we can take to extend hls4ml through the *extension API*.

This documentation will walk through a complete `complete end-to-end example <https://github.com/fastmachinelearning/hls4ml/blob/main/test/pytest/test_extensions.py>`_, which is part of our testing suite.
To implement a custom layer in ``hls4ml`` with the extension API, the required components are:

* Your custom layer class
* Equivalent hls4ml custom layer class
* Parser for the converter
* HLS implementation
* Layer config template
* Function config template
* Registration of layer, source code, and templates

Complete example
================

For concreteness, let's say our custom layer ``KReverse`` is implemented in Keras and reverses the order of the last dimension of the input.

.. code-block:: Python

    # Keras implementation of a custom layer
    class KReverse(tf.keras.layers.Layer):
        '''Keras implementation of a hypothetical custom layer'''

        def __init__(self):
            super().__init__()

        def call(self, inputs):
            return tf.reverse(inputs, axis=[-1])

        def get_config(self):
            return super().get_config()

Make sure you define a ``get_config()`` method for your custom layer as this is needed for correct parsing.
We can define the equivalent layer in hls4ml ``HReverse``, which inherits from ``hls4ml.model.layers.Layer``.

.. code-block:: Python

    # hls4ml layer implementation
    class HReverse(hls4ml.model.layers.Layer):
        '''hls4ml implementation of a hypothetical custom layer'''

        def initialize(self):
            inp = self.get_input_variable()
            shape = inp.shape
            dims = inp.dim_names
            self.add_output_variable(shape, dims)

A parser for the Keras to HLS converter is also required.
This parser reads the attributes of the Keras layer instance and populates a dictionary of attributes for the hls4ml layer.
It also returns a list of output shapes (one sjape for each output).
In this case, there a single output with the same shape as the input.

.. code-block:: Python

    # Parser for converter
    def parse_reverse_layer(keras_layer, input_names, input_shapes, data_reader):
        layer = {}
        layer['class_name'] = 'HReverse'
        layer['name'] = keras_layer['config']['name']
        layer['n_in'] = input_shapes[0][1]

        if input_names is not None:
            layer['inputs'] = input_names

        return layer, [shape for shape in input_shapes[0]]

Next, we need the actual HLS implementaton of the function, which can be written in a header file ``nnet_reverse.h``.

.. code-block:: C++

    #ifndef NNET_REVERSE_H_
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

Now, we can define the layer config and function call templates.
These two templates determine how to populate the config template based on the layer attributes and the function call signature for the layer in HLS, respectively.

.. code-block:: Python

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

Now, we need to tell hls4ml about the existence of this new layer by registering it.
We also need to register the parser (a.k.a. the layer handler), the template passes, and HLS implementation source code with the particular backend.
In this case, the HLS code is valid for both the Vivado and Quartus backends.

.. code-block:: Python

    # Register the converter for custom Keras layer
    hls4ml.converters.register_keras_layer_handler('KReverse', parse_reverse_layer)

    # Register the hls4ml's IR layer
    hls4ml.model.layers.register_layer('HReverse', HReverse)

    for backend_id in ['Vivado', 'Quartus']:
        # Register the optimization passes (if any)
        backend = hls4ml.backends.get_backend(backend_id)
        backend.register_pass('remove_duplicate_reverse', RemoveDuplicateReverse, flow=f'{backend_id.lower()}:optimize')

        # Register template passes for the given backend
        backend.register_template(HReverseConfigTemplate)
        backend.register_template(HReverseFunctionTemplate)

        # Register HLS implementation
        backend.register_source('nnet_reverse.h')

Finally, we can actually test the ``hls4ml`` custom layer compared to the Keras one.

.. code-block:: Python

    # Test if it works
    kmodel = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(8,)),
            KReverse(),
            tf.keras.layers.ReLU(),
        ]
    )

    x = np.random.randint(-5, 5, (8,), dtype='int32')
    kres = kmodel(x)

    for backend_id in ['Vivado', 'Quartus']:

        hmodel = hls4ml.converters.convert_from_keras_model(
            kmodel,
            output_dir=str(f'hls4mlprj_extensions_{backend_id}'),
            backend=backend_id,
            io_type='io_parallel',
            hls_config={'Model': {'Precision': 'ap_int<6>', 'ReuseFactor': 1}},
        )

        hmodel.compile()
        hres = hmodel.predict(x.astype('float32'))

        np.testing.assert_array_equal(kres, hres)
