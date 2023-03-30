====================
Flows and Optimizers
====================

The ``hls4ml`` package internally represents the model graph with the :py:class:`~hls4ml.model.graph.ModelGraph` class.
The nodes in this graph are represented by classes derived from the :py:class:`~hls4ml.model.layer.Layer` base class.

Layers have only inputs, outputs and attributes.
All information about the layer's state and configuration is stored in the attributes.
All weights, variables and data types are attributes and there are mapping views to sort through them.
Layers can define expected attributes and can be verified for correctness, or to produce a list of configurable attributes that user can tweak.

Optimizers
----------

All model/layer transformations should happen in the optimizers.
There are a number of types of optimizers:

* layer-specific:  These are special optimizations for a given layer. An example is the :py:class:`~hls4ml.model.optimizer.passes.fuse_biasadd`
  class that adds a bias to a :py:class:`~hls4ml.model.layer.Dense`, :py:class:`~hls4ml.model.layer.Conv1D`, or :py:class:`~hls4ml.model.layer.Conv2D` layer.

* backend-specific:  These are only used for particular backends. An example is :py:class:`~hls4ml.backends.vivado.passes.repack_stream.ReshapeStream`.

* whole-model:  These are run on every type of layer.

* templates:  These add the HLS code for a particular backend, e.g., :py:class:`~hls4ml.backends.vivado.passes.core_templates.DenseFunctionTemplate`.

* decorators

* ...

Flows
-----
A flow is an ordered set of optimizers that may depend on other flows.
There are common flows that can run regardless of the backend, and there are flows specific to given backend.
Each backend provides provides a default flow for processing.
For example, the Vivado backend defaults to an IP flow that applies all other flows and produces an IP.

Explain more flows

.. _fifo_depth:

FIFO Buffer Depth Optimization
------------------------------

With the ``io_stream`` IO type, each layer is connected with the subsequent layer through first-in first-out (FIFO) buffers.
The implementation of the FIFO buffers contribute to the overall resource utilization of the design, impacting in particular the BRAM or LUT utilization.
Because the neural networks can have complex architectures generally, it is hard to know a priori the correct depth of each FIFO buffer.
By default ``hls4ml`` choses the most conservative possible depth for each FIFO buffer, which can result in a an unnecessary overutilization of resources.

In order to reduce the impact on the resources used for FIFO buffer implementation, an optimization has been developed in `#509 <https://github.com/fastmachinelearning/hls4ml/pull/509>`_ that correctly sizes the depth of the FIFO buffers by analyzing the RTL cosimulation.
We implemented this FIFO buffer resizing as an optimization pass (or flow).
Through RTL simulation with large FIFO buffers (by default set to a depth of 100,000), we estimate the maximum occupation of each FIFO.
Once the maximum depth is determined, the optimization pass sets the FIFO buffer depth to that value plus 1.

As an example, we show below how to use the flow, inspired by this `GitHub Gist <https://gist.github.com/nicologhielmetti/3a268be32755448920e9f7d5c78a76d8>`_.
First, we can define a simple neural network in Keras

.. code-block:: Python

    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(Dense(64, input_shape=(16,), name='fc1', activation='relu')
    model.add(Dense(32, name='fc2', activation='relu'))
    model.add(Dense(32, name='fc3', activation='relu'))
    model.add(Dense(5, name='fc3', activation='softmax'))

Then, we can convert the model, including the flow

.. code-block:: Python

    import hls4ml

    config = hls4ml.utils.config_from_keras_model(model, granularity='model')
    config['Flows'] = ['vivado:fifo_depth_optimization']
    hls4ml.model.optimizer.get_optimizer('vivado:fifo_depth_optimization').configure(profiling_fifo_depth=100_000)


    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                           io_type='io_stream',
                                                           hls_config=config,
                                                           output_dir='hls4mlprj_fifo_depth_opt',
                                                           part='xc7z020clg400-1',
                                                           backend='Vivado')

    hls_model.build(reset=False, csim=True, synth=True, cosim=True)

For more details and results, see `H. Borras et al., "Open-source FPGA-ML codesign for the MLPerf Tiny Benchmark" (2022) <https://arxiv.org/abs/2206.11791>`_.
