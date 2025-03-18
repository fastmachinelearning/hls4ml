==============================
FIFO Buffer Depth Optimization
==============================

With the ``io_stream`` IO type, each layer is connected with the subsequent layer through first-in first-out (FIFO) buffers.
The implementation of the FIFO buffers contribute to the overall resource utilization of the design, impacting in particular the BRAM or LUT utilization.
Because the neural networks can have complex architectures generally, it is hard to know a priori the correct depth of each FIFO buffer.
By default ``hls4ml`` choses the most conservative possible depth for each FIFO buffer, which can result in a an unnecessary over-utilization of resources.

In order to reduce the impact on the resources used for FIFO buffer implementation, an optimization flow has been developed that correctly sizes the depth
of the FIFO buffers by analyzing the RTL co-simulation. This feature is currently available in ``Vitis`` and ``Vivado`` backends.

In ``Vivado`` backend, FIFO buffer resizing is implemented as a :py:class:`~hls4ml.backends.vivado.passes.fifo_depth_optimization` optimizer pass.
Through RTL simulation with large FIFO buffers (by default set to a depth of 100,000), we estimate the maximum occupation of each FIFO.
Once the maximum depth is determined, the optimizer pass sets the FIFO buffer depth to that value plus 1.

Below we show an example of the use of the FIFO depth optimization. First, we can define a simple neural network in Keras:

.. code-block:: Python

    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(Dense(64, input_shape=(16,), name='fc1', activation='relu'))
    model.add(Dense(32, name='fc2', activation='relu'))
    model.add(Dense(32, name='fc3', activation='relu'))
    model.add(Dense(5, name='fc4', activation='softmax'))

Then, we can convert the model, including the flow:

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

Similarly, the FIFO buffers can be optimized while using the ``Vitis`` backend with the following changes:

.. code-block:: Python

    config['Flows'] = ['vitis:fifo_depth_optimization']
    hls4ml.model.optimizer.get_optimizer('vitis:fifo_depth_optimization').configure(profiling_fifo_depth=100_000)

    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                        io_type='io_stream',
                                                        hls_config=config,
                                                        output_dir='hls4mlprj_fifo_depth_opt',
                                                        part='xc7z020clg400-1',
                                                        backend='Vitis')
