=======================
MultiModelGraph Class
=======================

This page documents the ``MultiModelGraph`` class, which enables handling multiple subgraphs (each represented as a ``ModelGraph``) derived from a single original model.
The central concept here is the division of a larger model into multiple smaller subgraphs at given layers which can be useful for:

* Very large models
* Step-wise optimization
* Modular design flows

A ``MultiModelGraph`` manages these subgraphs, facilitating:

* Parallel building and synthesis
* Stitched designs (merging the subgraphs in HW after synthesis)
* Simulation and performance estimation of the stitched design

--------------
Keras Example
--------------

For example, when converting a Keras model, you can specify the layers at which to split the model directly:

.. code-block:: python

   config = hls4ml.utils.config_from_keras_model(model, granularity='model')

   hls_model = hls4ml.converters.convert_from_keras_model(
       model,
       hls_config=config,
       backend='vitis',
       split_layer_names = ['layer3', 'layer7']
   )

Here, the ``hls_model`` is actually a ``MultiModelGraph`` containing three subgraphs. Each subgraph is a ``ModelGraph`` accessible via indexing: ``hls_model[i]``.


----------------------------------
Key Methods for MultiModelGraph
----------------------------------

* :ref:`compile <mmg-compile-method>`
* :ref:`predict <mmg-predict-method>`
* :ref:`build <mmg-build-method>`
* :ref:`trace <mmg-trace-method>`
* :ref:`make_multi_graph <make_multi_graph-method>`

----

.. _make_multi_graph-method:

``make_multi_graph`` method
===========================

The ``make_multi_graph`` method of ``ModelGraph`` takes a configuration, a full list of layers, the output shapes, and a list of split layers. It returns a ``MultiModelGraph`` that contains multiple ``ModelGraph`` instances.

.. code-block:: python

   from my_hls4ml_lib.modelgraph import ModelGraph
   multi_graph = ModelGraph.make_multi_graph(config, layer_list, output_shapes, split_layer_names=['fc2', 'fc3'])

This allows modular design flows and easier debugging of large models.

----

.. _mmg-compile-method:

``compile`` method
==================

Compiles all the individual ``ModelGraph`` subgraphs within the ``MultiModelGraph``. Also, compiles a chained bridge file with all the subgraphs linked together that can be used for the predict function.

.. code-block:: python

   multi_graph.compile()

----

.. _mmg-build-method:

``build`` method
================

Builds all subgraphs in parallel, each as if they were standalone ``ModelGraph`` projects. Returns reports for each subgraph. If configured, it then runs the stitching flow in Vivado, connecting the individual exported IPs and allowing you to simulate the stitched design at the RTL level.

.. code-block:: python

   report = multi_graph.build(export=True, stitch_design=True)

The returned ``report`` contains data from each subgraph's build and, if stitching was performed, a combined report of the stitched design.


----

.. _mmg-predict-method:

``predict`` method
==================

Performs a forward pass through the chained bridge file using the C-simulation (``sim='csim'``). Data is automatically passed from one subgraph's output to the next subgraph's input. For large stitched designs, you can also leverage RTL simulation (``sim='rtl'``) to perform the forward pass at the register-transfer level. In this case, a Verilog testbench is dynamically generated and executed against the stitched IP design, providing behavioral simulation to accurately verify latency and output at the hardware level. Note that the input data for the RTL simulation must have a single batch dimension.

.. code-block:: python

   # Perform prediction using C-simulation (default)
   y_csim = hls_model.predict(X, sim='csim')

   # Perform prediction using RTL simulation (behavioral)
   y_rtl = hls_model.predict(X, sim='rtl')


.. _mmg-trace-method:

``trace`` method [TODO]
================

Provides detailed layer-by-layer outputs across all sub-models, which is essential for debugging or tuning quantization and precision settings.

.. code-block:: python

   final_output, trace_outputs = hls_model.trace(X)

``trace_outputs`` includes intermediate results from each subgraph, enabling insights into the data flow.

--------------------------
Summary
--------------------------

The ``MultiModelGraph`` class is a tool for modular hardware design. By splitting a large neural network into multiple subgraphs, building each independently, and then stitching them together, you gain flexibility, parallelism, and facilitate hierarchical design, incremental optimization, and integrated system-level simulations.

--------------------------
Other Notes
--------------------------

* Branch Splitting Limitation: Splitting in the middle of a branched architecture (e.g., ResNet skip connections or multi-path networks) is currently unsupported. Also, each split subgraph must have a single input and a single output.
* Handling Multiple NN Inputs & Outputs: The final NN output can support multiple output layers. However, for networks with multiple input layers, proper synchronization is required to drive inputs—especially for stream interfaces. A fork-join mechanism in the Verilog testbench can help manage input synchronization effectively.
* RTL Simulation Issue: RTL simulation of stitched IPs with io_type='io_parallel' and a split at the flatten layer leads to improper simulation behavior and should be avoided.
* Array Partitioning for Parallel I/O: For io_parallel interfaces, all IPs must use the 'partition' pragma instead of 'reshape'.
