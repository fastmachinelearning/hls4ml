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
   )
   hls_multigraph_model = hls4ml.model.to_multi_model_graph(hls_model, ['layer3', 'layer7'])

Here, the ``hls_multigraph_model`` is a ``MultiModelGraph`` containing three subgraphs. Each subgraph is a ``ModelGraph`` accessible via indexing: ``hls_multigraph_model[i]``.


----------------------------------
Key Methods for MultiModelGraph
----------------------------------

* :ref:`compile <mmg-compile-method>`
* :ref:`predict <mmg-predict-method>`
* :ref:`build <mmg-build-method>`
* :ref:`trace <mmg-trace-method>`

----

.. _mmg-compile-method:

``compile`` method
==================

Compiles all the individual ``ModelGraph`` subgraphs within the ``MultiModelGraph``. Also, compiles a chained bridge file with all the subgraphs linked together that can be used for the predict function.

.. code-block:: python

   hls_multigraph_model.compile()

----

.. _mmg-build-method:

``build`` method
================

Builds all subgraphs in parallel, each as if they were standalone ``ModelGraph`` projects. Returns reports for each subgraph. If configured, it then runs the stitching flow in Vivado, connecting the individual exported IPs and allowing you to simulate the stitched design at the RTL level.

.. code-block:: python

   report = hls_multigraph_model.build(.., export=True, stitch_design=True, sim_stitched_design=True, export_stitched_design=True)

The returned ``report`` contains results from each subgraph's build and, if stitching was performed, a combined report of the stitched design. Reports for individual ``ModelGraph`` instances are always accessible via
``MultiModelGraph.graph_reports``.


----

.. _mmg-predict-method:

``predict`` method
==================

Performs a forward pass through the chained bridge file using the C-simulation (``sim='csim'``), providing 1-to-1 output with the original model. You can also leverage RTL simulation (``sim='rtl'``) to perform the forward pass at the register-transfer level. In this case, a Verilog testbench is dynamically generated and executed against the stitched IP design, providing behavioral simulation to accurately verify latency and output at the hardware level. Note that the input data for the RTL simulation must have a single batch dimension.

.. code-block:: python

   # Perform prediction using C-simulation (default)
   y_csim = hls_multigraph_model.predict(X, sim='csim')

   # Perform prediction using RTL simulation (behavioral)
   y_rtl = hls_multigraph_model.predict(X, sim='rtl')



--------------------------
Summary
--------------------------

The ``MultiModelGraph`` class is a tool for modular hardware design. By splitting a large neural network into multiple subgraphs, building each independently, and then stitching them together, you gain flexibility, parallelism, and facilitate hierarchical design, incremental optimization, and integrated system-level simulations.


Notes and Known Issues
=======================

Graph Splitting
---------------

-  Splitting in the middle of a branched architecture (e.g., ResNet skip connections) is currently unsupported.
-  Each split subgraph must have exactly one input.

Multiple Inputs & Outputs
-------------------------

- The final NN output can support multiple output layers.
- For networks with multiple input layers (a relatively uncommon case), proper synchronization is required in the testbench to drive inputs—especially for io_stream interfaces.

Simulation Discrepancies
------------------------

- Users should carefully verify functional equivalence (particularly for models that use ``io_stream`` interface)
- These discrepancies are more noticeable with raw output logits; applying a softmax layer at the model output can often help mask these differences, but this should be used with caution.

TODOs
-----------------------

- Currently tested with Vitis 2024.1. Investigate compatibility with other versions.
- Add support for Verilator-based simulation to enable faster RTL simulation.
- Investigate ``io_stream`` interface (output discrepancies, fifo optimization)
- Investigate differences in resource utilization for the ``io_parallel`` interface.
