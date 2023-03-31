==========================
Optimizer Passes and Flows
==========================

The ``hls4ml`` package internally represents the model graph with the :py:class:`~hls4ml.model.graph.ModelGraph` class.
The nodes in this graph are represented by classes derived from the :py:class:`~hls4ml.model.layers.Layer` base class.

Layers have only inputs, outputs and attributes.
All information about the layer's state and configuration is stored in the attributes.
All weights, variables and data types are attributes and there are mapping views to sort through them.
Layers can define expected attributes and can be verified for correctness, or to produce a list of configurable attributes that user can tweak.

Optimizer passes
----------------

An :py:class:`~hls4ml.model.optimizer.optimizer.OptimizerPass` transforms a model graph.
All model/layer transformations should happen in these optimizer passes.
There are a number of types of optimizer passes:

 * layer-specific: These are special optimizations for a given layer.
   An example is the :py:class:`~hls4ml.model.optimizer.passes.fuse_biasadd` class that adds a bias to a :py:class:`~hls4ml.model.layers.Dense`, :py:class:`~hls4ml.model.layers.Conv1D`, or :py:class:`~hls4ml.model.layers.Conv2D` layer.
 * backend-specific: These are only used for particular backends. An example is :py:class:`~hls4ml.backends.vivado.passes.repack_stream.ReshapeStream`.
 * model-level: These model-level optimizer passes are run on every type of layer.
 * templates: These add the HLS code for a particular backend, e.g., :py:class:`~hls4ml.backends.vivado.passes.core_templates.DenseFunctionTemplate`.
 * decorators

Flows
-----
A :py:class:`~hls4ml.model.flow.flow.Flow` is an ordered set of optimizers that may depend on other flows.
There are common model-level flows that can run regardless of the backend, and there are backend-specific flows.
Each backend provides provides a default flow for processing.
For example, the Vivado backend defaults to an IP flow that applies all other flows and produces an IP.
Another example is FIFO buffer depth optimization explained in the :ref:`FIFO Buffer Depth Optimization` section.

Explain more flows
