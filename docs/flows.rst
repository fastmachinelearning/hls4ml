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
All model/layer transformations should happen in these optimizer passes. There are two general types of optimizers.
Optimizers that inherit from :py:class:`~hls4ml.model.optimizer.optimizer.ModelOptimizerPass` run on the full model,
such as :py:class:`~hls4ml.model.optimizer.passes.stamp.MakeStamp`, while others
are only run on each node that passes the ``match`` criteria for the particular optimizer. Examples of the latter include
:py:class:`~hls4ml.model.optimizer.passes.fuse_biasadd` class that adds a bias to a :py:class:`~hls4ml.model.layers.Dense`,
:py:class:`~hls4ml.model.layers.Conv1D`, or :py:class:`~hls4ml.model.layers.Conv2D` layer. Optimizers can be general,
independent of the backend, in which case they are located in :py:mod:`hls4ml.model.optimizer.passes`, or they may be backend-specific,
in which case they are located in a folder dependent on the backend, e.g., :py:mod:`hls4ml.backends.vivado.passes` or
:py:mod:`hls4ml.backends.quartus.passes`. Passes for FPGAs in general are located in :py:mod:`hls4ml.backends.fpga.passes`.

Certain optimizers are used frequently enough that it makes sense to define special classes, which inherit from :py:class:`~hls4ml.model.optimizer.optimizer.OptimizerPass`

 * :py:class:`~hls4ml.model.optimizer.optimizer.GlobalOptimizerPass`: An optimizer pass that matches each node. This is useful, for example,
   to transform the types for a particular backend.
 * :py:class:`~hls4ml.model.optimizer.optimizer.LayerOptimizerPass`: An optimizer pass that matches each node of a particular layer type. This is
   useful, for example, to write out the HLS code for a particular node that remains in the final graph.
 * :py:class:`~hls4ml.model.optimizer.optimizer.ConfigurableOptimizerPass`:  An optimizer pass that has some configurable parameters.

Note that :py:class:`~hls4ml.model.optimizer.optimizer.LayerOptimizerPass` and :py:class:`~hls4ml.model.optimizer.optimizer.ModelOptimizerPass`
also exist as decorators that wrap a function.

Flows
-----
A :py:class:`~hls4ml.model.flow.flow.Flow` is an ordered set of optimizers that may depend on other flows. The function,
:py:func:`~hls4ml.model.flow.flow.register_flow` is used to register a new flow. There are common model-level flows that can run regardless
of the backend, and there are backend-specific flows. The `convert and optimize <https://github.com/fastmachinelearning/hls4ml/blob/7c0a065935904f50bd7e4c547f85354b36276092/hls4ml/model/optimizer/__init__.py#L14-L20>`_
flows do not depend on a backend.

Each backend provides provides a default flow that defines the default processing for that backend. For example, the Vivado backend defaults to an
`IP flow <https://github.com/fastmachinelearning/hls4ml/blob/7c0a065935904f50bd7e4c547f85354b36276092/hls4ml/backends/vivado/vivado_backend.py#L148-L160>`_
that requires additional flows and produces an IP. It runs no optimizers itself, but it requires that many other flows (subflows) to have run.
The convert and optimize flows defined above are some of these required subflows.

Another example is FIFO buffer depth optimization explained in the :ref:`FIFO Buffer Depth Optimization` section.
