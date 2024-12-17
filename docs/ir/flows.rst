==========================
Optimizer Passes and Flows
==========================

Optimizer passes
----------------

To reach a state from which the code can be generated, the internal model graph undergoes a series of optimizations (transformations), dubbed
*optimization passes*. All transformations of the model and any modification to any layer's attributes must be implemented through an optimization
pass. All optimizer passes derive from the :py:class:`~hls4ml.model.optimizer.optimizer.OptimizerPass` class. Optimizer passes are usually applied to
nodes/layers; however, a special class :py:class:`~hls4ml.model.optimizer.optimizer.ModelOptimizerPass` exists that is applied on the full model. An
example of a layer optimizer is :py:class:`~hls4ml.model.optimizer.passes.fuse_biasadd`, which adds a bias to a
:py:class:`~hls4ml.model.layers.Dense`, :py:class:`~hls4ml.model.layers.Conv1D`, or :py:class:`~hls4ml.model.layers.Conv2D` layer, while an example of
an optimizer pass that runs on the full model is :py:class:`~hls4ml.model.optimizer.passes.stamp.MakeStamp`, which creates a unique number (stamp).

Subclasses of :py:class:`~hls4ml.model.optimizer.optimizer.OptimizerPass` must provide a criteria in ``match`` function that, if satisfied, will
perform the transformation from ``transform`` function. The boolean return value of ``transform`` indicates if the optimizer pass made changes to the
model graph that may require running the optimizers again. In that case, optimizers in a flow are run again.

Optimizers can be general, independent of the backend, in which case they are located in :py:mod:`hls4ml.model.optimizer.passes`, or they may be backend-specific,
in which case they are located in a folder dependent on the backend, e.g., :py:mod:`hls4ml.backends.vivado.passes` or
:py:mod:`hls4ml.backends.quartus.passes`. A common set of optimizers that are used by FPGA backends are located in :py:mod:`hls4ml.backends.fpga.passes`.

Certain optimizers are used frequently enough that it makes sense to define special classes, which inherit from :py:class:`~hls4ml.model.optimizer.optimizer.OptimizerPass`

 * :py:class:`~hls4ml.model.optimizer.optimizer.GlobalOptimizerPass`: An optimizer pass that matches each node. This is useful, for example,
   to transform the types for a particular backend.
 * :py:class:`~hls4ml.model.optimizer.optimizer.LayerOptimizerPass`: An optimizer pass that matches each node of a particular layer type. This is
   useful, for example, to write out the HLS code for a particular node that remains in the final graph.
 * :py:class:`~hls4ml.model.optimizer.optimizer.ConfigurableOptimizerPass`:  An optimizer pass that has some configurable parameters.
 * :py:class:`~hls4ml.backends.template.Template`:  An optimizer pass that populates a code template and assigns it to an attribute of a given layer. This is commonly used
   to generate code blocks in later stages of the conversion.

Note that :py:class:`~hls4ml.model.optimizer.optimizer.LayerOptimizerPass` and :py:class:`~hls4ml.model.optimizer.optimizer.ModelOptimizerPass`
also exist as decorators that wrap a function.

New optimizers can be registered with the :py:func:`~hls4ml.model.optimizer.optimizer.register_pass`. Optimizers should be assigned to a flow (see below).

Flows
-----
A :py:class:`~hls4ml.model.flow.flow.Flow` is an ordered set of optimizers that represents a single stage in the conversion process. The optimizers
from a flow are applied in sequence until they no longer make changes to the model graph (controlled by the ``transform`` return value), after which
the next flow (stage) can start. Flows may require that other flows are applied before them, ensuring the model graph is in a desired state before a
flow starts. The function :py:func:`~hls4ml.model.flow.flow.register_flow` is used to register a new flow. Flows are applied on a model graph with
:py:func:`~hls4ml.model.graph.ModelGraph.apply_flow`.

There are common model-level flows that can run regardless of the backend, and there are backend-specific flows.
The `convert and optimize <https://github.com/fastmachinelearning/hls4ml/blob/7c0a065935904f50bd7e4c547f85354b36276092/hls4ml/model/optimizer/__init__.py#L14-L20>`_
flows do not depend on a backend.

Each backend provides provides a default flow that defines the default target for that backend. For example, the Vivado backend defaults to an
`IP flow <https://github.com/fastmachinelearning/hls4ml/blob/7c0a065935904f50bd7e4c547f85354b36276092/hls4ml/backends/vivado/vivado_backend.py#L148-L160>`_
that requires additional flows and produces an IP. It runs no optimizers itself, but it requires that many other flows (sub-flows) to have run.
The convert and optimize flows defined above are some of these required sub-flows.

Another example is FIFO buffer depth optimization explained in the :ref:`FIFO Buffer Depth Optimization` section.
