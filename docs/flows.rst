====================
Flows and Optimizers
====================

The ``hls4ml`` package internally represents the model graph with the :py:class:`~hls4ml.model.graph.ModelGraph` class. The nodes in this graph are represented by classes derived from the :py:class:`~hls4ml.model.layer.Layer` base class.

Layers have only inputs, outputs and attributes. All information about the layer's state and configuration is stored in the attributes. All weights, variables and data types are attributes and there are mapping views to sort through them. Layers can define expected attributes and can be verified for correctness, or to produce a list of configurable attributes that user can tweak.

Optimizers
----------

All model/layer transformations should happen in the optimizers. There are a number of types of optimizers:
* layer-specific:  describe
* backend-specific
* whole-model
* templates
* decorators
* ....

Flows
-----
A flow is an ordered set of optimizers that may depend on other flows. There are common flows that can run regardless of the backend, and there are flows specific to given backend. Each backend provides provides a default flow for processing. For example, the Vivado backend defaults to an ip flow that applies all other flows and produces an IP.

Explain more flows
