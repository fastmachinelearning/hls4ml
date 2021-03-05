========================
Release Notes
========================

See `here <https://github.com/fastmachinelearning/hls4ml/releases>`__ for official releases on Github.

----

**v0.5.0 / bartsia**

What's new:

* Streaming IO layer implementations, especially of Convolutional layers, accessed through the config with `IOType: io_stream`. Scales CNN support to much larger models than previously possible (see `arXiv:2101.05108 <https://arxiv.org/abs/2101.05108>`__)
* New `documentation and API reference <https://fastmachinelearning.org/hls4ml/>`__
* Further optimizations for QKeras / quantization aware training. A 'shift' operation is now used for `po2` quantizers
* Allow redefinition of weights directory for standalone project compilation
* ``profiling`` for PyTorch models

Deprecated:

* ``IOType : io_serial`` is deprecated, and superceded by new ``IOType: io_stream``

Bugfixes:

* Fix to Initiation Interval and different min/max latency for ``Strategy: Resource``
* Fix warnings in ``hls4ml`` command line script flow
* Write yml config from Python API - for mixed API / command line flow

----

**v0.4.0 / aster**

What's new:

* Support for GarNet layer (see `arXiv:2008.03601 <https://arxiv.org/abs/2008.03601>`__)
* Input layer precision added to config generator utility
* New 'SkipOptimizers' config option. Now you can run all Optimizers by default (as in v0.3.0) but subtract any specified by ``SkipOptimizer`` e.g. ``hls_config['SkipOptimizers'] = ['fuse_consecutive_batch_normalization']``
* Print out the latency report from Cosimulation

Bugfixes:

* Fixes related to tensorflow 2.3: new Functional API, changes to handling of Input layer
* Fix error with config generator utility and activation layers gor ``granularity='name'``
* Fix issue with reloading of emulation library after configuration change
* Fix to handling of layers with ``use_bias=False`` and merged Dense and BatchNormalization

----

**v0.3.0**


* Installing from ``PyPI``
* Create configuration dictionary from model object
* Run 'C Simulation' from Python with ``hls_model.predict(X)``
* Trace model layer output with ``hls_model.trace(X)``
* Write HLS project, run synthesis flow from Python
* QKeras support: convert models trained using layers and quantizers from QKeras
* Example models moved to separate repo, added API to retrieve them
* New Softmax implementations
* Minor fixes: weights exported at higher precision, concatenate layer shape corrected

----

**v0.2.0:**


* ``tf_to_hls`` tool for converting tensorflow models (protobufs ``.pb``\ )
* Support for larger ``Conv1D/2D`` layers
* Support for binary and ternary layers from `QKeras <https://github.com/google/qkeras>`_.
* API enhancements (custom layers, multiple backends)
* :doc:`Profiling <api/profiling>` support
* ``hls4ml report``\ command to gather HLS build reports, ``hls4ml build -l`` for Logic Synthesis
* Support for all-in-one Keras's ``.h5`` files (obtained with Keras's ``save()`` function, without the need for separate ``.json`` and ``.h5`` weight file).
* Fused Batch Normalisation into Dense layer optimsation.

----

**v0.1.6:**


* Support for larger Dense layers (enabled with Strategy: Resource in the configuration file)
* Binary/Ternary NN refinements
* Built-in optimization framework
* Optional C/RTL validation

----

**v0.1.5**\ : Per-layer precision and reuse factor

----

**v0.1.3**\ : Adding PyTorch support

----

**v0.1.2**\ : First beta release


* some bug fixes for pipelining and support for layer types

----

**v0.0.2**\ : first alpha release


* full translation of DNNs from Keras 
* an example Conv1D exists
* parallel mode is supported (serial mode, not yet)


