========================
Realease Notes
========================


Go to `here <https://github.com/hls-fpga-machine-learning/hls4ml/releases>`_ for official releases on Github.

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


