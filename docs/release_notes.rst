========================
Release Notes
========================

See `here <https://github.com/fastmachinelearning/hls4ml/releases>`__ for official releases on Github.

----

**v0.7.0 / TBD**

What's changed:

* GarNet and GarNetStack in config.py by @yiiyama in https://github.com/fastmachinelearning/hls4ml/pull/344
* support ZeroPadding layers by @jmduarte in https://github.com/fastmachinelearning/hls4ml/pull/480
* New backend development framework by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/395
* Register ``ApplyAlpha`` layer templates by @thesps in https://github.com/fastmachinelearning/hls4ml/pull/499
* Parsing extended by @nicologhielmetti in https://github.com/fastmachinelearning/hls4ml/pull/501
* Remove intermediate casting in product by @jmitrevs in https://github.com/fastmachinelearning/hls4ml/pull/490
* Add QKeras as a package dependency by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/511
* Copy flows from config by @thesps in https://github.com/fastmachinelearning/hls4ml/pull/510
* VivadoAccelerator backend updates by @thesps in https://github.com/fastmachinelearning/hls4ml/pull/508
* Optimized look-up table by @nemerchiedde in https://github.com/fastmachinelearning/hls4ml/pull/527
* Upsampling2D test case by @ChiRuiChen in https://github.com/fastmachinelearning/hls4ml/pull/520
* Support UpSampling1D by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/475
* RNN support (part 1) by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/521
* Quartus Custom Matrix Multiplication & Quantization by @bo3z in https://github.com/fastmachinelearning/hls4ml/pull/523
* Vivado-equivalent implementation of Softmax on Quartus by @bo3z in https://github.com/fastmachinelearning/hls4ml/pull/540
* Ensure 2 bits for scale in po2 quantizers by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/531
* Link update by @bkmgit in https://github.com/fastmachinelearning/hls4ml/pull/519
* Fix removal of nodes ingested by multiple downstream nodes by @jmduarte in https://github.com/fastmachinelearning/hls4ml/pull/544
* Enable SeparableConv2d by @jmduarte in https://github.com/fastmachinelearning/hls4ml/pull/547
* Extension API by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/528
* change string ReuseFactor to int by @jmitrevs in https://github.com/fastmachinelearning/hls4ml/pull/416
* Make the size of bn scale and bias what they really are by @jmitrevs in https://github.com/fastmachinelearning/hls4ml/pull/532
* Raise runtime error when a layer is named `input` by @jmduarte in https://github.com/fastmachinelearning/hls4ml/pull/482
* fix insertion before a node with multiple inputs + support additional broadcasting by @jmduarte in https://github.com/fastmachinelearning/hls4ml/pull/551
* Pointwise conv1d/2d resource by @jmduarte in https://github.com/fastmachinelearning/hls4ml/pull/471
* Quartus Embedding Layer by @bo3z in https://github.com/fastmachinelearning/hls4ml/pull/548
* Fix for QActivations passed as an argument by @AdrianAlan in https://github.com/fastmachinelearning/hls4ml/pull/553
* Don't override precision directly in the QKeras optimizer by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/567
* Remove the in/out size from top function by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/559
* Transpose2d, Concatenate2d, and up to 3 Clones for io_stream by @jmduarte in https://github.com/fastmachinelearning/hls4ml/pull/402
* Remove io_serial as io_stream and add some more info in docs. by @Duchstf in https://github.com/fastmachinelearning/hls4ml/pull/334
* Update docs for v0.6.0 by @thesps in https://github.com/fastmachinelearning/hls4ml/pull/453
* Use correct number of args for multiple outputs by @apfusco in https://github.com/fastmachinelearning/hls4ml/pull/487
* Fixed a few typos in the documentation  by @pitmonticone in https://github.com/fastmachinelearning/hls4ml/pull/467
* returning integer from _compute_n_samples by @JochiSt in https://github.com/fastmachinelearning/hls4ml/pull/537
* Providing support for Alveo boards by @selwyn96 in https://github.com/fastmachinelearning/hls4ml/pull/552
* Make layer names case sensitive in config. by @jmitrevs in https://github.com/fastmachinelearning/hls4ml/pull/577
* Add issue and PR templates by @jmduarte in https://github.com/fastmachinelearning/hls4ml/pull/582
* Vivado Backend GRU/LSTM support by @drankincms in https://github.com/fastmachinelearning/hls4ml/pull/560
* Update CI template syntax by @thesps in https://github.com/fastmachinelearning/hls4ml/pull/593
* Update flow dependencies by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/588
* Fix parsing of ZeroPadding layers by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/595
* remove cppname by @jmitrevs in https://github.com/fastmachinelearning/hls4ml/pull/562
* Remove email helpline from the docs by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/601
* Fixes for GRU/LSTM in Vivado backend by @drankincms in https://github.com/fastmachinelearning/hls4ml/pull/598
* Remove io_serial by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/609
* Fix test_graph by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/611
* Override parent backend optimizer passes with derived backend passes by @thesps in https://github.com/fastmachinelearning/hls4ml/pull/597
* Enforce function pipelining when using io_parallel with Resource strategy by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/605
* FIFO depth optimization by @nicologhielmetti in https://github.com/fastmachinelearning/hls4ml/pull/509
* Add tracing support for the quartus backend by @jmitrevs in https://github.com/fastmachinelearning/hls4ml/pull/583
* Quartus streaming support for Activations, Dense & Batch Normalization by @bo3z in https://github.com/fastmachinelearning/hls4ml/pull/557
* QConv alpha != 1 bug fix by @bo3z in https://github.com/fastmachinelearning/hls4ml/pull/612
* Quartus Stream Embedding by @bo3z in https://github.com/fastmachinelearning/hls4ml/pull/625
* change master to main by @jmitrevs in https://github.com/fastmachinelearning/hls4ml/pull/602
* Edit order of the optimizers in the flow so that BramFactor is followed by @jmitrevs in https://github.com/fastmachinelearning/hls4ml/pull/621
* Softmax LUT Optimization by @bo3z in https://github.com/fastmachinelearning/hls4ml/pull/570
* Quartus Synthesis Flow Improvement by @bo3z in https://github.com/fastmachinelearning/hls4ml/pull/618
* Quartus Extensions by @bo3z in https://github.com/fastmachinelearning/hls4ml/pull/628
* Quartus GRU by @bo3z in https://github.com/fastmachinelearning/hls4ml/pull/596
* Quartus Merge layers by @bo3z in https://github.com/fastmachinelearning/hls4ml/pull/634
* fix nondefault project name handling by @jmitrevs in https://github.com/fastmachinelearning/hls4ml/pull/626
* Fix parsing of logic synthesis reports by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/639
* Fix conv1d stream implementation hls directives by @Jonathan-Shoemaker in https://github.com/fastmachinelearning/hls4ml/pull/635
* Implementation and optimizations linked to Simple-RNN and LSTM for qu… by @nemerchiedde in https://github.com/fastmachinelearning/hls4ml/pull/575
* Softsign optimization by @nemerchiedde in https://github.com/fastmachinelearning/hls4ml/pull/585
* Parallel CNNs, Pooling & Image Layers for Quartus Backend by @bo3z in https://github.com/fastmachinelearning/hls4ml/pull/561
* Quartus Streaming Softsign (PR #585 contd.) by @bo3z in https://github.com/fastmachinelearning/hls4ml/pull/655
* Remove final reshapes even for Quartus by @jmitrevs in https://github.com/fastmachinelearning/hls4ml/pull/661
* Unrolled CNN implementation by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/600
* the strategy was not propagated in the pytest by @jmitrevs in https://github.com/fastmachinelearning/hls4ml/pull/663
* Fix keras model loading issue with loading model with KerasH5 by @calad0i in https://github.com/fastmachinelearning/hls4ml/pull/664
* append applied_flows container before filling instead of after by @jmitrevs in https://github.com/fastmachinelearning/hls4ml/pull/641
* set version using ``setuptools_scm`` by @jmduarte in https://github.com/fastmachinelearning/hls4ml/pull/479
* Argmax Softmax by @bo3z in https://github.com/fastmachinelearning/hls4ml/pull/627
* Fix version extraction in Sphinx config by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/669
* Add requested citations to README by @jmduarte in https://github.com/fastmachinelearning/hls4ml/pull/615
* skip BatchNorm fusion when input/output is used multiple times by @jmduarte in https://github.com/fastmachinelearning/hls4ml/pull/481
* Use wider accum_t for (average) pooling by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/681
* Quartus Streaming Conv, Pooling & Image layers by @bo3z in https://github.com/fastmachinelearning/hls4ml/pull/656
* Create branch on PR by @jmduarte in https://github.com/fastmachinelearning/hls4ml/pull/636
* Delete ``example-prjs`` directory by @jmduarte in https://github.com/fastmachinelearning/hls4ml/pull/682
* Adiabatically turn on `pre-commit` by @jmduarte in https://github.com/fastmachinelearning/hls4ml/pull/678
* Add causal padding by @cgutsche in https://github.com/fastmachinelearning/hls4ml/pull/688
* Update ``pre-commit`` GitHub Action by @jmduarte in https://github.com/fastmachinelearning/hls4ml/pull/689
* New config_from_keras_model by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/690
* remove obsolete np.int and np.float by @jmitrevs in https://github.com/fastmachinelearning/hls4ml/pull/703
* Update p-clang-format to work on mac by @jmduarte in https://github.com/fastmachinelearning/hls4ml/pull/704
* Fix function call in Alveo tcl script by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/694
* add readme for contrib by @jmitrevs in https://github.com/fastmachinelearning/hls4ml/pull/706
* WIP Add custom KL loss layer HLS implementation by @katyagovorkova in https://github.com/fastmachinelearning/hls4ml/pull/606
* Fix incorrectly linted build() command by @vloncar in https://github.com/fastmachinelearning/hls4ml/pull/709

New contributors:

* @nemerchiedde made their first contribution in https://github.com/fastmachinelearning/hls4ml/pull/527
* @ChiRuiChen made their first contribution in https://github.com/fastmachinelearning/hls4ml/pull/520
* @bo3z made their first contribution in https://github.com/fastmachinelearning/hls4ml/pull/523
* @bkmgit made their first contribution in https://github.com/fastmachinelearning/hls4ml/pull/519
* @apfusco made their first contribution in https://github.com/fastmachinelearning/hls4ml/pull/487
* @pitmonticone made their first contribution in https://github.com/fastmachinelearning/hls4ml/pull/467
* @JochiSt made their first contribution in https://github.com/fastmachinelearning/hls4ml/pull/537
* @selwyn96 made their first contribution in https://github.com/fastmachinelearning/hls4ml/pull/552
* @Jonathan-Shoemaker made their first contribution in https://github.com/fastmachinelearning/hls4ml/pull/635
* @calad0i made their first contribution in https://github.com/fastmachinelearning/hls4ml/pull/664
* @cgutsche made their first contribution in https://github.com/fastmachinelearning/hls4ml/pull/688

**Full Changelog**: https://github.com/fastmachinelearning/hls4ml/compare/v0.6.0...v0.7.0

----

**v0.6.0 / coris**

What's changed:

* ``VivadoAccelerator`` backend: target ``pynq-z2`` and ``zcu102`` boards directly from hls4ml by @nicologhielmetti
* Updated ``PyTorch`` and ``ONNX`` converters by @Duchstf 
* ``line_buffer`` Conv2D implementation for ``io_stream``: reduced resource usage and latency by @Keb-L, @violatingcp, @vloncar 
* Support ``QConv2DBatchnorm`` layer from ``QKeras`` by @nicologhielmetti 
* Improved profiling plots - easier to compare original vs ``hls4ml`` converted models by @maksgraczyk 
* Better derivation of data types for ``QKeras`` models by @jmduarte, @thesps 
* Improved CI by @thesps
* More support for models with branches, skip connections, ``Merge`` and ``Concatenate`` layers by @jmduarte, @vloncar 
* Support for ``Dense`` layers over multi-dimensional tensors by @vloncar 
* Overall improvements by @vloncar, @jmduarte, @thesps, @jmitrevs & others

New contributors:

* @siorpaes made their first contribution in https://github.com/fastmachinelearning/hls4ml/pull/424
* @jmitrevs made their first contribution in https://github.com/fastmachinelearning/hls4ml/pull/403
* @anders-wind made their first contribution in https://github.com/fastmachinelearning/hls4ml/pull/302
* @KOVI89alipes made their first contribution in https://github.com/fastmachinelearning/hls4ml/pull/318
* @maksgraczyk made their first contribution in https://github.com/fastmachinelearning/hls4ml/pull/323
* @Keb-L made their first contribution in https://github.com/fastmachinelearning/hls4ml/pull/332
* @ConsVin made their first contribution in https://github.com/fastmachinelearning/hls4ml/pull/307
* @nicologhielmetti made their first contribution in https://github.com/fastmachinelearning/hls4ml/pull/298

**Full Changelog**: https://github.com/fastmachinelearning/hls4ml/compare/v0.5.0...v0.6.0

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


