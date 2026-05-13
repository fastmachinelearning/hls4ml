================
Layer attributes
================


Input
=====
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Constant
========
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* value: ndarray

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Activation
==========
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* n_in: int

* activation: str

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Backend-specific attributes
---------------------------
* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* table_size: int (Default: 1024)

  * The size of the lookup table used to approximate the function.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* table_t: NamedType (Default: fixed<18,8,TRN,WRAP,0>)

  * The datatype (precision) used for the values of the lookup table.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

ParametrizedActivation
======================
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* param_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* n_in: int

* activation: str

* n_in: int

* activation: str

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* param_t: NamedType

Backend-specific attributes
---------------------------
* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* table_size: int (Default: 1024)

  * The size of the lookup table used to approximate the function.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* table_t: NamedType (Default: fixed<18,8,TRN,WRAP,0>)

  * The datatype (precision) used for the values of the lookup table.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

PReLU
=====
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* param_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* n_in: int

* activation: str

* n_in: int

* activation: str

Weight attributes
-----------------
* param: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* param_t: NamedType

Backend-specific attributes
---------------------------
* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* table_size: int (Default: 1024)

  * The size of the lookup table used to approximate the function.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* table_t: NamedType (Default: fixed<18,8,TRN,WRAP,0>)

  * The datatype (precision) used for the values of the lookup table.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

Softmax
=======
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* n_in: int

* activation: str

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Backend-specific attributes
---------------------------
* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* table_size: int (Default: 1024)

  * The size of the lookup table used to approximate the function.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* table_t: NamedType (Default: fixed<18,8,TRN,WRAP,0>)

  * The datatype (precision) used for the values of the lookup table.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* n_outer: int (Default: 1)

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* n_inner: int (Default: 1)

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* implementation: list [latency,stable,argmax,legacy] (Default: stable)

  * Choice of implementation of softmax function. "latency" provides good latency at the expense of extra resources. performs well on small number of classes. "stable" may require extra clock cycles but has better accuracy. "legacy" is the older implementation which has bad accuracy, but is fast and has low resource use. It is superseded by the "latency" implementation for most applications. "argmax" is a special implementation that can be used if only the output with the highest probability is important. Using this implementation will save resources and clock cycles.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* skip: bool (Default: False)

  * If enabled, skips the softmax node and returns the raw outputs.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* exp_table_t: NamedType (Default: fixed<18,8,RND,SAT,0>)

  * The datatype (precision) used for the values of the lookup table.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* inv_table_t: NamedType (Default: fixed<18,8,RND,SAT,0>)

  * The datatype (precision) used for the values of the lookup table.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* inv_inp_t: NamedType (Default: fixed<18,8,RND,SAT,0>)

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* accum_t: NamedType (Default: fixed<18,8,RND,SAT,0>)

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

TernaryTanh
===========
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* n_in: int

* activation: str

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Backend-specific attributes
---------------------------
* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* table_size: int (Default: 1024)

  * The size of the lookup table used to approximate the function.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* table_t: NamedType (Default: fixed<18,8,TRN,WRAP,0>)

  * The datatype (precision) used for the values of the lookup table.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

HardActivation
==============
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* slope_t: NamedType

* shift_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* n_in: int

* activation: str

* slope: float (Default: 0.2)

* shift: float (Default: 0.5)

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* slope_t: NamedType

* shift_t: NamedType

Backend-specific attributes
---------------------------
* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* table_size: int (Default: 1024)

  * The size of the lookup table used to approximate the function.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* table_t: NamedType (Default: fixed<18,8,TRN,WRAP,0>)

  * The datatype (precision) used for the values of the lookup table.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

Reshape
=======
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* target_shape: Sequence

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Dense
=====
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* n_in: int

* n_out: int

Weight attributes
-----------------
* weight: WeightVariable

* bias: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

Backend-specific attributes
---------------------------
* accum_t: NamedType

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* strategy: list [latency,resource] (Default: latency)

  * Available in: Libero

Conv
====
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Backend-specific attributes
---------------------------
* accum_t: NamedType

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

Conv1D
======
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* in_width: int

* out_width: int

* n_chan: int

* n_filt: int

* filt_width: int

* stride_width: int

* pad_left: int

* pad_right: int

Weight attributes
-----------------
* weight: WeightVariable

* bias: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

Backend-specific attributes
---------------------------
* accum_t: NamedType

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* parallelization_factor: int (Default: 1)

  * The number of outputs computed in parallel. Essentially the number of multiplications of input window with the convolution kernel occuring in parallel. Higher number results in more parallelism (lower latency and II) at the expense of resources used.Currently only supported in io_parallel.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult, oneAPI

* conv_implementation: list [LineBuffer,Encoded] (Default: LineBuffer)

  * "LineBuffer" implementation is preferred over "Encoded" for most use cases. This attribute only applies to io_stream.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult

Conv2D
======
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* in_height: int

* in_width: int

* out_height: int

* out_width: int

* n_chan: int

* n_filt: int

* filt_height: int

* filt_width: int

* stride_height: int

* stride_width: int

* pad_top: int

* pad_bottom: int

* pad_left: int

* pad_right: int

Weight attributes
-----------------
* weight: WeightVariable

* bias: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

Backend-specific attributes
---------------------------
* accum_t: NamedType

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* parallelization_factor: int (Default: 1)

  * The number of outputs computed in parallel. Essentially the number of multiplications of input window with the convolution kernel occuring in parallel. Higher number results in more parallelism (lower latency and II) at the expense of resources used.Currently only supported in io_parallel.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult, oneAPI

* conv_implementation: list [LineBuffer,Encoded] (Default: LineBuffer)

  * "LineBuffer" implementation is preferred over "Encoded" for most use cases. This attribute only applies to io_stream.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult

Conv2DBatchnorm
===============
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* in_height: int

* in_width: int

* out_height: int

* out_width: int

* n_chan: int

* n_filt: int

* filt_height: int

* filt_width: int

* stride_height: int

* stride_width: int

* pad_top: int

* pad_bottom: int

* pad_left: int

* pad_right: int

Weight attributes
-----------------
* weight: WeightVariable

* bias: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

Backend-specific attributes
---------------------------
* accum_t: NamedType

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* parallelization_factor: int (Default: 1)

  * The number of outputs computed in parallel. Essentially the number of multiplications of input window with the convolution kernel occuring in parallel. Higher number results in more parallelism (lower latency and II) at the expense of resources used.Currently only supported in io_parallel.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult, oneAPI

* conv_implementation: list [LineBuffer,Encoded] (Default: LineBuffer)

  * "LineBuffer" implementation is preferred over "Encoded" for most use cases. This attribute only applies to io_stream.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult

SeparableConv1D
===============
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* depthwise_t: NamedType

* pointwise_t: NamedType

* bias_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* in_width: int

* out_width: int

* n_chan: int

* n_filt: int

* depth_multiplier: int (Default: 1)

* filt_width: int

* stride_width: int

* pad_left: int

* pad_right: int

Weight attributes
-----------------
* depthwise: WeightVariable

* pointwise: WeightVariable

* bias: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* depthwise_t: NamedType

* pointwise_t: NamedType

* bias_t: NamedType

Backend-specific attributes
---------------------------
* depthwise_accum_t: NamedType

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* pointwise_accum_t: NamedType

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* depthwise_result_t: NamedType

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* depthwise_reuse_factor: int (Default: 1)

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* pointwise_reuse_factor: int (Default: 1)

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* conv_implementation: list [LineBuffer,Encoded] (Default: LineBuffer)

  * "LineBuffer" implementation is preferred over "Encoded" for most use cases. This attribute only applies to io_stream.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult

* dw_output_t: NamedType (Default: fixed<18,8,TRN,WRAP,0>)

  * Available in: Catapult

DepthwiseConv1D
===============
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

* weight_t: NamedType

* bias_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* in_width: int

* out_width: int

* n_chan: int

* n_filt: int

* filt_width: int

* stride_width: int

* pad_left: int

* pad_right: int

* in_width: int

* out_width: int

* n_chan: int

* depth_multiplier: int (Default: 1)

* n_filt: int

* filt_width: int

* stride_width: int

* pad_left: int

* pad_right: int

Weight attributes
-----------------
* weight: WeightVariable

* bias: WeightVariable

* weight: WeightVariable

* bias: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

* weight_t: NamedType

* bias_t: NamedType

Backend-specific attributes
---------------------------
* accum_t: NamedType

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* parallelization_factor: int (Default: 1)

  * The number of outputs computed in parallel. Essentially the number of multiplications of input window with the convolution kernel occuring in parallel. Higher number results in more parallelism (lower latency and II) at the expense of resources used.Currently only supported in io_parallel.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult, oneAPI

* conv_implementation: list [LineBuffer,Encoded] (Default: LineBuffer)

  * "LineBuffer" implementation is preferred over "Encoded" for most use cases. This attribute only applies to io_stream.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult

SeparableConv2D
===============
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* depthwise_t: NamedType

* pointwise_t: NamedType

* bias_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* in_height: int

* in_width: int

* out_height: int

* out_width: int

* n_chan: int

* n_filt: int

* depth_multiplier: int (Default: 1)

* filt_height: int

* filt_width: int

* stride_height: int

* stride_width: int

* pad_top: int

* pad_bottom: int

* pad_left: int

* pad_right: int

Weight attributes
-----------------
* depthwise: WeightVariable

* pointwise: WeightVariable

* bias: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* depthwise_t: NamedType

* pointwise_t: NamedType

* bias_t: NamedType

Backend-specific attributes
---------------------------
* depthwise_accum_t: NamedType

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* pointwise_accum_t: NamedType

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* depthwise_result_t: NamedType

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* depthwise_reuse_factor: int (Default: 1)

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* pointwise_reuse_factor: int (Default: 1)

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* conv_implementation: list [LineBuffer,Encoded] (Default: LineBuffer)

  * "LineBuffer" implementation is preferred over "Encoded" for most use cases. This attribute only applies to io_stream.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult

* dw_output_t: NamedType (Default: fixed<18,8,TRN,WRAP,0>)

  * Available in: Catapult

DepthwiseConv2D
===============
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

* weight_t: NamedType

* bias_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* in_height: int

* in_width: int

* out_height: int

* out_width: int

* n_chan: int

* n_filt: int

* filt_height: int

* filt_width: int

* stride_height: int

* stride_width: int

* pad_top: int

* pad_bottom: int

* pad_left: int

* pad_right: int

* in_height: int

* in_width: int

* out_height: int

* out_width: int

* n_chan: int

* depth_multiplier: int (Default: 1)

* n_filt: int

* filt_height: int

* filt_width: int

* stride_height: int

* stride_width: int

* pad_top: int

* pad_bottom: int

* pad_left: int

* pad_right: int

Weight attributes
-----------------
* weight: WeightVariable

* bias: WeightVariable

* weight: WeightVariable

* bias: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

* weight_t: NamedType

* bias_t: NamedType

Backend-specific attributes
---------------------------
* accum_t: NamedType

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* parallelization_factor: int (Default: 1)

  * The number of outputs computed in parallel. Essentially the number of multiplications of input window with the convolution kernel occuring in parallel. Higher number results in more parallelism (lower latency and II) at the expense of resources used.Currently only supported in io_parallel.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult, oneAPI

* conv_implementation: list [LineBuffer,Encoded] (Default: LineBuffer)

  * "LineBuffer" implementation is preferred over "Encoded" for most use cases. This attribute only applies to io_stream.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult

BatchNormalization
==================
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* scale_t: NamedType

* bias_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* n_in: int

* n_filt: int (Default: -1)

* use_gamma: bool (Default: True)

* use_beta: bool (Default: True)

Weight attributes
-----------------
* scale: WeightVariable

* bias: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* scale_t: NamedType

* bias_t: NamedType

Backend-specific attributes
---------------------------
* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

Pooling1D
=========
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* n_in: int

* n_out: int

* n_filt: int

* pool_width: int

* stride_width: int

* pad_left: int

* pad_right: int

* count_pad: bool (Default: False)

* pool_op: list [Max,Average]

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Backend-specific attributes
---------------------------
* accum_t: NamedType

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* conv_implementation: list [LineBuffer,Encoded] (Default: LineBuffer)

  * "LineBuffer" implementation is preferred over "Encoded" for most use cases. This attribute only applies to io_stream.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult

Pooling2D
=========
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* in_height: int

* in_width: int

* out_height: int

* out_width: int

* n_filt: int

* pool_height: int

* pool_width: int

* stride_height: int

* stride_width: int

* pad_top: int

* pad_bottom: int

* pad_left: int

* pad_right: int

* count_pad: bool (Default: False)

* pool_op: list [Max,Average]

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Backend-specific attributes
---------------------------
* accum_t: NamedType

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* conv_implementation: list [LineBuffer,Encoded] (Default: LineBuffer)

  * "LineBuffer" implementation is preferred over "Encoded" for most use cases. This attribute only applies to io_stream.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult

GlobalPooling1D
===============
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* n_in: int

* n_filt: int

* pool_op: list [Max,Average]

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Backend-specific attributes
---------------------------
* accum_t: NamedType

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

GlobalPooling2D
===============
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* in_height: int

* in_width: int

* n_filt: int

* pool_op: list [Max,Average]

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Backend-specific attributes
---------------------------
* accum_t: NamedType

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

ZeroPadding1D
=============
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* in_width: int

* out_width: int

* n_chan: int

* pad_left: int

* pad_right: int

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

ZeroPadding2D
=============
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* in_height: int

* in_width: int

* out_height: int

* out_width: int

* n_chan: int

* pad_top: int

* pad_bottom: int

* pad_left: int

* pad_right: int

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Cropping1D
==========
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* in_width: int

* out_width: int

* n_chan: int

* crop_left: int

* crop_right: int

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Cropping2D
==========
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* in_height: int

* in_width: int

* out_height: int

* out_width: int

* n_chan: int

* crop_top: int

* crop_bottom: int

* crop_left: int

* crop_right: int

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Merge
=====
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Backend-specific attributes
---------------------------
* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

MatMul
======
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Backend-specific attributes
---------------------------
* accum_t: NamedType

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

Dot
===
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Backend-specific attributes
---------------------------
* accum_t: NamedType

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, Vivado, VivadoAccelerator, VivadoAccelerator, Vitis, Vitis, Quartus, Quartus, Catapult, Catapult, SymbolicExpression, SymbolicExpression, oneAPI, oneAPI, Libero, Libero, XLS, XLS

Concatenate
===========
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Backend-specific attributes
---------------------------
* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

Resize
======
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* in_height: int

* in_width: int

* out_height: int

* out_width: int

* n_chan: int

* align_corners: bool (Default: False)

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* algorithm: list [nearest,bilinear] (Default: nearest)

Transpose
=========
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Embedding
=========
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* embeddings_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* n_in: int

* n_out: int

* vocab_size: int

Weight attributes
-----------------
* embeddings: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* embeddings_t: NamedType

Backend-specific attributes
---------------------------
* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

SimpleRNN
=========
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

* recurrent_weight_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* n_out: int

* activation: str

* return_sequences: bool (Default: False)

* return_state: bool (Default: False)

* pass_initial_states: bool (Default: False)

Weight attributes
-----------------
* weight: WeightVariable

* bias: WeightVariable

* recurrent_weight: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* direction: list [forward,backward] (Default: forward)

* weight_t: NamedType

* bias_t: NamedType

* recurrent_weight_t: NamedType

Backend-specific attributes
---------------------------
* accum_t: NamedType

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* recurrent_reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, oneAPI

* static: bool (Default: True)

  * If set to True, will reuse the the same recurrent block for computation, resulting in lower resource usage at the expense of serialized computation and higher latency/II.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult

* table_size: int (Default: 1024)

  * The size of the lookup table used to approximate the function.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, oneAPI

* table_t: NamedType (Default: fixed<18,8,TRN,WRAP,0>)

  * The datatype (precision) used for the values of the lookup table.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, oneAPI

LSTM
====
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

* recurrent_weight_t: NamedType

* recurrent_bias_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* n_out: int

* activation: str

* recurrent_activation: str

* return_sequences: bool (Default: False)

* return_state: bool (Default: False)

* pass_initial_states: bool (Default: False)

* direction: list [forward,backward] (Default: forward)

* time_major: bool (Default: False)

Weight attributes
-----------------
* weight: WeightVariable

* bias: WeightVariable

* recurrent_weight: WeightVariable

* recurrent_bias: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

* recurrent_weight_t: NamedType

* recurrent_bias_t: NamedType

Backend-specific attributes
---------------------------
* accum_t: NamedType

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* recurrent_reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, oneAPI

* static: bool (Default: True)

  * If set to True, will reuse the the same recurrent block for computation, resulting in lower resource usage at the expense of serialized computation and higher latency/II.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult

* table_size: int (Default: 1024)

  * The size of the lookup table used to approximate the function.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, oneAPI

* table_t: NamedType (Default: fixed<18,8,TRN,WRAP,0>)

  * The datatype (precision) used for the values of the lookup table.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, oneAPI

GRU
===
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

* recurrent_weight_t: NamedType

* recurrent_bias_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* n_out: int

* activation: str

* recurrent_activation: str

* return_sequences: bool (Default: False)

* return_state: bool (Default: False)

* pass_initial_states: bool (Default: False)

* direction: list [forward,backward] (Default: forward)

* time_major: bool (Default: False)

* apply_reset_gate: list [before,after] (Default: after)

Weight attributes
-----------------
* weight: WeightVariable

* bias: WeightVariable

* recurrent_weight: WeightVariable

* recurrent_bias: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

* recurrent_weight_t: NamedType

* recurrent_bias_t: NamedType

Backend-specific attributes
---------------------------
* accum_t: NamedType

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* recurrent_reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, oneAPI

* static: bool (Default: True)

  * If set to True, will reuse the the same recurrent block for computation, resulting in lower resource usage at the expense of serialized computation and higher latency/II.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult

* table_size: int (Default: 1024)

  * The size of the lookup table used to approximate the function.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, oneAPI

* table_t: NamedType (Default: fixed<18,8,TRN,WRAP,0>)

  * The datatype (precision) used for the values of the lookup table.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, oneAPI

Bidirectional
=============
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* forward_weight_t: NamedType

* forward_bias_t: NamedType

* forward_recurrent_weight_t: NamedType

* forward_recurrent_bias_t: NamedType

* backward_weight_t: NamedType

* backward_bias_t: NamedType

* backward_recurrent_weight_t: NamedType

* backward_recurrent_bias_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* n_out: int

* return_sequences: bool (Default: False)

* return_state: bool (Default: False)

* pass_initial_states: bool (Default: False)

* time_major: bool (Default: False)

* forward_activation: str

* forward_recurrent_activation: str

* backward_activation: str

* backward_recurrent_activation: str

Weight attributes
-----------------
* forward_weight: WeightVariable

* forward_bias: WeightVariable

* forward_recurrent_weight: WeightVariable

* forward_recurrent_bias: WeightVariable

* backward_weight: WeightVariable

* backward_bias: WeightVariable

* backward_recurrent_weight: WeightVariable

* backward_recurrent_bias: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* forward_weight_t: NamedType

* forward_bias_t: NamedType

* forward_recurrent_weight_t: NamedType

* forward_recurrent_bias_t: NamedType

* backward_weight_t: NamedType

* backward_bias_t: NamedType

* backward_recurrent_weight_t: NamedType

* backward_recurrent_bias_t: NamedType

Backend-specific attributes
---------------------------
* accum_t: NamedType

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* forward_reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis

* backward_reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis

* forward_recurrent_reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis

* backward_recurrent_reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis

* static: bool (Default: True)

  * If set to True, will reuse the the same recurrent block for computation, resulting in lower resource usage at the expense of serialized computation and higher latency/II.

  * Available in: Vivado, VivadoAccelerator, Vitis

* table_size: int (Default: 1024)

  * The size of the lookup table used to approximate the function.

  * Available in: Vivado, VivadoAccelerator, Vitis

* table_t: NamedType (Default: fixed<18,8,TRN,WRAP,0>)

  * The datatype (precision) used for the values of the lookup table.

  * Available in: Vivado, VivadoAccelerator, Vitis

TimeDistributed
===============
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* wrapped_layer: None

* n_time_steps: int

* output_shape: list

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Backend-specific attributes
---------------------------
* time_step_loop_parallelism: list [Off,Unroll,Pipeline] (Default: Off)

  * Controls the amont and type of parallelism in the loop over time steps. If set to "off", no parallelism will be used. If set to "unroll", the loop will be unrolled. This may result in excessive resource use and cannot be used in "io_stream" mode. If set to "pipeline", the loop will be pipelined.

  * Available in: Vivado, VivadoAccelerator, Vitis

GarNet
======
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Backend-specific attributes
---------------------------
* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, Vivado, VivadoAccelerator, VivadoAccelerator, Vitis, Vitis, Quartus, Quartus, Catapult, Catapult, SymbolicExpression, SymbolicExpression, oneAPI, oneAPI, Libero, Libero, XLS, XLS

GarNetStack
===========
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Backend-specific attributes
---------------------------
* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, Vivado, VivadoAccelerator, VivadoAccelerator, Vitis, Vitis, Quartus, Quartus, Catapult, Catapult, SymbolicExpression, SymbolicExpression, oneAPI, oneAPI, Libero, Libero, XLS, XLS

Quant
=====
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* narrow: bool

* rounding_mode: str

* signed: bool

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Backend-specific attributes
---------------------------
* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

BipolarQuant
============
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

ApplyAlpha
==========
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* scale_t: NamedType

* bias_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* n_in: int

* n_filt: int (Default: -1)

* use_gamma: bool (Default: True)

* use_beta: bool (Default: True)

Weight attributes
-----------------
* scale: WeightVariable

* bias: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* scale_t: NamedType

* bias_t: NamedType

Backend-specific attributes
---------------------------
* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

BatchNormOnnx
=============
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Backend-specific attributes
---------------------------
* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

LayerGroup
==========
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* layer_list: list

* input_layers: list

* output_layers: list

* data_reader: object

* output_shape: list

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

SymbolicExpression
==================
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* expression: list

* n_symbols: int

* lut_functions: list (Default: [])

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

LayerNormalization
==================
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* scale_t: NamedType

* bias_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* n_in: int

* seq_len: int

* axis: int (Default: 2)

* epsilon_power_of_10: int (Default: 3)

Weight attributes
-----------------
* scale: WeightVariable

* bias: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* scale_t: NamedType

* bias_t: NamedType

Backend-specific attributes
---------------------------
* accum_t: NamedType

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* table_range_power2: int (Default: 0)

  * The negative power of 2 that represents the range of the lookup table, e.g. a value of 1 would represent a range of 0.5.

  * Available in: Vivado, VivadoAccelerator, Vitis

* table_size: int (Default: 4096)

  * The size of the lookup table used to approximate the function.

  * Available in: Vivado, VivadoAccelerator, Vitis

* table_t: NamedType (Default: ufixed<8,5,RND_CONV,SAT,0>)

  * The datatype (precision) used for the values of the lookup table.

  * Available in: Vivado, VivadoAccelerator, Vitis

* accum_t: NamedType (Default: fixed<14,4,RND_CONV,SAT,0>)

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis

EinsumDense
===========
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

* accum_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* equation: str

* inp_shape: tuple

* out_shape: tuple

Weight attributes
-----------------
* weight: WeightVariable

* bias: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

* accum_t: NamedType

Einsum
======
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* accum_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* equation: str

* inp0_shape: tuple

* inp1_shape: tuple

* out_shape: tuple

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* accum_t: NamedType

BiasAdd
=======
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Backend-specific attributes
---------------------------
* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

DACombinational
===============
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

FixedPointQuantizer
===================
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

UnaryLUT
========
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* table_t: NamedType (Default: fixed<18,8,TRN,WRAP,0>)

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* n_in: int

Weight attributes
-----------------
* table: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* table_t: NamedType (Default: fixed<18,8,TRN,WRAP,0>)

Repack
======
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

Clone
=====
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

PointwiseConv1D
===============
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* in_width: int

* out_width: int

* n_chan: int

* n_filt: int

* filt_width: int

* stride_width: int

* pad_left: int

* pad_right: int

Weight attributes
-----------------
* weight: WeightVariable

* bias: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

Backend-specific attributes
---------------------------
* accum_t: NamedType

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* parallelization_factor: int (Default: 1)

  * The number of outputs computed in parallel. Essentially the number of multiplications of input window with the convolution kernel occuring in parallel. Higher number results in more parallelism (lower latency and II) at the expense of resources used.Currently only supported in io_parallel.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult, oneAPI

* conv_implementation: list [LineBuffer,Encoded] (Default: LineBuffer)

  * "LineBuffer" implementation is preferred over "Encoded" for most use cases. This attribute only applies to io_stream.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult

PointwiseConv2D
===============
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* in_height: int

* in_width: int

* out_height: int

* out_width: int

* n_chan: int

* n_filt: int

* filt_height: int

* filt_width: int

* stride_height: int

* stride_width: int

* pad_top: int

* pad_bottom: int

* pad_left: int

* pad_right: int

Weight attributes
-----------------
* weight: WeightVariable

* bias: WeightVariable

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* weight_t: NamedType

* bias_t: NamedType

Backend-specific attributes
---------------------------
* accum_t: NamedType

  * The datatype (precision) used to store intermediate results of the computation within the layer.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* reuse_factor: int (Default: 1)

  * The number of times each multiplier is used by controlling the amount of pipelining/unrolling. Lower number results in more parallelism and lower latency at the expense of the resources used.Reuse factor = 1 corresponds to all multiplications executed in parallel, and hence, the lowest possible latency.

  * Available in: Vivado, VivadoAccelerator, Vitis, Quartus, Catapult, SymbolicExpression, oneAPI, Libero, XLS

* parallelization_factor: int (Default: 1)

  * The number of outputs computed in parallel. Essentially the number of multiplications of input window with the convolution kernel occuring in parallel. Higher number results in more parallelism (lower latency and II) at the expense of resources used.Currently only supported in io_parallel.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult, oneAPI

* conv_implementation: list [LineBuffer,Encoded] (Default: LineBuffer)

  * "LineBuffer" implementation is preferred over "Encoded" for most use cases. This attribute only applies to io_stream.

  * Available in: Vivado, VivadoAccelerator, Vitis, Catapult

Broadcast
=========
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

BatchNormalizationQuantizedTanh
===============================
Base attributes
---------------
* result_t: NamedType

  * The datatype (precision) of the output tensor.

* accum_t: NamedType

Type attributes
---------------
* index: int

  * Internal node counter used for bookkeeping and variable/tensor naming.

* n_in: int

* n_filt: int (Default: 0)

Configurable attributes
-----------------------
* trace: int (Default: False)

  * Enables saving of layer output (tracing) when using hls_model.predict(...) or hls_model.trace(...)

* result_t: NamedType

  * The datatype (precision) of the output tensor.

* accum_t: NamedType

* reuse_factor: int (Default: 1)

