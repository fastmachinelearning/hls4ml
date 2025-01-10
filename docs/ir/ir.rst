=======================
Internal representation
=======================

The ``hls4ml`` library will parse models from Keras, PyTorch or ONNX into an internal execution graph. This model graph is represented with the
:py:class:`~hls4ml.model.graph.ModelGraph` class. The nodes in this graph, loosely corresponding to the layers and operations of the input model are represented
by classes derived from the :py:class:`~hls4ml.model.layers.Layer` base class.

Layers are required to have defined inputs and outputs that define how they are connected in the graph and what is the shape of their output. All information
about the layer's state and configuration is stored in its attributes. All weights, variables and data types are attributes and there are mapping views to sort through them.
Layers can define expected attributes and can be verified for correctness, or to produce a list of configurable attributes that user can tweak. The complete list of attributes can be found in the :doc:`Attributes <attributes>` page.


Layers
======

The backends of ``hls4ml`` are independent from each other and free to implement features in any suitable way, most implementations share common concepts which we will mention here.

Dense Layers
------------

One-dimensional Dense Layers
****************************

Dense layers over one-dimensional data perform a matrix-vector multiplication followed by elementwise addition of bias tensor. This routine is the underlying computation of many other layers as well and is reused as much as possible. It exists in several implementations across different backends, for different `io_type`'s and strategies.

io_parallel
^^^^^^^^^^^

All the backends have a ``Resource`` implementation, which divides the computation into a loop of ``reuse_factor`` iterations, each iteration simultaneously accessing a different part of the array partitioned in BRAM. There are different implementations depending on whether the reuse factor is smaller or bigger than the input size. The two Xilinx backends and Catapult also implement a ``Latency`` implementation, which uses the reuse factor to control the amount of pipelining/unrolling of the whole function while the weight array is fully partitioned in registers.

io_stream
^^^^^^^^^

The io_stream implementation only wraps the io_parallel implementation with streams or pipes for communication. Internally, data is still accessed in parallel as an array.

Multi-dimensional Dense Layers
******************************

Multi-dimensional Dense layers are converted to pointwise convolutions, and do not directly use the above implementation.


Convolution Layers
------------------

Standard convolution
********************

By *standard* convolution we refer to the operation represented by the ``Conv1D/2D`` layer in Keras (``Conv1d/2d`` in PyTorch). Depending on the ``io_type`` option used, there are two classes of implementations in ``hls4ml``.

io_parallel
^^^^^^^^^^^

Parallel IO is applicable to small models that require low latency implementation. Larger models face synthesizability limits very quickly.

In Vivado/Vitis backends, parallel convolution relies on the *im2col* transformation of the input, which turns convolution into a matrix-multiplication task. This task is then implemented as a sequence of matrix-vector multiplications using the routine mentioned above. The ``Latency`` and ``Resource`` strategies refer to the function used for matrix-vector multiplication routine, with ``Resource`` allowing for a slightly larger models to be synthesized. Parallelism can be further controlled via the ``ParallelizationFactor``. Catapult backend in turn uses a direct implementation of convolution via nested loops. The ``Quartus``, ``oneAPI``, and ``Catapult`` backends also implement a ``Winograd`` algorithm choosable by setting the ``implementation`` to ``Winograd`` or ``combination``. Winograd implementation is available for only a handful of filter size configurations, and it is less concerned about bit accuracy and overflow. In certain conditions it can be faster.

io_stream
^^^^^^^^^

There are two main classes of io_stream implementations, ``LineBuffer`` and  ``Encoded``. ``LineBuffer`` is the default, and generally produces marginally better results,
while ``Catapult`` and ``Vivado`` also implement ``Encoded``, choosable with the ``ConvImplementation`` configuration option. In all cases, the data is processed serially, one pixel at a time, with a pixel containing an array of all the channel values for the pixel.

Depthwise convolution
*********************

Depthwise implementation substitutes the matrix-vector multiplication in the kernel to the elementwise multiplication. The only implementation available is based on ``Latency`` strategy, used by both ``io_parallel`` and ``io_stream``.

Pointwise convolution
*********************

Pointwise convolutions are a special case of convolution where the filter size is ``1`` for 1D or ``1x1`` for 2D.

For the Vivado/Vitis backends, there is a dedicated ``io_parallel``/``Latency`` strategy implementation of 1D pointwise convolutional layers originally developed for `arXiv:2402.01876 <https://arxiv.org/abs/2402.01876>`_.
The reuse factor (RF) is used to split the layer execution and reuse the existing module RF times. The RF also limits the number of multipliers in each module.
The initiation interval scales as the RF. One limitation is that it assumes ``in_width`` is divisible by the RF.

Activations
-----------

Most activations without extra parameters are represented with the ``Activation`` layer, and those with single parameters (leaky ReLU, thresholded ReLU, ELU) as ``ParametrizedActivation``. ``PReLU`` has its own class because it has a parameter matrix (stored as a weight). The hard (piecewise linear) sigmoid and tanh functions are implemented in a ``HardActivation`` layer, and ``Softmax`` has its own layer class.

Backends have four softmax implementations that the user can choose from by setting the ``implementation`` parameter:

* **latency**:  Good latency, but somewhat high resource usage. It does not work well if there are many output classes.
* **stable**:  Slower but with better accuracy, useful in scenarios where higher accuracy is needed.
* **legacy**:  An older implementation with poor accuracy, but good performance. Usually the latency implementation is preferred.
* **argmax**:  If you don't care about normalized outputs and only care about which one has the highest value, using argmax saves a lot of resources. This sets the highest value to 1, the others to 0.

Vivado/Vitis backend additionally support completely skipping softmax activation and returning raw outputs.
