==================
Convolution Layers
==================

Standard convolutions
=====================

These are the standard 1D and 2D convolutions currently supported by hls4ml, and the fallback if there is no special pointwise implementation.

io_parallel
-----------

Parallel convolutions are for cases where the model needs to be small and fast, though synthesizability limits can be quickly reached. Also note that skip connections
are not supported in io_parallel.

For the Xilinx backends and Catapult, there is a very direct convolution implementation when using the ``Latency`` strategy. This is only for very small models because the
high number of nested loops. The ``Resource`` strategy in all cases defaults to an algorithm using the *im2col* transformation. This generally supports larger models. The ``Quartus``,
``oneAPI``, and ``Catapult`` backends also implement a ``Winograd`` algorithm choosable by setting the ``implementation`` to ``Winograd`` or ``combination``. Note that
the Winograd implementation is available for only a handful of filter size configurations, and it is less concerned about bit accuracy and overflow, but it can be faster.

io_stream
---------

There are two main classes of io_stream implementations, ``LineBuffer`` and  ``Encoded``. ``LineBuffer`` is always the default, and generally produces marginally better results,
while ``Catapult`` and ``Vivado`` also implement ``Encoded``, choosable with the ``convImplementation`` configuration option. In all cases, the data is processed serially, one pixel
at a time, with a pixel containing an array of all the channel values for the pixel.

Depthwise convolutions
======================

Pointwise convolutions
======================

Pointwise convolutions are a special case of convolution where the filter size is 1 for 1D or 1x1 for 2D.

For the Xilinx backend, there is a dedicated io_parallel ``Latency`` strategy implementation of 1D pointwise convolutional layers integrated in `#881 <https://github.com/fastmachinelearning/hls4ml/pull/881>`_ developed for `arXiv:2402.01876 <https://arxiv.org/abs/2402.01876>`_.
The reuse factor (RF) is used to split the layer execution and reuse the existing module RF times. The RF also limits the number of multipliers in each module.
The initiation interval scales as the RF. One limitation is that it assumes ``in_width`` is divisible by the RF.
