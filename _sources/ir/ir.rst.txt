=======================
Internal representation
=======================

The ``hls4ml`` library will parse models from Keras, PyTorch or ONNX into an internal execution graph. This model graph is represented with the
:py:class:`~hls4ml.model.graph.ModelGraph` class. The nodes in this graph, loosely corresponding to the layers and operations of the input model, are represented
by classes derived from the :py:class:`~hls4ml.model.layers.Layer` base class.

Layers are required to have defined inputs and outputs that define how they are connected in the graph and what the shape of their output is. All information
about the layer's state and configuration is stored in its attributes. All weights, variables and data types are attributes and there are mapping views to sort through them.
Layers can define expected attributes and can be verified for correctness, or to produce a list of configurable attributes that the user can tweak. The complete list of attributes can be found in the :doc:`Attributes <attributes>` page.


Layers
======

The backends of ``hls4ml`` are independent from each other and free to implement features in any suitable way; however, most implementations share common concepts which we will mention here.

Dense Layers
------------

One-dimensional Dense Layers
****************************

Dense layers over one-dimensional data perform a matrix-vector multiplication followed by elementwise addition of a bias tensor. This routine is the underlying computation of many other layers as well and is reused as much as possible. It exists in several implementations across different backends, for different ``io_type`` values and strategies.

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

Parallel IO is applicable to small models that require a low-latency implementation. Larger models face synthesizability limits very quickly.

In Vivado/Vitis backends, parallel convolution relies on the *im2col* transformation of the input, which turns convolution into a matrix-multiplication task. This task is then implemented as a sequence of matrix-vector multiplications using the routine mentioned above. The ``Latency`` and ``Resource`` strategies refer to the function used for the matrix-vector multiplication routine, with ``Resource`` allowing slightly larger models to be synthesized. Parallelism can be further controlled via the ``ParallelizationFactor``. The Catapult backend in turn uses a direct implementation of convolution via nested loops. The ``Quartus``, ``Altera``, and ``Catapult`` backends also implement a ``Winograd`` algorithm choosable by setting the ``implementation`` to ``Winograd`` or ``combination``. Winograd implementation is available for only a handful of filter size configurations, and it is less concerned about bit accuracy and overflow. In certain conditions it can be faster.

io_stream
^^^^^^^^^

There are two main classes of io_stream implementations, ``LineBuffer`` and ``Encoded``. ``LineBuffer`` is the default, and generally produces marginally better results,
while ``Catapult`` and ``Vivado`` also implement ``Encoded``, choosable with the ``ConvImplementation`` configuration option. In all cases, the data is processed serially, one pixel at a time, with a pixel containing an array of all the channel values for the pixel.

Depthwise convolution
*********************

Depthwise implementation substitutes the matrix-vector multiplication in the kernel with elementwise multiplication. The only implementation available is based on the ``Latency`` strategy, used by both ``io_parallel`` and ``io_stream``.

Pointwise convolution
*********************

Pointwise convolutions are a special case of convolution where the filter size is ``1`` for 1D or ``1x1`` for 2D.

For the Vivado/Vitis backends, there is a dedicated ``io_parallel``/``Latency`` strategy implementation of 1D pointwise convolutional layers originally developed for `arXiv:2402.01876 <https://arxiv.org/abs/2402.01876>`_.
The reuse factor (RF) is used to split the layer execution and reuse the existing module RF times. The RF also limits the number of multipliers in each module.
The initiation interval scales as the RF. One limitation is that it assumes ``in_width`` is divisible by the RF.

Transposed convolution
**********************

Transposed convolutions are the ``Conv1DTranspose`` and ``Conv2DTranspose`` layers in Keras, their quantized equivalents in QKeras-v3, and the
``ConvTranspose1d`` and ``ConvTranspose2d`` layers in PyTorch. The Vivado and Vitis backends implement them for ``io_parallel`` and ``io_stream``
with the ``Latency`` and ``Resource`` strategies. Strides, dilation, ``'valid'`` and ``'same'`` padding (any padding in PyTorch) and output padding
are supported, as is an ``output_size`` passed to ``forward()`` in PyTorch when it is a constant. The data must be in the ``channels_last`` format,
which PyTorch models are converted to as usual, and PyTorch layers with ``groups`` larger than 1 are not supported.

With stride ``s``, each input pixel produces a block of ``s`` output pixels along each axis. Kernel element ``t`` adds to position
``(t * dilation) % s`` of a block and reads the input ``(t * dilation) // s`` pixels before the block. The layer is computed one block at a time: the
inputs of all kernel elements of a block, ``filt_height * filt_width * n_chan`` values, are multiplied by the weights, and each product is added to
the output position of its kernel element. A block of up to ``s`` x ``s`` output pixels therefore costs as many multiplications as one output pixel of
a convolution with the same kernel and stride 1, and no input is multiplied by an inserted zero. (``UpSampling`` followed by a convolution, which
is sometimes used in place of a transposed convolution, is a different operation: it repeats the input pixels instead of inserting zeros.)

io_parallel
^^^^^^^^^^^

As for the standard convolution, a generated function gathers the inputs of each block, and a second generated function writes the results of a
block to the output and drops the positions that the padding removes. ``ReuseFactor`` is the number of clock cycles spent on one block; with the
``Resource`` strategy it can be at most ``filt_height * filt_width * n_chan``. ``ParallelizationFactor`` is the number of blocks computed at the same
time. The initiation interval is the number of blocks divided by ``ParallelizationFactor``, times ``ReuseFactor``, where the number of blocks is the
number of blocks that hold output pixels, usually one per input pixel.

The whole input and output are kept in registers, so this implementation suits small layers only. Vivado HLS 2020.1 stops at an output of more
than 4096 values, the largest array it partitions. With Vitis HLS, C synthesis and co-simulation become slow for outputs of a few thousand values
(14 and 77 minutes for an output of 16x16x8), and the ``Resource`` strategy uses many flip-flops (75 000 for the same layer). ``io_stream`` is the
better choice for larger layers.

io_stream
^^^^^^^^^

The layer reads and writes one pixel at a time, in two processes that run at the same time. The first process steps through the input pixels,
keeps the earlier rows and columns that the kernel needs, and computes a block in ``ReuseFactor`` clock cycles at each step. The second writes the
output pixels in row order, one per clock cycle, and keeps the blocks of a row of blocks until all of their rows are written. The initiation interval
is the larger of the number of steps times ``ReuseFactor`` and the number of output pixels. The number of steps is the number of input pixels, or a
few more when the output extends past the block of the last input pixel, for example with ``'valid'`` padding and a kernel larger than the stride.
A ``ReuseFactor`` up to the number of output pixels divided by the number of steps, about ``stride_height * stride_width``, therefore lowers the
number of multipliers without lengthening the interval. ``ParallelizationFactor`` has no effect. The depth of the FIFO between the two processes is
the smallest that gives the shortest interval, found from a model of the clock cycles of the two processes.

Recurrent Layers
----------------

``SimpleRNN``, ``LSTM`` and ``GRU`` layers, and ``Bidirectional`` layers of ``LSTM`` and ``GRU`` layers, carry a state from one time step to the
next. In each step they multiply the input and the state by their weights, apply the activations of their gates, and update the state.

In the Vivado/Vitis backends, the ``Latency`` strategy computes a step with two matrix-vector multiplications, one for the input and one for the
state, followed by the activations and the state update of all the units at once. With ``io_parallel``, setting ``Static`` to ``False`` builds a
separate block for each time step, which lowers the initiation interval at the cost of resources; otherwise one block is used for all of them.

The ``Resource`` strategy computes the units of a step in groups and updates the state of each unit as soon as its gates are done, so the
activations and the state update are built only for the units of one group. The reuse factor is the number of cycles a step spends on its
multiplications: a layer of ``n`` units computes ``u`` units at a time for ``c`` cycles each, with ``n / u * c`` equal to the reuse factor. It
uses about as many multipliers as the other strategies at the same reuse factor, a few more when the inputs or the units do not divide evenly
over the ``c`` cycles. Of the groupings that give the reuse factor, the one with the fewest
units at a time is used. A reuse factor that no grouping gives, with at most as many cycles as the larger of the number of inputs and of units,
is replaced by the closest one that does. A ``GRU`` with ``reset_after=False`` applies the reset gate before the recurrent multiplication, so
each step takes two passes over the units. One block is always used for all time steps, so ``Static`` has no effect.

Activations
-----------

Most activations without extra parameters are represented with the ``Activation`` layer, and those with a single parameter (leaky ReLU, thresholded ReLU, ELU) as ``ParametrizedActivation``. ``PReLU`` has its own class because it has a parameter matrix (stored as a weight). The hard (piecewise linear) sigmoid and tanh functions are implemented in a ``HardActivation`` layer, and ``Softmax`` has its own layer class.

Backends have four softmax implementations that the user can choose from by setting the ``implementation`` parameter:

* **latency**:  Good latency, but somewhat high resource usage. It does not work well if there are many output classes.
* **stable**:  Slower but with better accuracy, useful in scenarios where higher accuracy is needed.
* **legacy**:  An older implementation with poor accuracy, but good performance. Usually the latency implementation is preferred.
* **argmax**:  If you don't care about normalized outputs and only care about which one has the highest value, using argmax saves a lot of resources. This sets the highest value to 1, the others to 0.

Vivado/Vitis backends additionally support completely skipping softmax activation and returning raw outputs.
