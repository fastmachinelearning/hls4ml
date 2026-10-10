==============
Fused Strategy
==============

``Strategy: Fused`` is available in the **Vitis** backend, and in **Coyote**, which builds on it. The other strategies decide how each layer is computed on
its own; the fused strategy decides how a chain of ``Dense`` layers is computed together. The layers of a chain run at the same time in a ``DATAFLOW``
region, each starting as soon as the layer before it produces its first output, instead of one after another.

When to use it
==============

The strategy is meant for designs with a reuse factor greater than one, where each layer takes several cycles. Running the layers of a chain at the same
time reduces the latency of the model, at a much lower resource cost than unrolling the layers. The gain comes from that overlap, so it is largest for long
chains in which no single layer dominates the latency.

A reuse factor of 1 is not supported and stops the conversion. For fully parallel layers use the ``Latency`` strategy or, more efficiently,
``Strategy: distributed_arithmetic`` (see :doc:`Distributed Arithmetic <da>`). A fused layer uses at most ``n_in`` or ``n_out`` multipliers,
far fewer than a fully parallel layer.

The strategy requires ``io_parallel``.

How it works
============

A ``Dense`` layer needs all of its inputs before it can produce any output, so it can pass data one value at a time on one side only. Each layer of a chain
is therefore computed in one of three forms:

* ``dot`` reads the whole input array and writes its outputs one value at a time;
* ``axpy`` reads its input one value at a time and writes the whole output array;
* ``plain`` reads and writes whole arrays. It is used only for the first layer of a chain with an odd number of layers.

The layers of a chain alternate between ``dot`` and ``axpy``. Each ``axpy`` layer starts as soon as the ``dot`` layer before it writes its first output.
The connection between the two is an ``hls::stream`` of single values with the precision of the layer that writes it, so layers with different precisions
need no extra configuration. Between an ``axpy`` layer and the next ``dot`` layer the connection stays an array, because ``dot`` needs its whole input. A
chain therefore starts and ends with an array, and can sit between layers of any other type.

The reuse factor sets the number of multipliers a layer uses at the same time, ``n_in * n_out / ReuseFactor``, as it does for the other strategies. A
``dot`` layer can use at most ``n_in`` multipliers and the other forms at most ``n_out``, so every reuse factor below that point builds the same design,
and the conversion reports the reuse factor that was built instead. The ``dot`` and ``axpy`` layers of a pair get the lower of their two multiplier counts,
since the pair runs only as fast as the slower of the two.

Requirements
============

A chain is fused when all of the following hold:

* **The Vitis or Coyote backend.** Other backends report an error naming the layer and the backend.
* **io_parallel.** A model using ``io_stream`` is rejected during conversion.
* **Two or more** ``Dense`` **layers in sequence**, each using the strategy, where the output of each layer is read only by the next one.

A layer that asks for the strategy but cannot be fused, such as a single ``Dense`` layer, a layer whose output is read by more than one layer or is
also a model output, or a ``Conv1D`` or ``Conv2D`` layer, is built with the ``Resource`` strategy instead and reported during conversion. Depthwise
convolutions and other layer types keep the strategy they would otherwise have.

Two kinds of layer between ``Dense`` layers do not break a chain:

* **Elementwise activations**, which are computed at the end of the ``Dense`` layer before them, after which the activation layer is removed. Supported are
  ``relu``, ``sigmoid``, ``tanh``, ``selu``, ``softplus``, ``softsign``, ``binary_tanh``, ``leaky_relu``, ``thresholded_relu``, ``elu``, ``hard_sigmoid``
  and ``hard_tanh``. A ``linear`` activation is removed by ``hls4ml`` earlier.
* ``BatchNormalization`` **directly after a** ``Dense`` **layer**, which ``hls4ml`` merges into the weights and bias of that layer. This includes
  ``ApplyAlpha``, the scaling layer QKeras adds, when the merge applies.

Limitations
===========

The following end a chain. The ``Dense`` layers on either side can still be fused as separate chains if they meet the requirements above.

* ``Softmax``, which needs every output of a layer before it can produce any.
* ``PReLU``, whose parameters are stored as weights of the activation layer.
* ``ternary_tanh``, which is a layer type of its own with a threshold, not a plain activation.
* ``BatchNormalization`` that is not merged: when it does not directly follow a ``Dense`` layer, when the output type of the ``Dense`` layer is set, or
  when the weights of both layers are quantized. The most common case is a scaling layer after the activation instead of before it.
* Any other layer type, and any output read by more than one layer, since a value written to a stream can be read only once.

Configuration
=============

Set the strategy on the ``Dense`` layers of a chain:

.. code-block:: python

   config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Vitis')
   for layer in ['fc1', 'fc2', 'fc3']:
       config['LayerName'][layer]['Strategy'] = 'Fused'
       config['LayerName'][layer]['ReuseFactor'] = 4

It can also be set for a layer type or for the whole model. Every chain that qualifies is then fused, and every other layer that asked for the strategy is
built with the ``Resource`` strategy:

.. code-block:: python

   config = hls4ml.utils.config_from_keras_model(model, granularity='model', backend='Vitis')
   config['Model']['Strategy'] = 'Fused'
   config['Model']['ReuseFactor'] = 4

During conversion the strategy lists the layers of each chain, and every layer that asked for it but was not fused, with the strategy used instead. It also
reports when it sets the pipeline style of the model to ``dataflow``; if the configuration set a different ``PipelineStyle``, it is replaced with a
warning. Set ``FusedReport`` to ``False`` in the ``Model`` section to turn the report off; warnings are still printed:

.. code-block:: python

   config['Model']['FusedReport'] = False

The reuse factor and the interval
=================================

With the ``Resource`` strategy the initiation interval of a layer equals its reuse factor: a reuse factor of 128 gives an interval of 128 cycles. A fused
layer needs several cycles more, to fill its pipeline and to pass data to the other layers of its chain. How many more depends on the layer and on the
version of Vitis HLS.

If a design has to accept a new input at a fixed rate, set ``ReuseFactorAsInterval``. The reuse factor then gives the largest interval the layer may have,
in cycles, instead of setting the number of multipliers:

.. code-block:: python

   for layer in ['fc1', 'fc2', 'fc3']:
       config['LayerName'][layer]['Strategy'] = 'Fused'
       config['LayerName'][layer]['ReuseFactor'] = 128          # the interval, in cycles
       config['LayerName'][layer]['ReuseFactorAsInterval'] = True

The strategy uses the fewest multipliers that keep the layer within that interval, counting only numbers that divide the width the layer works through
(see below), and fills the remaining cycles with wait states. A ``dot`` layer and the ``axpy`` layer after it wait the same number of cycles, since the
faster of the two waits for the slower one. The interval is then the requested one or a few cycles less, never more. The wait states add latency, so the
layer is a little slower than it would be with the same number of multipliers and no wait states. They are added only in synthesis, so C simulation
results do not change.

A setting for a layer name takes precedence over one for a layer type, which takes precedence over one for the model. Note that ``granularity='name'``
writes a ``ReuseFactor`` for every layer, which then overrides one set for the model.

All layers of a chain must use ``ReuseFactorAsInterval`` the same way and request the same interval. If a layer cannot reach the requested interval even
with all of its multipliers, the conversion stops and gives the smallest interval it can reach. For each layer the conversion prints the multipliers and
wait cycles used, and the range the interval will fall in.

Estimating the interval
-----------------------

A chain is as slow as its slowest layer. A layer with ``m`` multipliers has an interval of

.. code-block::

   interval  =  passes * width / m  +  c

where ``width`` is the number of values the kernel works through in each pass, ``n_in`` for ``dot`` and ``n_out`` for the other forms, ``m`` divides
``width``, and ``passes`` is ``n_out`` for ``dot`` and ``n_in + 1`` for the other forms, the extra pass adding the bias and applying the activation.
``c`` is the time to fill the pipelines and pass data between layers. The strategy uses an overestimate of ``c``, which is why the interval can come
out a few cycles below the requested one but never above it.

This estimate of ``c`` was measured with Vitis HLS 2024.1. Vitis HLS 2025.1 is less predictable and is generally not supported.
