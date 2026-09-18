============
Vivado/Vitis
============

The **Vivado** and **Vitis** backends are aimed for use with AMD/Xilinx FPGAs. The **Vivado** backend targets the discontinued ``Vivado HLS`` compiler, while
the **Vitis** backend targets the ``Vitis HLS`` compiler. Both are designed to produce IP for incorporation in ``Vivado`` designs. (See :doc:`VivadoAccelerator <accelerator>`
for generating easily-deployable models with ``Vivado HLS``.) The ``Vitis`` accelerator flow is not directly supported, though HLS produced with the **Vitis**
backend can be easily incorporated into Vitis kernel.

Users should generally use the **Vitis** backend for new designs that target AMD/Xilinx FPGAs; new ``hls4ml`` developments will not necessarily be backported to
the **Vivado** backend.

Fused strategy
==============

``Strategy: Fused`` is available in the **Vitis** backend only. The remaining strategies determine how an individual layer is computed; this strategy determines
how a sequence of ``Dense`` layers is computed as a whole. Such a sequence is placed in a single ``DATAFLOW`` region and its layers execute concurrently, rather
than each layer completing before the next one begins.

Intended operating region
-------------------------

The strategy addresses designs that use a reuse factor greater than one, in which each layer is computed over several cycles and the layers of a model would
otherwise run one after another. It reduces the latency of the model by overlapping consecutive layers, at a resource cost well below that of unrolling them.

It is not intended for ``ReuseFactor = 1``. Fully unrolled constant-matrix multiplication is already served by the ``Latency`` strategy and, more efficiently,
by ``Strategy: distributed_arithmetic`` (see :doc:`Distributed Arithmetic <../advanced/da>`) together with the quantisation-aware flows built around it. A
layer that reads a complete input array uses at most ``n_in`` multipliers and an ``axpy`` layer at most ``n_out``, so the strategy cannot reach the parallelism
those paths provide. A reuse factor of 1 stops the conversion.

It also does not extend to models too large for ``io_parallel``, which require ``io_stream`` and are outside its scope for the reason given under
`Requirements`_.

A ``Dense`` layer requires all of its input values before it can produce any output value, and can therefore transfer data one value at a time on one of its two
sides, but not on both. Each layer of a chain is accordingly computed in one of two forms:

* ``dot`` reads the complete input array and writes its outputs one value at a time;
* ``axpy`` reads its input one value at a time and writes the complete output array.

A ``dot`` layer followed by an ``axpy`` layer executes concurrently, as the second layer can begin processing as soon as the first produces its first output
value. The connection between such a pair is emitted as an ``hls::stream`` carrying a single value, with the precision of the layer that writes it; models using
a different precision per layer therefore require no additional configuration. The connection between an ``axpy`` layer and the ``dot`` layer that follows it
remains an array, as the ``dot`` form requires the complete input. A chain containing an odd number of layers is given a leading layer computed in the
conventional form, so that the region as a whole reads an array and writes an array and can be placed among layers of any other type.

The reuse factor keeps its usual meaning and is the only setting: it is converted into ``multiplier_limit``, the number of multipliers the kernel uses
concurrently, as ``n_in * n_out / ReuseFactor``. The two layers of a streaming pair are assigned the lower of their two counts, as a pair is limited by its
slower half.

A layer that reads a complete input array uses at most ``n_in`` multipliers and an ``axpy`` layer at most ``n_out``, so reuse factors below that point all
produce the same design. A reuse factor that cannot be honoured is reported during conversion, together with the one that is built.

Requirements
------------

The strategy applies to a model when all of the following hold.

* **The Vitis backend.** The strategy is not available in the **Vivado** backend or in backends derived from it, which report an error naming the layer and the
  backend rather than substituting a different strategy.
* **io_parallel.** Under ``io_stream`` a single read of a stream carries an entire row, which the fused kernels cannot use; the combination is rejected during
  conversion.
* **Two or more** ``Dense`` **layers in sequence**, each selecting the strategy, with the output of one read only by the next. A single layer, a layer whose
  output is read by more than one layer, and any layer type other than ``Dense`` are computed as they would be under the other strategies. Selecting the
  strategy on a layer type that does not implement it is reported, and that layer is built with the strategy it would otherwise have used.

Two layer types are removed before the chain is identified and therefore do not interrupt a sequence of ``Dense`` layers:

* **Elementwise activations**, which are computed in the output stage of the ``Dense`` layer that produces the value, after which the activation layer is
  removed. The following are supported: ``relu``, ``sigmoid``, ``tanh``, ``selu``, ``softplus``, ``softsign``, ``binary_tanh``, ``leaky_relu``,
  ``thresholded_relu``, ``elu``, ``hard_sigmoid`` and ``hard_tanh``. A ``linear`` activation is removed by ``hls4ml`` at an earlier stage and never reaches the
  strategy.
* ``BatchNormalization`` **immediately following a** ``Dense`` **layer**, which ``hls4ml`` merges into the weights and bias of that layer before any backend
  processing. This includes ``ApplyAlpha``, the scaling layer introduced by QKeras models, provided the conditions for the merge are satisfied.

Limitations
-----------

The following terminate a chain. The ``Dense`` layers on either side are treated as separate chains, and are fused only where each side satisfies the
requirements above.

* ``Softmax``, which requires every output of a layer before it can produce any and therefore cannot be computed value by value.
* ``PReLU``, which is elementwise but holds one parameter per output, stored as weights of the activation layer.
* ``ternary_tanh``, which is a layer type of its own holding a threshold, rather than a plain activation.
* ``BatchNormalization`` that has not been merged. This occurs when it does not immediately follow a ``Dense`` layer, when the output type of the ``Dense``
  layer is specified, or when the weights of both layers are quantized. The most common case is a scaling layer placed after the activation rather than before
  it.
* Any other layer type, and any point at which a second layer reads the same output, as a value written to a stream can be read only once.

The reduction in latency is obtained from the overlap between consecutive layers. It is therefore largest for long chains whose individual layers are small
enough that the computation of a single layer does not dominate the latency of the model.

Configuration
-------------

The strategy is selected per layer and takes effect wherever two or more ``Dense`` layers selecting it are in sequence:

.. code-block:: python

   config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Vitis')
   for layer in ['fc1', 'fc2', 'fc3']:
       config['LayerName'][layer]['Strategy'] = 'Fused'
       config['LayerName'][layer]['ReuseFactor'] = 4

The reuse factor and the interval
---------------------------------

With the other strategies a layer takes as many cycles as its reuse factor, so a reuse factor of 128 gives an initiation interval of 128. This is not the
case with the fused strategy. A fused layer needs a few cycles more than its reuse factor: to fill its pipeline, and to pass data to the other layers of
its region. How many more depends on the layer and on the version of Vitis HLS.

If a design has to accept a new input at a fixed rate, set ``ReuseFactorAsInterval``. The reuse factor is then read as the largest interval the layer may
have, in cycles, rather than as a number its multiplier count is derived from:

.. code-block:: python

   for layer in ['fc1', 'fc2', 'fc3']:
       config['LayerName'][layer]['Strategy'] = 'Fused'
       config['LayerName'][layer]['ReuseFactor'] = 128          # the interval, in cycles
       config['LayerName'][layer]['ReuseFactorAsInterval'] = True

The strategy uses the fewest multipliers that keep the layer within that limit, and adds the cycles left over as wait states. The interval is then equal to
the requested one or a few cycles smaller, and never larger. Wait states cost latency, so a layer configured this way is a little slower than the same
layer would otherwise be.

Set the interval at the same level as the rest of the configuration. A value given for a layer by name takes precedence over one given for a layer type,
which takes precedence over one given for the model, and ``granularity='name'`` writes a reuse factor for every layer.

Every layer of a fused region must use the reuse factor the same way and request the same interval. If the requested interval is smaller than a layer can
achieve with all of its multipliers in use, the conversion stops and the message gives the smallest interval that layer can achieve. For each layer the
conversion prints the number of multipliers and wait cycles used, and the range the interval will be in.

What a layer costs
^^^^^^^^^^^^^^^^^^

A region is as slow as its slowest layer. A layer with ``m`` multipliers needs

.. code-block::

   interval  =  passes * ceil(lanes / m)  +  c

The kernel iterates over ``lanes`` values, which is ``n_in`` for the dot form and ``n_out`` for the other two, and makes ``passes`` passes over them:
``n_out`` for the dot form, and for the other two one pass per input plus one that applies the activation. ``c`` is the time to fill the pipeline and to
pass data between the layers of the region. The strategy uses an upper estimate of ``c``, which is why the interval can be a few cycles smaller than
requested but is never larger.

The wait states use ``ap_wait_n``. A ``LATENCY`` directive cannot be used instead: Vitis HLS ignores it on these kernels, because it compiles their loops
into separate functions whose duration it cannot determine in advance, and it reports this as ``HLS 200-893`` in ``solution1.log`` only. The wait states
are present during synthesis only, so results from C simulation do not change.

The estimate of ``c`` suits Vitis HLS 2023.2 and 2024.1. Vitis HLS 2025.1 needs more cycles, and less predictably, so check the interval with that version
before depending on it.
