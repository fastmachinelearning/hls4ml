========================================
Spiking Neural Networks (PyTorch/SNN)
========================================

This page describes the initial SNN support in the PyTorch frontend.

Install the SNN frontend dependencies with:

.. code-block:: bash

   pip install hls4ml[snn]

Backend support
===============

The SNN flow currently supports only the ``Vitis`` backend.

Execution model
===============

Current hls4ml SNN implementations are synchronous (clock-driven). Neuron state
updates and layer computations run in standard HLS pipelines/streams each cycle
according to interface handshakes. The generated design is not a native
asynchronous/event-routed neuromorphic architecture (yet!).

Reuse factor support
====================

Standard hls4ml layers used inside an SNN, such as ``Dense``/linear layers,
retain their normal ``ReuseFactor`` support. ``ReuseFactor`` can still be set at
the model, layer type, or layer name level for these layers, and each dense layer
uses its own configured value independently of the surrounding spiking neuron
layers.  The spiking neuron kernels themselves, ``IFNeuron`` and ``LIFNeuron``, do not
currently expose ``ReuseFactor``. They process one timestep at a time, keep
internal membrane state across timesteps, and unroll the per-neuron update loop
across ``n_out`` channels.

Supported PyTorch modules and readout wrappers
==============================================

The frontend currently supports direct parsing of:

* ``Leaky`` -> ``LIFNeuron`` (or ``IFNeuron`` when ``beta`` is effectively 1)

``SNNReadout`` is an hls4ml layer, not a ``snntorch`` module. To use the
built-in hls4ml readout from a PyTorch model, instantiate the provided PyTorch
marker module:

.. code-block:: python

   from hls4ml.contrib.snntorch import SNNReadout

The marker is an identity in PyTorch and is converted to the hls4ml
``SNNReadout`` layer by the PyTorch frontend. See Jupyter Notebook example.

`snntorch` tracing
==================

``snntorch`` modules are treated as leaf modules by the hls4ml PyTorch FX tracer.
This allows conversion models to use ``snntorch.Leaky`` directly without defining
conversion-only wrapper classes.

For ``Leaky``, the supported reset mechanisms are:

* ``subtract``
* ``zero``

``threshold`` supports scalar or per-neuron vectors (length ``n_out``) for both ``IFNeuron`` and ``LIFNeuron``.
``beta`` supports scalar or per-neuron vectors for ``LIFNeuron``.

Conversion selects the most memory-efficient representation automatically:

* scalar values are emitted as compile-time constants
* per-neuron values are emitted as parameter vectors

For trainable snntorch parameters, conversion uses the current parameter values from the model
at conversion time.

Readout and Decision Rules
==========================

The hls4ml ``SNNReadout`` layer implements programmable per-model decision policies.
By default, ``output_mode="spike"`` preserves the original spike-count behavior:

* ``argmax_spike_count``
* ``first_to_threshold``
* ``threshold_then_argmax``
* ``binary_logit`` (for binary classifiers with ``n_classes == 2``)

The layer accumulates class spikes over a window. For most decision rules it emits
a class ID. For ``binary_logit``, it emits a score equal to
``count(class_1) - count(class_0)``.

For non-spiking readout heads, set ``output_mode="membrane"`` and connect
``SNNReadout`` directly after the final dense/linear layer instead of after a
final spiking neuron. In this mode the readout owns the final membrane state:

.. code-block:: python

   x = self.fc2(x)
   return self.readout(x)

At each timestep, the generated readout computes:

.. code-block:: cpp

   mem[i] = beta * mem[i] + input[i];

No threshold or reset-on-spike is applied in membrane mode. The supported
membrane decision policies are:

* ``argmax_membrane``
* ``binary_logit`` (emits ``mem(class_1) - mem(class_0)`` for binary classifiers)

The example in the Jupyter Notebook follows this approach.

Do not place a final spiking neuron before ``SNNReadout(output_mode="membrane")``
unless you intentionally want the readout to consume that neuron's spike output.
The membrane mode does not recover or expose the internal membrane state of a
preceding ``Leaky``/``IFNeuron``/``LIFNeuron`` layer. If a final output neuron
has a learnable ``beta``, that learnable neuron membrane is not the same state
as the readout-owned membrane. The readout uses its own scalar ``beta``.

When using the default PyTorch parser, the wrapper module should expose these
attributes as needed:

* ``n_classes`` (defaults to the input feature count if omitted)
* ``window_size`` or ``stream_length`` (defaults to ``1``)
* ``class_threshold`` (defaults to ``1``)
* ``output_mode`` (defaults to ``spike``; use ``membrane`` for readout-owned membrane accumulation)
* ``beta`` (defaults to ``1.0`` for membrane readout)
* ``decision_rule`` (defaults to ``argmax_spike_count``)
* ``reset_policy`` or ``state_reset_policy`` (defaults to ``fixed_window``)

Window Boundary Semantics
=========================

The current implementation uses ``window_size`` timesteps as the sequence boundary
for generated HLS. During PyTorch conversion, the first fixed-window
``SNNReadout``'s ``window_size`` is propagated to all converted ``IFNeuron`` and
``LIFNeuron`` layers in the graph.

At each boundary:

* the class decision is emitted
* internal readout counters or readout membrane state are reset for the next sequence
* internal ``IFNeuron``/``LIFNeuron`` membrane state is reset for the next sequence

The reset happens after the final timestep has been processed and has contributed
to the output. This behavior is compatible with fixed-length time windows.

Only fixed-window reset is implemented in generated layer kernels today.
``state_reset_policy`` accepts future-facing values such as ``tlast``,
``host_pulse``, and ``never``, but the current layer kernels still use fixed
``window_size`` reset behavior.

Running ``hls_model.predict()``
==============================

Compiled SNN models are stateful across top-function calls. For fixed-window
SNN inference, call the compiled model once per timestep and pass exactly
``window_size`` timesteps for each independent sequence:

.. code-block:: python

   last = None
   for step in range(timesteps):
       x_step = x_sequence[step].astype("float32")[None, :]
       last = hls_model.predict(x_step)

After the last call in the window, generated HLS resets the neuron and readout
state for the next sequence. Avoid making stray single-timestep ``predict``
calls before evaluating a sequence, because those calls advance the state.

For membrane readout, the PyTorch reference should match the generated readout
accumulation:

.. code-block:: python

   mem = torch.zeros_like(currents[:, 0, :])
   for step in range(currents.shape[1]):
       mem = beta * mem + currents[:, step, :]
   pred = mem.argmax(dim=1)

Using only the final dense current, or using spike-count reduction for a
membrane readout, does not match generated HLS behavior.

Precision note
==============

Membrane readout accumulates dense currents over the full window, so very narrow
fixed-point types can reduce accuracy even when the floating-point PyTorch model
looks good.

``TLAST`` note
==============

True AXI sideband ``TLAST`` boundary handling requires top-level writer/interface support for packetized AXI stream types.
The current implementation does not yet expose ``TLAST`` to layer kernels directly.

For variable-length windows, a practical workaround is to keep the hls4ml core unchanged and perform ``TLAST`` to boundary conversion in a thin wrapper IP around the generated project.
