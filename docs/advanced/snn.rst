========================================
Spiking Neural Networks (PyTorch/SNN)
========================================

This page describes the initial SNN support in the PyTorch frontend.

Execution model
===============

Current hls4ml SNN implementations are synchronous (clock-driven). Neuron state
updates and layer computations run in standard HLS pipelines/streams each cycle
according to interface handshakes. The generated design is not a native
asynchronous/event-routed neuromorphic architecture (yet!).

Supported PyTorch modules and readout wrappers
==============================================

The frontend currently supports direct parsing of:

* ``Leaky`` -> ``LIFNeuron`` (or ``IFNeuron`` when ``beta`` is effectively 1)

``SNNReadout`` is an hls4ml layer, not a ``snntorch`` module. To use the
built-in hls4ml readout from a PyTorch model, define a lightweight PyTorch
module or custom extension layer, subclass ``hls4ml.utils.torch.HLS4MLModule``
so that FX treats it as a leaf module, and expose the readout configuration as
attributes. The default parser recognizes a wrapper whose class name is
``SNNReadout``. Alternatively, register a custom PyTorch layer handler that maps
your own module name to the hls4ml ``SNNReadout`` layer.

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

The hls4ml ``SNNReadout`` layer implements programmable per-model decision policies:

* ``argmax_spike_count``
* ``first_to_threshold``
* ``threshold_then_argmax``
* ``binary_logit`` (for binary classifiers with ``n_classes == 2``)

The layer accumulates class spikes over a window. For most decision rules it emits
a class ID. For ``binary_logit``, it emits a score equal to
``count(class_1) - count(class_0)``.

When using the default PyTorch parser, the wrapper module should expose these
attributes as needed:

* ``n_classes`` (defaults to the input feature count if omitted)
* ``window_size`` or ``stream_length`` (defaults to ``1``)
* ``class_threshold`` (defaults to ``1``)
* ``decision_rule`` (defaults to ``argmax_spike_count``)
* ``reset_policy`` or ``state_reset_policy`` (defaults to ``fixed_window``)

Window Boundary Semantics
=========================

The current implementation uses ``window_size`` timesteps as the sequence boundary.
At each boundary:

* the class decision is emitted
* internal readout counters are reset for the next sequence

This behavior is compatible with fixed-length time windows.

``TLAST`` note
==============

True AXI sideband ``TLAST`` boundary handling requires top-level writer/interface support for packetized AXI stream types.
The current implementation does not yet expose ``TLAST`` to layer kernels directly.

For variable-length windows, a practical workaround is to keep the hls4ml core unchanged and perform ``TLAST`` to boundary conversion in a thin wrapper IP around the generated project.
