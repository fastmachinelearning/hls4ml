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

Supported PyTorch modules
=========================

The frontend currently supports direct parsing of:

* ``Leaky`` -> ``LIFNeuron`` (or ``IFNeuron`` when ``beta`` is effectively 1)
* ``SNNReadout`` -> ``SNNReadout``

`snntorch` tracing
==================

``snntorch`` modules are treated as leaf modules by the hls4ml PyTorch FX tracer.
This allows conversion models to use ``snntorch.Leaky`` directly without defining
conversion-only wrapper classes.

For ``Leaky``, the supported reset mechanisms are:

* ``subtract``
* ``zero``

Only scalar ``threshold`` and scalar ``beta`` are currently supported.

Readout and Decision Rules
==========================

``SNNReadout`` implements programmable per-model decision policies:

* ``argmax_spike_count``
* ``first_to_threshold``
* ``threshold_then_argmax``
* ``binary_logit`` (for binary classifiers with ``n_classes == 2``)

The layer accumulates class spikes over a window. For most decision rules it emits
a class ID. For ``binary_logit``, it emits a score equal to
``count(class_1) - count(class_0)``.

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
