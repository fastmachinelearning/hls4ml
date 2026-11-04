============
Vivado/Vitis
============

The **Vivado** and **Vitis** backends are intended for use with AMD/Xilinx FPGAs. The **Vivado** backend targets the discontinued ``Vivado HLS`` compiler, while
the **Vitis** backend targets the ``Vitis HLS`` compiler. Both are designed to produce IP for incorporation in ``Vivado`` designs. (See :doc:`VivadoAccelerator <accelerator>`
for generating easily-deployable models with ``Vivado HLS``.) The ``Vitis`` accelerator flow is not directly supported, though HLS code produced with the **Vitis**
backend can be easily incorporated into a Vitis kernel.

Users should generally use the **Vitis** backend for new designs that target AMD/Xilinx FPGAs; new ``hls4ml`` developments will not necessarily be backported to
the **Vivado** backend.

Known issues
============

* With ``Vitis HLS`` 2023.2, recurrent layers (``SimpleRNN``, ``LSTM``, ``GRU``) may not meet the target clock at the default clock uncertainty when
  they use ``io_stream`` and the ``Latency`` strategy, or the ``Resource`` strategy with a reuse factor of 1. ``Vitis HLS`` 2024.1 is not affected. With
  2023.2, raising the clock uncertainty to 35% (``clock_uncertainty='35%'``) was enough for the models tested, at the cost of one to three extra
  clock cycles per time step.
* With ``Vivado HLS`` 2020.1, ``Conv1DTranspose`` and ``Conv2DTranspose`` layers with the ``Latency`` strategy may not meet the target clock at the
  default clock uncertainty. Raising the clock uncertainty to 27% (``clock_uncertainty='27%'``) was enough for most of the models tested, though some
  layers required more. The same can happen with ``io_stream`` and the ``Resource`` strategy at a small reuse factor. ``Vitis HLS`` 2024.1 is not affected.
* With ``Vitis HLS`` 2024.1, the ``io_stream`` implementation of ``Conv1DTranspose`` and ``Conv2DTranspose`` takes one clock cycle more per input
  than the interval given in :ref:`Transposed convolution` in some layers where the first of its two processes is the slower one.
* The following are not supported with ``Vitis HLS`` 2025.1:

  * ``Conv1D`` and ``Conv2D`` with ``io_parallel``: co-simulation does not finish when the model is a dataflow region, which is the default for
    models with convolutions.
  * ``Dense`` with ``io_parallel`` and the ``Resource`` strategy: C synthesis did not finish in six hours for most of the models tested,
    even small ones.
  * ``Conv1DTranspose`` and ``Conv2DTranspose``.
  * The ``Fused`` strategy, whose interval is less predictable with this version (see :doc:`Fused Strategy <../advanced/fused>`).

Fused strategy
==============

The **Vitis** backend also provides ``Strategy: Fused``, which computes a chain of ``Dense`` layers together so that consecutive layers run at the same
time. See :doc:`Fused Strategy <../advanced/fused>`.

Vivado synthesis and place and route
====================================

``build(vsynth=True)`` synthesizes the HLS design with Vivado, and ``build(pnr=True)`` also places and routes it. This works the same for the **Vivado**,
**Vitis** and :doc:`VivadoAccelerator <accelerator>` backends; for VivadoAccelerator the AXI wrapper ``<project_name>_axi`` is synthesized. The design is
synthesized alone, out of context, and only the clock is constrained. The slack therefore covers the paths between registers of the design, not the paths
from its input ports or to its output ports. After place and route, the slack includes the delay and skew of the clock from the first clock buffer site of
the device; another site changes it a little.
