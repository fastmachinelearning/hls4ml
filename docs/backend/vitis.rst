============
Vivado/Vitis
============

The **Vivado** and **Vitis** backends are aimed for use with AMD/Xilinx FPGAs. The **Vivado** backend targets the discontinued ``Vivado HLS`` compiler, while
the **Vitis** backend targets the ``Vitis HLS`` compiler. Both are designed to produce IP for incorporation in ``Vivado`` designs. (See :doc:`VivadoAccelerator <accelerator>`
for generating easily-deployable models with ``Vivado HLS``.) The ``Vitis`` accelerator flow is not directly supported, though HLS produced with the **Vitis**
backend can be easily incorporated into Vitis kernel.

Users should generally use the **Vitis** backend for new designs that target AMD/Xilinx FPGAs; new ``hls4ml`` developments will not necessarily be backported to
the **Vivado** backend.
