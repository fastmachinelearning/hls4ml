============
Dense Layers
============

One-dimensional Dense Layers
============================

One-dimensional dense layers implement a matrix multiply and bias add. The produced code is also used by other layers to implement the matrix multiplication.


io_parallel
-----------

All the backends implement a ``Resource`` implementation, which explicitly iterates over the reuse factor. There are different implementations depending on whether the reuse factor is
smaller or bigger than the input size. The two Xilinx backends and Catapult also implement a ``Latency`` implementation, which only uses the reuse factor in pragmas.

io_stream
---------

The io_stream implementation only wraps the io_parallel implementation with streams or pipes for communication. The data is still transferred in parallel.

Multi-dimensional Dense Layers
==============================

Multi-dimensional Dense layers are converted to pointwise convolutions, and do not directly use the above implementation
