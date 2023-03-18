==========================
Backends and Data Handling
==========================

Backends
--------

``hls4ml`` supports the concept of a *backend* that determines the target HLS language.
Currently, Vivado HLS, Intel HLS, and Vitis HLS (experimental) are supported.

Data Handling
-------------

``hls4ml`` supports multiple styles for handling data between layers.
Data is passed in parallel between the layers when using the the ``io_parallel`` type.
In contrast, data is passed one "pixel" at a time when using ``io_stream``.
What is actually streamed is an array of channels, which are always sent in parallel.
This definition is most pertinent when performing CNNs.
For ``Dense`` layers, all the inputs are streamed in parallel as a single array.
