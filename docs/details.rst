============
More Details
============

Backends
--------

``hls4ml`` supports the concept of a *backend* that determines what the output will look like....

Data handling
-------------

``hls4ml`` supports multiple styles for handling data between layers. Data is passed in parallel between the layers when using the the *io_parallel* type. In contrast, data is passed one "pixel" at a time when using *io_stream*. What is actually streamed is an array of channels, which are always sent in parallel. This definition is most pertinent when performing CNNs. (For Dense layers, all the inputs are streamed in parallel as a single array.)

FIFO stream optimization
^^^^^^^^^^^^^^^^^^^^^^^^

The FIFO buffer sizes can be automatically when using a Vivado or VivadoAccelerator Backend. The sizes are determined using cosimulation to determined the high water mark. FIFO depth optimization is done by selecting an appropriate flow. (Add more information, link to flows)
