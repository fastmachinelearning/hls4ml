======
oneAPI
======

The **oneAPI** backend of hls4ml is designed for deploying NNs on Intel/Altera FPGAs. It will eventually
replace the **Quartus** backend, which targeted Intel HLS. (Quartus continues to be used with IP produced by the
**oneAPI** backend.) This section discusses details of the **oneAPI** backend.

The **oneAPI** code uses SYCL kernels to implement the logic that is deployed on FPGAs. It naturally leads to the
accelerator style of programming. In the SYCL HLS (IP Component) flow, which is currently the only flow supported, the
kernel becomes the IP, and the "host code" becomes the testbench. An accelerator flow, with easier deployment on
PCIe accelerator boards, is planned to be added in the future.

The produced work areas use cmake to build the projects in a style based
`oneAPI-samples <https://github.com/oneapi-src/oneAPI-samples/tree/main/DirectProgramming/C%2B%2BSYCL_FPGA>`_.
The standard ``fpga_emu``, ``report``, ``fpga_sim``, and ``fpga`` make targets are supported. Additionally, ``make lib``
produces the library used for calling the ``predict`` function from hls4ml. The ``compile`` and ``build`` commands
in hls4ml interact with the cmake system, so one does not need to manually use the build system, but it there
if desired.

The **oneAPI** backend, like the **Quartus** backend, only implements the ``Resource`` strategy for the layers. There
is no ``Latency`` implementation of any of the layers.

Note:  currently tracing and external weights (i.e. setting BramFactor) are not supported.

io_parallel and io_stream
=========================

As mentioned in the :ref:`I/O Types` section, ``io_parallel`` is for small models, while ``io_stream`` is for
larger models. In ``oneAPI``, there is an additional difference: ``io_stream`` implements each layer on its
own ``task_sequence``. Thus, the layers run in parallel, with pipes connecting the inputs and outputs. This
is similar in style to the `dataflow` implementation on Vitis HLS, but more explicit. It is also a change
relative to the Intel HLS-based ``Quartus`` backend. On the other hand, ``io_parallel`` always uses a single task,
relying on pipelining within the task for good performance. In contrast, the Vitis backend sometimes uses dataflow
with ``io_parallel``.
