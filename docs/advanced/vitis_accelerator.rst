========================
VitisAccelerator Backend
========================

The ``VitsAccelerator`` backend leverages the `Vitis System Design Flow <https://www.xilinx.com/products/design-tools/vitis.html#design-flows>`_ to automate and simplify the creation of an hls4ml project targeting `AMD Alveo PCIe accelerators <https://www.amd.com/en/products/accelerators/alveo.html>`_.
The Vitis accelerator backend has been tested with the following boards:

* `Alveo u50 <https://www.xilinx.com/products/boards-and-kits/alveo/u50.html>`_
* `Alveo u55c <https://www.xilinx.com/products/boards-and-kits/alveo/u55c.html>`_
* `Alveo u250 <https://www.xilinx.com/products/boards-and-kits/alveo/u250.html>`_
* `Versal vck5000 <https://www.xilinx.com/products/boards-and-kits/vck5000.html>`_

Kernel wrapper
==============

To integrate with the Vitis System Design Flow and run on an accelerator, the generated ``hls4ml`` model must be encapsulated and built as a Vitis kernel (``.xo`` file) and linked into a binary file (``.xclbin``) during the implementation step. On the host side, standard C++ code using either `OpenCL <https://xilinx.github.io/XRT/master/html/opencl_extension.html>`_ or `XRT API <https://xilinx.github.io/XRT/master/html/xrt_native_apis.html>`_ can be used to download the ``.xclbin`` file to the accelerator card and use any kernel it contains.

The ``VitisAccelerator`` backend automatically generates a kernel wrapper, an host code example, and a Makefile to build the project.

**Note:** The current implementation of the kernel wrapper code is oriented toward throughput benchmarking and not general inference uses (See :ref:`here<hardware_predict-method>`). It can nonetheless be further customized to fit specific applications.

Options
=======

As PCIe accelerators are not suitable for ultra-low latency applications, it is assumed that they are used for high-throughput applications. To accommodate this, the backend supports the following options to optimize the kernel for throughput:

    * ``num_kernel``: Number of kernel instance to implement in the hardware architecture.
    * ``num_thread``: Number of host threads used to exercise the kernels in the host application.
    * ``batchsize``: Number of samples to be processed in a single kernel execution.

Additionaly, the backend proposes the following options to customize the implementation:

    * ``board``: The target board, must match one entry in ``supported_boards.json``.
    * ``clock_period``: The target clock period in ns.
    * ``hw_quant``: Is arbitrary precision quantization performed in hardware or not. If True, the quantization is performed in hardware and float are used at the kernel interface, otherwise it is performed in software and arbitrary precision types are used at the interface. (Defaults to  ``False``).
    * ``vivado_directives``: A list of strings to be added under the ``[Vivado]`` section of the generated ``accelerator_card.cfg`` link configuration file. Can be used to add custom directives to the Vivado project.

Build workflow
==============

At the call of the ``build`` method, the following option affect the build process:

    * ``reset``: If True, clears files generated during previous build processes (Equivalent to ``make clean`` in build folder).
    * ``target``: Can be one of ``hw``, ``hw_emu``, ``sw_emu``, to define which build target to use (Default is ``hw``).
    * ``debug``: If True, compiles the c++ host code and the HLS in debug mode.

Once the project is generated, it possible to run manually the build steps by using one of the following ``make`` targets in the generated project directory:

    * ``host``: Compiles the host application.
    * ``hls``: Produces only the kernel's object file.
    * ``xclbin``: Produces only the kernel's .xclbin file.
    * ``clean``: Removes all generated files.
    * ``run``: Run the host application using the .xclbin file and the input data present in ``tb_data/tb_input_features.dat``.

It is also possible to run the full build process by calling ``make`` without any target. Modifications to the ``accelerator_card.cfg`` file can be done manually before running the build process (e.g., to change the clock period, or add addition ``.xo`` kernel to the build).

The generated host code application and the xclbin file can be executed as such:

.. code-block:: Bash

    ./host <build_directory>/<myproject>.xclbin

Example
=======

The following example is a modified version of `hsl4ml example 7 <https://github.com/fastmachinelearning/hls4ml-tutorial/blob/master/part7_deployment.ipynb>`_.

.. code-block:: Python

    import hls4ml
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir='model_3/hls4ml_prj_vitis_accel',
        backend='VitisAccelerator',
        board='alveo-u55c',
        num_kernel=4,
        num_thread=8,
        batchsize=8192,
        hw_quant=False,
        vivado_directives=["prop=run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=Explore"]
    )
    hls_model.compile()
    hls_model.build()
    y = hls_model.predict_hardware(y) # Limited to batchsize * num_kernel * num_thread for now
