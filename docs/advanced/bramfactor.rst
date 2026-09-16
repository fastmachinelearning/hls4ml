==================================
Loading weights from external BRAM
==================================

``hls4ml`` can optionally store weights in BRAMs external to the design, so the surrounding FPGA design can supply or replace them without regenerating the HLS compute logic. This is supported in Vivado/Vitis and Catapult backends. It is the responsibility of the user to ensure the weights are properly loaded during the operation of the design.

The feature works as a threshold, exposed through a ``BramFactor`` config parameter. Layers with more weights above the threshold will be exposed as BRAM interface. Consider the following code:

.. code-block:: Python

    model = tf.keras.models.Sequential()
    model.add(Dense(10, activation="relu", input_shape=(12,), name="dense_1"))
    model.add(Dense(20, activation="relu", name="dense_2"))
    model.add(Dense(5, activation="softmax", name="dense_3"))
    model.compile(optimizer='adam', loss='mse')

    config = hls4ml.utils.config_from_keras_model(model)
    config["Model"]["Strategy"] = "Resource"
    config["Model"]["BramFactor"] = 100

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, io_type=io_type, backend=backend
    )

Having set ``BramFactor=100``, only layers with more than 100 weights will be exposed as external BRAM, in this case layers ``dense_1`` and ``dense_2``. ``BramFactor`` can currently be only set at the model level. The generated code will now have weights as part of the interface.

.. code-block:: C++

    void myproject(
        hls::stream<input_t> &dense_1_input,
        hls::stream<result_t> &layer7_out,
        model_default_t w2[120],
        model_default_t w4[200]
    ) {
        #pragma HLS INTERFACE axis port=dense_1_input,layer7_out
        #pragma HLS INTERFACE bram port=w2,w4
        ...

When integrating the design, users can use the exposed interface to implement weight reloading scheme. Supplying and driving those raw interfaces is otherwise left to the surrounding design; the rest of this page describes tooling that does it for a supported subset.

Runtime-selected weight banks
=============================

``hls4ml.contrib.runtime_weights`` allows several trained parameter sets of the
same network to reside on the FPGA at once. A bank is selected for each inference
without re-running HLS or modifying the synthesized compute IP.

A typical use case is one network architecture trained for several operating
conditions, configurations, or deployment modes.

The parameter sets are normally all known before the FPGA build. The flow is:
set ``BramFactor`` and ``build(synth=True)`` once; ``package()`` the wrapper with
``n_banks`` resident banks; ``pack_banks()`` the parameter sets; preload BRAM
parameters from ``.hex`` files; integrate the wrapper; after reset, load the
scalar parameters through the loader; select a bank with ``ext_bank_id`` per
inference.

End-to-end example
------------------

Assume three trained model variants with the same architecture::

    model_a
    model_b
    model_c

Choose one variant, normally the first, to create the hls4ml model. Parameters that
should change between variants must be externalized with ``BramFactor``::

    config["Model"]["Strategy"] = "Resource"
    config["Model"]["BramFactor"] = 0

    hls_model = ...
    hls_model.build(csim=False, synth=True)

The HLS design is synthesized only once.

Next, generate a wrapper with the required number of resident banks::

    from hls4ml.contrib.runtime_weights import package

    summary = package(hls_model, n_banks=3)

``package()`` checks the synthesized parameter interfaces and writes a
``runtime_weights/`` design containing the original compute IP together with the
banked parameter storage.

Preparing the banks
-------------------

Provide one complete parameter set for each trained model variant. Each parameter
set is keyed by ``(layer, role)``::

    params_a = {
        ("dense_1", "weight"): weights_a,
        ("dense_1", "bias"): bias_a,
        ("dense_2", "weight"): common_weights,
        ("dense_2", "bias"): common_bias,
    }

    params_b = {...}
    params_c = {...}

Pack all variants together::

    from hls4ml.contrib.runtime_weights import pack_banks

    images = pack_banks(
        hls_model,
        [params_a, params_b, params_c],
    )

Each dictionary must contain the complete model parameter set.

Only parameters externalized by ``BramFactor`` may differ between banks.
Parameters that remain compiled into the compute IP must be identical in every
variant. ``pack_banks()`` checks both conditions and rejects incomplete or
incompatible parameter sets.

``hls_model`` must be the model used to build the synthesized IP, normally
corresponding to bank 0.

Dense layers over rank-1, rank-2 and rank-3 inputs are supported. Parameter shapes
are checked exactly rather than silently reshaped.

Loading the banks
-----------------

``pack_banks()`` returns the packed banks keyed by port name. BRAM-backed
parameters (Dense weights) are preloaded into the bitstream: write each one with
``write_mem()`` and bind the file to the wrapper's ``<PORT>_INIT_HEX`` parameter
when instantiating it::

    from hls4ml.contrib.runtime_weights import write_mem

    for name, img in images.items():
        if img['kind'] == 'bram':
            write_mem(f'{name}.hex', img['image'], img['data_width'])

Scalar bundles (Dense biases) have no preload path. They power up as zero and
must be written after reset, while ``quiescent`` is high, through the wrapper's
single loader interface::

    ld_req, ld_param_id, ld_bank, ld_addr, ld_data  ->  ld_accept | ld_reject

One request writes one element of one bank; ``img['codes'][bank]`` are the
values, and ``runtime_weights.json`` gives each parameter's ``param_id`` and
address range. Out-of-range or busy requests are rejected and write nothing.
Getting the values from a host into these signals is the surrounding design's
job.

The same loader can also fill or later replace BRAM banks instead of
``INIT_HEX``, without regenerating the HLS IP or the bitstream, as long as the
new parameters pack into the already-built geometry. This is the advanced path;
see ``hls4ml/contrib/runtime_weights/README.md``.

Selecting a bank at runtime
---------------------------

Each bank has an integer id::

    0  -> parameter set A
    1  -> parameter set B
    2  -> parameter set C

For each inference, provide the desired ``ext_bank_id`` together with the normal
start request.

The wrapper samples ``ext_bank_id`` when the request is accepted and stores it
internally for the complete inference. Therefore ``ext_bank_id`` does not need to
remain unchanged while the inference is running; changing it only affects a later
request.

A controller may choose the bank from any external condition or configuration::

    if condition_a:
        bank_id = 0
    elif condition_b:
        bank_id = 1
    else:
        bank_id = 2

    if ext_ap_ready:
        ext_bank_id = bank_id
        ext_ap_start = 1

No fixed number of clock cycles needs to be known. A new request is issued when
``ext_ap_ready`` indicates that the wrapper can accept it.

Output handling
---------------

The wrapper preserves the synthesized IP's output-valid signals. Each output should
therefore be sampled using its corresponding ``ap_vld`` signal rather than assuming
that every output becomes valid on ``ap_done``.

Integrating the wrapper
-----------------------

``package()`` writes ``runtime_weights/rtl/`` next to the HLS project. Add it and
the HLS RTL (``<project>_prj/solution1/syn/verilog/*.v``) to your sources and
instantiate ``<project>_runtime_weights`` as one block in your existing design.
The surrounding design supplies data, transaction control
(``ext_ap_start``/``ext_ap_ready``/``ext_ap_done``), bank selection
(``ext_bank_id``) and any runtime parameter writes (``ld_*``).
``create_runtime_weights.tcl`` synthesizes the packaged design stand-alone as a
check; it does not create a Vivado project.

For Vivado IP Integrator compatibility, ``package()`` also generates a Verilog
shim ``<project>_runtime_weights_bd.v`` with identical ports. Port-level details
are in ``hls4ml/contrib/runtime_weights/README.md``.

API summary
-----------

``package(hls_model, n_banks)``
    Wrap an already-synthesized hls4ml design with ``n_banks`` resident parameter
    banks.

``pack_banks(hls_model, banks)``
    Validate complete model parameter sets and pack the parameters that were
    externalized by ``BramFactor``.

``write_mem(path, words, width_bits)``
    Write a packed BRAM ``image`` as a ``$readmemh`` file for ``<PORT>_INIT_HEX``.

The lower-level packing functions and the generated loader/RTL interfaces are
intended for integration and advanced use. Their details are documented in
``hls4ml/contrib/runtime_weights/README.md``.

Scope and limitations
---------------------

This version supports:

* the Vitis backend;
* ``io_parallel``;
* ``Strategy: Resource``;
* ``Dense`` layers over rank-1, rank-2 and rank-3 inputs;
* ``ap_fixed`` parameters using ``TRN``/``WRAP`` quantization;
* idle-time bank updates and one inference in flight at a time.

For Dense weights, reuse factor 1 is unsupported, and packed words wider than
4096 bits are outside the verified v1 scope. Scalar bundles (biases) are
loader-only; there is no ``INIT_HEX`` path for them in v1. The bank count is
fixed by ``package()``; changing it requires regenerating the wrapper, not
re-running HLS. AXI integration, a Vivado project or block design, board drivers
and automatic ``BramFactor`` selection are not provided. Unsupported layouts are
rejected rather than inferred.

Implementation details are documented in
`hls4ml/contrib/runtime_weights/README.md`.
