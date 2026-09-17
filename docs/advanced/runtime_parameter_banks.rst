=======================
Runtime parameter banks
=======================

``hls4ml.contrib.parameter_banks`` lets several trained parameter sets of the
same network reside on the FPGA at once. A bank is selected for each inference
without re-running HLS or modifying the synthesized compute IP.

A typical use case is one architecture trained for several operating conditions,
configurations, or deployment modes.

The flow is: mark the parameters that vary with ``ExternalParameters`` and
``build(synth=True)`` once; ``package()`` the wrapper; ``pack_banks()`` the
parameter sets; write BRAM images and load scalar parameters through the wrapper;
instantiate the wrapper in your design; select a bank with ``ext_bank_id`` per
inference.

Exposing parameters
-------------------

Name, per layer, the parameter roles that must be selectable at runtime. This
is per-layer configuration, so create the config with ``granularity='name'``::

    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Vitis')
    config['Model']['Strategy'] = 'Resource'
    config['LayerName']['dense_1']['ExternalParameters'] = ['weight', 'bias']
    config['LayerName']['dense_2']['ExternalParameters'] = ['weight', 'bias']

    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, backend='Vitis', ...)
    hls_model.build(csim=False, synth=True)

Those parameters leave the compute IP and become interfaces on it; everything
else is compiled in from this model as usual. The HLS design is synthesized only
once.

Packaging
---------

Generate the wrapper with the required number of resident banks::

    from hls4ml.contrib.parameter_banks import package

    summary = package(hls_model, n_banks=3)

``package()`` checks the synthesized interface of every exposed parameter against
what hls4ml asked for and writes ``parameter_banks/`` next to the project: the
original compute IP wrapped with bank storage, bank selection and a loader.

Preparing the banks
-------------------

Provide one complete parameter set per variant, keyed by ``(layer, role)``::

    from hls4ml.contrib.parameter_banks import pack_banks

    params_a = {('dense_1', 'weight'): w1_a, ('dense_1', 'bias'): b1_a, ('dense_2', 'weight'): w2_a, ...}
    images = pack_banks(hls_model, [params_a, params_b, params_c])

Only exposed parameters may differ between banks; the rest must be identical in
every set, and ``pack_banks()`` rejects incomplete or inconsistent sets. Tensor
shapes are checked exactly. ``hls_model`` must be the model the IP was built from.

Writing the bank contents
-------------------------

``images`` maps each exposed parameter's port name to its packed banks. A
parameter with a memory interface (for example, a Dense weight) has a
``$readmemh`` image to bind to the wrapper's ``<PORT>_INIT_HEX`` parameter::

    for name, img in images.items():
        if img.kind == 'bram':
            img.write_mem(f'{name}.hex')

Scalar bundles (Dense biases) have no preload path: they power up as zero in every
bank and must be written through the wrapper's loader while ``quiescent`` is
high, before the first inference. ``img.per_bank`` holds the values to write.

Selecting a bank at runtime
---------------------------

Banks are numbered in the order given to ``pack_banks()``. For each inference,
present ``ext_bank_id`` with ``ext_ap_start`` when ``ext_ap_ready`` is high; the
wrapper latches the id for the whole inference, so it may change afterwards. No
fixed cycle count is needed.

Sample each output on its own ``ap_vld``, as with the plain IP.

Updating bank contents
----------------------

One loader interface serves every parameter::

    ld_req, ld_param_id, ld_bank, ld_addr, ld_data  ->  ld_accept | ld_reject

Each request writes one word (memory interface) or one element (scalar bundle)
of one bank. ``parameter_banks.json`` gives every parameter its ``param_id``, valid
address range (``ld_depth``) and how many low bits of ``ld_data`` it uses
(``ld_data_width``). A request with an unknown id, an out-of-range bank or
address, or issued during an inference is rejected and writes nothing.

Integrating the wrapper
-----------------------

Add ``parameter_banks/rtl/*.sv`` and the HLS RTL
(``<project>_prj/solution1/syn/verilog/*.v``) to your sources and instantiate
``<project>_parameter_banks`` as one block in your existing design. The
surrounding design supplies data, transaction control
(``ext_ap_start``/``ext_ap_ready``/``ext_ap_done``), bank selection
(``ext_bank_id``) and any runtime parameter writes (``ld_*``).
``create_parameter_banks.tcl`` synthesizes the packaged design stand-alone as a
check; it does not create a Vivado project.

For Vivado IP Integrator compatibility, ``package()`` also generates a Verilog
shim ``<project>_parameter_banks_bd.v`` with identical ports. Port-level details
are in ``hls4ml/contrib/parameter_banks/README.md``.

Scope and limitations
---------------------

This version supports the Vitis backend, ``io_parallel``, ``Strategy: Resource``,
``Dense`` layers over rank-1, rank-2 and rank-3 inputs, ``ap_fixed`` parameters
with ``TRN``/``WRAP`` quantization, idle-time bank updates and one inference in
flight at a time.

Dense weights need a reuse factor above 1, and packed words wider than 4096 bits
are outside the verified scope. Scalar bundles are loader-only. The bank count is
fixed by ``package()``; changing it regenerates the wrapper, not the HLS. AXI
integration, a Vivado project or block design, and board drivers are not provided.
Unsupported layouts are rejected rather than inferred.
