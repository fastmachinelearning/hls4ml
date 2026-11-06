=======================
Runtime parameter banks
=======================

``hls4ml.contrib.parameter_banks`` lets several trained parameter sets of the
same network reside on the FPGA at once. A bank is selected for each inference
without re-running HLS or modifying the synthesized compute IP.

A typical use case is one architecture trained for several operating conditions,
configurations, or deployment modes.

The flow is: mark the parameters that vary with ``ExternalParameters`` and
``build(synth=True)`` once; ``package()`` the wrapper with the parameter sets;
use the wrapper in place of the IP and select a bank with ``bank_id`` per
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

Give ``package()`` one complete parameter set per bank, keyed by ``(layer, role)``::

    from hls4ml.contrib.parameter_banks import package

    params_a = {('dense_1', 'weight'): w1_a, ('dense_1', 'bias'): b1_a, ('dense_2', 'weight'): w2_a, ...}
    package(hls_model, banks=[params_a, params_b, params_c])

``package()`` checks the synthesized interface of every exposed parameter against
what hls4ml asked for and writes ``parameter_banks/`` next to the project: the
original compute IP wrapped with bank storage, bank selection and a loader, with
the parameter sets preloaded. The number of banks is a power of two, by default
the smallest that holds the sets (four here); banks beyond them start as zero.

Only exposed parameters may differ between sets; the rest must be identical in
every set, and incomplete or inconsistent sets are rejected. Tensor shapes are
checked exactly. ``hls_model`` must be the model the IP was built from.

Selecting a bank at runtime
---------------------------

Banks are numbered in the order of ``banks``. For each inference, present
``bank_id`` with ``ap_start`` and hold both until ``ap_ready``, as for
``ap_start`` on the plain IP. Requests for the bank
already selected go straight to the IP, so inferences overlap at the IP's own
initiation interval. A request for another bank waits until the IP is idle, so a
bank change costs about one latency of the IP.

Sample each output on its own ``ap_vld``, as with the plain IP.

Updating bank contents
----------------------

Banks can be rewritten at run time, without a new bitstream. ``pack_banks()``
packs new sets as ``package()`` does::

    from hls4ml.contrib.parameter_banks import pack_banks

    images = pack_banks(hls_model, [params_d])  # images[name].per_bank[0] holds the words to write

One loader interface serves every parameter::

    ld_req, ld_param_id, ld_bank, ld_addr, ld_data  ->  ld_accept | ld_reject

Each request writes one word (memory interface) or one element (scalar bundle)
of one bank. ``parameter_banks.json`` gives every parameter its ``param_id``, valid
address range (``ld_depth``) and how many low bits of ``ld_data`` it uses
(``ld_data_width``). A request with an unknown id or an out-of-range address, or
issued while the IP is busy, is rejected and writes nothing. Tie ``ld_req`` low
if banks are never rewritten.

Integrating the wrapper
-----------------------

Add ``parameter_banks/rtl/`` (the ``.sv`` sources and the ``.mem`` preload images)
and the HLS RTL (``<project>_prj/solution1/syn/verilog/*.v``) to your sources and instantiate
``<project>_parameter_banks`` as the single accelerator block in your design.
It already contains the original HLS compute IP, parameter storage, bank-selection
logic and loader, so the HLS IP should not be instantiated separately.
The surrounding design drives data and control exactly as for the plain IP
(``ap_start``, ``ap_done``, ``ap_idle``, ``ap_ready``), plus ``bank_id`` and
any runtime parameter writes (``ld_*``).
``create_parameter_banks.tcl`` synthesizes the packaged design stand-alone as a
check; it does not create a Vivado project.

For Vivado IP Integrator compatibility, ``package()`` also generates a Verilog
shim ``<project>_parameter_banks_bd.v`` with identical ports. Port-level details
are in ``hls4ml/contrib/parameter_banks/README.md``.

Scope and limitations
---------------------

This version supports the Vitis backend, ``io_parallel``, ``Strategy: Resource``,
``Dense`` layers over rank-1, rank-2 and rank-3 inputs, ``ap_fixed`` parameters
with ``TRN``/``WRAP`` quantization, and loader writes while the IP is idle.

Dense weights need a reuse factor above 1, and packed words wider than 4096 bits
are outside the verified scope. The bank count is
fixed by ``package()``; changing it regenerates the wrapper, not the HLS. AXI
integration, a Vivado project or block design, and board drivers are not provided.
Unsupported layouts are rejected rather than inferred.
