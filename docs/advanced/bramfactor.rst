==================================
Loading weights from external BRAM
==================================

.. note::
    This feature is being evaluated for re-implementation. We welcome feedback from users how to make the implementation more flexible.

``hls4ml`` can optionally store weights in BRAMs external to the design. This is supported in Vivado/Vitis and Catapult backends. It is the responsibility of the user to ensure the weights are properly loaded during the operation of the design.

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

When integrating the design, users can use the exposed interface to implement weight reloading scheme.

Runtime-selected weight banks
=============================

``hls4ml.contrib.runtime_weights`` gives a synthesized design several resident
weight banks, selected per inference, without re-running HLS. It runs *after*
``build(synth=True)``, reads the generated project and writes a wrapper around it;
the compute IP is never modified.

Enabling it
-----------

The parameters you want to switch must be outside the compute IP, so set
``BramFactor`` and ``Strategy: Resource`` before conversion::

    config["Model"]["Strategy"] = "Resource"
    config["Model"]["BramFactor"] = 0       # 0 externalizes every parameter

``hls4ml`` then also writes ``firmware/weights/external_parameters.json``, the
manifest describing each external parameter. It is the contract everything below
consumes.

Packaging
---------

::

    from hls4ml.contrib.runtime_weights import pack_banks, package

    hls_model.build(csim=False, synth=True)
    summary = package(hls_model, n_banks=4)

``package()`` inspects the synthesized interface, checks it against the manifest,
and writes ``runtime_weights/``: a generated top that instantiates the unmodified
IP alongside banked storage, the parameterized RTL, and a Vivado Tcl script. It
fails rather than guessing if anything disagrees.

Filling the banks
-----------------

``pack_banks()`` takes one **complete** parameter set per bank, keyed by
``(layer, role)``::

    images = pack_banks(hls_model, [params_a, params_b, params_c])

Only parameters ``BramFactor`` externalized may differ between banks; every other
parameter is compiled into the IP and must be identical in all of them. Passing the
model rather than the manifest lets ``pack_banks`` see the full parameter list, so a
fixed parameter that *disagrees* and one that is merely *missing* are both reported.
All externalized parameters share one ``bank_id``: banks switch as a whole, never
per layer.

Tensors use hls4ml's declared layout for the parameter -- for a Dense kernel that is
``(n_in, n_out)``, the Keras orientation. A PyTorch ``nn.Linear.weight`` is its
transpose and must be transposed first. Shapes are checked exactly.

``pack_tensor``, ``build_bank_image`` and ``write_mem`` are the per-port primitives
underneath, for callers that want one port at a time.

Selecting a bank
----------------

Selection is **idle-time only**. The wrapper takes a request only while the IP
reports ``ap_idle``, captures ``bank_id`` before raising ``ap_start``, and holds it
for the whole transaction. An out-of-range id is rejected and starts nothing.

Any bank may be written while idle; every write is rejected while an inference is
active, and reported on ``ld_<port>_reject`` rather than dropped. A BRAM port takes
``ld_<p>_req/_bank/_word/_wdata``; a scalar bundle takes
``ld_<p>_we/_bank/_idx/_data``. Both report ``_accept`` and ``_reject``.

A bank is usable once *all* of its parameters have been loaded. Weight memories can
also be preloaded at elaboration through a ``<PORT>_INIT_HEX`` parameter; scalar
bundles have no such path in v1 and are loader-only.

Sample outputs on their own ``ap_vld``. The wrapper passes the IP's port-level
validity signals through unchanged, and with more than one output head the slower
one can assert valid after ``ap_done``.

Scope and limitations
---------------------

Supported: the **Vitis** backend, ``io_parallel``, ``Strategy: Resource``, and
``Dense``, ``PointwiseConv1D`` and ``PointwiseConv2D`` layers. The pointwise classes
are where a ``Dense`` over a 2-D or 3-D input lands, so those are covered too.

* ``ap_fixed`` with ``TRN``/``WRAP`` only; other rounding or saturation is refused.
* Reuse factor must leave the memory more than one word deep, and a packed word
  must not exceed 4096 bits.
* The bank count is fixed when the wrapper is generated. Changing it is a packager
  run, not an HLS run.
* Overlapping transactions are not supported.
* No AXI, board support, drivers, or automatic ``BramFactor`` selection.

Anything outside this is refused, not approximated: the manifest claims no layout
for it and ``package()`` raises. See ``hls4ml/contrib/runtime_weights/README.md``
for the implementation and for how to add a layer.
