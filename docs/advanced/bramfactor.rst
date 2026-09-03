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

External parameter manifest
===========================

When ``BramFactor`` exposes parameters externally, ``hls4ml`` also writes
``firmware/weights/external_parameters.json``. It records, per external port, what
the generated design is expected to look like: the order in which ``hls4ml``
flattened the tensor, the ``ARRAY_RESHAPE`` it emitted, the resulting word width
and depth, and the fixed-point format. This is what a weight-reloading scheme
needs in order to know which logical scalar lands in which physical memory word
and lane.

It covers more than memories: a fully partitioned parameter such as a Dense bias
becomes a bundle of scalar ports, and is described here too.

.. code-block:: json

    {
      "schema": "hls4ml.external_parameter_manifest/v1",
      "ports": [
        {
          "name": "w2", "layer": "dense_1", "role": "weight",
          "kernel_variant": "dense_resource_rf_leq_nin",
          "flat_order": {"tensor_axes": ["n_in", "n_out"], "axes": ["n_out", "n_in"]},
          "layout": {"mode": "block", "block_size": 2, "lanes": 16},
          "expected_interface_kind": "bram",
          "expected_data_width": 256, "expected_depth": 2
        }
      ]
    }

``flat_order`` gives the tensor's own axis names and the order to enumerate them
in, so flattening is a transpose followed by a ravel. ``layout`` says how that
flat sequence maps into memory words: with ``mode: "block"``, lane is
``f // block_size`` and word is ``f % block_size``. Both are structured values, so
a consumer never has to parse an expression.

Two properties of the manifest are worth understanding:

* Every geometry field is named ``expected_*``. ``hls4ml`` knows what it *asked*
  HLS for; only C/RTL synthesis or export establishes what was actually built.
  Consumers should cross-check against the synthesis report before relying on it.
* Descriptions come from a registry of adapters keyed by
  ``(backend, io_type, strategy, layer_class, role)``. ``hls4ml`` sees every
  external parameter, including layers whose packing has not been checked, so an
  unregistered combination gets no interface kind, geometry or ordering - only a
  note, never a guess. Schema v1 registers the **Vitis backend only**, and within
  it ``Dense`` weights and biases with ``io_parallel`` and ``Strategy: Resource``.
  The Vivado backend also implements ``BramFactor`` but has not been verified, so
  it receives no claims.

Note that a ``Dense`` bias is *not* exposed as a memory even when ``BramFactor``
selects it: ``nnet_dense_resource.h`` fully partitions ``biases``, so it lowers to
individual scalar ports regardless of size. The manifest reports this as
``expected_interface_kind: "scalar_bundle"``.

Runtime-selected weight banks
=============================

``hls4ml.contrib.runtime_weights`` builds on the manifest to give a synthesized
design several pre-provisioned weight banks, selected per inference, without
resynthesizing the HLS IP. **This version supports the Vitis backend only.** It
runs *after* ``build(synth=True)`` and only reads the generated project, so the
compute IP is never modified. The summary records a fingerprint of the exported
RTL; comparing one taken before packaging with one taken after is what shows the
IP was left alone.

.. code-block:: Python

    from hls4ml.contrib.runtime_weights import package

    hls_model.build(csim=False, synth=True)
    summary = package(hls_model, n_banks=2)

This writes a ``runtime_weights/`` directory containing a generated top level that
instantiates the unmodified IP alongside banked storage per supported parameter
interface - a depth-stacked memory for a BRAM port, a scalar bank for a bundle
such as a Dense bias - plus the parameterized RTL and a Vivado Tcl script.
Bank images are produced with the same package:

.. code-block:: Python

    from hls4ml.contrib.runtime_weights import build_bank_image, write_mem

    image, stride = build_bank_image(port, [bank0_weights, bank1_weights])
    write_mem("w2_banks.mem", image, port["expected_data_width"])

Packing is driven by the manifest's structured ``flat_order`` and ``layout``.
Ports without both are rejected.

The image is bound to the wrapper through the generated top's per-port
``<PORT>_INIT_HEX`` parameter, which is read with ``$readmemh`` at elaboration:

.. code-block:: systemverilog

    my_prj_runtime_weights #(
        .N_BANKS(2),
        .W2_INIT_HEX("w2_banks.mem")
    ) u_wrapper ( ... );

Banks can equally be written at run time through the loader ports, which is the
point of the feature; ``INIT_HEX`` just gives the memory a defined starting state.

Bank selection is **idle-time only**. The wrapper captures ``bank_id`` when it
takes a request, makes it stable *before* asserting ``ap_start``, and holds it
through ``ap_done``. Acceptance follows the real ``ap_ctrl_hs`` handshake:
``ap_start`` is held until the IP asserts ``ap_ready``. Every parameter read
therefore belongs to the selected bank, including any read in the first cycle.
An out-of-range ``bank_id`` is rejected and starts nothing. While a transaction
is in flight a new start is gated away from the IP; overlapping transactions are
not supported, which gives up any back-to-back capability the IP may have.

Any valid bank may be written while idle; all writes are rejected while an
inference is active. Out-of-range bank ids, word addresses and scalar indices are
rejected too, so a write is either performed or reported as refused on
``ld_<port>_reject`` - never silently dropped.

Each banked parameter gets its own loader port on the generated top. A BRAM port
takes ``ld_<p>_req``, ``ld_<p>_bank``, ``ld_<p>_word`` and ``ld_<p>_wdata``; a
scalar bundle takes ``ld_<p>_we``, ``ld_<p>_bank``, ``ld_<p>_idx`` and
``ld_<p>_data``. Both report ``ld_<p>_accept`` and ``ld_<p>_reject``. To reload a
bank: wait for ``quiescent``, drive one write per word or scalar while checking
``ld_<p>_accept``, then select that bank on a later inference through
``ext_bank_id``.

The bank count is fixed when the wrapper is generated, since the packed image,
the bank stride and the generated RTL all encode it. Changing it means re-running
the packager - not re-running HLS.

A model whose external parameters are not all describable is **refused**: banking
only some of them would leave the rest as unconnected top-level ports.

Three boundaries are worth knowing:

* **Fully partitioned weights are out of scope by construction**, not by omission.
  ``Strategy: Latency`` (and ``ARRAY_PARTITION ... complete`` generally) lowers a
  weight to one port per element with no address, so there is nothing to
  concatenate a bank id onto. Such a parameter *can* be banked through the same
  scalar path used for biases, but the selection mux then scales as
  ``total_weights x n_banks`` and sits in front of the compute - fine for a bias
  vector, wrong for a weight matrix.
* **Losing constant folding is inherent to reloadable weights**, not a cost of
  banking. Once a weight is a function argument rather than a literal, HLS can no
  longer fold the multiplication into shift-adds, so full multipliers appear
  whether or not banks are used. ``BramFactor`` alone already pays this.
* **Selection is idle-time only.** Per-event switching with transactions in
  flight is a different and harder feature, currently marked as future work.

The number of banks, the memory geometry and the physical capacity are fixed when
the design is implemented. Changing values within an existing bank requires only
memory writes; adding a bank, or changing the network, precision or reuse factor,
requires regenerating and re-implementing.

Not provided: AXI or board support, any driver, a C++ wrapper, and any automatic
change to ``BramFactor`` - which parameters are exposed remains the user's choice.
