# Runtime-selected weight banks

Give a synthesized hls4ml IP several pre-provisioned weight banks and choose one
per inference, without re-running HLS.

This is **post-export packaging**. It reads a project that has already been
synthesized and writes a wrapper around it; the compute IP is never modified.

## Requirements

The IP must have been produced with parameters exposed outside it:

```python
config['Model']['BramFactor'] = 0      # or a threshold; nothing is automatic
config['Model']['Strategy'] = 'Resource'
```

`BramFactor` makes hls4ml write `firmware/weights/external_parameters.json`
describing each external parameter. That manifest is the contract this package
consumes.

## Use

```python
from hls4ml.contrib.runtime_weights import package

hls_model.build(csim=False, synth=True)
summary = package(hls_model, n_banks=4)
```

This writes, under `<output_dir>/runtime_weights/`:

| | |
| --- | --- |
| `rtl/<project>_runtime_weights.sv` | generated top: the unmodified IP plus banked storage |
| `rtl/*.sv` | the parameterized wrapper modules |
| `create_runtime_weights.tcl` | Vivado synthesis of the wrapper, no board constraints |
| `runtime_weights.json` | what was generated, including a fingerprint of the exported IP |

Bank images are packed from the manifest:

```python
from hls4ml.contrib.runtime_weights import build_bank_image, write_mem

port = manifest['ports'][0]
image, stride = build_bank_image(port, [w_bank0, w_bank1])
write_mem('w2_banks.mem', image, port['expected_data_width'])
```

and bound at elaboration through the per-port `<PORT>_INIT_HEX` parameter, or
written at run time through the native loader.

## Native loader

Each banked parameter gets a loader port on the generated top:

| BRAM port | Scalar bundle |
| --- | --- |
| `ld_<p>_req`, `ld_<p>_bank`, `ld_<p>_word`, `ld_<p>_wdata` | `ld_<p>_we`, `ld_<p>_bank`, `ld_<p>_idx`, `ld_<p>_data` |
| `ld_<p>_accept`, `ld_<p>_reject` | `ld_<p>_accept`, `ld_<p>_reject` |

A write is accepted only while `quiescent` is high. Out-of-range bank ids, word
addresses and scalar indices are refused on `ld_<p>_reject`, so every request is
either performed or reported as refused — it is never silently dropped.

Sequence to reload a bank:

1. wait for `quiescent`
2. drive one write per word (or per scalar) and check `ld_<p>_accept`
3. select the bank on a later inference via `ext_bank_id`

## Bank selection

`bank_id` is captured when the wrapper takes a request, made stable *before*
`ap_start` rises, and held through `ap_done`, so every parameter read belongs to
the selected bank. An out-of-range id is rejected and starts nothing.

Selection is **idle-time only**: a new start is gated away while a transaction is
in flight. That gives up any back-to-back capability the IP has. Overlapping
transactions are not supported.

The bank count is fixed when the wrapper is generated — the packed image, the
bank stride and `runtime_weights.json` all encode it. Changing it means re-running
this packager, not re-running HLS.

## Scope

Verified for the **Vitis** backend, `io_parallel`, `Strategy: Resource`, Dense
weights and biases, `ap_fixed` with `AP_TRN`/`AP_WRAP`, one inference at a time.

Anything outside that is **refused**, not guessed at: a model with an external
parameter this version cannot describe raises rather than banking only part of it.
Adding a layer means adding a manifest adapter in
`hls4ml/writer/external_parameters.py` plus evidence that its packing was checked
against generated RTL.

Not provided: AXI or board support, drivers, a C++ wrapper, automatic changes to
`BramFactor`, and overlapping transactions.

## Example

A runnable end-to-end example — C simulation, co-simulation, two-bank RTL
simulation and Vivado synthesis, all compared bit-exact — is described in
`docs/advanced/bramfactor.rst`.
