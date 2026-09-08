# Runtime-selected weight banks — implementation notes

User-facing guide: `docs/advanced/bramfactor.rst`. This file covers how the feature
works and what it takes to extend it.

Post-export packaging only: the generated project is read, never modified, and the
summary records a SHA-256 fingerprint of the exported compute artifacts; comparing
one taken before packaging with one taken after is what establishes it.

## Manifest

`BramFactor` makes the writer emit `firmware/weights/external_parameters.json`
(schema `hls4ml.external_parameter_manifest/v1`), one entry per external parameter.
For example, a 1-D Dense weight entry is:

```json
{"name": "w2", "layer": "dense_1", "role": "weight",
 "kernel_variant": "dense_resource_rf_leq_nin",
 "flat_order": {"tensor_axes": ["n_in","n_out"], "axes": ["n_out","n_in"], "shape": [8,4]},
 "layout": {"mode": "block", "block_size": 2, "lanes": 16},
 "expected_interface_kind": "bram", "expected_data_width": 256, "expected_depth": 2}
```

`flat_order` is a transpose then a ravel. `layout` maps that flat sequence into
words: for `mode: "block"`, scalar `f` lands in word `f % block_size` at lane
`f // block_size`. Both are structured, so no consumer parses an expression.

Every geometry field is named `expected_*`: the writer records what hls4ml *asked*
HLS for. Only synthesis establishes what was built, which is what `interface.py`
cross-checks before anything is generated.

## Adapters — the extension point

Descriptions come from `_ADAPTERS` in `hls4ml/writer/external_parameters.py`, keyed
`(backend, io_type, strategy, layer_class, role)`. The writer sees *every* external
parameter, so an unregistered combination gets no interface kind, geometry or
ordering — only a note. Nothing is ever guessed.

Two geometries are registered today, and they are genuinely different:

| | Dense | PointwiseConv1D/2D |
|---|---|---|
| reshape reaches the port | yes, `ARRAY_RESHAPE block factor=N` | **no** |
| word width | `block_factor × precision` | one scalar |
| depth | `block_size` (= reuse factor) | `n_chan × n_filt` |
| depends on reuse factor | yes | no |
| IP reads port B | no | **yes** |

A `Dense` over a 2-D/3-D input and a native `Conv*D` with a 1-wide kernel produce
the same layer with the same declared shape, so the pointwise adapter keys on the
class, not the origin. It still checks `filt_width`/`filt_height` and the
`linebuffer` implementation rather than trusting the class name.

### Adding a layer

1. Characterize it: synthesize a few configurations and read the real BRAM width,
   depth and address shift from the csynth report and generated RTL. Do not reason
   by analogy with an existing layer — the pointwise geometry is nothing like Dense.
2. Write a `_describe_*` function returning `flat_order` and `layout`, refusing with
   a `note` for anything it cannot prove.
3. Register it in `_ADAPTERS`.
4. Add a two-bank XSim test that shows switching banks changes the result.

A layer that satisfies an existing verified memory contract should need only an
adapter and tests. A genuinely different interface needs a new contract rather than
being forced into an existing one.

## Widths: logical, physical, stride

Three quantities, and only the first comes from the report:

```
logical width   the packed word this schema builds     e.g. 96
physical width  the RTL port that carries it           e.g. 128
byte stride     spacing between consecutive words      e.g. 16
```

Vitis rounds a parameter port up to a power-of-two byte count, so a 96-bit word
travels on a 128-bit port and strides by 16 rather than 12. `verify()` computes
`ceil(width/8)` — fixed-point words need not be byte-aligned — rounds up to the
stride, and requires the RTL's address shift to match. `parameter_bank` pads
logical→physical explicitly; nothing relies on implicit Verilog width extension.

## RTL inspection

`interface.py` is the only module that knows how Vitis spells anything.
`verify()` resolves each parameter's signals against the real port list and returns
the names; `package.py` consumes them and never rebuilds a name. It also checks
every signal's direction, the scalar-bundle members, and that the IP is
**read-only** on each memory (`WEN_A` and `WEN_B` proved inactive) — which is what
makes sharing a port with the loader sound. `Din`/`WEN` are then left dangling, so
the read interface (`Addr`, `EN`, `Dout`) is what the width checks use. Whether the IP *reads*
port B is deliberately not checked: Dense leaves it idle, pointwise uses it.

`solution_verilog_dir()` is the single place the Vitis project layout appears.

## Memories and the loader

`parameter_bank.sv` is a depth-stacked memory, `bank_id * BANK_STRIDE_WORDS +
local_word`, with `bank_addr_mapper` translating the IP's byte address on each port.
Both ports are read ports for the IP; the loader takes **port B only while
quiescent**, when the IP is not reading. That is why the wrapper is layer-agnostic.

`scalar_bank_mux.sv` handles fully partitioned parameters — a Dense bias lowers to
one `ap_none` port per element regardless of size. Its select is combinational, so
the selected bank is present on the cycle the IP starts. It has per-bank storage but
no `INIT_HEX` path in v1: loader-only.

`bank_select_latch.sv` owns the transaction. Acceptance requires `hls_ap_idle`, not
merely `!busy` — under `ap_ctrl_hs` idle follows done by a cycle — and a transaction
whose `ap_ready` and `ap_done` coincide (a pointwise convolution does) must not
latch `busy`, or nothing would ever clear it.

## Synthesis artifacts

Vitis initializes generated ROMs with `$readmemh("./<name>.dat")`, resolved against
the working directory. `dense_resource_rf_gt_nin_rem0` emits one. The packager
copies these next to the generated Tcl, the Tcl `cd`s there, and they are included
in the fingerprint — their contents change what the IP computes exactly as the
Verilog does. They are deliberately **not** `add_files`'d: Vivado deletes a data
file added as a source.
