# Runtime parameter banks — advanced use and implementation notes

User-facing guide: `docs/advanced/runtime_parameter_banks.rst`. This file is for
integrators and contributors: the objects that cross module boundaries, the
generated files, the loader contract, how the feature works and how to extend it.

Post-export packaging only: the generated project is read, never modified, and the
summary records a SHA-256 fingerprint of the exported compute artifacts; comparing
one taken before packaging with one taken after is what establishes it.

## Objects

Semantic state that crosses a module boundary is object-backed; dictionaries stay
at the JSON boundaries and in local plumbing such as the RTL port list.

- `ExternalParameter` (`hls4ml/model/external_parameters.py`) is what the writer
  claims about one exposed parameter and owns the rule that turns its tensor into
  interface words: `pack()` = `flat_order.flatten()` then `layout.pack()` of the
  quantized codes. It validates its own geometry on construction and on load.
  `ExternalParameterManifest` is the list of them plus the project facts a
  consumer needs; the written project carries its serialized form.
- `BramInterface` / `ScalarBundleInterface` (`interface.py`) are produced by
  `verify()` only after cross-checking a parameter against the synthesis report
  and the RTL; `Wrapper` and `LoaderGeometry` (`package.py`) consume them.
- `BankImage` (`pack.py`) is what `pack_banks()` hands the user.

Every geometry an `ExternalParameter` carries is what hls4ml *asked* HLS for;
only synthesis establishes what was built.

## Selection

`ExternalParameters: [roles]` on a layer (`LayerName` or `LayerType` config; no
model-wide form, since roles belong to a layer) makes `register_bram_weights`
convert the weight to a `BramWeightVariable`; the writer then emits an interface
port for it and records it in the manifest. Selection is the only thing the new
config changes: the lowering to a port is hls4ml's existing `storage == 'bram'`
contract. The legacy `BramFactor` size threshold reaches the same pass and is
kept for compatibility; the parameter-banks flow does not depend on it.

`BramInterface` names the HLS `bram` port protocol the IP was synthesized with,
not the storage behind it: `parameter_bank.sv` implements block RAM today, and a
LUTRAM or register implementation would sit behind the same interface.

## Manifest

The written project carries `firmware/weights/external_parameters.json` (schema
`hls4ml.external_parameter_manifest/v1`), the serialized manifest: one entry per
exposed parameter. A 1-D Dense weight entry is:

```json
{"name": "w2", "layer": "dense_1", "role": "weight",
 "kernel_variant": "dense_resource_rf_leq_nin",
 "precision": {"class_name": "hls4ml.model.types.FixedPrecisionType", "state": {"width": 16, "integer": 6, ...}},
 "flat_order": {"tensor_axes": ["n_in","n_out"], "axes": ["n_out","n_in"], "shape": [8,4]},
 "layout": {"mode": "block", "block_size": 2, "lanes": 16},
 "expected_interface_kind": "bram", "expected_data_width": 256, "expected_depth": 2}
```

`flat_order` is a transpose then a ravel. `layout` maps that flat sequence into
words: for `mode: "block"`, scalar `f` lands in word `f % block_size` at lane
`f // block_size`. Both are structured, so no consumer parses an expression.

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
2. Write a `_describe_*` function returning the `ExternalParameter` fields it can
   prove (`interface_kind`, `data_width`, `depth`, a `FlatOrder`, a layout), or a
   `note` for anything it cannot.
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

## Synthesis reports and RTL inspection

`hls4ml.report.parse_interface_summary()` reads the HW interface from the C
synthesis reports: every RTL port with its protocol, direction and physical width
and the block-level control protocol come from `<top>_csynth.xml`; the logical
BRAM data width, which the XML does not carry (its `Bits` is the rounded-up port),
comes from the `* BRAM` table of `csynth.rpt`. That is the only textual parse.

`interface.py` is the only module that knows how Vitis spells the BRAM signals.
`verify()` resolves each parameter's signals against the real RTL port list and
hands the names to the `BramInterface`; `Wrapper` connects them and never rebuilds
a name. It also checks every signal's direction, the scalar-bundle members, and
that the IP is **read-only** on each memory (`WEN_A` and `WEN_B` proved inactive
in the Verilog) — which is what makes sharing a port with the loader sound.
`Din`/`WEN` are then left dangling, so the read interface (`Addr`, `EN`, `Dout`)
is what the width checks use. Whether the IP *reads* port B is deliberately not
checked: Dense leaves it idle, pointwise uses it.

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

## Generated files

`package()` writes `parameter_banks/` next to the HLS project:

| file | purpose |
|---|---|
| `rtl/<project>_parameter_banks.sv` | the wrapper: compute IP + bank storage + control; this is the module to integrate |
| `rtl/*.sv` | the modules it instantiates |
| `rtl/<project>_parameter_banks_bd.v` | optional Verilog shim for IP Integrator (below) |
| `create_parameter_banks.tcl` | in-memory synthesis check; writes `utilization.rpt`/`timing.rpt`/`drc.rpt`, creates no project |
| `parameter_banks.json` | port summary: pass-through ports, each banked port's kind/width/depth, bank count |

## Integrating from HDL

Instantiate the wrapper from the user's top, binding each BRAM image to its
`INIT_HEX` parameter:

```verilog
myproject_parameter_banks #(
    .W2_INIT_HEX("/abs/path/to/w2.hex")
) u_nn (
    .ap_clk(clk), .ap_rst(rst),
    // transaction: replaces the IP's ap_start / ap_ready / ap_done
    .ext_ap_start(start), .ext_bank_id(bank), .ext_ap_ready(ready), .ext_ap_done(done),
    .ext_bank_id_bad(),
    // network data ports and their ap_vld, exactly as with the plain hls4ml IP
    ...
    // loader, shared by every parameter
    .ld_req(ld_req), .ld_param_id(ld_id), .ld_bank(ld_bank), .ld_addr(ld_addr), .ld_data(ld_data),
    .ld_accept(ld_ack), .ld_reject(ld_nak),
    // status
    .cur_bank_id(), .busy(), .quiescent(idle)
);
```

A VHDL top instantiates it as a component; Vivado resolves the mixed-language
boundary. `INIT_HEX` is read by `$readmemh` at elaboration, so an absolute path
avoids depending on the tool's working directory; an unbound parameter leaves the
memory initialized to zero (`parameter_bank.sv` clears every word before the
optional `$readmemh`).

## Loader

One interface serves every parameter; its widths are set by the widest one:

| port | width | meaning |
|---|---|---|
| `ld_req` | 1 | request; sampled on the clock edge, answered the same cycle |
| `ld_param_id` | `loader.param_id_width` | `banked_ports[i].param_id` |
| `ld_bank` | `bank_id_width` | bank to write |
| `ld_addr` | `loader.addr_width` | word (BRAM) or element (scalar bundle), `0 .. ld_depth-1` |
| `ld_data` | `loader.data_width` | value in the low `banked_ports[i].ld_data_width` bits; upper bits ignored |
| `ld_accept` / `ld_reject` | 1 | exactly one is high while `ld_req` is |

All of these numbers are in `parameter_banks.json`, which is authoritative for
the wrapper it was generated with: `param_id` is assigned by `package()` and must
be read from there, not assumed stable across regenerated designs. `ld_data` is as
wide as the widest packed word (up to 4096 bits in the verified scope); assembling
such words on the host side is the surrounding framework's concern. The wrapper compares
`ld_param_id` and the full-width `ld_addr` against the selected parameter *before*
narrowing the address to the port's own width, so an out-of-range value is
rejected rather than aliased onto a valid location; the bank modules check
`ld_bank` the same way. A request during an inference is rejected too. Rejected
requests write nothing.

From `pack_banks()` output, the write for `(param_id, bank, addr)` is
`img.per_bank[bank][addr]` for every kind of parameter.

### Runtime replacement of BRAM contents

`INIT_HEX` is the default for memory-interface parameters because the sets are normally known
at build time. The loader can do the same job, or replace a bank later, without
touching the HLS IP or the bitstream:

```python
images = pack_banks(hls_model, new_sets)      # same model, new parameter values
for name, img in images.items():
    pid = param_id[name]                      # parameter_banks.json
    for bank, words in enumerate(img.per_bank):
        for addr, data in enumerate(words):
            write(pid, bank, addr, data)      # one ld_req, while quiescent
```

Only the contents change: `pack_banks()` still checks the new sets against the
built model, so geometry, precision and layout stay what the IP was synthesized
for. Moving the words from a host to `ld_*` (AXI, PCIe, registers) belongs to the
surrounding framework and is out of scope here.

Scalar bundles power up as zero in every bank, including bank 0, so their values
must be loaded before the first inference (`BankImage.per_bank` holds them). Memory-interface parameters may instead be preloaded
through `INIT_HEX`.

## IP Integrator shim (optional)

Vivado 2025.2 synthesizes the SystemVerilog top without complaint but refuses it
as the top of a Module Reference in IP Integrator (`filemgmt 56-195`); that
behaviour was verified manually on 2025.2 only. `package.py` therefore also emits
`<top>_bd.v`, a plain Verilog-2001 module forwarding every port and `*_INIT_HEX`
parameter 1:1 with no logic of its own. It is generated unconditionally: it has no
functional content and a flag would cost more than the file. Use it with
**Add Module** or `create_bd_cell -type module -reference <top>_bd`; the
`*_INIT_HEX` parameters appear in the customization dialog. Both files are printed
from the same `Wrapper.port_lines()` so they cannot drift; the generated Tcl
synthesizes through the shim, the XSim testbench instantiates it, and
`test_vivado_ip_integrator_accepts_the_shim` adds it to a block design with
whatever Vivado the test environment provides.

## Synthesis artifacts

Vitis initializes generated ROMs with `$readmemh("./<name>.dat")`, resolved against
the working directory. `dense_resource_rf_gt_nin_rem0` emits one. The packager
copies these next to the generated Tcl, the Tcl `cd`s there, and they are included
in the fingerprint — their contents change what the IP computes exactly as the
Verilog does. They are deliberately **not** `add_files`'d: Vivado deletes a data
file added as a source.
