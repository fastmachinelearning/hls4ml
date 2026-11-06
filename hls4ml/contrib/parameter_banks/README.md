# Runtime parameter banks: implementation notes

User guide: `docs/advanced/runtime_parameter_banks.rst`. This file is for integrators and contributors.

`package()` only reads the synthesized project; the compute IP is never modified, and `parameter_banks.json`
records a SHA-256 fingerprint of its RTL and ROM data to show it.

## Objects

- `ExternalParameter` (`hls4ml/model/external_parameters.py`) describes one exposed parameter and turns its
  tensor into interface words: `pack()` is `flat_order.flatten()` then `layout.pack()` of the quantized codes.
  `ExternalParameterManifest` holds them with the project facts, and is saved as
  `firmware/weights/external_parameters.json`.
- `BramInterface` / `ScalarBundleInterface` (`interface.py`) exist only after a parameter has been checked
  against the synthesis report and the RTL; `Wrapper` (`package.py`) generates the RTL from them.
- `BankImage` (`pack.py`) is what `pack_banks()` returns.

A manifest entry for a Dense weight:

```json
{"name": "w2", "layer": "dense_1", "role": "weight", "kernel_variant": "dense_resource_rf_leq_nin",
 "flat_order": {"tensor_axes": ["n_in","n_out"], "axes": ["n_out","n_in"], "shape": [8,4]},
 "layout": {"mode": "block", "block_size": 2, "lanes": 16},
 "expected_interface_kind": "bram", "expected_data_width": 256, "expected_depth": 2, ...}
```

`flat_order` is a transpose then a ravel; with a `block` layout, scalar `f` lands in word `f % block_size`
at lane `f // block_size`.

## Layouts: the extension point

The Vitis pass `vitis:describe_external_parameters` (`hls4ml/backends/vitis/passes/external_parameters.py`)
attaches an `ExternalParameter` to every external weight from `_LAYOUTS`, keyed
`(io_type, strategy, layer_class, role)`. Anything else gets only a note, and warns at conversion when it
was selected with `ExternalParameters`. Two geometries are verified:

| | Dense | PointwiseConv1D/2D |
|---|---|---|
| word | `ARRAY_RESHAPE block` lanes of one scalar each | one scalar |
| depth | the reuse factor | `n_chan × n_filt` |
| IP reads port B | no | yes |

A Dense over 2-D/3-D input is lowered to a pointwise convolution, and its Keras kernel is accepted as is.

To add a layer: synthesize a few configurations and read the real BRAM width, depth and address shift from
the reports and RTL; write a function returning the `ExternalParameter` it can prove, raising `Unbankable`
otherwise; register it in `_LAYOUTS`; extend `test_banks_match_predict` with a case for it.

## What `package()` checks

`parse_vivado_report()` gives the RTL ports (`RtlPorts`, from `<top>_csynth.xml`) and the logical BRAM
widths (`BramInterfaces`, from `csynth.rpt`; the XML only has the port width, which Vitis rounds up to a
power-of-two number of bytes). `interface.py` then checks, against the generated Verilog, every BRAM signal
and its direction, the address shift that sets the byte stride between words, that the IP never writes the
memory and clocks both ports with `ap_clk`, and whether it reads port B. `interface.py` is the only place
that knows how Vitis names these signals.

## Memories and control

- `parameter_bank.sv`: the banks of one parameter stacked in depth, `bank * BANK_STRIDE_WORDS + word`.
  `memory='auto'` makes it LUTRAM when shallow and block RAM otherwise. When the IP never reads port B,
  that port is the loader's write port only, so block RAM needs only simple dual port.
- `scalar_bank_mux.sv`: a fully partitioned parameter (a Dense bias is one `ap_none` port per element),
  selected combinationally so the bank is present on the cycle the IP starts.
- `bank_select_latch.sv`: requests for the selected bank go straight to the IP; a request for another bank
  waits until the IP is idle. The loader writes only while the IP is idle and no request is presented.

## Generated files

| file | purpose |
|---|---|
| `rtl/<project>_parameter_banks.sv` | the wrapper to integrate |
| `rtl/*.sv` | the modules it instantiates |
| `rtl/*.mem` | preload images, the wrapper's `<PORT>_INIT_HEX` defaults |
| `rtl/<project>_parameter_banks_bd.v` | Verilog shim with the same ports, for IP Integrator, which refuses a SystemVerilog top as a Module Reference |
| `*.dat` | ROM data of the HLS IP, read by `$readmemh` |
| `create_parameter_banks.tcl` | stand-alone place-and-route check |
| `parameter_banks.json` | bank count, memory styles, loader geometry and `param_id` of each parameter |

## Integrating from HDL

Add `rtl/` (sources and `.mem` files), the `.dat` files and the HLS RTL to the project, then:

```verilog
myproject_parameter_banks u_nn (
    .ap_clk(clk), .ap_rst(rst),
    .ap_start(start), .ap_done(done), .ap_idle(idle), .ap_ready(ready), .bank_id(bank),
    ...  // data ports and their ap_vld, as with the plain IP
    .ld_req(ld_req), .ld_param_id(ld_id), .ld_bank(ld_bank), .ld_addr(ld_addr), .ld_data(ld_data),
    .ld_accept(ld_ack), .ld_reject(ld_nak)  // tie ld_req low if banks are never rewritten
);
```

With one bank there are no `bank_id` and `ld_bank` ports. In IP Integrator, add the `_bd.v` shim with
**Add Module** or `create_bd_cell -type module -reference <project>_parameter_banks_bd`; its `*_INIT_HEX`
parameters appear in the customization dialog.

## Loader

| port | width | meaning |
|---|---|---|
| `ld_req` | 1 | write request, answered in the same cycle |
| `ld_param_id` | `loader.param_id_width` | `param_id` from `parameter_banks.json` |
| `ld_bank` | `bank_id_width` | bank to write |
| `ld_addr` | `loader.addr_width` | word (memory) or element (scalars), below `ld_depth` |
| `ld_data` | `loader.data_width` | value in the low `ld_data_width` bits |
| `ld_accept` / `ld_reject` | 1 | exactly one is high while `ld_req` is |

An unknown id, an out-of-range address or a request while the IP is busy is rejected and writes nothing.
Read `param_id` from `parameter_banks.json`: it can change when the wrapper is regenerated. `ld_data` is as
wide as the widest packed word (up to 4096 bits); assembling it on the host side is up to the surrounding
framework.
New parameter sets are packed by `pack_banks()` as for preloading:

```python
images = pack_banks(hls_model, [new_set])
for name, image in images.items():
    for addr, word in enumerate(image.per_bank[0]):
        write(param_id[name], bank, addr, word)  # one ld_req while the IP is idle
```
