# xcku040 example

End-to-end conversion, csim validation and optional C synthesis of a
small, single-conv CTLearn-style model, sized to fit comfortably on a
Xilinx XCKU040 and specifically used to validate real C synthesis of
this backbone (not just csim). The neighbor map and input data are
generated synthetically in the script, and the model uses random,
untrained weights, so this example needs no external files (see
`run_io_stream.py`'s own docstring, "Synthetic data").

## Model

```
Input(163, 20)
-> BatchNormalization
-> NeighborGatherLayer         K=7
-> IndexedConvolutionLayer     8 filters
-> BatchNormalization
-> GlobalAveragePooling1D
-> Dense(64, relu)
-> Dense(32, relu)
-> Dense(2, linear)
-> Softmax
```

This is a separate, smaller model from the one in `../ctlearn/`, with
a single conv block and no pooling, sized specifically to fit real C
synthesis on this target.

## Contents

| File | Description |
|------|-------------|
| `run_io_stream.py` | Self-contained conversion, validation, and optional synthesis script |

## Running the example

From this directory:

```bash
python run_io_stream.py                    # conversion + csim only
python run_io_stream.py --synth            # + C synthesis
python run_io_stream.py --fifo-opt         # + cosim-based FIFO depth optimization (implies --synth)
```

Other options: `--rf` (global reuse factor), `--layer-rf` (per-layer
override, repeatable), `--pack-neighbors`. Run with `--help` for full
details.

Only `io_stream` is supported for this model (see the package
`README.md`, "Known limitations", for why `io_parallel` is not used
here).

## Expected results

```
Mean absolute difference < 0.05
Classification agreement >= 95 %
```

The HLS project is written to `hls4mlprj_xcku040_stream/` (git-ignored).

## Reference results (manual comparison)

This section compares the automatic, general-purpose `io_stream`
configuration produced by `precision_utils.get_hls_config()` against a
hand-optimized, model-specific HLS implementation of the same
architecture, both synthesized for the same target: Xilinx XCKU040,
330 MHz (3.03 ns), 0.5 ns clock uncertainty, Vitis HLS 2025.2. Both
were measured with a trained model of this architecture, not the
random-weight version this script now builds. Resource usage and
timing for a fixed architecture, precision and `ReuseFactor`
configuration are not directly sensitive to the specific weight
values, but `precision_utils.get_hls_config()` infers precision per
layer from profiled activation ranges, which do depend on the
weights; a run against the synthetic model here may infer slightly
different per-layer widths, and therefore may not reproduce these
exact numbers.

Configuration used for that comparison (equivalent to `--rf 1
--layer-rf fc_type_2:64:Resource --layer-rf fc_type_1:8:Resource
--pack-neighbors --fifo-opt`):

```python
hls_config['LayerName']['neighbor_gather_layer']['PackNeighbors'] = True
hls_config['LayerName']['SingleCNNIndexed_block_conv_1_1']['PackNeighbors'] = True
hls_config['LayerName']['fc_type_1']['ReuseFactor'] = 8
hls_config['LayerName']['fc_type_1']['Strategy'] = 'Resource'
hls_config['LayerName']['fc_type_2']['ReuseFactor'] = 64
hls_config['LayerName']['fc_type_2']['Strategy'] = 'Resource'
hls_config['InputData'] = 'path/to/real_input_sample.npy'
hls_config['Flows'] = ['vitis:fifo_depth_optimization']
```

| Metric | `io_stream` (this package) | Manual HLS (reference) |
|--------|------------------------------|---------------------------|
| Timing | 2.781 ns (meets 3.03 ns target) | 2.524 ns (meets target) |
| Latency | 337 cycles (1.021 us) | 343 cycles (1.039 us) |
| Interval (II) | 173 cycles | 164 cycles |
| DSP | 1453 (75 %) | 1205 (62 %) |
| FF | 127 273 (26 %) | 51 583 (10 %) |
| LUT | 32 675 (13 %) | 24 390 (10 %) |
| BRAM | 412 (34 %) | 310 (25 %) |

Both designs close timing at the target clock, and latency and
throughput (II) are essentially the same, within 2% and 5%
respectively. The main gap is resource usage, particularly FF (roughly
2.5x) and DSP (roughly 20% more), attributable mainly to the manual
design's hand-tuned, narrower per-weight precisions and its fusion of
`GlobalAveragePooling1D` directly into the surrounding accumulation
logic, neither of which the generic layer structure used here does
automatically. See the package `README.md`, "Future work", for
possible ways to close this gap. csim/synthesis
correctness is unaffected by event count, but exact resource numbers
were not re-measured against the reduced dataset.
