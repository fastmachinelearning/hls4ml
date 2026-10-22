# CTLearn example

End-to-end conversion example using a model architecturally modeled
after a real [CTLearn](https://github.com/ctlearn-project/ctlearn) 2D
classifier. The neighbor map and input data are generated
synthetically in the script, and the model uses random, untrained
weights, so this example needs no external files (see
`run_io_stream.py`'s own docstring, "Synthetic data").

## Model

The model is a two-stage hexagonal CNN classifier, processing a
single telescope camera frame consisting of 163 pixels with 20
waveform time samples each. With trained weights it would predict the
probability of the event being a gamma-ray shower (class 1) versus
background (class 0); here, with random weights, its output has no
physical meaning and only the numerical agreement between the Keras
and HLS computations is being validated.

```
Input(163, 20)
-> BatchNormalization
-> NeighborGatherLayer         K=7
-> IndexedConvolutionLayer     8 filters
-> NeighborGatherLayer         K=7
-> IndexedPoolingLayer(max)
-> BatchNormalization
-> NeighborGatherLayer         K=7
-> IndexedConvolutionLayer     16 filters
-> BatchNormalization
-> GlobalAveragePooling1D
-> Dense(64, relu)
-> Dense(32, relu)
-> Dense(2, linear)
-> Softmax
```

| Property | Value |
|----------|-------|
| Input pixels | 163 (SiPM hexagonal camera) |
| Neighbors per pixel | 7 |
| Input features | 20 (waveform time samples) |
| Border pixels | 90 slots with index -1 |
| Trainable parameters | 5,362 |

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

Only `io_stream` is supported for this model. `io_parallel` does not
scale to this pixel count (163 pixels, three gather stages): the
fully-unrolled `io_parallel` kernels do not complete C synthesis at
this scale, a limitation already observed on the smaller single-conv
model in `../xcku040/` (see that example's README, "Reference
results", and the package `README.md`, "Known limitations").

Synthesis, when run with `--synth`, targets a Xilinx XCKU115
(`part='xcku115-flvb2104-2-i'`), at 330 MHz (`clock_period=3.03`,
`clock_uncertainty='0.5ns'`), matching the original manual
implementation this model was validated against.

## Expected results

```
Mean absolute difference < 0.05
Classification agreement >= 95 %
```

The HLS project is written to `hls4mlprj_ctlearn_stream/` (git-ignored).

## Synthesis results

With `--rf 1 --layer-rf fc_type_2:64:Resource --layer-rf fc_type_1:8:Resource --pack-neighbors --fifo-opt`:

| Metric | Value |
|--------|-------|
| Timing | 2.643 ns (meets 3.03 ns target) |
| Latency | 513 cycles (1.554 us) |
| Interval (II) | 173 cycles |
| DSP | 2527 (45 %) |
| FF | 204 268 (15 %) |
| LUT | 43 347 (6 %) |
| BRAM | 740 (17 %) |

These figures were measured with a trained model of this same
architecture, not the random-weight version this script now builds.
Resource usage and timing for a fixed architecture, precision and
`ReuseFactor` configuration are not directly sensitive to the specific
weight values, but `precision_utils.get_hls_config()` infers precision
per layer from profiled activation ranges, which do depend on the
weights; a run against the synthetic model here may infer slightly
different per-layer widths, and therefore may not reproduce these
exact numbers.

A hand-optimized, model-specific manual HLS implementation of this
model, synthesized for the same target, uses BRAM 18 %, DSP 39 %, FF
6 %, LUT 5 %. BRAM and LUT usage are close to the manual design; DSP
and FF run higher here (45 % vs. 39 %, 15 % vs. 6 %), the same
trade-off already documented for the smaller single-conv model in
`../xcku040/` (see that example's README, "Reference results"): the
two `IndexedConvolutionLayer` instances account for the large majority
of both (79 % of DSP, 55 % of FF between them), and `ReuseFactor` has
no effect on them here since `pack_neighbors=True` (see the package
`README.md`, "Known limitations"). The remaining gap is attributable
to the same general-purpose, safety-margined precision inference and
lack of cross-layer fusion (BatchNormalization is not fused into the
gather/pooling stages here, unlike in the manual design) described
there.

## Notes

- The Softmax layer is excluded from the hls4ml model and applied in
  Python after inference (see the package `README.md`, "Known
  limitations", "Softmax").
- Precision is inferred automatically via `precision_utils.get_hls_config()`
  (one directory up), which profiles the model being converted (`-1`
  indices included) for activation ranges, then converts that same
  model to HLS. See the package `README.md`, "Choosing precision", for
  why this example-only helper is used here instead of hls4ml's own
  `'auto'` inference.
