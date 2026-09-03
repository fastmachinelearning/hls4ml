---
name: precision-and-debugging
description: >-
  Choose and debug fixed-point precision in hls4ml, and inspect a generated project when the result is wrong.
  Use when predict() or C simulation does not match the trained model, when setting ap_fixed types or
  accumulator widths, or when you need to know which generated file to read. Applies to every backend.
globs:
  - "hls4ml/model/types.py"
  - "hls4ml/model/optimizer/passes/infer_precision.py"
---

# Precision, numerics, and reading a generated project

Paths are relative to the package directory `hls4ml/hls4ml/`. For the kernels these types flow through, see
[kernels](kernels.md).

## Precision and numerics

- `FixedPrecisionType` (`model/types.py`) defaults to `width=16, integer=6, signed=True`, rounding
  `TRN` (truncate) and saturation `WRAP`. **Overflow wraps silently**, which turns a slightly-too-small
  integer width into a large sign-flipped error rather than a clipped value. When a layer's output is wildly
  wrong but neighbouring layers look fine, suspect the integer width first.
- The accumulator is a separate type. `Layer._set_accum_t` fills `accum_t` from the config; a wide-enough
  accumulator matters more than a wide output type, because every partial sum passes through it.
- Types left unset arrive as `UnspecifiedPrecisionType` and are filled by
  `model/optimizer/passes/infer_precision.py`, which has per-layer-class rules. Automatic inference is
  layer-local: it cannot know the data range, so it is conservative for weights and bias but not a substitute
  for measuring.
- **Mixed precision is a first-class case.** Each inter-layer connection carries the *producer's* output
  type, not a single model-wide type. A kernel that assumes one type for input, weights, accumulator and
  output will fail as soon as layers differ; template on all four.
- Expected accuracy: for `ap_fixed<16,6>` on a small dense network, the maximum absolute difference against
  the float reference lands at a few times `1e-3`. Errors at that scale are quantization and are not a bug.
  Errors of order 0.1 or larger are a real defect: wrong weight layout, overflow, or an uninitialized
  accumulator.
- Tools: `hls4ml.model.profiling.numerical(model=keras_model, hls_model=hls_model, X=X)` plots weight and
  activation ranges against the chosen types; `compare(keras_model, hls_model, X)` shows per-layer
  differences. For a layer-by-layer dump, set `Trace: True` in the layer config and call
  `hls_model.trace(X)`, which recompiles with tracing and returns the intermediate tensors.

## Reading a generated project

File names below are the Vivado-family writer's. Other writers differ — Quartus and oneAPI put the headers
under `firmware/`, ship a Makefile or CMakeLists instead of tcl scripts, and Quartus writes separate parallel
and stream testbenches. The working method underneath the table holds regardless.

| File | What it tells you |
| --- | --- |
| `firmware/myproject.cpp` | the real dataflow: stream declarations, pragmas, the order of layer calls |
| `firmware/parameters.h` | every config struct — the fastest check that an attribute reached C++ |
| `firmware/defines.h` | the concrete `ap_fixed<>` widths for each layer and variable |
| `firmware/weights/*.h` | the actual constants, in the layout the kernel receives |
| `myproject_bridge.cpp` | what `compile()` builds and `predict()` calls |
| `hls4ml_config.yml` | the fully resolved configuration, after all defaults and overrides |
| `vitis_hls.log`, `*_csynth.rpt` | scheduling, initiation interval and timing messages |

Working method:

1. **Check correctness before anything else.** `hls_model.compile()` then `hls_model.predict(X)` against the
   float model — see [**running hls4ml**](running-hls4ml.md). C simulation is fast; do not synthesize a wrong design.
2. **Diff two generated projects** when comparing implementations. Generate one project per variant and
   `diff -r` the `firmware/` directories. Everything except the intended difference should be identical; this
   is how you prove a comparison is fair, and it repeatedly finds unintended differences.
3. **Use a fresh `output_dir`, or delete it first.** hls4ml does not clean the output directory on
   re-conversion, so files from a previous run persist and you may be reading stale generated code.
4. **Read the logs, not only the parsed report.** A design that meets its latency target in C synthesis can
   still fail timing in logic synthesis, and initiation-interval violations appear only in the log text.
5. **Which number to trust:** C synthesis is an estimate and is directional only; co-simulation gives the
   true cycle count; logic synthesis gives the true resource use. Do not report C synthesis numbers as
   final. See the [toolchain access](toolchain-access.md) and [evaluating implementations](evaluating-implementations.md) skills.

## Anti-checklist

- Do not treat a large numerical error as a precision problem before checking the weight layout, and do not
  treat a small one as a bug.
- Do not conclude anything from a project directory you did not regenerate cleanly.
