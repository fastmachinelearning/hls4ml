---
name: running-hls4ml
description: >-
  Convert, compile and run an hls4ml model — the conversion API, the arguments that decide the outcome, and
  the traps that make it fail. Use whenever you need to turn a Keras/PyTorch/ONNX model into an hls4ml
  project, run predict() or trace(), check that a change works, or when `import hls4ml` behaves oddly. Read
  this before any task that needs a model converted, because verifying anything else starts here.
globs:
  - "hls4ml/converters/**"
  - "hls4ml/utils/config.py"
---

# Running hls4ml

## Environment

hls4ml needs Python 3.10 or newer. Install it in editable mode (`pip install -e .`) when working on the tree,
so edits take effect without reinstalling. Frontend and tooling dependencies are optional extras, declared in
`pyproject.toml`: `keras-v3`, `onnx`, `profiling`, `da`, `hgq`, and others. Only the frontend you use needs to
be installed.

For Keras, set `KERAS_BACKEND` (`tensorflow`, `torch` or `jax`) before importing, in the environment or at the
top of the script.

Your site's own setup — where the interpreter lives, which extras are installed, how the HLS toolchain is
reached — belongs in a local file; see `local-setup.template.md` for a skeleton, and
[**toolchain access**](toolchain-access.md) for what a build needs on `PATH`.

**Do not run Python from a directory that contains the `hls4ml` checkout directory.** It shadows the installed
package: `import hls4ml` then succeeds but yields an empty namespace package with `__file__` set to `None` and
no `converters` attribute. Run from anywhere else. If an attribute that obviously exists is missing, check the
working directory before anything else.

## The recipe

This runs as written:

```python
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np, hls4ml
from tensorflow import keras

model = keras.Sequential([keras.layers.Input((8,)),
                          keras.layers.Dense(6, activation='relu'),
                          keras.layers.Dense(3)])

cfg = hls4ml.utils.config_from_keras_model(
    model, granularity='name', backend='Vitis', default_precision='fixed<16,6>')

hmodel = hls4ml.converters.convert_from_keras_model(
    model, hls_config=cfg, backend='Vitis', io_type='io_stream', output_dir='/tmp/prj')

hmodel.compile()
X = np.random.rand(20, 8).astype('float32')
print(np.max(np.abs(hmodel.predict(X) - model.predict(X, verbose=0))))
```

`compile()` writes the project and builds the shared library that `predict()` calls; it does not run the
vendor toolchain. `build()` does that, and is a separate, much slower step.

## Arguments that decide the outcome

- **`granularity`** in `config_from_keras_model`: `'model'` (one set of keys for everything, the default),
  `'type'` (per layer class), `'name'` (per layer instance). Use `'name'` whenever a task involves setting
  anything per layer; the other two silently give you no place to put it.
- **`backend`** defaults to `'Vivado'` in `convert_from_keras_model`, so it must be passed explicitly for
  anything else, and passed to `config_from_keras_model` as well — the config helper asks the backend which
  attributes are configurable, so a config built without it lacks backend-specific keys.
- **`io_type`** defaults to `'io_parallel'`. `'io_stream'` is a different code path in most layer families,
  not a variation on the same one.
- **`part`**, **`clock_period`**, **`clock_uncertainty`** fall back to backend defaults when not given
  (Vitis: `xcvu13p-flga2577-2-e`, 5 ns, 27%). Any comparison must fix them explicitly.
- **`default_precision`** accepts `'fixed<16,6>'` or a backend-specific spelling; `'auto'` is not allowed as
  the default, but individual layer types may be set to `'auto'` to be inferred.
- Registered backends: `vivado`, `vivadoaccelerator`, `vitis`, `quartus`, `catapult`, `symbolicexpression`,
  `oneapi`, `libero`. `hls4ml.backends.get_available_backends()` lists them.

## What the numbers should look like

For a small untrained dense network at `fixed<16,6>`, the maximum absolute difference against the Keras
prediction lands at a few times `1e-3`. That is quantization, not a bug. An error of order `0.1` or larger
means something real: overflow from too few integer bits, a wrong weight layout, or an uninitialized
accumulator. See [**kernels**](kernels.md) for the precision rules.

## Inspecting what happened

- The generated project is under `output_dir`; `firmware/parameters.h` is the fastest check that a config
  value reached the generated code.
- **hls4ml does not clean `output_dir` on re-conversion.** Stale files from a previous run persist, and you
  can spend a long time reading generated code that no longer corresponds to your change. Use a fresh
  directory or delete it first.
- Per-layer intermediate values: set `Trace: True` in the layer config and call `hmodel.trace(X)`, which
  recompiles with tracing and returns the intermediates alongside the prediction.
- Distributions against the chosen types: `hls4ml.model.profiling.numerical(model=keras_model,
  hls_model=hmodel, X=X)`.

## Common failures

| Symptom | Cause |
| --- | --- |
| `hls4ml.__file__` is `None`, `converters` missing | running from a directory containing the checkout |
| `KeyError` on a layer class name during `config_from_keras_model` | the layer is parsed but not registered in `layer_map`; see [**frontends**](frontends.md) |
| "installation not found" from `build()` | the toolchain command is not on `PATH`; see [**toolchain access**](toolchain-access.md) |
| generated code does not match the change | stale `output_dir` |
| prediction is far from the reference | precision or layout, not the toolchain — check before synthesizing anything |

Related: [**frontends**](frontends.md) for adding a layer the parser does not know, [**architecture map**](architecture-map.md) for
what happens between convert and write, [**evaluating implementations**](evaluating-implementations.md) for turning runs into a verdict.
