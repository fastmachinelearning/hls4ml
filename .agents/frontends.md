---
name: frontends
description: >-
  Teach hls4ml to parse a layer it does not know — the Keras v3 and v2, PyTorch and ONNX handler patterns,
  registering the hls4ml Layer class, what the config helpers need, and the extension API for out-of-tree
  layers. Use whenever the task is "add support for layer/op X", a model fails to convert with an unsupported
  or unknown layer, or you need to know where a parsed attribute comes from. Ends with the end-to-end
  checklist for adding a layer, which sequences the other hls4ml skills.
globs:
  - "hls4ml/converters/**"
  - "hls4ml/model/layers.py"
  - "hls4ml/utils/config.py"
---

# Frontends: getting a layer into hls4ml

Paths are relative to the package directory `hls4ml/hls4ml/`.

A frontend turns a trained model into a **list of plain dictionaries**, one per hls4ml layer. Each dictionary
carries at least `name`, `class_name`, and the input/output tensor names; `class_name` selects the `Layer`
subclass through `model/layers.py`'s `layer_map`, and the remaining keys become that layer's attributes.
Everything downstream — passes, templates, writer — sees only the graph built from those dictionaries.

## First decide: in-tree or extension API

A layer of general interest belongs in the tree. A layer specific to one project can be added from outside it
with the **extension API**, which registers the same pieces at runtime without patching hls4ml:

```python
hls4ml.converters.register_keras_v2_layer_handler('KReverse', parse_reverse_layer)
hls4ml.model.layers.register_layer('KReverse', HReverse)
backend.register_template(HReverseConfigTemplate)
backend.register_template(HReverseFunctionTemplate)
backend.register_source('/path/to/nnet_reverse.h')
```

`docs/advanced/extension.rst` walks a complete example, and `test/pytest/test_extensions.py` and
`test_extensions_pytorch.py` are runnable versions of it. The component list is the same either way, which
makes the extension example the best template for in-tree work too.

## Keras v3 — the current path

Handlers live in `converters/keras_v3/`. A handler is a class; defining it registers it, through the
metaclass, for every Keras class named in `handles`:

```python
class DenseHandler(KerasV3LayerHandler):
    handles = ('keras.src.layers.core.dense.Dense',)

    def handle(self, layer, in_tensors, out_tensors):
        kernel = self.load_weight(layer, 'kernel')
        bias = self.load_weight(layer, 'bias') if layer.use_bias else None
        n_in, n_out = kernel.shape
        return {'data_format': 'channels_last', 'weight_data': kernel,
                'bias_data': bias, 'n_out': n_out, 'n_in': n_in}
```

Things the base class does for you, worth knowing before writing a handler:

- `handles` entries are **fully qualified Keras module paths**, not display names. Get them from the layer's
  `__module__` and class name, not from the Keras docs.
- When `handle` returns a **single dict**, the base fills in `name`, `class_name`, `module`,
  `input_keras_tensor_names`, `input_shape` and `output_keras_tensor_names`, and copies `epsilon`,
  `use_bias`, `data_format` if the layer has them. Anything you return overrides those.
- When `handle` returns a **tuple of dicts** — one Keras layer becoming several hls4ml layers — none of that
  is automatic: every dict must carry `name`, `class_name`, `input_keras_tensor_names` and
  `output_keras_tensor_names` itself, and an assertion fires if one is missing.
- **A layer with an `activation` attribute is split automatically** into your layer plus a following
  activation layer, with an intermediate tensor threaded between them. This is why one Keras `Dense(...,
  activation='relu')` becomes two nodes in the graph. Softmax, hard sigmoid, leaky ReLU and ELU map to
  dedicated classes with their parameters filled in; everything else becomes `Activation`.
- `self.load_weight(layer, 'kernel')` returns a NumPy array regardless of the Keras backend in use.
- Weight arrays go into the dictionary under the names the `Layer` class expects, conventionally
  `weight_data` and `bias_data`, because `add_weights_variable` looks up `<name>_data`.

## Keras v2, PyTorch, ONNX

Same idea, function-based instead of class-based:

- **Keras v2** (`converters/keras_v2_to_hls.py`): a function decorated with `@keras_handler('ClassName')`,
  taking the layer's serialized config and returning `(layer_dict, output_shape)`. `parse_default_keras_layer`
  fills the common keys. Registration also works at runtime through
  `register_keras_v2_layer_handler(name, func)`. Conversion tries v3 first and falls back to v2 unless
  `allow_v2_fallback=False`, so a model may reach either path — when a layer parses in one and not the other,
  this is why.
- **PyTorch** (`converters/pytorch_to_hls.py`): `@pytorch_handler('ClassName', ...)`.
- **ONNX** (`converters/onnx_to_hls.py`): `@onnx_handler('OpName', ...)`.

Supporting a layer in one frontend does not support it in the others. Decide explicitly which frontends the
task covers, and say so.

## Registering the hls4ml layer

`register_layer(name, clazz)` in `model/layers.py` adds an entry to `layer_map`, keyed by the `class_name`
your handler emits. Without it two things break: `ModelGraph.make_node` cannot resolve the layer, and
`config_from_keras_model` raises a `KeyError` before conversion even starts, because it looks the class up to
discover which attributes are configurable.

That lookup is also why the config helpers need nothing layer-specific: at `'type'` or `'name'` granularity
they walk the parsed list, take each layer class's `expected_attributes`, and emit a key for every attribute
marked configurable — a `TypeAttribute` becomes a `Precision` entry (`'auto'` when it has no default), a
`reuse_factor` becomes `ReuseFactor`, and the rest are copied from the parsed dictionary or their default. A
new layer therefore appears in the generated config automatically, provided its attributes are declared
properly. See [**optimizer passes**](optimizer-passes.md) for declaring them.

## Adding a layer end to end

The full path, with the skill that covers each step:

1. In-tree or extension API — this skill.
2. Write the frontend handler for each frontend in scope — this skill.
3. Define the `Layer` subclass: `_expected_attributes` and `initialize()` (output shape, weights) —
   [**architecture map**](architecture-map.md).
4. Register it in `layer_map` — this skill.
5. Add a branch to `model/optimizer/passes/infer_precision.py` if the layer's types should be inferable;
   without one it falls to the default rule — [**optimizer passes**](optimizer-passes.md).
6. Backend initializer (`@layer_optimizer`) to set and validate attributes, plus any backend-specific
   configurable attributes — [**optimizer passes**](optimizer-passes.md).
7. Config and function templates for each backend in scope — [**architecture map**](architecture-map.md).
8. The C++ kernel, for each io_type you claim to support — [**kernels**](kernels.md).
9. Convert, compile, predict against the reference model — [**running hls4ml**](running-hls4ml.md).
10. Synthesize — [**toolchain access**](toolchain-access.md).
11. Tests and pull request — [**contributing changes**](contributing-changes.md).

Steps 3, 4, 7 and 8 are the ones that fail loudly. Steps 5 and 6 fail quietly: the layer converts and
computes, with types or a strategy nobody chose.

## Anti-checklist

- Do not assume the display name works in `handles`; it takes the fully qualified module path.
- Do not hand-fill the automatic keys when returning a single dict, and do not forget them when returning a
  tuple.
- Do not be surprised by the extra activation node — check whether your layer should keep its activation
  fused instead, and handle that in a pass rather than in the parser.
- Do not add a handler without registering the layer class; the failure appears in the config helper, far
  from the cause.
- Do not claim support for a frontend you did not implement or test.
