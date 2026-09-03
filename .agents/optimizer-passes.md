---
name: optimizer-passes
description: >-
  Write or modify an hls4ml optimizer pass, a Strategy, a layer initializer, a validation check, or a new
  user-settable config attribute. Use whenever the task is "make hls4ml do X to the graph / set X on a layer
  / expose X in the config" rather than changing the C++ kernel itself. Covers the pass types, automatic
  discovery, flow registration and ordering, the transform return value, graph edit APIs, and the traps that
  cause a pass to silently never run or to loop forever.
globs:
  - "hls4ml/model/optimizer/**"
  - "hls4ml/backends/*/passes/**"
---

# Writing hls4ml optimizer passes

Paths are relative to the package directory `hls4ml/hls4ml/`. Passes are the only sanctioned way to change a
model between parsing and writing. If a change can be expressed as a pass, it should be.

## Pick the right pass type

All of these live in `model/optimizer/optimizer.py`.

| Type | Runs on | Use for |
| --- | --- | --- |
| `OptimizerPass` | every node where `match(node)` is true | the normal case: rewriting, checking, annotating a layer |
| `GlobalOptimizerPass` | every node (`match` always true) | sweeping work such as type conversion (`transform_types.py`) |
| `ModelOptimizerPass` | the whole model once, `transform(model)` | decisions needing the full graph, such as chain detection |
| `ConfigurableOptimizerPass` | as `OptimizerPass`, plus `configure(**kwargs)` | passes with options, e.g. `InferPrecisionTypes` |
| `Template` subclasses | matching nodes | generating C++ text; see the architecture-map skill |
| `@layer_optimizer(LayerClass)` on a backend method | instances of that layer class | backend-specific layer initialization, e.g. `init_dense` |

A backend method decorated with `@layer_optimizer(Dense)` becomes a `LayerOptimizerPass` named after the
method (`vitis:init_dense`). Layer initializers are sorted by the length of the layer class's method
resolution order, so `init_base_layer` runs before `init_dense`.

## Discovery and naming

- **File passes:** any class in `backends/<backend>/passes/*.py` that subclasses `OptimizerPass` and is
  *defined in that module* (not imported into it) is registered automatically. Alternatively, define a
  function `register_<module_filename>(backend)` in the module and it is called instead, which lets you
  register several passes explicitly.
- The registered name is `snake_case(ClassName)` unless the class sets `name`, and it is always prefixed with
  the backend name in lower case: `DistributedArithmeticCodegen` in `backends/vivado/passes/` becomes
  `vivado:distributed_arithmetic_codegen`.
- Discovery walks `[*self.__class__.__bases__, self.__class__]`, so a `VitisBackend(VivadoBackend)` gets
  `backends/vivado/passes/` and `backends/vitis/passes/`, but a backend two levels down does **not** inherit
  its grandparent's passes.
- Model-level passes in `model/optimizer/passes/` are registered without a backend prefix and are available
  to every backend.

**A pass that is registered but not in a flow never runs.** This is the most common reason new pass code
appears to do nothing.

## Flow registration and ordering

Register a flow with `register_flow(name, optimizers, requires=[...], backend=self.name)` inside the
backend's `_register_flows`. To insert your flow at a specific point in an existing pipeline, copy the
requirement list of the flow you are extending and insert into it — this is what Vitis does:

```python
validation_flow = register_flow('validation', validation_passes, requires=['vivado:init_layers'], backend=self.name)
ip_flow_requirements = get_flow('vivado:ip').requires.copy()
ip_flow_requirements.insert(ip_flow_requirements.index('vivado:init_layers'), validation_flow)
self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)
```

Backend flow names that appear in every FPGA backend, each under its own prefix: `init_layers`,
`specific_types`, `apply_templates`, `write` and `ip`. Others are optional and vary — Vivado, Quartus, oneAPI
and Catapult add `optimize`, `streaming` and `quantization`, while Libero registers only the five above and
Vitis adds `validation` on top of the ones it inherits from Vivado. Libero is the useful minimal example.

Do not confuse a backend's own `<backend>:optimize` with the unprefixed core `optimize` flow registered in
`model/optimizer/__init__.py`. Every backend's `init_layers` requires the **core** one.

Ordering rules worth memorizing:

- Anything that reads or sets **types** must run before `*:specific_types` (which runs `transform_types` and
  turns tensors into arrays or streams). After that pass the variables are backend C++ objects.
- Anything that changes **which C++ kernel is selected** must run before `*:apply_templates`.
- Precision inference is the pass `infer_precision_types`
  (`model/optimizer/passes/infer_precision.py`). It runs at the end of the core `convert` flow, and some
  backends list it again in their own `optimize` flow to catch types that became inferable later. Before it,
  types may still be `UnspecifiedPrecisionType`; do not read a final bit width earlier than that.

## The transform return value

`optimize_model` loops over the passes repeatedly:

- Returning **`True` means "the graph changed"**, which aborts the current sweep and restarts from the first
  pass. Return `True` only after adding or removing nodes.
- Returning `False` (or `None`) means "I edited attributes only, keep going". Template passes deliberately
  return `False`.
- Returning `True` when nothing actually changed makes conversion hang forever. If a conversion never
  finishes, suspect this first.
- A `ModelOptimizerPass` is applied at most once per flow, and only recorded as applied if it returns a
  truthy value.

## Editing the graph

`ModelGraph` (`model/graph.py`) provides `make_node`, `insert_node`, `remove_node`, `replace_node`,
`split_node`. Constraints that are enforced with exceptions or assertions:

- `make_node(kind, name, attributes, inputs, outputs=None)` creates a detached node; the layer type must be
  in `layer_map` (`register_layer(name, clazz)` in `model/layers.py`), and the backend wraps the class to add
  its own expected attributes.
- `insert_node` accepts only nodes with a single input; pass `before=` when the predecessor has several
  consumers.
- `remove_node` requires at most one input and one output, and asserts that the input and output shapes have
  the same number of elements.
- `replace_node` requires the same number of inputs and outputs as the node being replaced.

All of these rewrite the tensor-name lists of neighbouring nodes. After any of them, return `True`.

## Adding a user-settable attribute

Two options:

1. **In the layer class** — add a `ConfigurableAttribute` to `_expected_attributes` in `model/layers.py`.
   Applies to every backend.
2. **In the backend** — append to `self.attribute_map[LayerClass]` in the backend's
   `_register_layer_attributes`. `FPGABackend.create_layer_class` then builds a subclass of the layer that
   carries these extra expected attributes, so the knob exists only for that backend. This is the right place
   for anything toolchain-specific.

Give every attribute a default. A `ConfigurableAttribute` with no default and no user value raises during
layer construction, which breaks every existing model.

`infer_precision_types` is the other place a new layer class needs an entry. `_infer_precision` dispatches on
`node.class_name` through a chain of class-name lists; a class named in none of them falls through to the
default rule, which converts but gives types nobody chose. Adding a layer type means adding its branch there
too — a quiet omission, not a loud one.

The user writes the name in pascal case in the config (`RecurrentReuseFactor`); it arrives as snake case
(`recurrent_reuse_factor`) in `layer.attributes`, and templates read it from there by the same name.

## Worked example in the tree: the distributed arithmetic strategy

A strategy implemented end to end as passes, useful as a template to copy. The machinery it uses is core to
every backend; the vocabulary it manipulates — `strategy`, `reuse_factor`, a `kernel` typedef chosen by a
config template — is a Vivado-family convention. Quartus and oneAPI have no `strategy` attribute at all, and
a backend under development may share none of these names. Passes remain the right tool there; only the
attribute names change.

- `backends/vivado/passes/distributed_arithmetic.py` holds the whole feature: `DistributedArithmeticCodegen`
  generates a per-layer C++ kernel from the weight values, `FuseQuantizerIntoDALayers` folds a neighbouring
  operation in, and three template classes emit the layer's code.
- Ordering is the interesting part. The codegen pass is registered in the `optimize` flow **after**
  `infer_precision_types`, because it needs final types to generate code from. The template passes are
  registered in `specific_types`, after `transform_types` and `set_pipeline_style`.
- `VivadoBackend.init_dense` has a `distributed_arithmetic` branch that sets the attribute and raises when
  the strategy is combined with a reuse factor it cannot support — validation belongs at the point the
  attribute is set, not in the generated code.
- `DenseConfigTemplate.match` returns `False` for this strategy, so the common Dense template steps aside and
  the strategy's own template runs instead. This is how a strategy replaces generated code wholesale.
- `backends/vitis/passes/feature_check.py` shows the shape of validation passes: match the unsupported
  combination, then either warn and correct the attribute or raise.

`backends/vivado/passes/unrolled_codegen.py` with the `resource_unrolled` strategy is a second example of the
same pattern at smaller scale.

## Anti-checklist

- Do not put model logic in the writer. If the writer needs to branch on a layer property, add a pass that
  sets an attribute and let the writer copy it.
- Do not read `node.get_output_variable().type` for a final bit width in a pass that runs before precision
  inference.
- Do not assume a pass in another backend's directory is available to you. Cross-backend names resolve, but
  they create a dependency on that backend's flow; copy the pass into your own `passes/` instead.
- Do not return `True` from a pass that only sets attributes.
- Do not add an expected attribute without a default.
- After adding a pass, confirm it actually ran (print inside `transform`, or check the applied set) before
  concluding that the effect you expected is impossible.
