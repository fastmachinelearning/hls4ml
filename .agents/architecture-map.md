---
name: architecture-map
description: >-
  Map of how hls4ml turns a trained model into an HLS project — the graph and attribute system, the flow /
  optimizer-pass machinery, the C++ templates, and the writer. Read this BEFORE changing anything inside the
  hls4ml Python tree, or when you need to find which file controls a given piece of generated code. Covers
  what each stage owns, how a config value reaches a C++ struct field, and where to look for a given task.
globs:
  - "hls4ml/model/**"
  - "hls4ml/backends/**"
  - "hls4ml/writer/**"
---

# How hls4ml is put together

All paths below are relative to the package directory `hls4ml/hls4ml/` inside the repo checkout.

The conversion is a five-stage pipeline. Every stage only communicates with the next through **layer
attributes**, so almost every change you will make is "set a different attribute" or "read an attribute and
emit different text".

```
frontend converter  →  ModelGraph  →  flows of optimizer passes  →  templates  →  writer
(converters/)          (model/graph.py)  (model/optimizer/, backends/*/passes/)   (writer/)
```

1. **Frontend** (`converters/`) parses Keras / PyTorch / ONNX into a plain list of layer dictionaries, whose
   `class_name` selects the `Layer` subclass — see [**frontends**](frontends.md).
2. **`ModelGraph.from_layer_list`** builds the graph of `Layer` nodes (`model/layers.py`).
3. **`model.apply_flow(backend.get_default_flow())`** runs the registered optimizer passes in order.
4. Template passes fill in the code attributes of each node — in the Vivado family `config_cpp` and
   `function_cpp`; these strings *are* the generated C++.
5. The backend's **writer** (`writer/<backend>_writer.py`) writes the project directory.

## The graph and its nodes

- `model.graph` is an `OrderedDict` of `name -> Layer`. Insertion order is the emission order.
- Nodes are connected by **tensor name strings**, not object references: `node.inputs` and `node.outputs` are
  lists of names. `node.get_input_node()`, `node.get_input_variable()`, `node.get_output_variable()` resolve
  them through `model.output_vars`. When you rewire the graph, you rewrite these name lists.
- `Layer.initialize()` is where a layer declares its output shape and weights
  (`add_output_variable`, `add_weights`, `add_bias`). `Dense.initialize` is the shortest useful example.

## The attribute system — the single source of truth

Everything about a layer lives in `layer.attributes` (`model/attributes.py`), a dict with side effects:

- Storing a `TensorVariable` registers it as a model output variable and also sets `result_t`.
- Storing a `WeightVariable` named `weight` automatically creates the type attribute `weight_t`.
- Four filtered views exist: `layer.weights`, `layer.variables`, `layer.types`, `layer.code`.
- `Layer.expected_attributes` is collected over the whole class hierarchy from each class's
  `_expected_attributes` list. Anything expected but unset and without a default raises at construction.
- Attribute kinds: `Attribute`, `ConfigurableAttribute` (user may set it), `TypeAttribute` (name always ends
  in `_t`), `ChoiceAttribute`, `WeightAttribute`, `CodeAttrubute` (spelling as in the source).

**How a user config value reaches a layer:** `HLSConfig` (`model/graph.py`) reads the `HLSConfig` dict; in
`Layer.__init__` each key of the layer's config is converted from pascal case to snake case
(`ReuseFactor` -> `reuse_factor`), and any key ending in `_t` whose value is a string is converted into a
`NamedType` through the backend's `convert_precision_string`. So a user knob and a layer attribute are the
same thing under two spellings.

## Flows and passes

- A `Flow` (`model/flow/flow.py`) is a named list of pass names plus a `requires` list of other flows.
  `apply_flow` walks requirements depth first and skips flows already applied.
- Three flows are registered by the core, without a backend prefix, in `model/optimizer/__init__.py`:
  `parse_qonnx`, `convert` (frontend cleanups, ending with `infer_precision_types`), and `optimize`. Every
  backend's `init_layers` requires the core `optimize`. Note the name collision: an unprefixed `optimize` in a
  `requires` list is the core flow, while `vivado:optimize` is a different, backend-owned flow.
- Passes are registered globally in `optimizer_map`, prefixed with the backend name:
  `vivado:distributed_arithmetic_codegen`. Backends discover them automatically — `Backend._init_file_optimizers`
  (`backends/backend.py`) scans the `passes/` directory of **`[*self.__class__.__bases__, self.__class__]`**,
  not the full method resolution order, and `_init_class_optimizers` picks up decorated backend methods.
- A registered pass that is in no flow never runs.
- `VitisBackend._register_flows` (`backends/vitis/vitis_backend.py`) is the best example to copy: it builds
  its own flows and inserts them into a copy of `vivado:ip`'s requirement list at chosen positions.

## Templates — where C++ text comes from

`backends/template.py` defines two template base classes, both of which are optimizer passes:

- `LayerConfigTemplate` sets the node's `config_cpp` attribute — the `struct configN { ... };` block.
- `FunctionCallTemplate` sets `function_cpp` and `include_header` — the one-line call in `myproject.cpp`.

Both format a Python string against `node.attributes` plus a few extras (`_default_config_params` adds
`iotype`, `reuse`, `namespace`; `_default_function_params` adds `config`, `input_t`, `output_t`, `input`,
`output`). `backends/vivado/passes/core_templates.py` holds the Dense and activation templates and shows how
`strategy` selects the C++ kernel class name.

Types and variables are converted from the backend-independent form into backend code form by that backend's
own `passes/transform_types.py`, using converters built on `backends/fpga/fpga_types.py`. Vivado, Quartus,
oneAPI, Catapult and Libero each have their own copy; Vitis has none and uses Vivado's through inheritance.
This pass is where the backend decides what a tensor becomes — an array with a partition pragma, an
`hls::stream`, or a pipe. io_type behaviour is decided there, not in the writer.

A backend may add template kinds of its own by subclassing `Template` with a different `attribute_name`; see
`backends/oneapi/oneapi_template.py`. The writer then reads those extra attributes.

## What is universal and what is only a convention

Stages 1 to 3 above — the graph, the attribute system, flows and passes — are core and identical for every
backend. From stage 4 onward, backends diverge, and how much they follow the Vivado pattern varies:

| Universal | Vivado/Vitis convention that others follow only partly |
| --- | --- |
| `ModelGraph`, `Layer`, attributes, `expected_attributes` | the `config_cpp` / `function_cpp` attribute pair as the only generated-code channel |
| flows, `register_flow`, pass discovery and `backend:` naming | `LayerConfigTemplate` and `FunctionCallTemplate` as the only template kinds |
| `Template` as the mechanism that produces code attributes | `nnet::` headers, the `strategy` enum, the `kernel` typedef, `nnet_utils/` layout |
| a registered `Writer` subclass that emits the project | the file names and directory layout listed below |
| variable and type converters applied by a `transform_types` pass | `hls::stream<nnet::array<T,N>>` for io_stream, arrays plus partition pragmas for io_parallel |

Concrete counter-examples in this tree: oneAPI defines two extra template kinds
(`StreamFunctionCallTemplate`, `TaskSequenceTemplate` in `backends/oneapi/oneapi_template.py`), carries data
in pipes with `pipe_name` rather than `hls::stream`, and keeps its headers under
`templates/oneapi/firmware/nnet_utils/`. Quartus and oneAPI have no `strategy` field and no kernel typedef.
The symbolic backend has no `nnet_utils` at all.

Read as: the Python machinery is a contract, the C++ shape is a precedent. A backend may define its own
template kinds, its own variable types, and its own project layout; doing so is a supported extension, not a
workaround.

## The writer (Vivado family)

`VivadoWriter.write_hls` calls, in order: `write_project_dir`, `write_project_cpp`, `write_project_header`,
`write_weights`, `write_defines`, `write_parameters`, `write_test_bench`, `write_bridge`,
`write_build_script`, `write_nnet_utils`, `write_generated_code`, `write_yml`, `write_tar`.

Resulting project layout:

| Path | Contents |
| --- | --- |
| `firmware/myproject.cpp` | top function: stream declarations, `#pragma HLS DATAFLOW` or pipeline, one `function_cpp` line per layer |
| `firmware/myproject.h` | top function signature |
| `firmware/defines.h` | typedefs for every layer type and variable |
| `firmware/parameters.h` | every `config_cpp` struct, plus includes |
| `firmware/weights/w2.h, b2.h …` | weight arrays |
| `firmware/nnet_utils/` | the C++ kernel headers, copied from `templates/<backend>/nnet_utils/` |
| `myproject_test.cpp`, `tb_data/` | C simulation testbench |
| `myproject_bridge.cpp` | the shared library used by `compile()` / `predict()` |
| `build_prj.tcl`, `project.tcl` | synthesis driver scripts |
| `hls4ml_config.yml` | the resolved configuration |

The writer should stay simple: it copies text that passes and templates already produced. If you find
yourself adding model logic to the writer, it belongs in a pass instead.

Three of those outputs are easy to forget when changing the project structure, and each has a consumer:

- **The bridge** (`write_bridge`, from `templates/<backend>/myproject_bridge.cpp`) is what `compile()` builds
  and `predict()` calls through ctypes, and it also carries the trace-collection entry points used by
  `trace()`. Change the top function's interface and the bridge must change with it, or Python-side
  verification breaks while synthesis still succeeds.
- **The testbench** (`write_test_bench`, plus `tb_data/`) is what C simulation and co-simulation run.
- **The build scripts** (`write_build_script`) are copied from the backend's `templates/` directory and are
  what `build()` ultimately invokes; the report that `build()` returns is parsed by `hls4ml/report/`.

## Where to look for a given task

| I want to change… | Open |
| --- | --- |
| what a layer's generated struct contains | `backends/<backend>/passes/*_templates.py` |
| which C++ kernel is called | the same template's `format()` — in the Vivado family, its `strategy` branch |
| the compute itself | `templates/<backend>/nnet_utils/nnet_*.h` |
| graph shape, layer fusion, layer removal | a pass in `backends/<backend>/passes/` or `model/optimizer/passes/` |
| a new user-settable knob | a `ConfigurableAttribute` in the layer class or `_register_layer_attributes` |
| whether a tensor is an array or a stream | `passes/transform_types.py` plus `backends/fpga/fpga_types.py` |
| default precision, accumulator width | `model/optimizer/passes/infer_precision.py`, `HLSConfig.get_precision` |
| the order things run in | the backend's `_register_flows` |
| files written to disk | `writer/<backend>_writer.py` |

## Reading order for a first session

`model/layers.py` (the `Layer` base class and `Dense`), `model/attributes.py`, `backends/backend.py`,
`backends/template.py`, `backends/vivado/passes/core_templates.py` (Dense part only), then one generated
project's `parameters.h` and `myproject.cpp` side by side with the templates that produced them. That last
step is worth more than reading any further Python.

Dense and the Vivado backend are used here because they are the shortest complete path, not because they are
representative. Once the mechanism is clear, read the files of the backend and layer family you are actually
changing: that backend's `_register_flows`, its `passes/`, its `transform_types.py`, its writer, and the
templates for your layer. Assume nothing carries over until you have seen it there.

Related: [**frontends**](frontends.md) for getting a layer in and for the end-to-end checklist,
[**optimizer passes**](optimizer-passes.md) for writing the passes, [**kernels**](kernels.md) for the generated code and the
precision rules, [**running hls4ml**](running-hls4ml.md) for actually converting a model, [**new backend**](new-backend.md) for standing
up a backend of your own.
