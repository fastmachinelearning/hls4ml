---
name: new-backend
description: >-
  Create a new in-tree hls4ml FPGA backend — its Backend subclass, Writer, passes, and code templates — by
  forking an existing backend and pruning it to what is actually supported. Use when standing up a backend
  for a new toolchain or a variant of an existing one. Covers deciding whether a new backend is warranted,
  what to subclass, the pass discovery and namespacing rules, registration, how to avoid carrying dead
  copied code, the rules for deriving from a concrete backend, and the contracts (bridge symbols, build
  report, relocatable output) every backend must honour.
globs:
  - "hls4ml/backends/**"
  - "hls4ml/writer/**"
---

# Standing up a new hls4ml backend

Paths are relative to the package directory `hls4ml/hls4ml/`.

## First, decide whether it should be a backend at all

A new backend is warranted by a different toolchain, a different code language or memory model, or a project
structure the existing writers cannot express. It is not warranted by a better kernel or a different
scheduling idea: those are a **Strategy plus passes** inside an existing backend, which is far less code to
maintain and far easier to get reviewed upstream.

The test: if the change could be rolled into an existing backend without contorting it, it is not a new
backend. Answer this before writing code, because the answer determines everything below.

## Fork, then prune

hls4ml has no scaffolding generator, and copying an existing backend is the sanctioned starting point. The
trap is that a fork silently carries the entire feature surface of its source. A backend that supports two
layer types should not ship headers, passes and writer paths for twenty. Prune to what is supported, and copy
the next piece fresh from the source backend when you add each new feature.

## What to subclass

- **Backend:** inherit from `FPGABackend` (`backends/fpga/fpga_backend.py`), not from `VivadoBackend` or
  `VitisBackend`, unless you genuinely want their whole pass set. `FPGABackend` provides the shared FPGA
  plumbing without toolchain-specific passes. If you do inherit from a concrete backend, read
  "Deriving from a concrete backend" below.
- **Writer:** there is no `FPGAWriter`. Writers subclass either `Writer` (`writer/writers.py`) or an existing
  backend's writer. For a backend that is not a variant of another, subclass `Writer` directly.
- **Types, passes, templates:** fork the closest existing backend's `*_types.py`, `passes/` and its
  `templates/` directory, renaming throughout.

You are not obliged to keep the source backend's C++ conventions. The `nnet::` namespace, the strategy enum,
the kernel typedef dispatch and the `hls::stream` representation are conventions of the Vivado family that
other backends in the tree follow only partly. Defining your own variable types, your own `Template`
subclasses and your own project layout is a supported extension, not a workaround. Say explicitly which
convention you are diverging from and why.

## Deriving from a concrete backend

A backend that wraps another backend's flow — an accelerator integration, a tool variant — legitimately
subclasses that backend and its writer instead of `FPGABackend`. That inheritance is where derived backends
have accumulated most of their defects, and each rule below closes a failure that reached users.

- **The constructor.** Parent backends hard-code their names in `__init__`, so derived backends bypass the
  direct parent with `super(GrandParent, self).__init__(name='Yours')` and then repeat the parent's
  registration calls by hand. Every step the parent later gains is silently missing in the subclass, and
  the breakage surfaces as an unrelated model class failing to convert. If you must bypass, state in a
  comment exactly which parent calls you are repeating, and add a test that compares your registered flows
  and layer attributes against a fresh parent instance, so that a new parent step becomes a test failure
  instead of a latent hole.
- **The writer.** The parent's `write_hls` runs a fixed sequence of steps (listed in the
  [architecture map](architecture-map.md)). A derived writer must account for every one of them: use it as
  is, override it, or override it with a documented no-op. The two known failure modes are opposites of
  each other — re-opening a file the parent wrote and patching it by string matching (after which any
  parent change breaks the subclass silently), and calling `super().write_hls()` whole and leaving parent
  output the flow does not use as dead files beside your own (after which nobody can tell which files
  matter). Own templates for your own files; explicit no-ops for parent steps you replace.
- **`create_initial_config`.** Name your own parameters and forward the rest:
  `super().create_initial_config(..., **kwargs)`. A bare `**_` sink swallows the shared writer options
  (`namespace`, `write_tar`, ...) without a message — the user sets an option, nothing happens, and no
  test that only checks numbers will notice.
- **Validation timing.** Validate at conversion, not at write time. Checks that need no graph (board,
  interface, interface types) belong in `create_initial_config`; checks that need the graph (input and
  output counts against the interface) belong in a pass in your default flow. Raise exceptions, not
  asserts — asserts disappear under `python -O`. A configuration error that first appears in `compile()`
  has already cost the user the whole conversion.
- **`build()`.** Keep the keyword surface of the sibling backends; a flag you accept must either act or
  raise, never be ignored. Run tools with `subprocess` and check exit codes — `os.system` inside
  `try/except` never raises on a failing tool. Return the report dictionary that `hls4ml/report/` and the
  CI synthesis helper expect; returning `None` fails the shared synthesis test on its report assertion.

## Pass discovery and namespacing — the rule that governs everything

`Backend._init_file_optimizers` walks `[*self.__class__.__bases__, self.__class__]` — your direct bases plus
your own class, **not** the full method resolution order — and registers each `passes/` directory it finds,
prefixed with `self.name.lower() + ':'`.

Consequences to design around:

- `class YourBackend(FPGABackend)` registers `backends/fpga/passes/` and `backends/yourbackend/passes/` as
  `yourbackend:*`. It does not pull in another backend's passes. Anything you need from an existing backend
  must be copied into your own `passes/`.
- Because only direct bases are scanned, a deep inheritance chain does not stack passes. Keep `FPGABackend` as
  the base and put everything else in your own directory.
- Cross-backend pass names resolve globally, but referencing one creates a dependency on that backend's flows.
  Prefer your own copies.

## Registration

- `backends/__init__.py`: `register_backend('YourName', YourBackend)`.
- `writer/__init__.py`: register the writer.

## The backend class — minimum viable shape

- `_register_flows`: register the flows you support. Five names appear in every FPGA backend —
  `init_layers`, `specific_types`, `apply_templates`, `write` and `ip` — and following them makes your
  backend legible to anyone who knows another one. `optimize`, `streaming` and `quantization` are optional
  additions; Libero registers only the five and is the smallest example to copy. A minimal backend may leave
  shared FPGA passes unwired — the warning about optimizers not in any flow is informational, not an error.
- `create_initial_config`: defaults for part, io_type and clock, plus your own knobs.
- `build()`: the synthesis entry point for your toolchain.
- `init_<layer>` methods decorated with `@layer_optimizer`: set and validate the attributes your kernels need.
  Watch for interactions with any wrapper your io_stream path uses — in the Vivado family the stream wrapper
  branches on `strategy` and will force-pipeline a kernel that reaches the wrong branch.

## Prune aggressively

After forking, delete what the backend does not support and trim what remains:

- **Kernel headers:** keep only those the supported layers use. Trim a kept header to the configuration
  struct and the dispatch for the kernels you actually ship.
- **Passes:** keep the template passes for supported layers, the type transformation pass, and the pipeline
  style pass. Delete passes for layers and strategies you do not ship.
- **Writer:** remove the paths for features you do not support, and make sure the writer copies the build
  scripts you do use.
- **Configuration structs:** emit only the fields the kernel reads. Fields inherited from a base config
  struct and never used mislead readers into thinking the kernel honours them.

## Contracts the rest of hls4ml imposes

- **The bridge exports two symbols.** `ModelGraph._get_top_function` selects `<project>_float` or
  `<project>_double` by the dtype of the incoming array. Generate working bodies for both, converting at
  the boundary; an empty body makes `predict()` return zeros with no error for that dtype, which is the
  worst failure mode a backend can have.
- **The top function's ports are not fixed.** Configuration features add ports — weights above
  `BramFactor` become top-level arguments. Anything that wraps the top function must forward every port it
  exposes, or the backend must reject the feature at conversion with a message that names it.
- **Generated projects must be relocatable.** Users archive them, move them and build on other machines.
  No absolute paths in generated scripts or tool configuration files; make every path relative to the
  project.

## If the backend targets a board

- Boards are data, not code. One driver template per interface kind, filled by the writer; the board list
  supplies the part and the platform. Accept the platform (or board support file) path as a configuration
  argument, so an unlisted board works without editing the package. Byte-identical driver or script files
  copied per board are the pattern that has made board-support requests stall as unmergeable pull
  requests.
- Do not put vendor headers or backend-specific helpers in another backend's `templates/` tree. Your
  backend owns `templates/<yourbackend>/`, and a writer step copies from there into the project. A vendor
  file needs a license that permits redistribution, recorded at the top of the file together with where it
  came from.
- Write down the external interface contract the wrapper implements — beats consumed and produced per
  invocation, TLAST behaviour in both directions, which side channels exist and who sets them — in the
  documentation and in the driver. Boards that hang or return zeros with no way to debug them are the
  dominant class of accelerator issue report, and the written contract is what makes them debuggable.

## Verify end to end

Gate on a real convert, write, compile, predict and synthesize run of a small model in your target io_type,
comparing against the reference framework at a tolerance consistent with the precision. Confirm that each
custom attribute actually threads from the configuration through to the generated code — read the generated
file, do not assume.

hls4ml does not clean `output_dir` on re-conversion. Files from an earlier run persist, so inspect only a
freshly created or deleted-and-recreated directory.

Related: [**architecture map**](architecture-map.md) and [**optimizer passes**](optimizer-passes.md) for the machinery you are wiring into,
[**evaluating implementations**](evaluating-implementations.md) for benchmarking the kernels, [**toolchain access**](toolchain-access.md) for reaching the
toolchain your `build()` invokes.
