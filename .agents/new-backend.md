---
name: new-backend
description: >-
  Create a new in-tree hls4ml FPGA backend — its Backend subclass, Writer, passes, and code templates — by
  forking an existing backend and pruning it to what is actually supported. Use when standing up a backend
  for a new toolchain or a variant of an existing one. Covers deciding whether a new backend is warranted,
  what to subclass, the pass discovery and namespacing rules, registration, and how to avoid carrying dead
  copied code.
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
  plumbing without toolchain-specific passes.
- **Writer:** there is no `FPGAWriter`. Writers subclass either `Writer` (`writer/writers.py`) or an existing
  backend's writer. For a backend that is not a variant of another, subclass `Writer` directly.
- **Types, passes, templates:** fork the closest existing backend's `*_types.py`, `passes/` and its
  `templates/` directory, renaming throughout.

You are not obliged to keep the source backend's C++ conventions. The `nnet::` namespace, the strategy enum,
the kernel typedef dispatch and the `hls::stream` representation are conventions of the Vivado family that
other backends in the tree follow only partly. Defining your own variable types, your own `Template`
subclasses and your own project layout is a supported extension, not a workaround. Say explicitly which
convention you are diverging from and why.

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
