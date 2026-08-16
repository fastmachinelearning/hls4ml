---
name: toolchain-access
description: >-
  Find, select and run the vendor toolchain an hls4ml backend needs (Vitis, Vivado, Quartus, oneAPI and
  others), including when the tool is not installed on the host. Use whenever you need to synthesize or build
  an hls4ml project, choose a tool version, launch a long build, or when a build fails with "installation not
  found". Covers what each backend looks for, where a toolchain can come from, writing wrappers, version
  selection, and which reported numbers to trust.
globs:
  - "hls4ml/backends/*/*_backend.py"
---

# Getting an HLS toolchain and running a build

## What hls4ml actually requires

Every backend's `build()` looks for one command on `PATH` and runs it in the generated project directory.
Nothing more is required: satisfy the name on `PATH` and the backend is happy, whether the command is a real
installation, a module, or a wrapper script.

| Backend | Command it looks for | What `build()` runs |
| --- | --- | --- |
| Vitis | `vitis-run` | `vitis-run --tcl build_prj.tcl --mode hls` |
| Vivado | `vivado_hls` | `vivado_hls -f build_prj.tcl "reset=… csim=… …"` |
| Quartus | `quartus_sh` | `make <project>-fpga` in the project directory |
| oneAPI | the compiler toolchain used by the generated build files | `make <build_type>` in the build directory |

Check the backend's `build()` in `hls4ml/backends/<backend>/<backend>_backend.py` rather than assuming; the
command and its arguments change between releases. When the command is missing, the backend raises an
exception naming it — that message is the fastest diagnosis of a toolchain problem.

## Where a toolchain can come from

Vendor HLS tools are rarely on `PATH` by default. In order of how often they work:

1. **A wrapper script your site already provides.** Check the local setup notes for the machine you are on;
   many groups keep wrappers in a directory that only needs to be added to `PATH`.
2. **A local installation**, reached through a module system or by sourcing the vendor's settings script.
3. **A container image.** Vendor tools are commonly distributed as container images, sometimes as unpacked
   directories on a shared filesystem. Which images exist is site-specific — list what is available rather
   than assuming a path from documentation written elsewhere.
4. **Your own container image**, if nothing above provides the tool or the version you need.

Record whichever applies in your site's local setup notes; see `local-setup.template.md`.

## Writing a wrapper

A wrapper is a script named exactly what the backend looks for, placed on `PATH`. For a container image:

```bash
#!/usr/bin/env bash
exec <container runtime> exec --home "$HOME" -B <paths your project lives on> \
    <image> <command> "$@"
```

Three details matter:

- **Bind the filesystems your project lives on.** Paths not bound are invisible inside the container.
  Runtimes usually bind `/tmp` and the current directory automatically; anything else has to be requested.
- **Give each run its own `HOME`.** Xilinx tools keep state in `$HOME/.Xilinx`. When several runs share it
  they interfere and fail, either unable to load a Tcl package or crashing during logic synthesis.
- **Keep the wrapper's filename equal to the command name**, so both hls4ml's `build()` and generated build
- **Keep the wrapper's filename equal to the command name**, so both hls4ml's `build()` and generated build
  scripts resolve it.

Selecting a different version means pointing a wrapper at a different version directory. Keeping one wrapper
per version, named distinctly, and a plain-named wrapper for the default is a workable arrangement.

## Choosing a version deliberately

Results depend on the tool version: scheduling, resource estimates, achieved clock frequency and even
whether a design compiles all change between releases. So:

- Use the **same version across every run of a comparison**, and say which one in any report of results.
- When reproducing an earlier result, reproduce its version too.
- When trying a newer version, treat it as a separate axis, not as a free upgrade — rerun the baseline.

## Running a build

Either call `model.build(...)` from Python, which runs the command in the table above, or run the same
command by hand in the generated project directory. Drive the tool through its scripted entry point rather
than a graphical session.

Long builds take minutes to hours, so run them in a way that survives the calling process rather than
blocking on them, and avoid stacking extra shell layers around the launch — the intermediate process tends to
be orphaned and the build lost. Run one build per process, and never let two processes append to the same
results file, because interleaved writes corrupt it. Give each run its own output file.

## Working directory, environment, storage

- **Run from the generated project directory**, and never from a directory containing the hls4ml checkout —
  it shadows the installed package. See [**running hls4ml**](running-hls4ml.md).
- Put build scratch on fast node-local storage rather than a shared or network filesystem, and copy the small
  result files back afterwards. HLS builds write continuously, and write latency on shared storage is often
  what limits them.

## Which number to trust

Four levels, in order; a design passes only when it clears all of them:

1. **C simulation** — builds and runs, correct against the reference model.
2. **C synthesis** — early latency and resource *estimates*. Directional only; it under-reports multi-layer
   io_stream latency. Never quote it as a final number.
3. **Co-simulation** — the true cycle count. Latency in any verdict comes from here.
4. **Logic synthesis** — the true LUT, FF, DSP and BRAM use, and the achieved clock.

Read the logs, not only the parsed report: timing-closure failures, scheduling problems and initiation
interval violations appear as log text while the summary table still looks acceptable. Reports and logs are
under the solution directory of the generated project. Match part, clock period and tool version across any
two runs being compared.

Related: [**evaluating implementations**](evaluating-implementations.md) for building a fair comparison on top of these runs,
[**kernels**](kernels.md) for reading the generated project itself.
