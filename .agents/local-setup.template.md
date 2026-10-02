---
name: local-setup
description: >-
  Site-specific setup for this machine or group — interpreter, installed extras, how the HLS toolchain is
  reached, and where builds are run. Copy this template, fill it in, and keep it out of version control unless
  your group shares one machine. The other documents in this directory deliberately contain no local paths.
globs:
  - "**"
---

# Local setup

<!--
This is a template. Copy it to local-setup.md (git-ignored) or wherever your group keeps such notes, and
replace every placeholder. Delete sections that do not apply.

Everything here is specific to a machine, a cluster or a group. Nothing in it should be copied into the other
documents in this directory, which are written to be true anywhere.
-->

## Python environment

- Interpreter: <!-- path, version, and whether hls4ml is installed editable -->
- Optional extras installed: <!-- keras-v3, onnx, profiling, da, hgq, ... -->
- Frontend backend variable: <!-- e.g. KERAS_BACKEND=tensorflow -->
- How to add a package here: <!-- pip/uv command, and any cache or temp directory the site requires -->
- Directories to avoid running Python from: <!-- anything containing the hls4ml checkout -->

## HLS toolchain

- Tools and versions available: <!-- how to list them, not a fixed list -->
- How they are reached: <!-- module load, settings script, container image, wrapper directory -->
- Wrapper scripts: <!-- where they live, and what to add to PATH -->
- Container binds needed: <!-- filesystems your projects live on -->
- Per-run state: <!-- how each run gets its own HOME, if the tools need it -->

## Running builds

- Default part and clock used for comparisons: <!-- so results are comparable across runs -->
- Where to put build scratch: <!-- fast node-local storage, and where results are copied back to -->
- How long a typical build takes: <!-- so an agent knows what is normal -->
- Job submission, if builds go to a batch system: <!-- command and constraints -->

## Anything else that surprises a newcomer

<!--
Quotas, shared caches, filesystems that are fast to read and slow to write, machines that must not be used
for long jobs, licence servers, proxies.
-->
