## Description

<!--
What does this change do, and why? Include motivation, context, and any new
dependencies. Keep PRs focused — one logical change per PR reviews far faster.

Pull requests are squash-merged, so this description becomes the permanent
record of the change. Please write it for someone reading the history later.
-->

Fixes #  <!-- or: Relates to # -->

## Type of change <!-- Tick all that apply -->

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (existing configs, APIs or generated code behave differently)
- [ ] New frontend / backend
- [ ] New layer / operator support
- [ ] New configuration option — a new `io_type`, Strategy, or config attribute
- [ ] Research paper implementation
- [ ] Documentation
- [ ] Build, CI or tooling
- [ ] Refactor / cleanup (no functional change)

## Affected areas

<!-- Tick everything this PR touches. This tells reviewers which parts of the CI matter. -->

**Backends:** <!-- e.g. Vitis, Vivado, VivadoAccelerator, Quartus, oneAPI, Catapult, Libero, SymbolicExpression -->
- [ ]
- [ ] Backend-independent (core IR, optimizer, `hls4ml.model`)

**Frontends:** <!-- e.g. Keras v3, Keras v2, PyTorch, ONNX/QONNX -->
- [ ]
- [ ] Not related to a frontend

**Components:**
- [ ] IR (layers + model graph) / optimizer passes
- [ ] C++ templates / HLS sources under `hls4ml/templates/`
- [ ] Profiling, reporting or the CLI
- [ ] Packaging, build or CI

## Configurations affected and exercised

<!--
Fill one row per configuration this change actually touches.

These axes are not universal: not every backend supports every `io_type` or
Strategy, and some have no such concept at all — write `n/a` rather than
picking the nearest match. If a row applies to all values of an axis, write
`all`. Add columns if your change is keyed on something else entirely
(precision, reuse factor, a new attribute).

Under "Verified", say what you actually ran: pytest / csim / csynth / cosim /
hardware — and leave it blank if you ran nothing, rather than implying a run.
-->

| Backend | `io_type` | Strategy | Verified |
|---|---|---|---|
|  |  |  |  |
|  |  |  |  |

**New configuration axis introduced by this PR** (if any): <!-- name it, say which backends support it, and note what happens on the backends that don't -->

## Impact on generated HLS

<!-- Delete only if this PR cannot change generated code (e.g. docs-only). -->

- [ ] This PR **does not change** the generated HLS for existing models.
- [ ] This PR **changes** the generated HLS. Numbers below.

**Numerical behaviour:** <!-- bit-exact vs main / bounded difference / intentionally changed — say which -->

| Model / test | Backend & version | Part | Latency (cycles) | II | LUT | FF | DSP | BRAM |
|---|---|---|---|---|---|---|---|---|
| before |  |  |  |  |  |  |  |  |
| after  |  |  |  |  |  |  |  |  |

<!-- Csim-only is fine for many PRs — if you did not run synthesis, say so here
     explicitly rather than leaving the table empty. -->

## Tests

<!--
Which tests you added or ran under `test/pytest`, and anything a reviewer needs
in order to reproduce them.
-->

**Test configuration** (OS, Python, ML framework version, HLS tool version):

## Breaking changes and migration

<!-- Delete if not a breaking change. -->

**What breaks:**

**How users migrate:**

## AI assistance disclosure

<!--
REQUIRED. hls4ml welcomes AI-assisted contributions. We ask two things in
return: disclose the assistance here, and own the result. Tick exactly one
level below.

Disclosure belongs here, in the pull request, where a reviewer will read it.
Do not credit AI tools as authors in commit metadata — authorship carries
copyright, which a tool cannot hold. Some assistants add such trailers
automatically; please remove them before opening the PR.
-->

- [ ] **None** — no AI tool was used.
- [ ] **Assisted** — completion, refactoring, docstrings, tests; design and code are mine.
- [ ] **Substantial** — significant AI-generated portions, reviewed and edited by me.
- [ ] **Agentic** — produced largely end-to-end by an AI agent from my prompts.

Tool(s) and model(s): <!-- e.g. Claude Code (Opus 4.5), GitHub Copilot, Cursor, Codex -->

Where it was used: <!-- one line: which files or which part of the change -->

If anything other than *None* is ticked, confirm all of the following:

- [ ] **I am the author of this contribution and take full responsibility for
      it.** I have read and understood every line and can explain and defend it
      in review.
- [ ] I verified the generated code against real hls4ml and HLS semantics —
      no invented APIs, config keys, pragmas or citations.
- [ ] All numbers, logs and test results quoted in this PR come from runs I
      actually performed, not from agent output.
- [ ] I have the right to submit this work under the project's licence, and to
      my knowledge it does not reproduce third-party code that would conflict
      with it.
- [ ] No AI tool is credited as an author in any commit in this branch.

<!-- The same expectations apply to AI-generated review comments and issue
     replies: disclose them, and don't post output you haven't checked. -->

## Checklist

**Required:**
- [ ] I have read the [contributing guidelines](https://github.com/fastmachinelearning/hls4ml/blob/main/CONTRIBUTING.md).
- [ ] I installed and ran `pre-commit` on the files I edited.
- [ ] I added tests under `test/pytest` covering this change (a bug fix should
      add a test that fails on `main` and passes here).
- [ ] I self-reviewed the full diff and it contains no leftover debug code,
      commented-out blocks or unrelated changes.
- [ ] The AI assistance disclosure above is complete and accurate.

**If applicable:**
- [ ] Documentation under `docs/` updated.
- [ ] No new build or synthesis warnings.
- [ ] Public API / config schema changes are documented.

## Release note

<!--
One user-facing sentence for the release notes, or the word "none".
Write it for someone who has never seen this PR.
-->

```release-note

```
