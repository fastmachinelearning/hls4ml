# Working on hls4ml as an AI agent

This file is for AI coding agents. Humans should read [CONTRIBUTING.md](CONTRIBUTING.md), which this file
follows; where the two appear to disagree, CONTRIBUTING.md wins.

hls4ml compiles trained neural networks into HLS projects for FPGAs. The Python package is in `hls4ml/`, its
C++ kernel sources in `hls4ml/templates/`, tests in `test/pytest/`, documentation in `docs/`.

Conversion is a pipeline, and almost every change belongs to exactly one stage of it:

```
frontend converter  ->  model graph  ->  optimizer passes  ->  code templates  ->  writer
hls4ml/converters/      hls4ml/model/    hls4ml/model/optimizer/, hls4ml/backends/*/passes/    hls4ml/writer/
```

## Where to read before you work

Detailed notes live in [`.agents/`](.agents/README.md). Open the one that matches the task rather than
exploring the tree from scratch:

| Task | Document |
| --- | --- |
| convert a model, run `predict()`, check a change works | [.agents/running-hls4ml.md](.agents/running-hls4ml.md) |
| anything inside the Python tree — what runs when, which file to open | [.agents/architecture-map.md](.agents/architecture-map.md) |
| add support for a layer or operator; a model fails to parse | [.agents/frontends.md](.agents/frontends.md) |
| change the graph, add a Strategy, an initializer or a config attribute | [.agents/optimizer-passes.md](.agents/optimizer-passes.md) |
| write or modify a C++ compute kernel | [.agents/kernels.md](.agents/kernels.md) |
| choose fixed-point types, or the numbers are wrong | [.agents/precision-and-debugging.md](.agents/precision-and-debugging.md) |
| claim one implementation is faster or smaller than another | [.agents/evaluating-implementations.md](.agents/evaluating-implementations.md) |
| stand up a backend for a new toolchain | [.agents/new-backend.md](.agents/new-backend.md) |
| synthesize, pick a tool version, or a build cannot find its tool | [.agents/toolchain-access.md](.agents/toolchain-access.md) |
| shape a change, and before opening a pull request | [.agents/contributing-changes.md](.agents/contributing-changes.md) |
| report a bug or a performance problem, or propose a feature | [.agents/reporting-issues.md](.agents/reporting-issues.md) |

## Ground rules

**Verify, do not assume.** hls4ml has several backends and they do not share conventions. A pragma, config
field, C++ class or attribute that exists for one backend may not exist for another. Read the file you are
about to rely on. Never invent an API, configuration key, pragma or citation.

**Numbers come from runs you performed.** Do not quote latency, resource usage or accuracy that you did not
measure. If you did not run something, say so plainly. C synthesis reports estimates; co-simulation gives
true latency; logic synthesis gives true resource usage. Do not present an estimate as a measurement.

**Match the surrounding code.** Comment density, naming and idiom should look like the file you are editing.
Do not add narration, banner comments, or explanations of what the next line does. Generated verbosity is a
common reason contributions are sent back.

**Stay inside the requested scope.** Do not reformat unrelated code, rename things you were not asked to
rename, or fix unrelated problems in the same change. If you notice something else worth doing, mention it
rather than doing it.

**Do not add dependencies casually.** The runtime dependency list is deliberately small and most extras are
optional. A new hard dependency needs a reason, and the default answer is an optional extra.

## Authorship and disclosure

Do not add `Co-authored-by` or any similar trailer crediting an AI tool in a commit message. Authorship
carries copyright, which a tool cannot hold, and such trailers distort contributor statistics. Some harnesses
add them automatically; do not.

Disclosure belongs in the pull request description, which has a section for it. Fill in the tool and model
honestly, and state where assistance was used.

The pull request template contains attestations that a human contributor makes about their own review and
rights. **Do not tick those on the human's behalf.** Fill in the factual parts, leave the attestations for
the person opening the pull request, and tell them what remains to be confirmed.

## Before proposing a change

- Run `pre-commit run --files <edited files>` and commit what it changes. It formats Python and C++ and will
  reject the pull request otherwise.
- Add or update a test under `test/pytest/`. For a bug fix, the test should fail without the fix.
- Check the whole diff yourself, hunk by hunk, before presenting it. Remove debug output, commented-out code
  and stray changes.
- Do not commit generated HLS projects, model files or logs.
- New functionality is discussed before it is built. If there is no issue for it, propose one rather than
  opening a large unsolicited pull request.

## Reporting back

State what you actually did, including what you could not do. If a test fails, say so and show the output. If
you skipped a step, say which. Do not describe work as complete until it is, and do not summarize an intended
change as though it had been made.
