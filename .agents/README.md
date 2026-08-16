# Agent documentation

Working notes about how hls4ml is built, written to be read by an AI coding assistant and by anyone new to
the codebase. [AGENTS.md](../AGENTS.md) in the repository root is the short version that agents read first;
these documents are the detail it points to.

They are not a substitute for [CONTRIBUTING.md](../CONTRIBUTING.md), which states the rules contributions
have to follow. Where the two appear to disagree, CONTRIBUTING.md wins.

## The documents

| Document | Read it when |
| --- | --- |
| [running-hls4ml.md](running-hls4ml.md) | you need to convert a model, run `predict()`, or check that a change works |
| [architecture-map.md](architecture-map.md) | before changing anything in the Python tree — what each stage owns and which file to open |
| [frontends.md](frontends.md) | adding support for a layer or operator, or a model fails to parse |
| [optimizer-passes.md](optimizer-passes.md) | changing the graph, adding a Strategy, a layer initializer or a config attribute |
| [kernels.md](kernels.md) | writing or modifying the C++ compute kernels |
| [precision-and-debugging.md](precision-and-debugging.md) | choosing fixed-point types, or the numerical result is wrong |
| [evaluating-implementations.md](evaluating-implementations.md) | claiming one implementation is faster or smaller than another |
| [new-backend.md](new-backend.md) | standing up a backend for a new toolchain |
| [toolchain-access.md](toolchain-access.md) | synthesizing, choosing a tool version, or a build cannot find its tool |
| [contributing-changes.md](contributing-changes.md) | shaping a change, and again before opening a pull request |
| [reporting-issues.md](reporting-issues.md) | reporting a bug, a performance problem, or proposing a feature |
| [local-setup.template.md](local-setup.template.md) | recording how your own machine is set up |

`frontends.md` ends with an end-to-end checklist for adding a layer, which sequences the others.

## Using these with your assistant

Each document carries a short front matter block with a `name`, a `description` saying when it applies, and
`globs` listing the paths it covers. Different assistants consume that differently, so the files are kept in
one neutral place and adapted rather than duplicated:

```
python .agents/agent_adapters.py --list          # what can be generated
python .agents/agent_adapters.py claude cursor   # generate those views
```

Generated views are ignored by git. Nothing prevents you from pointing your assistant at `.agents/` directly —
the files are plain Markdown and the front matter is harmless.

## Keeping them true

These describe mechanisms that change. Two rules keep them from rotting:

- A change to the machinery a document describes updates that document in the same pull request.
- `test/pytest/test_agent_docs.py` checks that every repository path mentioned in these files still exists.
  It runs in the normal test suite; a renamed module makes it fail.

Statements should be checkable. Prefer naming the file that proves a claim over asserting it, and do not
record numbers from a specific machine or a specific project — the point is what stays true.
