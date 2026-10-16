---
name: reporting-issues
description: >-
  File a bug report, performance issue, feature request or RFC against hls4ml that a maintainer can act on.
  Use when something appears broken, slower than expected, or missing, before opening anything on the issue
  tracker. Covers deciding whether there is a bug at all, identifying which stage failed, reducing to a
  minimal reproducer, and what each issue form actually needs.
globs:
  - ".github/ISSUE_TEMPLATE/**"
---

# Reporting a problem in hls4ml

The counterpart to [contributing changes](contributing-changes.md): that one covers submitting a fix, this
one covers reporting the problem. An issue is a request for someone else's time, so the work of narrowing it
down belongs to whoever files it.

## Before filing anything

1. **Reproduce on current `main`.** Bugs get fixed; the one you found may already be gone. Note the commit
   you tested.
2. **Check it is not a configuration mistake.** Wrong precision, a reuse factor the layer cannot use, a
   strategy the backend does not support, an `io_type` the layer family does not implement — these produce
   confusing results and are not bugs. See [running hls4ml](running-hls4ml.md) for the arguments that decide
   behaviour.
3. **Separate unsupported from broken.** Backends differ in what they implement. A layer that has no kernel
   for a given backend or `io_type` is a missing feature, not a defect; file it as a feature request and say
   which backend.
4. **Search the tracker.** Add to an existing issue rather than opening a second one.
5. **If you can fix it in a small, obvious change, send the pull request instead.** The contributing
   guidelines ask contributors to spend effort on real improvements rather than on issue volume.

Do not file speculative or theoretical problems — something that could go wrong in principle, or that a
reading of the code suggests, but that you have not observed. If you cannot reproduce it, you do not yet have
a report.

## Identify which stage failed

hls4ml is a pipeline, and the stage that fails determines what the report needs. Establish this before
writing anything.

| Stage | Symptom | What the report needs |
| --- | --- | --- |
| Parsing | conversion fails on an unknown layer or attribute | frontend and its version, the layer, the model definition |
| Graph and passes | conversion succeeds but the graph is wrong; an exception from a pass | the failing pass name from the traceback, the config used |
| Writing | conversion succeeds, generated code is missing or malformed | the generated file and the part of it that is wrong |
| C simulation | `predict()` disagrees with the reference model | both outputs, the precision, and the tolerance you expected |
| Synthesis | the build fails, or the design does not meet timing | tool and version, the error from the log, the part and clock |

A traceback usually names the stage directly. Include the **whole** traceback, not the last line.

## Reduce to a minimal reproducer

A report that arrives with a small script gets fixed; one that arrives with a research model usually does
not. Shrink it:

- One layer if possible, with the smallest shapes that still fail, and a fixed random seed.
- Replace loaded weights with generated ones unless the values matter — if they do, say so, because that is
  itself a clue.
- Put the model definition inline in the report rather than attaching a file.

While shrinking, note which axes matter, because that is most of the diagnosis:

- Does it depend on `io_type`? On the backend? On the strategy or reuse factor?
- Does it survive at a much wider precision such as `ap_fixed<32,16>`? If it disappears, the problem is
  numerical rather than structural.

## Telling quantization from a bug

A fixed-point result never matches the float reference exactly. At `ap_fixed<16,6>` a small dense network
lands within a few times `1e-3`, and differences at that scale are quantization, not defects. Errors of order
`0.1` or larger indicate something real — overflow from too few integer bits, a wrong weight layout, an
uninitialized accumulator. Say which of the two you believe it is and why. See
[precision and debugging](precision-and-debugging.md).

## Performance and resource issues

These need numbers, and numbers need provenance. Say which stage produced them: C synthesis reports
estimates, co-simulation gives true latency, logic synthesis gives true resource usage. Quoting an estimate
as a measurement wastes a maintainer's time.

Include the part, the clock period, the tool version, the model shape, the precision, the strategy and the
reuse factor, plus what you expected and where that expectation comes from. A report that some model is
"slow" without a configuration cannot be acted on. See
[evaluating implementations](evaluating-implementations.md) for producing comparable numbers.

## Feature requests and RFCs

A **feature request** describes a capability you need: the problem, why it matters for real users, and the
user-facing surface it would add. A **plan or RFC** proposes a design: motivation, mechanism, the components
affected, what happens on backends that cannot support it, and the migration for existing users. If your
proposal changes the intermediate representation, the optimizer machinery, or behaviour that existing
configurations depend on, it is an RFC and the guidelines ask for that discussion before implementation.

## What not to put in an issue

- Whole log files. Excerpt the error and the lines around it.
- Whole generated projects. Name the file and quote the part that matters.
- Screenshots of text.
- Guesses presented as findings. If you suspect a cause, mark it as a suspicion.

## For agents specifically

- **Never fabricate a traceback, an error message, a log line or a number.** Everything quoted must come from
  a run that actually happened. If you have not run it, say so.
- If reproduction failed, report that outcome rather than filing an issue that implies it succeeded.
- Fill the issue form's fields; each one exists because reports kept arriving without it.
- Tick the AI assistance box on the form, and let the human check the content before it is posted.
- One issue per problem, and no duplicates — search first.

## Anti-checklist

- Do not open an issue you have not reproduced on current `main`.
- Do not report a configuration error as a bug without first checking the arguments you passed.
- Do not report a quantization difference as incorrect output.
- Do not quote C synthesis estimates as measured latency or resource usage.
- Do not attach a large model when a ten-line one shows the same failure.
- Do not file an issue for something a small pull request would fix.
