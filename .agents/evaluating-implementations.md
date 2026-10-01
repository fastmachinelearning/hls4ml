---
name: evaluating-implementations
description: >-
  Fairly benchmark a new or modified hls4ml compute kernel / Strategy / layer implementation against the
  baseline it wants to replace. Use whenever deciding go/no-go on a kernel change, a new strategy, an
  upstream PR, or any claim of the form "X is faster/smaller than Y" in hls4ml or its HLS backends. Covers
  building a fair comparison, the sweep matrix (shapes, io_type, reuse_factor), dependent-layer fallout,
  synthesis artifacts that fake wins, measuring with co-simulation, tool versions, and reporting a
  defensible verdict.
globs:
  - "hls4ml/templates/**"
  - "hls4ml/backends/**"
---

# Evaluating an hls4ml implementation (kernel / strategy / layer)

A benchmark is worth only as much as its fairness. Treat the first result as wrong until you have shown that
the two implementations differ in **exactly one** variable.

## The cardinal rule: compare through real hls4ml, not a standalone harness

Do not write a hand-rolled C++ and C-synthesis harness that calls the two kernels and compares the reports.
It silently handicaps one side. The usual mechanism: a harness passes weights as a function argument, so the
tool infers a memory interface with one or two ports, while real hls4ml emits them as partitioned constant
arrays. Whichever kernel issues more weight reads per cycle is then throttled by the harness rather than by
its own design, and the difference can be an order of magnitude. Similar distortions come from a missing
top-level pragma, a different interface mode, or weights that are constant on one side and not the other.

Instead, integrate **both** implementations as first-class options inside hls4ml — a new Strategy, or a
kernel selected by the layer config — and generate one project per variant with the same backend, io_type,
precision, part and clock. hls4ml then emits identical scaffolding around each kernel, so the kernel is the
only difference. Confirm this by diffing the two generated `firmware/` directories: everything except the
intended change should be identical.

**Calibrate before sweeping.** Check at least one configuration against a number you trust — the same layer
built through the unmodified path, or a published figure. If a single point is far from expectation, stop and
find the cause. A sweep built on an unfair setup produces a large, self-consistent, wrong answer.

## Verify correctness before looking at quality of results

For every configuration: convert, `compile()`, `predict()`, and compare against the reference model. The
maximum absolute error should be consistent with the precision in use — of order `1e-3` for
`ap_fixed<16,6>`. A kernel that synthesizes small and fast but computes the wrong answer is not a win.

Sign flips and transposes hide behind symmetric test data. Use non-square shapes and asymmetric weights and
inputs, so an index error cannot cancel itself out.

## The sweep matrix — square shapes at the default reuse factor are not enough

Rankings flip across regimes, so vary every axis:

- **Shapes:** `n_in == n_out`, `n_in > n_out`, `n_in < n_out`, powers of two, and odd sizes. Odd and
  non-square shapes expose padding and divisibility assumptions.
- **io_type:** both `io_parallel` and `io_stream`. They wrap kernels differently, and for some layer families
  io_stream is a separate implementation rather than a wrapper.
- **reuse_factor:** the full valid range, not the default alone. Only certain values are valid; respect the
  backend's `_validate_reuse_factor`. Report results grouped by regime, because that is where rankings
  diverge: `rf < n_in`, `rf == n_in` (the common default — a loss here usually decides the verdict), and
  `rf >> n_in` (large layers folded to save multipliers).
- **Precision:** at least two widths. The balance between logic and multiplier use moves with bit width.
- **Backend:** if the change touches shared code, every backend that reaches it.

When the two implementations expose different parallelism knobs, match resources rather than parameters:
choose settings that give both the same multiplier count, then compare logic, registers, clock frequency and
latency at that operating point.

## Establish the weight storage model first

Before drawing any conclusion about resources, know how the design under test is meant to hold its weights,
because the answer changes what a result means:

- **Baked into the design as constants.** This is the default and, for most implementations and backends, the
  intended deployment. The tool then replaces multipliers with fixed-coefficient logic wherever it can, and
  multiplier counts fall. **That saving is real** — it is part of what the implementation delivers, not an
  artifact — as long as the design ships with constant weights.
- **Held in memory, on-chip or off-chip, or reloadable at runtime.** Weights are read rather than folded into
  the arithmetic, so strength reduction does not apply and the multipliers come back. hls4ml exposes part of
  this today through `BramFactor`, which turns weights above a size threshold into a BRAM interface on the
  Vivado, Vitis and Catapult backends; the default threshold is high enough that weights stay constant unless
  you ask otherwise. Work aimed at off-chip or reloadable weights lives in this mode by design.

The mistake is carrying a result across the boundary. A constant-weight comparison says nothing about a
design that must load weights at runtime, and vice versa. So state which mode you measured in, measure the
implementation in the mode it is actually aimed at, and if a claim is meant to hold in both, measure both.

## Synthesis artifacts that fake a win

- **Pipeline or dataflow directives not applied.** If a new strategy is not recognized by the pass that sets
  the pipeline style, the kernel is left serial or fully unrolled and the reported cost describes the
  scaffolding, not the design. A resource use far from what the parallelism setting implies is the signal.
- **Estimates instead of measurements.** C synthesis under-reports multi-layer io_stream latency for both
  sides. Use it for relative comparison at matched conditions only.

## Measuring with co-simulation

Co-simulation runs the generated testbench, `myproject_test.cpp`, which reads its inputs from the project's
`tb_data/` directory. Give it the inputs explicitly — the `input_data_tb` and `output_data_tb` arguments of
the convert call take `.npy` or `.dat` files — and check the build log afterwards. When the testbench cannot
open those files it says so in the log and falls back to a small default input set chosen by the backend; the
run still completes and the report still has numbers in it.

Run enough inputs for the design to reach its steady pace. In a design whose stages run at the same time, the
buffers between stages accept the first inputs at the pace of the first stage, so with too few inputs the
measured interval is the first stage's rather than the slowest one's. Several times as many inputs as there
are buffering stages is a safe number. Where the report gives the smallest and largest value over the
simulated inputs, take:

- **latency** as the smallest — one input entering an idle design. In a design whose stages run at different
  speeds, later inputs also wait in the buffers, so the largest latency describes that waiting, not the
  design;
- **interval** as the largest — the pace the design keeps once its buffers are full.

Then check each number against what the design has to do: the interval cannot be shorter than the time its
slowest stage needs for one input. A measurement that beats such a bound is wrong, whichever side it favors.

## Tool versions

Use one tool version for every configuration of a comparison wherever it can build them. Releases fail on
designs that neighbouring releases build, so a version may not build some configurations. Build both sides of
those configurations with another version, mark them in the results with the version they come from, and
never compare a design built with one version against a design built with another.

## Check the layers that share the algorithm

A change to the matrix-vector code path is not local: the convolution family reuses it, and recurrent layers
call a dense-like kernel per gate. Before declaring a result, generate a layer from each dependent family
through the changed path and confirm correctness and cost there too. A win in one layer that regresses
another is not a win.

## Deliverables

- A results table with ratios per configuration (logic, registers, latency, clock frequency, multipliers),
  averaged **within** each regime and never across regimes. State the convention explicitly, for example
  that a ratio below one means smaller or faster.
- Plots of those ratios against size and regime.
- A short report covering: why the comparison is fair and which point was calibrated, what holds across all
  configurations, the regime split, the artifacts discounted, and a one-line verdict naming the operating
  point it applies to. "Wins in a narrow regime" is a legitimate verdict — say which regime, and what
  adopting it would cost.

## Anti-checklist

1. Concluding from a standalone harness rather than from real hls4ml output.
2. Reporting square shapes at the default reuse factor and calling it a sweep.
3. Comparing under constant weights and claiming the result for a design that loads its weights, or the
   reverse.
4. Forgetting the layer families that share the algorithm.
5. Averaging across regimes, which hides a loss in the most common one.
6. Reporting C synthesis estimates as final latency or resource numbers.
7. Quoting a co-simulation without checking that it ran the inputs you gave it, and enough of them to reach
   the steady interval.
8. Comparing a design built with one tool version against one built with another.

Latency numbers belong to co-simulation and resource numbers to logic synthesis. See
[**toolchain access**](toolchain-access.md) for running those here, and
[**contributing changes**](contributing-changes.md) once a verdict says the change is worth proposing.
