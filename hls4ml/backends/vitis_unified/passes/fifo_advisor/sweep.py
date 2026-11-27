import time
from pathlib import Path

from . import ls_compat
from .dedup import dedup_env, restrict_env


class AdvisorResult:
    """Selected FIFO depths plus the Pareto front, per-solver results and timings of a sweep."""

    def __init__(self, depths, winner, front, per_solver, fifo_stats, compat_stats, timings):
        self.depths = depths
        self.winner = winner
        self.front = front
        self.per_solver = per_solver
        self.fifo_stats = fifo_stats
        self.compat_stats = compat_stats
        self.timings = timings

    def to_dict(self):
        """Serialise the result for ``fifo_depths_advisor.json``."""
        return {
            'winner': self.winner,
            'pareto_front': self.front,
            'solvers': self.per_solver,
            'fifo_stats': self.fifo_stats,
            'ls_compat_stats': self.compat_stats,
            'timings_s': self.timings,
            'depths': self.depths,
        }


def _solver_classes():
    from fifo_advisor.solvers import (
        DiscreteSimulatedAnnealingOptimizer,
        GroupedDiscreteSimulatedAnnealingOptimizer,
        GroupRandomSearchOptimizer,
        HeuristicOptimizer,
        RandomSearchOptimizer,
    )

    return {
        'heuristic': HeuristicOptimizer,
        'sa': DiscreteSimulatedAnnealingOptimizer,
        'group-sa': GroupedDiscreteSimulatedAnnealingOptimizer,
        'random': RandomSearchOptimizer,
        'group-random': GroupRandomSearchOptimizer,
    }


def unique_points(env, evals, searched=None):
    """Drop repeated design points and, if ``searched`` is given, re-cost each point on those FIFO ids only.

    FIFO-Advisor's heuristic sizes every FIFO in the simulation, not only ``env.fifos``, and repeats points.
    """
    points, seen = [], set()
    for r in evals:
        sizes = r.fifo_sizes if searched is None else {i: d for i, d in r.fifo_sizes.items() if i in searched}
        key = frozenset(sizes.items())
        if key not in seen:
            seen.add(key)
            points.append((r, sizes))
    if searched is None:
        return [r for r, _ in points]
    return env.eval_solution_parallel([sizes for _, sizes in points]) if points else []


def select_from_front(front, selection):
    """Pick one point of ``front`` by ``'min_bram'``, ``'min_latency'`` or a callable."""
    if callable(selection):
        return selection(front)
    if selection == 'min_bram':
        return min(front, key=lambda r: (r.bram_usage_total, r.latency))
    if selection == 'min_latency':
        return min(front, key=lambda r: (r.latency, r.bram_usage_total))
    raise ValueError(f'unknown selection rule: {selection!r}')


def analyze_solution(
    solution_dir,
    *,
    solvers,
    selection='min_bram',
    restrict_names=None,
    retrace=True,
    verbose=True,
):
    """Trace the solution with LightningSim, run the solvers and select depths.

    Args:
        solution_dir (str or pathlib.Path): The synthesised HLS solution directory.
        solvers (tuple[str]): Solvers to run, in addition to the baseline.
        selection (str or callable): Passed to :func:`select_from_front`.
        restrict_names (set[str], optional): If given, only these channels are searched; the others keep their
            generated depths but are still simulated.
        retrace (bool): Discard a cached ``trace.pkl`` before tracing.
        verbose (bool): Print progress.

    Raises:
        RuntimeError: If every evaluated assignment deadlocks.

    Returns:
        AdvisorResult: The selected depths and sweep results.
    """
    from fifo_advisor.main import collect_baseline_results, fifo_id_to_name_map_from_env
    from fifo_advisor.opt_env import LSEnv, is_pareto_efficient_simple

    solution_dir = Path(solution_dir)
    timings = {}

    if retrace:
        # A cached trace may come from another build of the design.
        (solution_dir / 'trace.pkl').unlink(missing_ok=True)

    t0 = time.time()
    env, fifo_stats = dedup_env(LSEnv(solution_dir))
    searched = None
    if restrict_names is not None:
        env, fifo_stats['searched'] = restrict_env(env, restrict_names)
        searched = {f.id for f in env.fifos}
    timings['trace'] = round(time.time() - t0, 3)
    if verbose:
        print(
            f'  trace            : {timings["trace"]}s '
            f'({fifo_stats["distinct"]} channels x {fifo_stats["invocations"]} invocations '
            f'= {fifo_stats["total"]} entries; searching {fifo_stats.get("searched", fifo_stats["distinct"])})'
        )

    names = fifo_id_to_name_map_from_env(env)

    def named(sizes):
        return {names[f]: int(v) for f, v in sizes.items() if f in names}

    everything, per_solver, solver_of = [], {}, {}
    classes = _solver_classes()

    t_sweep = time.time()
    for label in ('baseline', *solvers):
        t0 = time.time()
        try:
            evals = collect_baseline_results(env) if label == 'baseline' else classes[label](env).solve()
        except Exception as exc:  # a failing solver should not abort the sweep
            per_solver[label] = {'failed': f'{type(exc).__name__}: {exc}'}
            if verbose:
                print(f'  {label:<13} FAILED - {type(exc).__name__}: {exc}')
            continue
        evals = unique_points(env, evals, searched)
        dt = round(time.time() - t0, 3)
        ok = [r for r in evals if not r.deadlock]
        if not ok:
            per_solver[label] = {'runtime_s': dt, 'n_evals': len(evals), 'all_deadlock': True}
            if verbose:
                print(f'  {label:<13} runtime={dt:7.3f}s  evals={len(evals):>4}  all deadlock')
        else:
            best = min(ok, key=lambda r: (r.bram_usage_total, r.latency))
            per_solver[label] = {
                'runtime_s': dt,
                'n_evals': len(evals),
                'best_latency': best.latency,
                'best_bram': best.bram_usage_total,
                'best_depths': named(best.fifo_sizes),
            }
            if verbose:
                print(
                    f'  {label:<13} runtime={dt:7.3f}s  evals={len(evals):>4}  '
                    f'best latency={best.latency} bram={best.bram_usage_total}'
                )
        for r in evals:
            solver_of[id(r)] = label
        everything.extend(evals)
    timings['sweep'] = round(time.time() - t_sweep, 3)

    nd = [r for r in everything if not r.deadlock]
    if not nd:
        raise RuntimeError('every evaluated depth assignment deadlocked; nothing to select')

    flags = is_pareto_efficient_simple(nd)
    seen, front = set(), []
    for r in sorted((r for r, f in zip(nd, flags) if f), key=lambda r: (r.bram_usage_total, r.latency)):
        if (r.latency, r.bram_usage_total) not in seen:
            seen.add((r.latency, r.bram_usage_total))
            front.append(r)

    def solvers_reaching(result):
        key = (result.latency, result.bram_usage_total)
        found = [
            lbl
            for lbl in ('baseline', *solvers)
            if any(solver_of.get(id(r)) == lbl and (r.latency, r.bram_usage_total) == key for r in nd)
        ]
        return found or ['unknown']

    win = select_from_front(front, selection)
    if verbose:
        print(f'  pareto front     : {[(r.latency, r.bram_usage_total) for r in front]}')
        for r in front:
            print(f'    ({r.latency}, {r.bram_usage_total}) found by: {", ".join(solvers_reaching(r))}')
        print(f'  selected         : latency={win.latency} bram={win.bram_usage_total}')
        print(f'  selected found by: {", ".join(solvers_reaching(win))}')

    return AdvisorResult(
        depths=named(win.fifo_sizes),
        winner={
            'latency': win.latency,
            'bram': win.bram_usage_total,
            'found_by': solvers_reaching(win),
        },
        front=[
            {
                'latency': r.latency,
                'bram': r.bram_usage_total,
                'found_by': solvers_reaching(r),
                'depths': named(r.fifo_sizes),
            }
            for r in front
        ],
        per_solver=per_solver,
        fifo_stats=fifo_stats,
        compat_stats=ls_compat.stats(),
        timings=timings,
    )
