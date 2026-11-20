from collections import Counter


class DedupError(RuntimeError):
    """Raised when the FIFO list does not have the expected repeated-invocation shape."""


def dedup_env(env, *, verify_compiled=True):
    """Keep the first instance of each FIFO name in ``env.fifos``, in place, and return ``(env, stats)``.

    The testbench calls the design once per sample, and only the first call's FIFOs are modelled by the simulation.

    Raises:
        DedupError: If the names do not repeat uniformly, or a kept FIFO is not modelled by the simulation.
    """
    fifos = list(env.fifos)
    if not fifos:
        raise DedupError('environment lists no FIFOs at all')

    ids = [f.id for f in fifos]
    if len(set(ids)) != len(ids):
        raise DedupError('FIFO ids in the trace are not unique')

    counts = Counter(f.name for f in fifos)
    multiplicities = set(counts.values())
    if len(multiplicities) != 1:
        raise DedupError(
            f'channel names do not repeat uniformly (counts {sorted(multiplicities)}), cannot deduplicate the trace'
        )
    (multiplicity,) = multiplicities
    distinct = len(counts)

    seen, kept = set(), []
    for fifo in fifos:
        if fifo.name in seen:
            continue
        seen.add(fifo.name)
        kept.append(fifo)

    if verify_compiled:
        compiled = env.trace_base.compiled
        unmodelled = []
        for fifo in kept:
            try:
                compiled.get_fifo_design_space([fifo.id], fifo.width)
            except Exception:
                unmodelled.append(fifo.name)
        if unmodelled:
            raise DedupError(
                f'{len(unmodelled)} of {len(kept)} channels are not modelled by the compiled simulation '
                f'(e.g. {unmodelled[:3]})'
            )

    env.fifos = kept
    env.num_fifos = len(kept)
    return env, {'total': len(fifos), 'distinct': distinct, 'invocations': multiplicity}


def restrict_env(env, names):
    """Limit ``env.fifos`` to the named channels, in place, and return ``(env, n_kept)``.

    The other channels keep their generated depths and are still simulated.
    """
    names = set(names)
    kept = [f for f in env.fifos if f.name in names]
    if not kept:
        raise DedupError(f'none of the {len(names)} hls4ml stream names matches a channel in the trace')
    env.fifos = kept
    env.num_fifos = len(kept)
    return env, len(kept)
