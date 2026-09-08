"""Runtime-selected weight banks for a stock hls4ml IP.

Post-export packaging only: the generated HLS compute IP is read, never modified,
and its exported compute artifacts stay byte-identical (verified by the
fingerprint in the summary).

Typical use, after ``hls_model.build(synth=True)`` on a project written with
``BramFactor`` set::

    from hls4ml.contrib.runtime_weights import package

    summary = package('my-hls-test', n_banks=2)

``pack_banks`` turns one complete parameter set per bank into the memory images the
wrapper expects; ``pack_tensor`` / ``build_bank_image`` are the primitives under it.

Bank selection is idle-time only: ``bank_id`` is captured when the wrapper takes a
request, made stable before ``ap_start`` rises, and held through ``ap_done``. Any
valid bank may be written while idle; all writes are rejected while an inference
is active.

Deliberately out of scope: AXI and board support, drivers, a C++ wrapper, any
automatic change to ``BramFactor``, and overlapping transactions.
"""

from hls4ml.contrib.runtime_weights.interface import InterfaceMismatch  # noqa: F401
from hls4ml.contrib.runtime_weights.pack import (  # noqa: F401
    PackingUnsupported,
    build_bank_image,
    pack_banks,
    pack_flat,
    pack_tensor,
    write_mem,
)
from hls4ml.contrib.runtime_weights.package import InterfaceUnsupported, package  # noqa: F401

# Internal helpers (interface.verify, pack.quantize_port, package.fingerprint_ip,
# the schema constants) are deliberately not re-exported: import them from their
# module if you need them.
