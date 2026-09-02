"""Runtime-selected weight banks for a stock hls4ml IP.

Post-export packaging only: the generated HLS compute IP is read, never modified,
and its RTL stays byte-identical (verified by the fingerprint in the summary).

Typical use, after ``hls_model.build(synth=True)`` on a project written with
``BramFactor`` set::

    from hls4ml.contrib.runtime_weights import package

    summary = package('my-hls-test', n_banks=2)

Bank selection is idle-time only: ``bank_id`` is latched when the IP accepts a
transaction and held until ``ap_done``. Writes to the selected bank are rejected.

Deliberately out of scope: AXI and board support, drivers, a C++ wrapper, any
automatic change to ``BramFactor``, and overlapping transactions.
"""

from hls4ml.contrib.runtime_weights.interface import InterfaceMismatch, verify  # noqa: F401
from hls4ml.contrib.runtime_weights.pack import (  # noqa: F401
    PackingUnsupported,
    build_bank_image,
    pack_scalar_bank,
    pack_weight_bank,
    quantize,
    write_mem,
)
from hls4ml.contrib.runtime_weights.package import fingerprint_ip, package  # noqa: F401
