"""Runtime-selected parameter banks for synthesized hls4ml designs.

The compute IP is left unchanged: this package reads an already-synthesized project
and generates bank storage plus a wrapper around it.

See ``docs/advanced/bramfactor.rst`` for usage, and this package's README.md for the
manifest, the adapter model and how to add a layer.
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
