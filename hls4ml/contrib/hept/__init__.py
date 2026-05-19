"""HEPT custom-layer support for hls4ml.

Call :func:`register` to enable the extension for selected backends.
"""

from .registration import register, vendor_external_headers

__all__ = ["register", "vendor_external_headers"]
