from __future__ import absolute_import

from hls4ml.backends.backend import Backend, register_backend, get_backend, get_available_backends
from hls4ml.backends.fpga.fpga_backend import FPGABackend
from hls4ml.backends.vivado.vivado_backend import VivadoBackend
from hls4ml.backends.quartus.quartus_backend import QuartusBackend

register_backend('Vivado', VivadoBackend)
register_backend('Quartus', QuartusBackend)
