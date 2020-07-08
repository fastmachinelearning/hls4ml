from __future__ import absolute_import

from hls4ml.templates.templates import Backend, register_backend, get_backend
from hls4ml.templates.vivado_template import VivadoBackend
from hls4ml.templates.quartus_template import QuartusBackend

register_backend('Vivado', VivadoBackend)
register_backend('Quartus', QuartusBackend)
