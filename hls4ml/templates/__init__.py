from __future__ import absolute_import

from hls4ml.templates.templates import Backend, register_backend, get_backend
from hls4ml.templates.vivado_template import VivadoBackend
from hls4ml.templates.pynq_template import PynqBackend

register_backend('Vivado', VivadoBackend)
register_backend('Pynq', PynqBackend)
