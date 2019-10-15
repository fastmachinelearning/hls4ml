from __future__ import absolute_import

from .templates import Backend, register_backend, get_backend
from .vivado_template import VivadoBackend

register_backend('Vivado', VivadoBackend)
