from __future__ import absolute_import

from hls4ml.templates.templates import Backend, register_backend, get_backend, get_available_backends, get_supported_boards_dict
from hls4ml.templates.vivado_template import VivadoBackend
from hls4ml.templates.vivado_accelerator_template import VivadoAcceleratorBackend

register_backend('Vivado', VivadoBackend)
register_backend('VivadoAccelerator', VivadoAcceleratorBackend)