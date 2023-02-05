from __future__ import absolute_import

from hls4ml.backends.backend import Backend, register_backend, get_backend, get_available_backends
from hls4ml.backends.fpga.fpga_backend import FPGABackend
from hls4ml.backends.vivado.vivado_backend import VivadoBackend
from hls4ml.backends.vivado_accelerator.vivado_accelerator_backend import VivadoAcceleratorBackend
from hls4ml.backends.vivado_accelerator.vivado_accelerator_config import VivadoAcceleratorConfig
from hls4ml.backends.quartus.quartus_backend import QuartusBackend
from hls4ml.backends.catapult.catapult_backend import CatapultBackend

register_backend('Vivado', VivadoBackend)
register_backend('VivadoAccelerator', VivadoAcceleratorBackend)
register_backend('Quartus', QuartusBackend)
register_backend('Catapult', CatapultBackend)
