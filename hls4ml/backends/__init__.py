from hls4ml.backends.backend import Backend, get_available_backends, get_backend, register_backend  # noqa: F401
from hls4ml.backends.fpga.fpga_backend import FPGABackend  # noqa: F401
from hls4ml.backends.quartus.quartus_backend import QuartusBackend
from hls4ml.backends.symbolic.symbolic_backend import SymbolicExpressionBackend
from hls4ml.backends.vivado.vivado_backend import VivadoBackend
from hls4ml.backends.vivado_accelerator.vivado_accelerator_backend import VivadoAcceleratorBackend
from hls4ml.backends.vivado_accelerator.vivado_accelerator_config import VivadoAcceleratorConfig  # noqa: F401

from hls4ml.backends.vitis.vitis_backend import VitisBackend  # isort: skip

register_backend('Vivado', VivadoBackend)
register_backend('VivadoAccelerator', VivadoAcceleratorBackend)
register_backend('Vitis', VitisBackend)
register_backend('Quartus', QuartusBackend)
register_backend('SymbolicExpression', SymbolicExpressionBackend)
