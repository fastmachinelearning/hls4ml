from hls4ml.writer.altera_writer import AlteraWriter
from hls4ml.writer.catapult_writer import CatapultWriter
from hls4ml.writer.coyote_writer import CoyoteWriter
from hls4ml.writer.libero_writer import LiberoWriter
from hls4ml.writer.quartus_writer import QuartusWriter
from hls4ml.writer.symbolic_writer import SymbolicExpressionWriter
from hls4ml.writer.vitis_writer import VitisWriter
from hls4ml.writer.vivado_accelerator_writer import VivadoAcceleratorWriter
from hls4ml.writer.vivado_writer import VivadoWriter
from hls4ml.writer.writers import Writer, get_writer, register_writer  # noqa: F401
from hls4ml.writer.xls_writer import XLSWriter

register_writer('Vivado', VivadoWriter)
register_writer('VivadoAccelerator', VivadoAcceleratorWriter)
register_writer('Vitis', VitisWriter)
register_writer('Quartus', QuartusWriter)
register_writer('Altera', AlteraWriter)
register_writer('Catapult', CatapultWriter)
register_writer('Libero', LiberoWriter)
register_writer('SymbolicExpression', SymbolicExpressionWriter)
register_writer('XLS', XLSWriter)
register_writer('Coyote', CoyoteWriter)
