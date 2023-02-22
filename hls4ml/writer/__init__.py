from hls4ml.writer.quartus_writer import QuartusWriter
from hls4ml.writer.symbolic_writer import SymbolicExpressionWriter
from hls4ml.writer.vivado_accelerator_writer import VivadoAcceleratorWriter
from hls4ml.writer.vivado_writer import VivadoWriter
from hls4ml.writer.writers import Writer, get_writer, register_writer

register_writer('Vivado', VivadoWriter)
register_writer('VivadoAccelerator', VivadoAcceleratorWriter)
register_writer('Quartus', QuartusWriter)
register_writer('SymbolicExpression', SymbolicExpressionWriter)
