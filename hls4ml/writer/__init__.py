from __future__ import absolute_import

from .writers import Writer, register_writer, get_writer
from .vivado_writer import VivadoWriter
from .brisk_writer import BriskWriter

register_writer('Vivado', VivadoWriter)
register_writer('Brisk', BriskWriter)
