from __future__ import absolute_import

from .writers import Writer, register_writer, get_writer
from .vivado_writer import VivadoWriter

register_writer('Vivado', VivadoWriter)
