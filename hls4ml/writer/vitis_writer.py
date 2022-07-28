import os
from shutil import copyfile, copytree
from distutils.dir_util import copy_tree
from hls4ml.writer.vivado_writer import VivadoWriter

class VitisWriter(VivadoWriter):

    def __init__(self):
        super().__init__()

    def write_hls(self, model):
        """
        Write the HLS project. Calls the steps from VivadoWriter, adapted for Vitis
        """
        super(VitisWriter, self).write_hls(model)
