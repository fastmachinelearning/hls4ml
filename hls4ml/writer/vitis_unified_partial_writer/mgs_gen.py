import os
import stat
from pathlib import Path


from hls4ml.writer.vitis_unified_writer.meta import VitisUnifiedWriterMeta
from .meta_gen import VitisUnifiedPartial_MetaGen as mg

class VitisUnifiedPartial_MgsGen():

    @classmethod
    def write_mgs(self, meta: VitisUnifiedWriterMeta, model):

        filedir = os.path.dirname(os.path.abspath(__file__))
        fin = open(os.path.join(filedir, '../../templates/vitis_unified_partial/myproject_mgs.cpp'), 'r')
        fout = open(f'{model.config.get_output_dir()}/firmware/myproject_mgs.cpp', 'w')


