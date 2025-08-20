import os
import stat
from pathlib import Path


from hls4ml.writer.vitis_unified_writer.meta import VitisUnifiedWriterMeta
from hls4ml.writer.vitis_unified_writer.driver_gen import VitisUnified_DriverGen

class VitisUnifiedPartial_DriverGen(VitisUnified_DriverGen):


    @classmethod
    def write_driver(self, meta: VitisUnifiedWriterMeta, model, mg):
        filedir = os.path.dirname(os.path.abspath(__file__))
        fin     = open(os.path.join(filedir, '../../templates/vitis_unified_partial/driver/pynq/pynq_driver.py'), 'r')
        fout    = open(f'{model.config.get_output_dir()}/export/pynq_driver.py', 'w')


        for line in fin.readlines():
            newline = line
            ##### TODO
            fout.write(newline)


        fin.close()
        fout.close()