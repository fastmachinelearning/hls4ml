import glob
import os
from pathlib import Path
from shutil import copy

from hls4ml.writer.vivado_writer import VivadoWriter


class VitisWriter(VivadoWriter):
    def __init__(self):
        super().__init__()

    def write_nnet_utils_overrides(self, model):
        ###################
        # nnet_utils
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/vitis/nnet_utils/')
        dstpath = f'{model.config.get_output_dir()}/firmware/nnet_utils/'

        headers = [os.path.basename(h) for h in glob.glob(srcpath + '*.h')]

        for h in headers:
            copy(srcpath + h, dstpath + h)

    def write_board_script_override(self, model):
        '''
        Write the tcl scripts and kernel sources to create a Vitis IPI
        '''

        ###################
        # project.tcl
        ###################

        prj_tcl_file = Path(f'{model.config.get_output_dir()}/project.tcl')
        with open(prj_tcl_file) as f:
            prj_tcl_contents = f.readlines()
            for line_num, line in enumerate(prj_tcl_contents):
                if 'set backend' in line:
                    prj_tcl_contents[line_num] = 'set backend "vitis"\n'
                if 'set clock_uncertainty' in line:
                    prj_tcl_contents[line_num] = 'set clock_uncertainty {}\n'.format(
                        model.config.get_config_value('ClockUncertainty', '27%')
                    )

        with open(prj_tcl_file, 'w') as f:
            f.writelines(prj_tcl_contents)

    def write_hls(self, model):
        """
        Write the HLS project. Calls the steps from VivadoWriter, adapted for Vitis
        """
        super().write_hls(model)
        self.write_nnet_utils_overrides(model)
        self.write_board_script_override(model)
        self.write_tar(model)
