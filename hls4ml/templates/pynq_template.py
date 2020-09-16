from hls4ml.templates.vivado_template import VivadoBackend
import os
from shutil import copyfile

class PynqBackend(VivadoBackend):
    def __init__(self):
        super(PynqBackend, self).__init__(name='Pynq')

    def make_bitfile(model):
        curr_dir = os.getcwd()
        os.chdir(model.config.get_output_dir())
        try:
            os.system('vivado -mode batch -source pynq_design.tcl')
        except:
            print("Something went wrong, check the Vivado logs")
        # These should work but Vivado seems to return before the files are written...
        #copyfile('{}_pynq/project_1.runs/impl_1/design_1_wrapper.bit'.format(model.config.get_project_name()), './{}.bit'.format(model.config.get_project_name()))
        #copyfile('{}_pynq/project_1.srcs/sources_1/bd/design_1/hw_handoff/design_1.hwh'.format(model.config.get_project_name()), './{}.hwh'.format(model.config.get_project_name()))
        os.chdir(curr_dir)
