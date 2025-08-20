import os

from hls4ml.writer.vitis_unified_writer.meta import VitisUnifiedWriterMeta
from hls4ml.writer.vitis_unified_writer.test_bridge_gen import VitisUnified_BridgeGen

class VitisUnifiedPartial_BridgeGen(VitisUnified_BridgeGen):

    @classmethod
    def write_bridge(self, meta: VitisUnifiedWriterMeta, model, mg):

        filedir = os.path.dirname(os.path.abspath(__file__))
        #### we will use the same bridge template file as VitisUnified_BridgeGen
        fin = open(os.path.join(filedir, '../../templates/vitis_unified/myproject_bridge.cpp'))
        fout = open(f"{model.config.get_output_dir()}/{model.config.get_project_name()}_bridge.cpp", 'w')

        indent = '    '


        for line in fin.readlines():

            #### TODO we will do the code next time
            newline = line

            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))

            elif 'myproject' in line:
                newline = line.replace('myproject', format(model.config.get_project_name()))

            elif 'PROJECT_FILE_NAME' in line:
                newline = line.replace('PROJECT_FILE_NAME', format(mg.get_wrapper_file_name(model)))

            fout.write(newline)


        fin.close()
        fout.close()