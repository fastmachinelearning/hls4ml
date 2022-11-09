from shutil import copyfile, copytree, rmtree
import os
import glob

from hls4ml.writer.vivado_writer import VivadoWriter
from hls4ml.backends import get_backend


class SymbolicExpressionWriter(VivadoWriter):

    def write_nnet_utils(self, model):
        ###################
        ## nnet_utils
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir,'../templates/vivado/nnet_utils/')
        dstpath = '{}/firmware/nnet_utils/'.format(model.config.get_output_dir())

        if not os.path.exists(dstpath):
            os.mkdir(dstpath)

        headers = [os.path.basename(h) for h in glob.glob(srcpath + '*.h')]

        for h in headers:
            copyfile(srcpath + h, dstpath + h)

        ###################
        ## HLS includes
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = model.config.get_config_value('HLSIncludePath')
        dstpath = '{}/firmware/ap_types/'.format(model.config.get_output_dir())

        if os.path.exists(dstpath):
            rmtree(dstpath)

        copytree(srcpath, dstpath)

        ###################
        ## custom source
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))

        custom_source = get_backend('Vivado').get_custom_source()
        for dst, srcpath in custom_source.items():
            dstpath = '{}/firmware/{}'.format(model.config.get_output_dir(), dst)
            copyfile(srcpath, dstpath)

    def write_build_script(self, model):

        filedir = os.path.dirname(os.path.abspath(__file__))

        ###################
        # project.tcl
        ###################
        f = open('{}/project.tcl'.format(model.config.get_output_dir()), 'w')
        f.write('variable project_name\n')
        f.write('set project_name "{}"\n'.format(model.config.get_project_name()))
        f.write('variable backend\n')
        f.write('set backend "vivado"\n')
        f.write('variable part\n')
        f.write('set part "{}"\n'.format(model.config.get_config_value('Part')))
        f.write('variable clock_period\n')
        f.write('set clock_period {}\n'.format(model.config.get_config_value('ClockPeriod')))
        f.close()

        ###################
        # build_prj.tcl
        ###################

        srcpath = os.path.join(filedir,'../templates/vivado/build_prj.tcl')
        dstpath = '{}/build_prj.tcl'.format(model.config.get_output_dir())
        copyfile(srcpath, dstpath)

        ###################
        # vivado_synth.tcl
        ###################

        srcpath = os.path.join(filedir,'../templates/vivado/vivado_synth.tcl')
        dstpath = '{}/vivado_synth.tcl'.format(model.config.get_output_dir())
        copyfile(srcpath, dstpath)

        ###################
        # build_lib.sh
        ###################

        f = open(os.path.join(filedir,'../templates/symbolic/build_lib.sh'),'r')
        fout = open('{}/build_lib.sh'.format(model.config.get_output_dir()),'w')

        for line in f.readlines():
            line = line.replace('myproject', model.config.get_project_name())
            line = line.replace('mystamp', model.config.get_config_value('Stamp'))
            line = line.replace('mylibspath', model.config.get_config_value('HLSLibsPath'))

            fout.write(line)
        f.close()
        fout.close()

    def write_hls(self, model):
        print('Writing HLS project')
        self.write_project_dir(model)
        self.write_project_cpp(model)
        self.write_project_header(model)
        #self.write_weights(model) # No weights to write
        self.write_defines(model)
        self.write_parameters(model)
        self.write_test_bench(model)
        self.write_bridge(model)
        self.write_build_script(model)
        self.write_nnet_utils(model)
        self.write_generated_code(model)
        self.write_yml(model)
        self.write_tar(model)
        print('Done')
