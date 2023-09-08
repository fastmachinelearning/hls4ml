import glob
import os
from shutil import copyfile, copytree, rmtree

from hls4ml.backends import get_backend
from hls4ml.writer.vivado_writer import VivadoWriter


class SymbolicExpressionWriter(VivadoWriter):
    def write_nnet_utils(self, model):
        """Copy the nnet_utils, AP types headers and any custom source to the project output directory

        Args:
            model (ModelGraph): the hls4ml model.
        """

        # nnet_utils
        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/vivado/nnet_utils/')
        dstpath = f'{model.config.get_output_dir()}/firmware/nnet_utils/'

        if not os.path.exists(dstpath):
            os.mkdir(dstpath)

        headers = [os.path.basename(h) for h in glob.glob(srcpath + '*.h')]

        for h in headers:
            copyfile(srcpath + h, dstpath + h)

        # ap_types
        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = model.config.get_config_value('HLSIncludePath')
        if not os.path.exists(srcpath):
            srcpath = os.path.join(filedir, '../templates/vivado/ap_types/')
        dstpath = f'{model.config.get_output_dir()}/firmware/ap_types/'

        if os.path.exists(dstpath):
            rmtree(dstpath)

        copytree(srcpath, dstpath)

        # custom source
        filedir = os.path.dirname(os.path.abspath(__file__))

        custom_source = get_backend('Vivado').get_custom_source()
        for dst, srcpath in custom_source.items():
            dstpath = f'{model.config.get_output_dir()}/firmware/{dst}'
            copyfile(srcpath, dstpath)

    def write_build_script(self, model):
        """Write the TCL/Shell build scripts (project.tcl, build_prj.tcl, vivado_synth.tcl, build_lib.sh)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))

        # build_prj.tcl
        f = open(f'{model.config.get_output_dir()}/project.tcl', 'w')
        f.write('variable project_name\n')
        f.write(f'set project_name "{model.config.get_project_name()}"\n')
        f.write('variable backend\n')
        f.write('set backend "vivado"\n')
        f.write('variable part\n')
        f.write('set part "{}"\n'.format(model.config.get_config_value('Part')))
        f.write('variable clock_period\n')
        f.write('set clock_period {}\n'.format(model.config.get_config_value('ClockPeriod')))
        f.close()

        # build_prj.tcl
        srcpath = os.path.join(filedir, '../templates/vivado/build_prj.tcl')
        dstpath = f'{model.config.get_output_dir()}/build_prj.tcl'
        copyfile(srcpath, dstpath)

        # vivado_synth.tcl
        srcpath = os.path.join(filedir, '../templates/vivado/vivado_synth.tcl')
        dstpath = f'{model.config.get_output_dir()}/vivado_synth.tcl'
        copyfile(srcpath, dstpath)

        # build_lib.sh
        f = open(os.path.join(filedir, '../templates/symbolic/build_lib.sh'))
        fout = open(f'{model.config.get_output_dir()}/build_lib.sh', 'w')

        for line in f.readlines():
            line = line.replace('myproject', model.config.get_project_name())
            line = line.replace('mystamp', model.config.get_config_value('Stamp'))
            line = line.replace('mylibspath', model.config.get_config_value('HLSLibsPath'))

            if 'LDFLAGS=' in line and not os.path.exists(model.config.get_config_value('HLSLibsPath')):
                line = 'LDFLAGS=\n'

            fout.write(line)
        f.close()
        fout.close()

    def write_hls(self, model):
        print('Writing HLS project')
        self.write_project_dir(model)
        self.write_project_cpp(model)
        self.write_project_header(model)
        # self.write_weights(model) # No weights to write
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
