import glob
import os
import stat
from pathlib import Path
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

        filedir = Path(__file__).parent

        # project.tcl
        prj_tcl_dst = Path(f'{model.config.get_output_dir()}/project.tcl')
        with open(prj_tcl_dst, 'w') as f:
            f.write('variable project_name\n')
            f.write(f'set project_name "{model.config.get_project_name()}"\n')
            f.write('variable backend\n')
            f.write('set backend "vivado"\n')
            f.write('variable part\n')
            f.write('set part "{}"\n'.format(model.config.get_config_value('Part')))
            f.write('variable clock_period\n')
            f.write('set clock_period {}\n'.format(model.config.get_config_value('ClockPeriod')))
            f.write('variable clock_uncertainty\n')
            f.write('set clock_uncertainty {}\n'.format(model.config.get_config_value('ClockUncertainty', '0%')))
            f.write('variable version\n')
            f.write('set version "{}"\n'.format(model.config.get_config_value('Version', '1.0.0')))

        # build_prj.tcl
        srcpath = (filedir / '../templates/vivado/build_prj.tcl').resolve()
        dstpath = f'{model.config.get_output_dir()}/build_prj.tcl'
        copyfile(srcpath, dstpath)

        # vivado_synth.tcl
        srcpath = (filedir / '../templates/vivado/vivado_synth.tcl').resolve()
        dstpath = f'{model.config.get_output_dir()}/vivado_synth.tcl'
        copyfile(srcpath, dstpath)

        # build_lib.sh
        build_lib_src = (filedir / '../templates/symbolic/build_lib.sh').resolve()
        build_lib_dst = Path(f'{model.config.get_output_dir()}/build_lib.sh').resolve()
        with open(build_lib_src) as src, open(build_lib_dst, 'w') as dst:
            for line in src.readlines():
                line = line.replace('myproject', model.config.get_project_name())
                line = line.replace('mystamp', model.config.get_config_value('Stamp'))
                line = line.replace('mylibspath', model.config.get_config_value('HLSLibsPath'))

                if 'LDFLAGS=' in line and not os.path.exists(model.config.get_config_value('HLSLibsPath')):
                    line = 'LDFLAGS=\n'

                dst.write(line)
        build_lib_dst.chmod(build_lib_dst.stat().st_mode | stat.S_IEXEC)

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
