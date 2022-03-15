import os
import shutil
import tarfile
import time
import numpy as np

from hls4ml.backends import VivadoBackend
from hls4ml.model.flow import register_flow
from hls4ml.report import parse_vivado_report

class VivadoAcceleratorBackend(VivadoBackend):
    def __init__(self):
        super(VivadoBackend, self).__init__(name='VivadoAccelerator')
        self._register_flows()

    @staticmethod
    def package(model, X=None, y=None, sleep_before_retry=60):
        '''Package the hardware build results for HW inference, including test data'''

        odir = model.config.get_output_dir()
        name = model.config.get_project_name()

        if os.path.isdir(f'{odir}/package/'):
            print(f'Found existing package "{odir}/package/", overwriting')
        os.makedirs(f'{odir}/package/', exist_ok=True)
        if not X is None:
            np.save(f'{odir}/package/X.npy', X)
        if not y is None:
            np.save(f'{odir}/package/y.npy', y)

        src = f'{odir}/{name}_vivado_accelerator/project_1.runs/impl_1/design_1_wrapper.bit'
        dst = f'{odir}/package/{name}.bit'
        _copy_wait_retry(src, dst, sleep=sleep_before_retry)

        src = f'{odir}/{name}_vivado_accelerator/project_1.srcs/sources_1/bd/design_1/hw_handoff/design_1.hwh'
        dst = f'{odir}/package/{name}.hwh'
        _copy_wait_retry(src, dst, sleep=sleep_before_retry)

        driver = model.config.backend.writer.vivado_accelerator_config.get_driver_file()
        shutil.copy(f'{odir}/{driver}', f'{odir}/package/{driver}')

        _make_tarfile(f'{odir}/{name}.tar.gz', f'{odir}/package')

    def build(self, model, reset=False, csim=True, synth=True, cosim=False, validation=False, export=False, vsynth=False, bitfile=False):
        # run the VivadoBackend build
        report = super().build(model, reset=reset, csim=csim, synth=synth, cosim=cosim, validation=validation, export=export, vsynth=vsynth)
        # now make a bitfile
        if bitfile:
            curr_dir = os.getcwd()
            os.chdir(model.config.get_output_dir())
            success = False
            try:
                os.system('vivado -mode batch -source design.tcl')
                success = True
            except:
                print("Something went wrong, check the Vivado logs")
            os.chdir(curr_dir)
            if success:
                VivadoAcceleratorBackend.package(model)

        return parse_vivado_report(model.config.get_output_dir())

    def create_initial_config(self, board='pynq-z2', part=None, clock_period=5, io_type='io_parallel', interface='axi_stream',
                              driver='python', input_type='float', output_type='float'):
        '''
        Create initial accelerator config with default parameters
        Args:
            board: one of the keys defined in supported_boards.json
            clock_period: clock period passed to hls project
            io_type: io_parallel or io_stream
            interface: `axi_stream`: generate hardware designs and drivers which exploit axi stream channels.
                       `axi_master`: generate hardware designs and drivers which exploit axi master channels.
                       `axi_lite` : generate hardware designs and drivers which exploit axi lite channels. (Don't use it
                       to exchange large amount of data)
            driver: `python`: generates the python driver to use the accelerator in the PYNQ stack.
                    `c`: generates the c driver to use the accelerator bare-metal.
            input_type: the wrapper input precision. Can be `float` or an `ap_type`. Note: VivadoAcceleratorBackend
                             will round the number of bits used to the next power-of-2 value.
            output_type: the wrapper output precision. Can be `float` or an `ap_type`. Note:
                              VivadoAcceleratorBackend will round the number of bits used to the next power-of-2 value.

        Returns:
            populated config
        '''
        board = board if board is not None else 'pynq-z2'
        config = super(VivadoAcceleratorBackend, self).create_initial_config(part, clock_period, io_type)
        config['AcceleratorConfig'] = {}
        config['AcceleratorConfig']['Board'] = board
        config['AcceleratorConfig']['Interface'] = interface  # axi_stream, axi_master, axi_lite
        config['AcceleratorConfig']['Driver'] = driver
        config['AcceleratorConfig']['Precision'] = {}
        config['AcceleratorConfig']['Precision']['Input'] = {}
        config['AcceleratorConfig']['Precision']['Output'] = {}
        config['AcceleratorConfig']['Precision']['Input'] = input_type  # float, double or ap_fixed<a,b>
        config['AcceleratorConfig']['Precision']['Output'] = output_type  # float, double or ap_fixed<a,b>
        return config

    def _register_flows(self):
        vivado_writer = ['vivado:write']
        vivado_accel_writer = ['vivadoaccelerator:write_hls']
        self._writer_flow = register_flow('write', vivado_accel_writer, requires=vivado_writer, backend=self.name)
        self._default_flow = 'vivado:ip'

def _make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def _copy_wait_retry(src, dst, sleep=60):
    if not os.path.isfile(src):
        print(f'File {src} not found, waiting {sleep}s before retry')
        time.sleep(sleep)
    shutil.copy(src, dst)