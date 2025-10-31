import os
import subprocess
from hls4ml.model.flow import get_flow, register_flow
from hls4ml.backends import VitisBackend, VivadoBackend

class CoyoteAcceleratorBackend(VitisBackend):
    """
    The CoyoteAccelerator backend, which deploys hls4ml models on a PCIe-attached Alveo FPGA
    Underneath it uses the Coyote shell: https://github.com/fpgasystems/Coyote,
    which offers high-performance data movement, networking capabilities, multi-tenancy,
    partial reconfiguration etc. This backend has some similarities with the VitisAccelerator
    backend, but the underlying platforms are different. The implementation of this backend
    remains mostly simple, inheriting most of the functionality from the Vitis backend and
    providing the necessary infrastructure to run model inference on Alveo boards.

    Currently, this backend supports batched inference of a single model on hardware.
    In the future, it can easily be extended with the following capabilities, leveraging
    Coyote's features:
        - Distributed inference 
        - Multiple parallel instances of hls4ml models (same or distinct models)
        - Dynamic, run-time reconfiguration of models

    Generic examples of Coyote can be found at the above-mentioned repository, under examples/
    """

    def __init__(self):
        super(VivadoBackend, self).__init__(name='CoyoteAccelerator')
        self._register_layer_attributes()
        self._register_flows()

    def _register_flows(self):
        writer_passes = ['make_stamp', 'coyoteaccelerator:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['vitis:ip'], backend=self.name)

        ip_flow_requirements = get_flow('vitis:ip').requires.copy()
        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

    def compile(self, model):
        """
        Compiles the hls4ml model for software emulation

        Args:
            model (ModelGraph): hls4ml model to synthesize

        Return:
            lib_name (str): The name of the compiled library
        """
        lib_name = None
        ret_val = subprocess.run(
            ['./build_lib.sh'],
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=model.config.get_output_dir(),
        )
        if ret_val.returncode != 0:
            print(ret_val.stdout)
            raise Exception(f'Failed to compile project "{model.config.get_project_name()}"')
        lib_name = '{}/build/{}-{}.so'.format(
            model.config.get_output_dir(), model.config.get_project_name(), model.config.get_config_value('Stamp')
        )

        return lib_name

    def build(
        self,
        model,
        device: str = 'u55c',
        reset: bool = False,
        csim: bool = True,
        synth: bool = True,
        cosim: bool = False,
        validation: bool = False,
        csynth: bool = False,
        bitfile: bool = False,
        timing_opt: bool = False,
        hls_clock_period: float = 4,
        hls_clock_uncertainty: float = 27
    ):
        """
        Synthesizes the hls4ml model bitstream as part of the Coyote shell
        and compiles the host-side software to control the FPGA and run model inference

        Args:
            model (ModelGraph): hls4ml model to synthesize
            device (str, optional): Target Alveo FPGA card; currently supported u55c, u280 and u250
            reset (bool, optional): Reset HLS project, if a previous one is found
            csim (bool, optional): Run C-Simulation of the HLS project
            synth (bool, optional): Run HLS synthesis
            cosim (bool, optional): Run HLS co-simulation
            validation (bool, optional): Validate results between C-Sim and Co-Sim
            csynth (bool, optional): Run Coyote synthesis using Vivado, which will synthesize the model in a vFPGA
            bitfile (bool, optional): Generate Coyote bitstream
            timing_opt (bool, optional): Run additional optimizations when running PnR during bitstream generation
            hls_clock_period (float, optional): Clock period to be used for HLS synthesis
            hls_clock_uncertainty (float, optional): Clock uncertainty to be used for HLS synthesis

        NOTE: Currently, the hardware will synthesize with a default clock period of 4ns / 250 MHz frequency,
        since this is the default frequency of Coyote (since the XDMA core defaults to 250 MHz). Coyote allows
        one to specify a different clock period for the model and use a clock-domain crossing (CDC) between the 
        XDMA region and the model. This option is currently not exposed as part of the hls4ml backend, but advanced
        users can easily set in the the CMake configuration of Coyote.

        NOTE: While the hardware will synthesize at 250 MHz, users can optionally pass a different HLS clock period
        This is primarily a work-around when HLS synthesize a kernel that doesn't meet timing during PnR.
        The "trick" is to run HLS synthesis at a higher clock frequency then (or provide higher uncertainty)

        TODO: Add functionality to parse synthesis reports
        """
        curr_dir = os.getcwd()

        # Synthesize hardware
        cmake_cmd = (
            f'cmake ../../  '
            f'-DFLOW=hw '
            f'-DFDEV_NAME={device} '
            f'-DBUILD_OPT={int(timing_opt)} '
            f'-DEN_HLS_RESET={int(reset)} '
            f'-DEN_HLS_CSIM={int(csim)} '
            f'-DEN_HLS_SYNTH={int(synth)} '
            f'-DEN_HLS_COSIM={int(cosim)} '
            f'-DEN_HLS_VALIDATION={int(validation)} '
            f'-DHLS_CLOCK_PERIOD={hls_clock_period} '
            f'-DHLS_CLOCK_UNCERTAINTY="{str(hls_clock_uncertainty)}%"'
        )

        if not os.path.exists(f'{model.config.get_output_dir()}/build/{model.config.get_project_name()}_cyt_hw'):
            os.mkdir(f'{model.config.get_output_dir()}/build/{model.config.get_project_name()}_cyt_hw')
        os.chdir(f'{model.config.get_output_dir()}/build/{model.config.get_project_name()}_cyt_hw')
        os.system(cmake_cmd)

        if bitfile:
            os.system('make project && make bitgen')
        elif csynth:
            os.system('make project && make synth')
        else:
            os.system('make project')
            
        os.chdir(curr_dir)
        
        # Compile host software
        cmake_cmd = 'cmake ../../ -DFLOW=sw'
        if not os.path.exists(f'{model.config.get_output_dir()}/build/{model.config.get_project_name()}_cyt_sw'):
            os.mkdir(f'{model.config.get_output_dir()}/build/{model.config.get_project_name()}_cyt_sw')
        os.chdir(f'{model.config.get_output_dir()}/build/{model.config.get_project_name()}_cyt_sw')
        os.system(cmake_cmd)
        os.system('make')
        os.chdir(curr_dir)

