import os
import sys
import subprocess


from hls4ml.backends import VitisBackend, VivadoBackend
from hls4ml.model.flow import register_flow
from hls4ml.report import parse_vivado_report

from hls4ml.writer.vitis_unified_writer import meta_gen as mg


class VitisUnifiedBackend(VitisBackend):
    def __init__(self):
        super(VivadoBackend, self).__init__(name='VitisUnified')
        self._register_layer_attributes()
        self._register_flows()


    def run_term_command(self, model, taskName: str,command: str, logOutput: bool, cwd):


        output_dir = model.config.get_output_dir()

        out_log_path = os.path.join(output_dir, f'{taskName}_out.log')
        err_log_path = os.path.join(output_dir, f'{taskName}_err.log')
        out_target   = None if logOutput else open(out_log_path, 'w')
        err_target   = None if logOutput else open(err_log_path, 'w')

        runningProcess = subprocess.Popen(
            command, shell=True, cwd=cwd, stdout=out_target, stderr=err_target, text=True
        )
        runningProcess.communicate()
        if runningProcess.returncode != 0:
            raise Exception(f'Package failed for {taskName} for project {model.config.get_project_name()}. See logs for details.')

        stdout, stderr = runningProcess.communicate()
        return stdout, stderr



    def build(
        self,
        model,
        reset=False,
        csim=False,
        synth=True,
        cosim=False,
        validation=False,
        export=False,
        vsynth=False,
        fifo_opt=False,
        bitfile=False,
        log_to_stdout=True
    ):
        ##### it builds and return vivado reports
        if 'linux' in sys.platform:
            found = os.system('command -v vitis > /dev/null')
            if found != 0:
                raise Exception('Vitis installation not found. Make sure "vitis" is on PATH.')

        ##### TODO support this system
        if csim:
            raise Exception("Current Vitis Unified not support csim. Please set csim=False to run Vitis Unified.")
        if validation:
            raise Exception("Current Vitis Unified not support validation. Please set validation=False to run Vitis Unified.")
        if export:
            raise Exception("Current Vitis Unified not support export. Please set export=False to run Vitis Unified.")
        if fifo_opt:
            raise Exception("Current Vitis Unified not support fifo_opt. Please set fifo_opt=False to run Vitis Unified.")

        output_dir = model.config.get_output_dir()

        ##### build command
        csynth_cmd = (
            "v++ -c --mode hls --config {configPath} --work_dir unifiedPrj"
        ).format(configPath=(os.path.join(output_dir, "hls_kernel_config.cfg")))
        csynth_cwd = mg.getVitisHlsDir(model)

        ##### util template (used in csim/cosim/package)
        util_command = "vitis-run --mode hls --{op} --config {configPath} --work_dir unifiedPrj"


        ##### package command

        package_command = util_command.format(op="package", configPath=os.path.join(output_dir, "hls_config.cfg"))
        cosim_command = util_command.format(op="cosim", configPath=os.path.join(output_dir, "hls_config_cosim.cfg"))
        csim_command = util_command.format(op="csim", configPath=os.path.join(output_dir, "hls_config_csim.cfg"))

        ##### co-sim command



        ##### c-sim command

        # cosim_command += ' --flag "RTL_SIM"'
        cs_out_log_path = os.path.join(output_dir, 'csim_out.log')
        cs_err_log_path = os.path.join(output_dir, 'csim_err.log')
        cs_out_target = None if log_to_stdout or (not csim) else open(cs_out_log_path, 'w')
        cs_err_target = None if log_to_stdout or (not csim) else open(cs_err_log_path, 'w')

        try:
            if synth:
                print("---------------------------------------------------")
                print("-----------   start build command   ---------------")
                print("---------------------------------------------------")
                ##### run build project
                process = subprocess.Popen(
                    build_command, shell=True, cwd=output_dir, stdout=stdout_target, stderr=stderr_target, text=True
                )
                process.communicate()
                if process.returncode != 0:
                    raise Exception(f'Build failed for {model.config.get_project_name()}. See logs for details.')
                ##### run package project
                print("---------------------------------------------------")
                print("-----------   start package command ---------------")
                print("---------------------------------------------------")

                packageProcess = subprocess.Popen(
                    package_command, shell=True, cwd=output_dir, stdout=pk_out_target, stderr=pk_err_target, text=True
                )
                packageProcess.communicate()
                if packageProcess.returncode != 0:
                    raise Exception(f'Package failed for {model.config.get_project_name()}. See logs for details.')

            if cosim:
                print("---------------------------------------------------")
                print("-----------   start cosim command ---------------")
                print("---------------------------------------------------")
                cosimProcess = subprocess.Popen(
                    cosim_command, shell=True, cwd=output_dir, stdout=cos_out_target, stderr=cos_err_target, text=True
                )
                cosimProcess.communicate()
                if cosimProcess.returncode != 0:
                    raise Exception(f'Cosim failed for {model.config.get_project_name()}. See logs for details.')

            if csim:
                print("---------------------------------------------------")
                print("-----------   start csim command ---------------")
                print("---------------------------------------------------")
                csimProcess = subprocess.Popen(
                    csim_command, shell=True, cwd=output_dir, stdout=cs_out_target, stderr=cs_err_target, text=True
                )
                csimProcess.communicate()
                if csimProcess.returncode != 0:
                    raise Exception(f'Csim failed for {model.config.get_project_name()}. See logs for details.')




        finally:
            if not log_to_stdout:
                if synth:
                    stdout_target.close()
                    stderr_target.close()
                    pk_out_target.close()
                    pk_err_target.close()

                if cosim:
                    cos_out_target.close()
                    cos_err_target.close()

                if csim:
                    cs_out_target.close()
                    cs_err_target.close()

        # now make a bitfile
        if bitfile:
            curr_dir = os.getcwd()
            os.chdir(model.config.get_output_dir())
            try:
                os.system('vivado -mode batch -source design.tcl')  # check if this is accepted as a command
            except Exception:
                print("Something went wrong, check the Vivado logs")
            os.chdir(curr_dir)

        return parse_vivado_report(model.config.get_output_dir())

    def create_initial_config(
        self,
        board               ='pynq-z2',
        part                =None,
        clock_period        =5,
        clock_uncertainty   ='12.5%',
        io_type             ='io_parallel',
        interface           ='axi_stream',
        driver              ='python',
        input_type          ='float',
        output_type         ='float',
        gmemBuf_in_size     =12,
        gmemBuf_out_size    =12,
        xpfmPath            ='/tools/Xilinx/Vitis/2023.2/base_platforms/'
                             'xilinx_zcu102_base_202320_1/xilinx_zcu102_base_202320_1.xpfm',
        input_interim_type  ='io_stream',    #### it should be io_stream or io_free_stream/ io_stream
        output_interim_type ='io_stream'
    ):
        board = board if board is not None else 'pynq-z2'

        if input_interim_type not in ['io_free_stream', 'io_stream']:
            raise Exception(f'input_interim_type should be io_free_stream or io_stream, but got {input_interim_type}')
        if output_interim_type not in ['io_free_stream', 'io_stream']:
            raise Exception(f'output_interim_type should be io_free_stream or io_stream, but got {output_interim_type}')

        config = super().create_initial_config(part, clock_period, clock_uncertainty, io_type)
        config['AcceleratorConfig'] = {}
        config['AcceleratorConfig']['Board'] = board
        config['AcceleratorConfig']['Interface'] = interface  # axi_stream, axi_master, axi_lite
        config['AcceleratorConfig']['Driver'] = driver
        config['AcceleratorConfig']['Precision'] = {}
        config['AcceleratorConfig']['Precision']['Input'] = {}
        config['AcceleratorConfig']['Precision']['Output'] = {}
        config['AcceleratorConfig']['Precision']['Input'] = input_type  # float, double or ap_fixed<a,b>
        config['AcceleratorConfig']['Precision']['Output'] = output_type  # float, double or ap_fixed<a,b>

        config['UnifiedConfig'] = {}
        config['UnifiedConfig']['bufInSize']    = gmemBuf_in_size
        config['UnifiedConfig']['bufOutSize']   = gmemBuf_out_size
        config['UnifiedConfig']['XPFMPath']     = xpfmPath

        config['MultiGraphConfig'] = {}
        config['MultiGraphConfig']['IOInterimType'] = {}
        config['MultiGraphConfig']['IOInterimType']['Input'] = input_interim_type
        config['MultiGraphConfig']['IOInterimType']['Output'] = output_interim_type

        return config

    def get_default_flow(self):
        return self._default_flow

    def get_writer_flow(self):
        return self._writer_flow


    def _register_flows(self):
        vitis_ip = 'vitis:ip'
        writer_passes = ['make_stamp', 'vitisunified:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['vitis:ip'], backend=self.name)
        self._default_flow = vitis_ip

