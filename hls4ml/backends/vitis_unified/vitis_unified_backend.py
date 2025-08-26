import os
import sys
import subprocess
from shutil import copy2


from hls4ml.backends import VitisBackend, VivadoBackend
from hls4ml.model.flow import register_flow
from hls4ml.report import parse_vivado_report

from hls4ml.writer.vitis_unified_writer.meta_gen import VitisUnified_MetaGen  as mg


class VitisUnifiedBackend(VitisBackend):
    def __init__(self):
        super(VivadoBackend, self).__init__(name='VitisUnified')
        self._register_layer_attributes()
        self._register_flows()


    def run_term_command(self, model, taskName: str, command: str, logStdOut: bool, cwd):

        print("-------------------------------------------------------")
        print(f"start running task : {taskName}")
        print(f"    with command: {command}")
        print("-------------------------------------------------------")

        output_dir = model.config.get_output_dir()

        out_log_path = os.path.join(output_dir, f'{taskName}_out.log')
        err_log_path = os.path.join(output_dir, f'{taskName}_err.log')
        out_target   = None if logStdOut else open(out_log_path, 'w')
        err_target   = None if logStdOut else open(err_log_path, 'w')

        try:
            runningProcess = subprocess.Popen(
                command, shell=True, cwd=cwd, stdout=out_target, stderr=err_target, text=True
            )
            runningProcess.communicate()
            if runningProcess.returncode != 0:
                raise Exception(f'Package failed for {taskName} for project {model.config.get_project_name()}. See logs for details.')

            stdout, stderr = runningProcess.communicate()
            print(f"stdout: {stdout}")
            print(f"stderr: {stderr}")

            print(f"task {taskName} finished")

        except Exception as e:
            print(f"task {taskName} failed")
            print(e)
            raise e
        finally:
            if out_target:
                out_target.close()
            if err_target:
                err_target.close()



    def build(
        self,
        model,
        reset=False,
        csim=False,
        synth=False,
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

        hls_config_file = os.path.join(output_dir, "hls_kernel_config.cfg")
        ##### build command
        csynth_cmd = (
            "v++ -c --mode hls --config {configPath} --work_dir unifiedPrj"
        ).format(configPath=hls_config_file)
        csynth_cwd = mg.get_vitis_hls_dir(model)

        ##### util template (used in csim/cosim/package)
        util_command = "vitis-run --mode hls --{op} --config {configPath} --work_dir unifiedPrj"
        ##### package command

        package_cmd = util_command.format(op="package", configPath=hls_config_file)
        package_cwd = mg.get_vitis_hls_dir(model)
        cosim_cmd   = util_command.format(op="cosim"  , configPath=hls_config_file)
        cosim_cwd   = mg.get_vitis_hls_dir(model)
        csim_cmd    = util_command.format(op="csim"   , configPath=hls_config_file)
        csim_cwd = mg.get_vitis_hls_dir(model)

        kerlink_cmd = "./buildAcc.sh"
        kerlink_cwd = mg.get_vitis_linker_dir(model)

        if synth:
            self.run_term_command(model, "csynth", csynth_cmd, log_to_stdout, csynth_cwd)
            self.run_term_command(model, "package", package_cmd, log_to_stdout, package_cwd)

        if csim:
            self.run_term_command(model, "csim", csim_cmd, log_to_stdout, csim_cwd)

        if cosim:
            self.run_term_command(model, "cosim", cosim_cmd, log_to_stdout, cosim_cwd)

        ##if bitfile
        if bitfile:
            self.run_term_command(model, "kerlink", kerlink_cmd, log_to_stdout, kerlink_cwd)




    def create_initial_config(
        self,
        board               ='zcu102',
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
        **_
    ):

        config = super().create_initial_config(part, clock_period, clock_uncertainty, io_type)

        config['UnifiedConfig'] = {}
        config['UnifiedConfig']['bufInSize'  ]  = gmemBuf_in_size
        config['UnifiedConfig']['bufOutSize' ]  = gmemBuf_out_size
        config['UnifiedConfig']['XPFMPath'   ]  = xpfmPath
        config['UnifiedConfig']['Board'      ]  = board
        config['UnifiedConfig']['Driver'     ]  = driver
        config['UnifiedConfig']['InputDtype' ]  = input_type  # float, double or ap_fixed<a,b>
        config['UnifiedConfig']['OutputDtype']  = output_type  # float, double or ap_fixed<a,b>

        if input_type not in ["double", "float"]:
            raise Exception("input_type must be float or double")
        if output_type not in ["double", "float"]:
            raise Exception("output_type must be float or double")

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

