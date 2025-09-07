import os
import stat
from pathlib import Path

from .meta import VitisUnifiedWriterMeta


class VitisUnified_BuildGen:

    @classmethod
    def write_bridge_build_script(self, meta: VitisUnifiedWriterMeta, model, mg):
        filedir = os.path.dirname(os.path.abspath(__file__))
        fin = open(os.path.join(filedir, '../../templates/vitis_unified/build_lib.sh'))
        fout = open(f"{model.config.get_output_dir()}/build_lib.sh", 'w')

        for line in fin.readlines():
            if 'myprojectBaseName' in line:
                line = line.replace('myprojectBaseName', format(model.config.get_project_name()))
            if 'myprojectWrapName' in line:
                line = line.replace('myprojectWrapName', mg.get_wrapper_file_name(model))
            if 'mystamp' in line:
                line = line.replace('mystamp', model.config.get_config_value('Stamp'))

            fout.write(line)

        fin.close()
        fout.close()

        # change permission
        build_lib_dst = Path(f'{model.config.get_output_dir()}/build_lib.sh').resolve()
        build_lib_dst.chmod(build_lib_dst.stat().st_mode | stat.S_IEXEC)

    @classmethod
    def write_hls_kernel_cfg(self, meta, model, mg, is_csim=False):  # True is_csim else is cosim+fifo_optimization
        # This will gen hls_kernel_config_<csim/cosim>.cfg file which Vitis_hls unified will use it to config
        # the synthesizer
        filedir = os.path.dirname(os.path.abspath(__file__))
        sufix = "csim" if is_csim else "cosim"
        fin = open(os.path.join(filedir, '../../templates/vitis_unified/hls_kernel_config.cfg'))
        fout = open(f"{model.config.get_output_dir()}/hls_kernel_config_{sufix}.cfg", 'w')

        for line in fin.readlines():
            if "{PART}" in line:
                line = line.replace("{PART}", model.config.get_config_value('Part'))
            if "{CLK}" in line:
                line = line.replace("{CLK}", model.config.get_config_value('ClockPeriod'))
            if "{CLK_UC}" in line:
                line = line.replace("{CLK_UC}", model.config.get_config_value('ClockUncertainty'))
            if "{OUTDIR}" in line:
                line = line.replace("{OUTDIR}", model.config.get_output_dir())
            if "{TOP_NAME}" in line:
                line = line.replace("{TOP_NAME}", mg.get_top_wrap_func_name(model))
            if "{FILE_NAME_WRAP}" in line:
                line = line.replace("{FILE_NAME_WRAP}", mg.get_wrapper_file_name(model))
            if "{SIM_FILE_NAME}" in line:
                line = line.replace("{SIM_FILE_NAME}", mg.get_sim_file_name())
            if "{FILE_NAME_BASE}" in line:
                line = line.replace("{FILE_NAME_BASE}", mg.get_main_file_name(model))
            if "{OUTPUT_KERNEL_TYPE}" in line:
                line = line.replace("{OUTPUT_KERNEL_TYPE}", mg.get_output_kernel_type())
            if is_csim and (("enable_fifo_sizing" in line) or ("-DRTL_SIM" in line)):
                line = "#" + line

            fout.write(line)

        fin.close()
        fout.close()

    @classmethod
    def build_unified_project_ske(self, meta, model, mg, workspaceDir=None):
        # this will generate the vitis-comp.json file, the file will enable vitis ide gui to see it
        # as a project
        if workspaceDir is None:
            workspaceDir = mg.get_vitis_unified_working_directory_dir(model)
        hlsDir = mg.get_vitis_hls_dir(model)
        execDir = mg.get_vitis_hls_dir(model)
        vitisComp = os.path.join(str(hlsDir), "vitis-comp.json")

        # create my own project for this graph
        os.makedirs(workspaceDir, exist_ok=True)
        os.makedirs(hlsDir, exist_ok=True)
        os.makedirs(execDir, exist_ok=True)
        # create project vitis-comp.json to
        filedir = os.path.dirname(os.path.abspath(__file__))
        fin = open(os.path.join(filedir, "../../templates/vitis_unified/workspace/projectName/vitis-comp.json"))
        fout = open(vitisComp, 'w')

        for line in fin.readlines():
            if "{HLS_NAME}" in line:
                line = line.replace("{HLS_NAME}", model.config.get_project_name())
            if "{CONFIG_FILE}" in line:
                line = line.replace("{CONFIG_FILE}", f"{model.config.get_output_dir()}/hls_kernel_config.cfg")
            fout.write(line)

        fin.close()
        fout.close()

    @classmethod
    def write_launch_vitis_linker_dir(self, meta, model, mg):
        os.makedirs(mg.get_vitis_linker_dir(model), exist_ok=True)

    @classmethod
    def write_launch_vitis_linker_launcher(self, meta, model, mg):
        # This section generate buildAcc.sh file to combine the platform and the hls kernel together
        filedir = os.path.dirname(os.path.abspath(__file__))
        fin = open(os.path.join(filedir, '../../templates/vitis_unified/workspace/sysProj/buildAcc.sh'))
        fout = open(f"{mg.get_vitis_linker_dir(model)}/buildAcc.sh", 'w')

        for line in fin.readlines():
            if "{PLATFORM_XPFM}" in line:
                line = line.replace("{PLATFORM_XPFM}", meta.vitis_unified_config.get_XPFMPath())
            if "{KERNEL_XO}" in line:
                line = line.replace("{KERNEL_XO}", mg.get_xo_file_path(model))
            if "{PROJECT_NAME}" in line:
                line = line.replace("{PROJECT_NAME}", model.config.get_project_name())

            fout.write(line)

        fin.close()
        fout.close()

        link_lib_dst = Path(f"{mg.get_vitis_linker_dir(model)}/buildAcc.sh").resolve()
        link_lib_dst.chmod(link_lib_dst.stat().st_mode | stat.S_IEXEC)

    @classmethod
    def write_launch_vitis_linker_cfg(self, meta, model, mg):
        # this will generate the config file that linker (platform + vitis)
        filedir = os.path.dirname(os.path.abspath(__file__))
        fin = open(os.path.join(filedir, '../../templates/vitis_unified/workspace/sysProj/buildConfig.cfg'))
        fout = open(f"{mg.get_vitis_linker_dir(model)}/buildConfig.cfg", 'w')

        for line in fin.readlines():
            if "{CLK}" in line:
                line = line.replace("{CLK}", str(100_000_000))  # model.config.get_config_value('ClockPeriod'))
            if "{KERNEL_NAME}" in line:
                line = line.replace("{KERNEL_NAME}", mg.get_top_wrap_func_name(model))
            if "{GUI_STATUS}" in line:
                line = line.replace("{GUI_STATUS}", "true")
            line = ""
            fout.write(line)

        fin.close()
        fout.close()
