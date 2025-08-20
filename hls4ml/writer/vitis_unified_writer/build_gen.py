import os
import stat
from pathlib import Path


from .meta import VitisUnifiedWriterMeta

class VitisUnified_BuildGen:

    def write_bridge_build_script(self, meta: VitisUnifiedWriterMeta, model, mg):
        filedir = os.path.dirname(os.path.abspath(__file__))
        fin = open(os.path.join(filedir, '../../templates/vitis_unified/build_lib.sh'))
        fout = open(f"{model.config.get_output_dir()}/build_lib.sh", 'w')

        for line in fin.readlines():
            if 'myproject' in line:
                line = line.replace('myproject', format(model.config.get_project_name()))
            if 'mystamp' in line:
                line = line.replace('mystamp', model.config.get_config_value('Stamp'))

            fout.write(line)

        fin.close()
        fout.close()

        #### change permission
        build_lib_dst = Path(f'{model.config.get_output_dir()}/build_lib.sh').resolve()
        build_lib_dst.chmod(build_lib_dst.stat().st_mode | stat.S_IEXEC)

    def write_hls_kernel_cfg(self, meta, model, mg):
        filedir = os.path.dirname(os.path.abspath(__file__))
        fin     = open(os.path.join(filedir, '../../templates/vitis_unified/hls_kernel_config.cfg'), 'r')
        fout    = open(f"{model.config.get_output_dir()}/hls_kernel_config.cfg", 'w')

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
            if "{FILE_NAME_DM}" in line:
                line = line.replace("{FILE_NAME_DM}", mg.get_wrapper_file_name(model))
            if "{FILE_NAME_AXIS}" in line:
                line = line.replace("{FILE_NAME_AXIS}", mg.get_wrapper_file_name(model))
            if "{FILE_NAME_BASE}" in line:
                line = line.replace("{FILE_NAME_BASE}", mg.get_main_file_name(model))


            fout.write(line)

        fin.close()
        fout.close()

    def build_unified_project_ske(self, meta, model, mg, workspaceDir = None):
        if workspaceDir is None:
            workspaceDir = mg.get_vitis_unified_working_directory_dir(model)
        hlsDir    = mg.get_vitis_hls_dir(model)
        execDir   = mg.get_vitis_hls_dir(model)
        vitisComp = os.path.join(str(hlsDir), "vitis-comp.json")

        ###### create my own project for this graph
        os.makedirs(workspaceDir, exist_ok=True)
        os.makedirs(hlsDir      , exist_ok=True)
        os.makedirs(execDir      , exist_ok=True)
        ###### create project vitis-comp.json to
        filedir = os.path.dirname(os.path.abspath(__file__))
        fin = open(os.path.join(filedir, "../../templates/vitis_unified/workspace/projectName/vitis-comp.json"), 'r')
        fout = open(vitisComp, 'w')

        for line in fin.readlines():
            if "{HLS_NAME}" in line:
                line = line.replace("{HLS_NAME}", model.config.get_project_name())
            if "{CONFIG_FILE}" in line:
                line = line.replace("{CONFIG_FILE}", f"{model.config.get_output_dir()}/hls_kernel_config.cfg")
            fout.write(line)

        fin.close()
        fout.close()

    def write_launch_vitis_linker_dir(self, meta, model, mg):
        os.makedirs(mg.get_vitis_linker_dir(model), exist_ok=True)

    def write_launch_vitis_linker_launcher(self, meta, model, mg):
        filedir = os.path.dirname(os.path.abspath(__file__))
        fin = open(os.path.join(filedir, '../../templates/vitis_unified/workspace/sysProj/buildAcc.sh'), 'r')
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



    def write_launch_vitis_linker_cfg(self, meta, model, mg):
        filedir = os.path.dirname(os.path.abspath(__file__))
        fin = open(os.path.join(filedir, '../../templates/vitis_unified/workspace/sysProj/buildConfig.cfg'), 'r')
        fout = open(f"{mg.get_vitis_linker_dir(model)}/buildConfig.cfg", 'w')

        for line in fin.readlines():
            if "{CLK}" in line:
                line = line.replace("{CLK}", str(100_000_000))#model.config.get_config_value('ClockPeriod'))
            if "{KERNEL_NAME}" in line:
                line = line.replace("{KERNEL_NAME}", mg.get_top_wrap_func_name(model))
            if "{GUI_STATUS}" in line:
                line = line.replace("{GUI_STATUS}", "true")
            line=""
            fout.write(line)

        fin.close()
        fout.close()