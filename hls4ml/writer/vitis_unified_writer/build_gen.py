import os

from . import meta_gen as mg

def write_driver(meta, model):
    print("[partial reconfig] we are not supporting write_driver this yet")

def write_hls_kernel_cfg(meta, model):
    filedir = os.path.dirname(os.path.abspath(__file__))
    fin     = open(os.path.join(filedir, '../../templates/vitis_unified/hls_config.h'), 'r')
    fout    = open(f"{model.config.get_output_dir()}/firmware/hls_kernel_config.h", 'w')

    for line in fin.readlines():
        if "{PART}" in line:
            line = line.replace("{PART}", model.config.get_config_value('Part'))
        if "{CLK}" in line:
            line = line.replace("{CLK}", model.config.get_config_value('ClockPeriod'))
        if "{CLK_UC}" in line:
            line = line.replace("{UNCERT}", model.config.get_config_value('ClockUncertainty'))
        if "{OUTDIR}" in line:
            line = line.replace("OUTDIR", model.config.get_output_dir())
        if "{TOP_NAME}" in line:
            line = line.replace("TOP_NAME", mg.getGemTopFuncName(model))
        if "{FILE_NAME_DM}" in line:
            line = line.replace("FILE_NAME_DM", mg.getGmemWrapperFileName(model))
        if "{FILE_NAME_AXIS}" in line:
            line = line.replace("FILE_NAME_AXIS", mg.getAxiWrapperFileName(model))
        if "{FILE_NAME_BASE}" in line:
            line = line.replace("FILE_NAME_BASE", mg.getMainFileName(model))

        fout.write(line)

    fin.close()
    fout.close()

def build_unified_project_ske(meta, model, workspaceDir = None):
    if workspaceDir is None:
        workspaceDir = os.path.join(model.config.get_output_dir(), "unifiedWorkspace")
    hlsDir    = os.path.join(workspaceDir, model.config.get_project_name())
    execDir   = os.path.join(str(hlsDir), "unifiedPrj")
    vitisComp = os.path.join(str(hlsDir), "vitis-comp.json")

    ###### create my own project for this graph
    os.makedirs(workspaceDir, exist_ok=True)
    os.makedirs(hlsDir      , exist_ok=True)
    os.makedirs(execDir      , exist_ok=True)
    ###### create project vitis-comp.json to
    fin = open("../../templates/vitis_unified/vitis-comp.json", 'r')
    fout = open(vitisComp, 'w')

    for line in fin.readlines():
        if "{HLS_NAME}" in line:
            line = line.replace("{HLS_NAME}", model.config.get_project_name())
        if "{CONFIG_FILE}" in line:
            line = line.replace("{CONFIG_FILE}", f"{model.config.get_output_dir()}/hls_config.h")
        fout.write(line)

    fin.close()
    fout.close()




def write_new_tar(meta, model):
    super().write_tar(model)