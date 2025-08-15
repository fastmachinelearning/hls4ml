import os
from pathlib import Path
import stat
from shutil import copyfile

from .meta import VitisUnifiedWriterMeta
from .     import meta_gen as mg

################################################################################
###### main function ###########################################################
################################################################################


def write_gmem_wrapper(meta: VitisUnifiedWriterMeta, model):

    inp_axi_t, out_axi_t, inps, outs = meta.vitis_unified_config.get_corrected_types()
    indent = '      '

    ######################################
    ###### start write myproject_dm.cpp ##
    ######################################

    filedir = os.path.dirname(os.path.abspath(__file__))
    fin     = open(os.path.join(filedir, '../../templates/vitis_unified/myproject_dm.cpp'), 'r')
    fout    = open(f'{model.config.get_output_dir()}/firmware/myproject_dm.cpp', 'w')

    for line in fin.readlines():

        if "MY_PROJECT_AXI_INC" in line:
            line = line.replace("MY_PROJECT_AXI_INC", mg.getAxiWrapperFileName(model))
        elif "DMX_BUF_IN_SZ" in line:
            line = line.replace("VAL", str(meta.vitis_unified_config.get_gmem_in_sz()))
        elif "DMX_BUF_OUT_SZ" in line:
            line = line.replace("VAL", str(meta.vitis_unified_config.get_gmem_out_sz()))
        elif "load_input" in line:
            line = line.replace("ATOMIC_TYPE", inp_axi_t)
        elif "store_output" in line:
            line = line.replace("ATOMIC_TYPE", out_axi_t)
        elif "MY_PROJECT_CON" in line:
            line = line.replace("MY_PROJECT_CON", mg.getAxisTopFuncName(model))
        elif "MY_PROJECT_TOP_FUNC" in line:
            line = line.replace("ATOMIC_TYPE* in", f"{inp_axi_t}* in")
            line = line.replace("ATOMIC_TYPE* out", f"{out_axi_t}* out")
            line = line.replace("MY_PROJECT_TOP_FUNC", mg.getGemTopFuncName(model))
        fout.write(line)


    fin.close()
    fout.close()

    ######################################
    ###### start write myproject_dm.h   ##
    ######################################

    filedir = os.path.dirname(os.path.abspath(__file__))
    fin = open(os.path.join(filedir, '../../templates/vitis_unified/myproject_dm.h'), 'r')
    fout = open(f'{model.config.get_output_dir()}/firmware/myproject_dm.h', 'w')

    for line in fin.readlines():

        if "FILENAME" in line:
            line = line.replace("FILENAME", mg.getGmemWrapperFileName(model).upper())
        elif "MY_PROJECT_TOP_FUNC" in line:
            line = line.replace("ATOMIC_TYPE* in", f"{inp_axi_t}* in")
            line = line.replace("ATOMIC_TYPE* out", f"{out_axi_t}* out")
            line = line.replace("MY_PROJECT_TOP_FUNC", mg.getGemTopFuncName(model))


        fout.write(line)

    fin.close()
    fout.close()