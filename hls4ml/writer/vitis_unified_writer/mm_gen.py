import os
from pathlib import Path
import stat
from shutil import copyfile

from .meta import VitisUnifiedWriterMeta
from .     import meta_gen as mg

################################################################################
###### main function ###########################################################
################################################################################

def genIoStr(indent, inp_gmem_t, out_gmem_t, inps, outs):

    inputPtrList   = []
    outputPtrList  = []
    inputSizeList  = []
    outputSizeList = []

    for inp_idx, inp in enumerate(inps):
        inputPtrList.append(f"{indent} {inp_gmem_t}* {mg.getGmemIOPortName(inp, True, inp_idx)}")
        inputSizeList.append(f"{indent} int {mg.getGmemIOPortSizeName(inp, True, inp_idx)}")

    for out_idx, out in enumerate(outs):
        outputPtrList.append(f"{indent} {inp_gmem_t}* {mg.getGmemIOPortName(out, False, out_idx)}")
        outputSizeList.append(f"{indent} int {mg.getGmemIOPortSizeName(out, False, out_idx)}")


    line = ", ".join(inputPtrList)   + ",\n"
    line += ", ".join(outputPtrList) + ",\n"
    line += ", ".join(inputSizeList) + ",\n"
    line += ", ".join(outputSizeList)

    return line


def write_gmem_wrapper(meta: VitisUnifiedWriterMeta, model):

    inp_gmem_t, out_gmem_t, inps, outs = meta.vitis_unified_config.get_corrected_types()
    indent = '      '

    ######################################
    ###### start write myproject_dm.cpp ##
    ######################################

    filedir = os.path.dirname(os.path.abspath(__file__))
    fin     = open(os.path.join(filedir, '../../templates/vitis_unified/myproject_dm.cpp'), 'r')
    fout    = open(f'{model.config.get_output_dir()}/firmware/myproject_dm.cpp', 'w')

    for line in fin.readlines():

        if "MY_PROJECT_AXI_INC" in line:
            line = line.replace("MY_PROJECT_AXI_INC", mg.getMainWrapperFileName(model))
        if "MY_PROJECT_TOP_FUNC" in line:
            line = line.replace("MY_PROJECT_TOP_FUNC", mg.getGemTopFuncName(model))
        elif "DMX_BUF_IN_SZ" in line:
            line = line.replace("VAL", str(meta.vitis_unified_config.get_gmem_in_bufferSz()))
        elif "DMX_BUF_OUT_SZ" in line:
            line = line.replace("VAL", str(meta.vitis_unified_config.get_gmem_out_bufferSz()))
        elif "// vitis-unified-wrapper-io" in line:
            line = genIoStr(indent, inp_gmem_t, out_gmem_t, inps, outs) + "\n"
        elif "// vitis-unified-wrapper-interface" in line:
            for inp_idx, inp in enumerate(inps):
                line += f"{indent} #pragma HLS INTERFACE m_axi     port={mg.getGmemIOPortName(inp, True, inp_idx)} bundle = gmem_in{inp_idx}\n"
                line += f"{indent} #pragma HLS INTERFACE s_axilite port={mg.getGmemIOPortSizeName(inp, True, inp_idx)} bundle = control\n"
            for out_idx, out in enumerate(outs):
                line += f"{indent} #pragma HLS INTERFACE m_axi     port={mg.getGmemIOPortName(out, False, out_idx)} bundle = gmem_out{out_idx}\n"
                line += f"{indent} #pragma HLS INTERFACE s_axilite port={mg.getGmemIOPortSizeName(out, False, out_idx)} bundle = control\n"
        elif "// vitis-unified-wrapper-stream-dec"    in line:

            for inp_idx, inp in enumerate(inps):
                line += f"{indent} static hls::stream<{inp.type.name}> {mg.getGmemLocalStreamName(inp, True, inp_idx)};\n"
            for out_idx, out in enumerate(outs):
                line += f"{indent} static hls::stream<{out.type.name}> {mg.getGmemLocalStreamName(out, False, out_idx)};\n"

        elif "// vitis-unified-wrapper-stream-config" in line:
            for inp_idx, inp in enumerate(inps):
                line += f"#pragma HLS STREAM variable={mg.getGmemLocalStreamName(inp, True, inp_idx)} depth=DMX_BUF_IN_SZ\n"
            for out_idx, out in enumerate(outs):
                line += f"#pragma HLS STREAM variable={mg.getGmemLocalStreamName(out, False, out_idx)} depth=DMX_BUF_OUT_SZ\n"

        elif "// vitis-unified-wrapper-load"    in line:
            for inp_idx, inp in enumerate(inps):
                line += f"load_input({mg.getGmemIOPortName(inp, True, inp_idx)}, {mg.getGmemLocalStreamName(inp, True, inp_idx)}, {mg.getGmemIOPortSizeName(inp, True, inp_idx)});\n"
        elif "// vitis-unified-wrapper-compute"       in line:
            poolList = []
            for inp_idx, inp in enumerate(inps):
                poolList.append(f"{mg.getGmemLocalStreamName(inp, True, inp_idx)}")
            for out_idx, out in enumerate(outs):
                poolList.append(f"{mg.getGmemLocalStreamName(out, False, out_idx)}")
            joinedIo = ", \n".join(poolList)
            line += f"{indent} {mg.getTopModelName(model)}({joinedIo});\n"

        elif "// vitis-unified-wrapper-store"  in line:
            for out_idx, out in enumerate(outs):
                line += f"store_result({mg.getGmemIOPortName(out, False, out_idx)}, {mg.getGmemLocalStreamName(out, False, out_idx)}, {mg.getGmemIOPortSizeName(out, False, out_idx)});\n"

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
            line = line.replace("MY_PROJECT_TOP_FUNC", mg.getGemTopFuncName(model))
        elif "// vitis-unified-wrapper-io" in line:
            line += genIoStr(indent, inp_gmem_t, out_gmem_t, inps, outs) + "\n"
        fout.write(line)

    fin.close()
    fout.close()