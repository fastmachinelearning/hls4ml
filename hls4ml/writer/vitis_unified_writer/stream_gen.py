import os
from .meta import VitisUnifiedWriterMeta
from . import meta_gen as mg


def write_axi_wrapper_io(meta: VitisUnifiedWriterMeta, inps, outs):
    inputList = []
    outputList = []
    for inp in inps:
        streamType = mg.get_axi_wrapper_type(inp) if meta.vitis_unified_config.isFreeInterimInput() else "dma_data_packet"
        inputList.append(f'hls::stream<{streamType}>& {mg.getWrapperPortName(inp, True)}')
    for out in outs:
        streamType = mg.get_axi_wrapper_type(out) if meta.vitis_unified_config.isFreeInterimOutput() else "dma_data_packet"
        outputList.append(f'hls::stream<{streamType}> & {mg.getWrapperPortName(out, False)}')

    if len(inputList) == 0 or len(outputList) == 0:
        raise Exception("No input or output stream found")
    newline = "/////// inputs\n" +  ",\n ".join(inputList) + ",\n\n ///outputs\n " + ", ".join(outputList) + "\n"
    return newline
##### content in axi_wrapper.cpp
def write_axi_wrapper_interface(meta: VitisUnifiedWriterMeta, model, inps, outs):
    if meta.vitis_unified_config.get_interface() == 'axi_stream':
        newline = ''
        indent = "      "
        for inp in inps:
            portname = mg.getWrapperPortName(inp, True)
            newline += indent + f'#pragma HLS INTERFACE axis port={portname}\n'
        for out in outs:
            portname = mg.getWrapperPortName(out, False)
            newline += indent + f'#pragma HLS INTERFACE axis port={portname}\n'
        if model.config.get_config_value("IOType") == 'io_stream':
                newline += indent + '#pragma HLS INTERFACE ap_ctrl_none port=return\n'
                newline += indent + '#pragma HLS DATAFLOW\n'
        return newline
    else:
        raise Exception("vitis_unified supports only axi_stream @ interface retriever")

def write_axi_local_vars(meta: VitisUnifiedWriterMeta, model, inps, outs):

    ####### build local stream variable

    newline = '///// wrinting local stream vars /////\n'
    if meta.vitis_unified_config.get_interface() == 'axi_stream':
        indent = "      "
        ##### loop to build local stream to send data into the system
        newline += '///////// build input vars ///////////\n'
        for idx, inp in enumerate(inps):
            newline += f"    bool {mg.getWrapperIsLastCnt(idx)} = false;\n"
            portname = mg.getWrapperPortNameLocal(inp, True)
            newline += indent + f'hls::stream<{inp.type.name}> {portname}("{portname}");\n'
        newline += '///////// build output vars ///////////\n'
        for out in outs:
            portname = mg.getWrapperPortNameLocal(out, False)
            newline += indent + f'hls::stream<{out.type.name}> {portname}("{portname}");\n'

    ####### set stream DEPTH

        newline += '///////// set the stream depth ///////////\n'
        ##### loop to set depth

        for inpIdx, inp in enumerate(inps):
            portname = mg.getWrapperPortNameLocal(inp, True)
            newline += indent + f'#pragma HLS STREAM variable={portname} depth={inps[inpIdx].pragma[1]}\n'
        for outIdx, out in enumerate(outs):
            portname = mg.getWrapperPortNameLocal(out, False)
            newline += indent + f'#pragma HLS STREAM variable={portname} depth={model.get_output_variables()[outIdx].pragma[1]}\n'

        return newline

    else:
        raise Exception("vitis_unified supports only axi_stream @ local vars")


def write_axi_wrapper_each_enqueue(meta: VitisUnifiedWriterMeta, model, inps, idx):

    io_type = model.config.get_config_value("IOType")
    indent = "      "
    newline = "\n\n\n"
    if io_type == 'io_stream':
        newline += '////////////// enqueue number ' + str(idx) + ' //////////////\n'
        newline += indent + "///// temp var \n"
        newline += indent + f'dma_data_packet {mg.getWrapperTmpName(inps[idx], True)};\n'
        newline += indent + f"{mg.getWrapperTmpName(inps[idx], True)}.last = 0;\n"
        ### newline += indent + f'{inps[idx].type.name}\n'
        newline += indent + f'for(unsigned i = 0; i < {mg.get_inputSizeArrName(model)}[' +str(idx) +']/' + inps[idx].type.name + '::size; ++i){\n'
        newline += indent + indent + inps[idx].type.name + ' ctype;\n'
        newline += indent + indent + 'for(unsigned j = 0; j < '+ inps[idx].type.name + '::size; ++j){\n'
        if meta.vitis_unified_config.get_interface() == 'axi_stream':
            newline += indent + indent + indent + mg.getWrapperPortName(inps[idx], True) + f'.read({mg.getWrapperTmpName(inps[idx], True)});\n'
            newline += indent + indent + indent + "ctype[j] = " + mg.getWrapperTmpName(inps[idx], True) + ".data;\n"
            newline += indent + indent + indent + mg.getWrapperIsLastCnt(idx) + " = " + mg.getWrapperTmpName(inps[idx], True) + ".last;\n"
        else:
            raise Exception("vitis_unified supports only axi_stream @ each enqueue")

        newline += indent + indent + '}\n'
        newline += indent + indent + mg.getWrapperPortNameLocal(inps[idx], True) + ".write(ctype);\n"
        newline += indent + '}\n'
        newline += indent + mg.getWrapperTmpName(inps[idx], True) + ".last = 0;\n"

    else:
        raise Exception("vitis_unified supports only io_stream @ each enqueue")

    return newline

def write_free_axi_wrapper_each_enqueue(meta: VitisUnifiedWriterMeta, model, inps, idx):
    io_type = model.config.get_config_value("IOType")
    indent = "      "
    newline = "\n\n\n"
    if io_type == 'io_stream':
        newline += '////////////// enqueue number ' + str(idx) + ' //////////////\n'
        newline += indent + "///// temp var \n"
        newline += indent + f'{mg.get_axi_wrapper_type(inps[idx])} {mg.getWrapperTmpName(inps[idx], True)};\n'
        newline += indent + f"{mg.getWrapperTmpName(inps[idx], True)}.last = 0;\n"
        newline += indent + f'for(unsigned i = 0; i < {mg.get_inputSizeArrName(model)}[' + str(idx) + ']/' + inps[
            idx].type.name + '::size; ++i){\n'
        newline += indent + indent + inps[idx].type.name + ' ctype;\n'
        newline += indent + indent + mg.getWrapperPortName(inps[idx], True) + f'.read({mg.getWrapperTmpName(inps[idx], True)});\n'
        newline += indent + indent + "ctype = " + mg.getWrapperTmpName(inps[idx], True) + ".data;\n"
        newline += indent + indent + mg.getWrapperIsLastCnt(idx) + " = " + mg.getWrapperTmpName(inps[idx], True) + ".last;\n"
        newline += indent + indent + mg.getWrapperPortNameLocal(inps[idx], True) + ".write(ctype);\n"
        newline += indent + '}\n'
        newline += indent + mg.getWrapperTmpName(inps[idx], True) + ".last = 0;\n"
    else:
        raise Exception("vitis_unified supports only io_stream @ each free axi enqueue")

    return newline

def write_axi_wrapper_dequeue(meta: VitisUnifiedWriterMeta, model, inputs, outs, idx, out_axi_t):

    io_type = model.config.get_config_value("IOType")
    indent = "      "
    newline = "\n\n\n"
    if io_type == 'io_stream':
        newline += '////////////// dequeue number ' + str(idx) + ' //////////////\n'
        newline += indent + "///// temp var \n"
        newline += indent + f'dma_data_packet {mg.getWrapperTmpName(outs[idx], False)};\n'
        newline += indent + f"{mg.getWrapperTmpName(outs[idx], False)}.last = 0;\n"
        ####### the tmp must copy from input to prevent dma get stuck
        newline += indent + f'for(unsigned i = 0; i < {mg.get_outputSizeArrName(model)}[' +str(idx) +']/' + outs[idx].type.name + '::size; ++i){\n'
        newline += indent + indent + outs[idx].type.name + ' ctype = ' + mg.getWrapperPortNameLocal(outs[idx], False) + '.read();\n'
        newline += indent + indent + 'for(unsigned j = 0; j < ' + outs[idx].type.name + '::size; ++j){\n'
        if meta.vitis_unified_config.get_interface() == 'axi_stream':
            newline += indent + indent + indent + mg.getWrapperTmpName(outs[idx], False) + f'.data = ({out_axi_t}) (ctype[j]);\n'
            poolLastCondition = " & ".join([mg.getWrapperIsLastCnt(condIdx) for condIdx  in range(len(inputs))])
            newline += indent + indent + indent + f"if({poolLastCondition}){{\n"
            newline += indent + indent + indent + indent + mg.getWrapperTmpName(outs[idx], False) + f".last = (((i+1)*(j+1))=={mg.get_outputSizeArrName(model)}[{str(idx)}]);\n"
            newline += indent + indent + indent + "}\n"
            newline += indent + indent + indent + mg.getWrapperPortName(outs[idx], False) + f'.write({mg.getWrapperTmpName(outs[idx], False)});\n'
            newline += indent + indent + "}\n"
            newline += indent + "}\n"
            newline += indent + mg.getWrapperTmpName(outs[idx], False) + ".last = 0;\n"
        else:
            raise Exception("vitis_unified supports only axi_stream @ each dequeue")
    else:
        raise Exception("vitis_unified supports only io_stream @ each dequeue")

    return newline

def write_free_axi_wrapper_dequeue(meta: VitisUnifiedWriterMeta, model, inputs, outs, idx, out_axi_t):

    io_type = model.config.get_config_value("IOType")
    indent = "      "
    newline = "\n\n\n"
    if io_type == 'io_stream':
        newline += '////////////// dequeue number ' + str(idx) + ' //////////////\n'
        newline += indent + "///// temp var \n"
        newline += indent + f'{mg.get_axi_wrapper_type(outs[idx])} {mg.getWrapperTmpName(outs[idx], False)};\n'
        newline += indent + f"{mg.getWrapperTmpName(outs[idx], False)}.last = 0;\n"
        ####### the tmp must copy from input to prevent dma get stuck
        newline += indent + f'for(unsigned i = 0; i < {mg.get_outputSizeArrName(model)}[' +str(idx) +']/' + outs[idx].type.name + '::size; ++i){\n'
        newline += indent + indent + outs[idx].type.name + ' ctype = ' + mg.getWrapperPortNameLocal(outs[idx], False) + '.read();\n'
        newline += indent + indent + mg.getWrapperTmpName(outs[idx], False) + ".data = ctype;\n"
        poolLastCondition = " & ".join([mg.getWrapperIsLastCnt(condIdx) for condIdx in range(len(inputs))])
        newline += indent + indent + f"if({poolLastCondition}){{\n"
        newline += indent + indent + indent + mg.getWrapperTmpName(outs[idx], False) + f".last = ((i+1) == ({mg.get_outputSizeArrName(model)}[{str(idx)}] / {outs[idx].type.name + '::size'} ));\n"
        newline += indent + indent + "}\n"
        newline += indent + indent + mg.getWrapperPortName(outs[idx],False) + f'.write({mg.getWrapperTmpName(outs[idx], False)});\n'
        newline += indent + "}\n"
    else:
        raise Exception("vitis_unified supports only io_stream @ each dequeue")

    return newline

def write_axi_wrapper_insert_call(meta, model, inps, outs):
    io_type = model.config.get_config_value("IOType")
    indent = "      "
    newline = indent + f'{model.config.get_project_name()}' + "("
    inputList = []
    outputList = []
    for inp in inps:
        inputList.append(mg.getWrapperPortNameLocal(inp, True))
    for out in outs:
        outputList.append(mg.getWrapperPortNameLocal(out, False))
    newline += ", ".join(inputList) + ", " + ", ".join(outputList) + ");\n"
    return newline

########################################################
##### main function ####################################
########################################################

def write_axi_wrapper(meta: VitisUnifiedWriterMeta, model):
    '''
        We we want to have multi io system
    '''
    inp_axi_t, out_axi_t, inps, outs = meta.vitis_unified_config.get_corrected_types()
    indent = '    '

    print("------------------------------- input write wrapper is -------------------------")
    print([inp.name for inp in inps])
    print(model.inputs)
    print("------------------------------- output write wrapper is -------------------------")
    print([out.name for out in outs])
    print(model.outputs)
    print("-----------------------------------------------------------------------------------")

    ######################
    # myproject_axi.h
    ######################
    filedir = os.path.dirname(os.path.abspath(__file__))
    f       = open(os.path.join(filedir, '../../templates/vitis_unified/myproject_axi.h'))
    fout    = open(f'{model.config.get_output_dir()}/firmware/{mg.getAxiWrapperFileName(model)}.h', 'w')

    for line in f.readlines():
        if 'MYPROJECT' in line:
            newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
        elif '// hls-fpga-machine-learning insert include' in line:
            newline = f'#include "{model.config.get_project_name()}.h"\n'
            newline += '#include "ap_axi_sdata.h"\n'
        elif 'myproject' in line:
            newline =  line.replace('myproject', model.config.get_project_name())
        elif '// hls-fpga-machine-learning insert definitions' in line:

            ##### make input
            newline = ''
            inputSizeStr = "{ " + ", ".join([str(inp.size()) for inp in inps]) +  " }"
            newline += f'constexpr unsigned {mg.get_inputSizeArrName(model)}  [{len(inps)}] = {inputSizeStr};\n'

            ##### make output
            outputSizeStr = "{ " + ", ".join([str(out.size()) for out in outs]) +  " }"
            newline += f'constexpr unsigned {mg.get_outputSizeArrName(model)} [{len(outs)}] = {outputSizeStr};\n'
            if meta.vitis_unified_config.get_interface() == 'axi_stream':
                newline += 'typedef hls::axis<float, 0, 0, 0, AXIS_ENABLE_LAST> dma_data_packet;\n'
            else:
                newline += f'typedef {inp_axi_t} input_axi_t;\n'
                newline += f'typedef {out_axi_t} output_axi_t;\n'
            #### incase the io is interim input
            if meta.vitis_unified_config.isFreeInterimInput():
                for inp in inps:
                    newline += mg.get_axi_wrapper_dec(inp) + "\n"
            #### incase the io is interim output
            if meta.vitis_unified_config.isFreeInterimOutput():
                for out in outs:
                    newline += mg.get_axi_wrapper_dec(out) + "\n"
        elif '// hls-fpga-machine-learning insert multi-io' in line:
            newline = ''
            if meta.vitis_unified_config.get_interface() == 'axi_stream':
                newline += write_axi_wrapper_io(meta, inps, outs)
            else:
                raise Exception("vitis_unified supports only axi_stream")

        else:
            newline = line

        #### TODO add stream

        fout.write(newline)
    f.close()
    fout.close()

    ######################
    # myproject_axi.cpp
    ######################
    f     = open(os.path.join(filedir, '../../templates/vitis_unified/myproject_axi.cpp'))
    fout  = open(f'{model.config.get_output_dir()}/firmware/{mg.getAxiWrapperFileName(model)}.cpp', 'w')

    io_type = model.config.get_config_value("IOType")

    for line in f.readlines():
        if 'myproject' in line:
            newline = line.replace('myproject', model.config.get_project_name())
        elif '// hls-fpga-machine-learning insert include' in line:
            newline = f'#include "{model.config.get_project_name()}_axi.h"\n'
        elif '// hls-fpga-machine-learning insert multiIo' in line:
            newline = ''
            if meta.vitis_unified_config.get_interface() == 'axi_stream':
                newline += write_axi_wrapper_io(meta, inps, outs)
            else:
                raise Exception("vitis_unified supports only axi_stream")
        elif '// hls-fpga-machine-learning insert interface' in line:
            newline = write_axi_wrapper_interface(meta, model, inps, outs)
        elif '// hls-fpga-machine-learning insert local vars' in line:
            newline = write_axi_local_vars(meta, model, inps, outs)
        elif '// hls-fpga-machine-learning insert enqueue' in line:
            newline = ''
            if meta.vitis_unified_config.isFreeInterimInput():
                for idx, inp in enumerate(inps):
                    newline += write_free_axi_wrapper_each_enqueue(meta, model, inps, idx) + '\n'
            else:
                for idx, inp in enumerate(inps):
                    newline += write_axi_wrapper_each_enqueue(meta, model, inps, idx) + '\n'
        elif '// hls-fpga-machine-learning insert call' in line:
            newline = '////// call the main variable\n'
            newline += write_axi_wrapper_insert_call(meta, model, inps, outs)
        elif '// hls-fpga-machine-learning insert dequeue' in line:
            newline = ''
            if meta.vitis_unified_config.isFreeInterimOutput():
                for idx, out in enumerate(outs):
                    newline += write_free_axi_wrapper_dequeue(meta, model, inps, outs, idx, out_axi_t)
            else:
                for idx, out in enumerate(outs):
                    newline += write_axi_wrapper_dequeue(meta, model, inps, outs, idx, out_axi_t)
        else:
            newline = line
        fout.write(newline)
    f.close()
    fout.close()