import os
from pathlib import Path
import stat

from shutil import copyfile

from hls4ml.writer.vitis_writer import VitisWriter

class VitisUnifiedWriter(VitisWriter):

    def __init__(self):
        super().__init__()
        self.vitis_unified_config = None

    #######################################################
    ## naming of variable function helper #################
    #######################################################

    def getDmaTypeName(self):
        return "dma_data_packet"

    def getWrapperPortName(self, tensorVar, isInput: bool):
        ioStr = "in" if isInput else "out"
        return f"par_{ioStr}_{tensorVar.name}"

    def getTopModelName(self, model):
        return f"{model.config.get_project_name()}_axi"
    ### it is renamed for stitch layer
    def renameType(self, tensorVar, layerIdx:int, isInput: bool):
        return "result_" + tensorVar.type.name + f"_at_layer_{str(layerIdx)}"

    def get_inputSizeArrName(self, model):
        return "N_IN_" + model.config.get_project_name()

    def get_outputSizeArrName(self, model):
        return "N_OUT_" + model.config.get_project_name()

    def get_axi_wrapper_type(self, tensorVar):
        return f"{tensorVar.type.name}_packet"

    def get_axi_wrapper_dec(self, tensorVar):
        return f"typedef hls::axis<{tensorVar.type.name}, 0,0,0, AXIS_ENABLE_LAST> {self.get_axi_wrapper_type(tensorVar)};"


    ########################################################
    ## axi_wrapper.h & axi_wrapper.cpp  function helper ####
    ########################################################
    ##### variable
    def getWrapperPortNameLocal(self, tensorVar, isInput: bool):
        ioStr = "in" if isInput else "out"
        return f"par_{ioStr}_{tensorVar.name}_local"

    def getWrapperTmpName(self, tensorVar, isInput: bool):
        ioStr = "in" if isInput else "out"
        return f"par_{ioStr}_{tensorVar.name}_tmp"

    def getWrapperIsLastCnt(self, idx):
        return f"isLastCnt_{str(idx)}"
    ##### io
    def write_axi_wrapper_io(self, inps, outs):
        inputList = []
        outputList = []
        for inp in inps:
            streamType = self.get_axi_wrapper_type(inp) if self.vitis_unified_config.isFreeInterimInput() else "dma_data_packet"
            inputList.append(f'hls::stream<{streamType}>& {self.getWrapperPortName(inp, True)}')
        for out in outs:
            streamType = self.get_axi_wrapper_type(out) if self.vitis_unified_config.isFreeInterimOutput() else "dma_data_packet"
            outputList.append(f'hls::stream<{streamType}> & {self.getWrapperPortName(out, False)}')

        if len(inputList) == 0 or len(outputList) == 0:
            raise Exception("No input or output stream found")
        newline = "/////// inputs\n" +  ",\n ".join(inputList) + ",\n\n ///outputs\n " + ", ".join(outputList) + "\n"
        return newline
    ##### content in axi_wrapper.cpp
    def write_axi_wrapper_interface(self, model, inps, outs):
        if self.vitis_unified_config.get_interface() == 'axi_stream':
            newline = ''
            indent = "      "
            for inp in inps:
                portname = self.getWrapperPortName(inp, True)
                newline += indent + f'#pragma HLS INTERFACE axis port={portname}\n'
            for out in outs:
                portname = self.getWrapperPortName(out, False)
                newline += indent + f'#pragma HLS INTERFACE axis port={portname}\n'
            if model.config.get_config_value("IOType") == 'io_stream':
                    newline += indent + '#pragma HLS INTERFACE ap_ctrl_none port=return\n'
                    newline += indent + '#pragma HLS DATAFLOW\n'
            return newline
        else:
            raise Exception("vitis_unified supports only axi_stream @ interface retriever")

    def write_axi_local_vars(self, model, inps, outs):

        ####### build local stream variable

        newline = '///// wrinting local stream vars /////\n'
        if self.vitis_unified_config.get_interface() == 'axi_stream':
            indent = "      "
            ##### loop to build local stream to send data into the system
            newline += '///////// build input vars ///////////\n'
            for idx, inp in enumerate(inps):
                newline += f"    bool {self.getWrapperIsLastCnt(idx)} = false;\n"
                portname = self.getWrapperPortNameLocal(inp, True)
                newline += indent + f'hls::stream<{inp.type.name}> {portname}("{portname}");\n'
            newline += '///////// build output vars ///////////\n'
            for out in outs:
                portname = self.getWrapperPortNameLocal(out, False)
                newline += indent + f'hls::stream<{out.type.name}> {portname}("{portname}");\n'

        ####### set stream DEPTH

            newline += '///////// set the stream depth ///////////\n'
            ##### loop to set depth

            for inpIdx, inp in enumerate(inps):
                portname = self.getWrapperPortNameLocal(inp, True)
                newline += indent + f'#pragma HLS STREAM variable={portname} depth={inps[inpIdx].pragma[1]}\n'
            for outIdx, out in enumerate(outs):
                portname = self.getWrapperPortNameLocal(out, False)
                newline += indent + f'#pragma HLS STREAM variable={portname} depth={model.get_output_variables()[outIdx].pragma[1]}\n'

        else:
            raise Exception("vitis_unified supports only axi_stream @ local vars")


        return newline

    def write_axi_wrapper_each_enqueue(self, model, inps, idx):

        io_type = model.config.get_config_value("IOType")
        indent = "      "
        newline = "\n\n\n"
        if io_type == 'io_stream':
            newline += '////////////// enqueue number ' + str(idx) + ' //////////////\n'
            newline += indent + "///// temp var \n"
            newline += indent + f'dma_data_packet {self.getWrapperTmpName(inps[idx], True)};\n'
            newline += indent + f"{self.getWrapperTmpName(inps[idx], True)}.last = 0;\n"
            ### newline += indent + f'{inps[idx].type.name}\n'
            newline += indent + f'for(unsigned i = 0; i < {self.get_inputSizeArrName(model)}[' +str(idx) +']/' + inps[idx].type.name + '::size; ++i){\n'
            newline += indent + indent + inps[idx].type.name + ' ctype;\n'
            newline += indent + indent + 'for(unsigned j = 0; j < '+ inps[idx].type.name + '::size; ++j){\n'
            if self.vitis_unified_config.get_interface() == 'axi_stream':
                newline += indent + indent + indent + self.getWrapperPortName(inps[idx], True) + f'.read({self.getWrapperTmpName(inps[idx], True)});\n'
                newline += indent + indent + indent + "ctype[j] = " + self.getWrapperTmpName(inps[idx], True) + ".data;\n"
                newline += indent + indent + indent + self.getWrapperIsLastCnt(idx) + " = " + self.getWrapperTmpName(inps[idx], True) + ".last;\n"
            else:
                raise Exception("vitis_unified supports only axi_stream @ each enqueue")

            newline += indent + indent + '}\n'
            newline += indent + indent + self.getWrapperPortNameLocal(inps[idx], True) + ".write(ctype);\n"
            newline += indent + '}\n'
            newline += indent + self.getWrapperTmpName(inps[idx], True) + ".last = 0;\n"

        else:
            raise Exception("vitis_unified supports only io_stream @ each enqueue")

        return newline

    def write_free_axi_wrapper_each_enqueue(self, model, inps, idx):
        io_type = model.config.get_config_value("IOType")
        indent = "      "
        newline = "\n\n\n"
        if io_type == 'io_stream':
            newline += '////////////// enqueue number ' + str(idx) + ' //////////////\n'
            newline += indent + "///// temp var \n"
            newline += indent + f'{self.get_axi_wrapper_type(inps[idx])} {self.getWrapperTmpName(inps[idx], True)};\n'
            newline += indent + f"{self.getWrapperTmpName(inps[idx], True)}.last = 0;\n"
            newline += indent + f'for(unsigned i = 0; i < {self.get_inputSizeArrName(model)}[' + str(idx) + ']/' + inps[
                idx].type.name + '::size; ++i){\n'
            newline += indent + indent + inps[idx].type.name + ' ctype;\n'
            newline += indent + indent + self.getWrapperPortName(inps[idx], True) + f'.read({self.getWrapperTmpName(inps[idx], True)});\n'
            newline += indent + indent + "ctype = " + self.getWrapperTmpName(inps[idx], True) + ".data;\n"
            newline += indent + indent + self.getWrapperIsLastCnt(idx) + " = " + self.getWrapperTmpName(inps[idx], True) + ".last;\n"
            newline += indent + indent + self.getWrapperPortNameLocal(inps[idx], True) + ".write(ctype);\n"
            newline += indent + '}\n'
            newline += indent + self.getWrapperTmpName(inps[idx], True) + ".last = 0;\n"
        else:
            raise Exception("vitis_unified supports only io_stream @ each free axi enqueue")

        return newline

    def write_axi_wrapper_dequeue(self, model, inputs, outs, idx, out_axi_t):

        io_type = model.config.get_config_value("IOType")
        indent = "      "
        newline = "\n\n\n"
        if io_type == 'io_stream':
            newline += '////////////// dequeue number ' + str(idx) + ' //////////////\n'
            newline += indent + "///// temp var \n"
            newline += indent + f'dma_data_packet {self.getWrapperTmpName(outs[idx], False)};\n'
            newline += indent + f"{self.getWrapperTmpName(outs[idx], False)}.last = 0;\n"
            ####### the tmp must copy from input to prevent dma get stuck
            newline += indent + f'for(unsigned i = 0; i < {self.get_outputSizeArrName(model)}[' +str(idx) +']/' + outs[idx].type.name + '::size; ++i){\n'
            newline += indent + indent + outs[idx].type.name + ' ctype = ' + self.getWrapperPortNameLocal(outs[idx], False) + '.read();\n'
            newline += indent + indent + 'for(unsigned j = 0; j < ' + outs[idx].type.name + '::size; ++j){\n'
            if self.vitis_unified_config.get_interface() == 'axi_stream':
                newline += indent + indent + indent + self.getWrapperTmpName(outs[idx], False) + f'.data = ({out_axi_t}) (ctype[j]);\n'
                poolLastCondition = " & ".join([self.getWrapperIsLastCnt(condIdx) for condIdx  in range(len(inputs))])
                newline += indent + indent + indent + f"if({poolLastCondition}){{\n"
                newline += indent + indent + indent + indent + self.getWrapperTmpName(outs[idx], False) + f".last = (((i+1)*(j+1))=={self.get_outputSizeArrName(model)}[{str(idx)}]);\n"
                newline += indent + indent + indent + "}\n"
                newline += indent + indent + indent + self.getWrapperPortName(outs[idx], False) + f'.write({self.getWrapperTmpName(outs[idx], False)});\n'
                newline += indent + indent + "}\n"
                newline += indent + "}\n"
                newline += indent + self.getWrapperTmpName(outs[idx], False) + ".last = 0;\n"
            else:
                raise Exception("vitis_unified supports only axi_stream @ each dequeue")
        else:
            raise Exception("vitis_unified supports only io_stream @ each dequeue")

        return newline

    def write_free_axi_wrapper_dequeue(self, model, inputs, outs, idx, out_axi_t):

        io_type = model.config.get_config_value("IOType")
        indent = "      "
        newline = "\n\n\n"
        if io_type == 'io_stream':
            newline += '////////////// dequeue number ' + str(idx) + ' //////////////\n'
            newline += indent + "///// temp var \n"
            newline += indent + f'{self.get_axi_wrapper_type(outs[idx])} {self.getWrapperTmpName(outs[idx], False)};\n'
            newline += indent + f"{self.getWrapperTmpName(outs[idx], False)}.last = 0;\n"
            ####### the tmp must copy from input to prevent dma get stuck
            newline += indent + f'for(unsigned i = 0; i < {self.get_outputSizeArrName(model)}[' +str(idx) +']/' + outs[idx].type.name + '::size; ++i){\n'
            newline += indent + indent + outs[idx].type.name + ' ctype = ' + self.getWrapperPortNameLocal(outs[idx], False) + '.read();\n'
            newline += indent + indent + self.getWrapperTmpName(outs[idx], False) + ".data = ctype;\n"
            poolLastCondition = " & ".join([self.getWrapperIsLastCnt(condIdx) for condIdx in range(len(inputs))])
            newline += indent + indent + f"if({poolLastCondition}){{\n"
            newline += indent + indent + indent + self.getWrapperTmpName(outs[idx], False) + f".last = ((i+1) == ({self.get_outputSizeArrName(model)}[{str(idx)}] / {outs[idx].type.name + '::size'} ));\n"
            newline += indent + indent + "}\n"
            newline += indent + indent + self.getWrapperPortName(outs[idx],False) + f'.write({self.getWrapperTmpName(outs[idx], False)});\n'
            newline += indent + "}\n"
        else:
            raise Exception("vitis_unified supports only io_stream @ each dequeue")

        return newline

    def write_axi_wrapper_insert_call(self, model, inps, outs):
        io_type = model.config.get_config_value("IOType")
        indent = "      "
        newline = indent + f'{model.config.get_project_name()}' + "("
        inputList = []
        outputList = []
        for inp in inps:
            inputList.append(self.getWrapperPortNameLocal(inp, True))
        for out in outs:
            outputList.append(self.getWrapperPortNameLocal(out, False))
        newline += ", ".join(inputList) + ", " + ", ".join(outputList) + ");\n"
        return newline

    ########################################################
    ##### main function ####################################
    ########################################################

    def write_axi_wrapper(self, model):
        '''
            We we want to have multi io system
        '''
        inp_axi_t, out_axi_t, inps, outs = self.vitis_unified_config.get_corrected_types()
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
        f       = open(os.path.join(filedir, '../templates/vitis_unified/myproject_axi.h'))
        fout    = open(f'{model.config.get_output_dir()}/firmware/{model.config.get_project_name()}_axi.h', 'w')

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
                newline += f'constexpr unsigned {self.get_inputSizeArrName(model)}  [{len(inps)}] = {inputSizeStr};\n'

                ##### make output
                outputSizeStr = "{ " + ", ".join([str(out.size()) for out in outs]) +  " }"
                newline += f'constexpr unsigned {self.get_outputSizeArrName(model)} [{len(outs)}] = {outputSizeStr};\n'
                if self.vitis_unified_config.get_interface() == 'axi_stream':
                    newline += 'typedef hls::axis<float, 0, 0, 0, AXIS_ENABLE_LAST> dma_data_packet;\n'
                else:
                    newline += f'typedef {inp_axi_t} input_axi_t;\n'
                    newline += f'typedef {out_axi_t} output_axi_t;\n'
                #### incase the io is interim input
                if self.vitis_unified_config.isFreeInterimInput():
                    for inp in inps:
                        newline += self.get_axi_wrapper_dec(inp) + "\n"
                #### incase the io is interim output
                if self.vitis_unified_config.isFreeInterimOutput():
                    for out in outs:
                        newline += self.get_axi_wrapper_dec(out) + "\n"
            elif '// hls-fpga-machine-learning insert multi-io' in line:
                newline = ''
                if self.vitis_unified_config.get_interface() == 'axi_stream':
                    newline += self.write_axi_wrapper_io(inps, outs)
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
        f     = open(os.path.join(filedir, '../templates/vitis_unified/myproject_axi.cpp'))
        fout  = open(f'{model.config.get_output_dir()}/firmware/{model.config.get_project_name()}_axi.cpp', 'w')

        io_type = model.config.get_config_value("IOType")

        for line in f.readlines():
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            elif '// hls-fpga-machine-learning insert include' in line:
                newline = f'#include "{model.config.get_project_name()}_axi.h"\n'
            elif '// hls-fpga-machine-learning insert multiIo' in line:
                newline = ''
                if self.vitis_unified_config.get_interface() == 'axi_stream':
                    newline += self.write_axi_wrapper_io(inps, outs)
                else:
                    raise Exception("vitis_unified supports only axi_stream")
            elif '// hls-fpga-machine-learning insert interface' in line:
                newline = self.write_axi_wrapper_interface(model, inps, outs)
            elif '// hls-fpga-machine-learning insert local vars' in line:
                newline = self.write_axi_local_vars(model, inps, outs)
            elif '// hls-fpga-machine-learning insert enqueue' in line:
                newline = ''
                if self.vitis_unified_config.isFreeInterimInput():
                    for idx, inp in enumerate(inps):
                        newline += self.write_free_axi_wrapper_each_enqueue(model, inps, idx) + '\n'
                else:
                    for idx, inp in enumerate(inps):
                        newline += self.write_axi_wrapper_each_enqueue(model, inps, idx) + '\n'
            elif '// hls-fpga-machine-learning insert call' in line:
                newline = '////// call the main variable\n'
                newline += self.write_axi_wrapper_insert_call(model, inps, outs)
            elif '// hls-fpga-machine-learning insert dequeue' in line:
                newline = ''
                if self.vitis_unified_config.isFreeInterimOutput():
                    for idx, out in enumerate(outs):
                        newline += self.write_free_axi_wrapper_dequeue(model, inps, outs, idx, out_axi_t)
                else:
                    for idx, out in enumerate(outs):
                        newline += self.write_axi_wrapper_dequeue(model, inps, outs, idx, out_axi_t)
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    ########################################################
    ## write test (for co simulation)/ and bridge file () script  function helper    ###############
    ########################################################

    def write_wrapper_test(self, model):

        filedir = os.path.dirname(os.path.abspath(__file__))
        f    = open(os.path.join(filedir, '../templates/vitis_unified/myproject_test.cpp'))
        fout = open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_test.cpp', 'w')

        model_inputs  = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        fout.write("//// generated by partial backend\n")

        for line in f.readlines():
            indent = ' ' * (len(line) - len(line.lstrip(' ')))

            #Insert numbers
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            elif '// hls-fpga-machine-learning insert bram' in line:
                newline = line
                for bram in model_brams:
                    newline += f'#include \"firmware/weights/{bram.name}.h\"\n'

            elif '// hls-fpga-machine-learning insert data' in line:
                newline = line
                offset = 0
                for inputIdx, inp in enumerate(model_inputs):
                    streamPktType = self.get_axi_wrapper_type(inp) if self.vitis_unified_config.isFreeInterimInput() else self.getDmaTypeName()

                    newline += "      hls::stream<{desType}> {inputPortName};\n".format(
                        desType = streamPktType, inputPortName = self.getWrapperPortName(inp, True)
                    )
                    if self.vitis_unified_config.isFreeInterimInput():
                        newline += '      nnet::copy_data_axi_w_offset<float, {underlyType}, {wrapType}, {offset}, {inputSize}>(in, {inputPortName});\n'.format(
                            underlyType = inp.type.name,
                            wrapType=streamPktType,
                            offset=offset,
                            inputSize=str(inp.size()),
                            inputPortName=self.getWrapperPortName(inp, True)
                        )
                    else:
                        newline += '      nnet::copy_data_axi_w_offset<float, {destype}, {offset}, {inputSize}>(in, {inputPortName});\n'.format(
                            destype = streamPktType, offset=offset, inputSize=str(inp.size()),
                            inputPortName=self.getWrapperPortName(inp, True)
                        )
                    offset += inp.size()
                for out in model_outputs:
                    streamPktType = self.get_axi_wrapper_type(out) if self.vitis_unified_config.isFreeInterimOutput() else self.getDmaTypeName()
                    newline += '      ' + f"hls::stream<{streamPktType}> {self.getWrapperPortName(out, False)};\n"

            elif '// hls-fpga-machine-learning insert top-level-function' in line:
                newline = line

                input_vars  = ','.join([self.getWrapperPortName(inp, True) for inp in model_inputs])
                output_vars = ','.join([self.getWrapperPortName(out, False) for out in model_outputs])
                bram_vars   = ','.join([b.name for b in model_brams])

                # Concatenate the input, output, and bram variables. Filter out empty/null values
                all_vars = ','.join(filter(None, [input_vars, output_vars, bram_vars]))

                top_level = indent + f'{self.getTopModelName(model)}({all_vars});\n'

                newline += top_level
            elif '// hls-fpga-machine-learning insert predictions' in line:
                newline = line
                for outIdx, out in enumerate(model_outputs):
                    #newline += indent + f'for(int i = 0; i < {out.size_cpp()}; i++) {{\n'
                    newline += indent + f'for(int i = 0; i < {self.get_outputSizeArrName(model)}[{outIdx}]; i++) {{\n'
                    newline += indent + '  std::cout << pr[i] << " ";\n'
                    newline += indent + '}\n'
                    newline += indent + 'std::cout << std::endl;\n'
            # elif '// hls-fpga-machine-learning insert tb-output' in line:
            #     newline = line
            #     tb_stream = model.config.get_writer_config().get('TBOutputStream', 'both')
            #     if tb_stream != 'stdout':
            #         for outIdx, out in enumerate(model_outputs):
            #             # newline += indent + 'nnet::print_result<{}, {}>({}, fout);\n'.format(
            #             #     out.type.name, out.size_cpp(), out.name
            #             # )  # TODO enable this
            #             newline += indent + 'nnet::print_result<{actualType}, {dmaType}, {arrName}[{arrSize}]>({portName}, fout);\n'.format(
            #                 actualType = out.type.name, dmaType = self.getDmaTypeName(), arrName = self.get_outputSizeArrName(model),arrSize = str(outIdx), portName = self.getWrapperPortName(out, False)
            #             )  # TODO enable this
            elif '// hls-fpga-machine-learning insert zero' in line:
                newline = line
                for inpIdx, inp in enumerate(model_inputs):
                    streamPktType = self.get_axi_wrapper_type(inp) if self.vitis_unified_config.isFreeInterimInput() else self.getDmaTypeName()
                    fillZeroFunc  = "fill_zero_toArr" if self.vitis_unified_config.isFreeInterimInput() else "fill_zero"
                    newline += "        " + f"hls::stream<{streamPktType}> {self.getWrapperPortName(inp, True)};\n"
                    newline += "        " + (f'nnet::{fillZeroFunc}<{inp.type.name}, {streamPktType},{self.get_inputSizeArrName(model)}[{str(inpIdx)}]>'
                                             f'({self.getWrapperPortName(inp,True)});\n')
                for out in model_outputs:
                    #newline += indent + out.definition_cpp() + ';\n'
                    streamPktType = self.get_axi_wrapper_type(out) if self.vitis_unified_config.isFreeInterimOutput() else self.getDmaTypeName()
                    newline += "        " + f"hls::stream<{streamPktType}> {self.getWrapperPortName(out, False)};\n"

            elif (
                   '// hls-fpga-machine-learning insert output'    in line
                or '// hls-fpga-machine-learning insert quantized' in line
                or '// hls-fpga-machine-learning insert tb-output' in line
            ):
                newline = line
                tb_stream = model.config.get_writer_config().get('TBOutputStream', 'both')
                dest =  'fout' if ((tb_stream == 'file') or ('// hls-fpga-machine-learning insert tb-output' in line) ) else 'std::cout'
                keep_output = "true" if ("// hls-fpga-machine-learning insert tb-output" in line) else "false"
                #keep_output = str(tb_stream != 'stdout').lower()  # We keep output if we need to write it to file too.
                if tb_stream != 'file': ### it mean cout
                    for outIdx, out in enumerate(model_outputs):
                        #     newline += indent + 'nnet::print_result<{}, {}>({}, std::cout, {});\n'.format(
                        #         out.type.name, out.size_cpp(), out.name, keep_output
                        #     )
                        streamPktType = self.get_axi_wrapper_type(out) if self.vitis_unified_config.isFreeInterimOutput() else self.getDmaTypeName()

                        newline += (indent + 'nnet::print_result<{actualType}, {pktType}, {arrName}[{arrIdx}]>({portName}, {des}, {keepOutput});\n'
                                    .format( actualType = out.type.name,
                                             pktType    = streamPktType,
                                             arrName    = self.get_outputSizeArrName(model),
                                             arrIdx     = str(outIdx),
                                             portName   = self.getWrapperPortName(out, False),
                                             des        = dest,
                                             keepOutput = keep_output))

            elif '// hls-fpga-machine-learning insert namespace' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += indent + f'using namespace {namespace};\n'

            else:
                newline = line

            fout.write(newline)
        f.close()
        fout.close()


        ####################################################
        ### write myproject_bridge.cpp #####################
        ####################################################
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/vitis_unified/myproject_bridge.cpp'))
        fout = open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_bridge.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))

            elif 'myproject' in line:
                newline = line.replace('myproject', format(model.config.get_project_name()))

            elif '// hls-fpga-machine-learning insert bram' in line:
                newline = line
                for bram in model_brams:
                    newline += f'#include \"firmware/weights/{bram.name}.h\"\n'

            elif '// hls-fpga-machine-learning insert header' in line:
                dtype = line.split('#', 1)[1].strip()
                inputs_str = ', '.join([f'{dtype} {i.name}[{i.size_cpp()}]' for i in model_inputs])
                outputs_str = ', '.join([f'{dtype} {o.name}[{o.size_cpp()}]' for o in model_outputs])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str + '\n'

            elif '// hls-fpga-machine-learning insert wrapper' in line:
                dtype = line.split('#', 1)[1].strip()
                newline = ''
                for inpIdx, inp in enumerate(model_inputs): ## former is i
                    if self.vitis_unified_config.isFreeInterimInput():
                        newline += indent + f"hls::stream<{inp.type.name}> {self.getWrapperPortName(inp, True)};\n"
                        newline += indent + "nnet::convert_data_pkt<{srcType}, {underlying_data_T}, {data_T}, {sz}>({inpRaw}, {inp_wrapper});\n".format(
                            srcType           =dtype,
                            underlying_data_T=inp.type.name,
                            data_T=self.get_axi_wrapper_type(inp),
                            sz=str(inp.size()),
                            inpRaw=inp.name,
                            inp_wrapper=self.getWrapperPortName(inp, True),
                        )
                    else:
                        newline += indent + f"hls::stream<dma_data_packet> {self.getWrapperPortName(inp, True)};\n"
                        newline += indent + f"nnet::convert_data<{dtype}, {dtype}, {self.get_inputSizeArrName(model)}[{str(inpIdx)}]>({inp.name}, {self.getWrapperPortName(inp, True)});\n"
                    # newline += indent + '{var};\n'.format(var=i.definition_cpp(name_suffix='_ap'))
                    # newline += indent + 'nnet::convert_data<{}, {}, {}>({}, {}_ap);\n'.format(
                    #     dtype, i.type.name, i.size_cpp(), i.name, i.name
                    # )
                newline += '\n'

                for out in model_outputs:
                    #newline += indent + '{var};\n'.format(var=o.definition_cpp(name_suffix='_ap'))
                    outStreamType = self.get_axi_wrapper_type(out) if self.vitis_unified_config.isFreeInterimOutput() else self.getDmaTypeName()
                    newline += indent + f"hls::stream<{outStreamType}> {self.getWrapperPortName(out, False)};\n"

                newline += '\n'

                input_vars = ','.join([self.getWrapperPortName(inp, True)for inp in model_inputs])
                bram_vars = ','.join([b.name for b in model_brams])
                output_vars = ','.join([self.getWrapperPortName(out, False) for out in model_outputs])

                # Concatenate the input, output, and bram variables. Filter out empty/null values
                all_vars = ','.join(filter(None, [input_vars, output_vars, bram_vars]))

                top_level = indent + f'{self.getTopModelName(model)}({all_vars});\n'
                newline += top_level

                newline += '\n'

                for outIdx, out in enumerate(model_outputs):
                    # newline += indent + 'nnet::convert_data<{}, {}, {}>({}_ap, {});\n'.format(
                    #     o.type.name, dtype, o.size_cpp(), o.name, o.name
                    # )
                    if self.vitis_unified_config.isFreeInterimOutput():
                        newline += indent + f"nnet::convert_data_pkt<{dtype}, {out.type.name}, {self.get_outputSizeArrName(model)}[{str(outIdx)}]>({self.getWrapperPortName(out, False)}, {out.name});\n"
                    else:
                        newline += indent + (f"nnet::convert_data<{dtype}, {dtype}, {self.get_axi_wrapper_type(out)},{self.get_outputSizeArrName(model)}[{str(outIdx)}]>"
                                             f"({self.getWrapperPortName(out, False)}, {out.name});\n")

            elif '// hls-fpga-machine-learning insert trace_outputs' in line:
                newline = ''
                for layer in model.get_layers():
                    func = layer.get_attr('function_cpp', None)
                    if func and model.config.trace_output and layer.get_attr('trace', False):
                        vars = layer.get_variables()
                        for var in vars:
                            newline += (
                                    indent
                                    + 'nnet::trace_outputs->insert(std::pair<std::string, void *>('
                                    + f'"{layer.name}", (void *) malloc({var.size_cpp()} * element_size)));\n'
                            )

            elif '// hls-fpga-machine-learning insert namespace' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += indent + f'using namespace {namespace};\n'

            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    ########################################################
    ## write test script  function helper    ###############
    ########################################################


    def write_board_script(self, model):
        '''
                Write the tcl scripts and kernel sources to create a Vivado IPI project for the VitisAcceleratorIPFlow
                '''
        ### I am not sure yet what it is
        # filedir = os.path.dirname(os.path.abspath(__file__))
        # copyfile(
        #     os.path.join(filedir, self.vitis_accelerator_ip_flow_config.get_tcl_file_path()),
        #     f'{model.config.get_output_dir()}/design.tcl',
        # )

        ###################
        # project.tcl
        ###################
        f = open(f'{model.config.get_output_dir()}/project.tcl', 'w')
        f.write('variable project_name\n')
        f.write(f'set project_name "{model.config.get_project_name()}"\n')
        f.write('variable backend\n')
        f.write('set backend "vitisacceleratoripflowpartial"\n')
        f.write('variable part\n')
        f.write("set part \"xc7z020clg400-1\"\n")
        #f.write(f'set part "{self.vitis_accelerator_ip_flow_config.get_part()}"\n')
        f.write('variable clock_period\n')
        f.write('set clock_period {}\n'.format(model.config.get_config_value('ClockPeriod')))
        f.write('variable clock_uncertainty\n')
        f.write('set clock_uncertainty {}\n'.format(model.config.get_config_value('ClockUncertainty', '12.5%')))
        f.write('variable version\n')
        f.write('set version "{}"\n'.format(model.config.get_config_value('Version', '1.0.0')))
        # if self.vitis_accelerator_ip_flow_config.get_interface() == 'axi_stream':
        #     in_bit, out_bit = self.vitis_accelerator_ip_flow_config.get_io_bitwidth()
        #     f.write(f'set bit_width_hls_output {in_bit}\n')
        #     f.write(f'set bit_width_hls_input {out_bit}\n')
        f.close()
        return

    def write_driver(self, model):
        print("[partial reconfig] we are not supporting write_driver this yet")

    def modify_build_script(self, model):
        '''
        Modify the build_prj.tcl and build_lib.sh scripts to add the extra wrapper files and set the top function
        '''
        filedir = os.path.dirname(os.path.abspath(__file__))
        oldfile = f'{model.config.get_output_dir()}/build_prj.tcl'
        newfile = f'{model.config.get_output_dir()}/build_prj_axi.tcl'
        f = open(oldfile)
        fout = open(newfile, 'w')

        for line in f.readlines():
            if 'set_top' in line:
                newline = line[:-1] + '_axi\n'  # remove the newline from the line end and append _axi for the new top
                newline += f'add_files firmware/{model.config.get_project_name()}_axi.cpp -cflags "-std=c++0x"\n'
            elif f'{model.config.get_project_name()}_cosim' in line:
                newline = line.replace(
                    f'{model.config.get_project_name()}_cosim',
                    f'{model.config.get_project_name()}_axi_cosim',
                )
            elif '${project_name}.tcl' in line:
                newline = line.replace('${project_name}.tcl', '${project_name}_axi.tcl')
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()
        os.rename(newfile, oldfile)

        ###################
        # build_lib.sh
        ###################

        f = open(os.path.join(filedir, '../templates/vitis_unified/build_lib.sh'))
        fout = open(f'{model.config.get_output_dir()}/build_lib.sh', 'w')

        for line in f.readlines():
            line = line.replace('myproject', model.config.get_project_name())
            line = line.replace('mystamp', model.config.get_config_value('Stamp'))

            fout.write(line)
        f.close()
        fout.close()

    def write_new_tar(self, model):
        super().write_tar(model)

    def write_bridge_multigraph(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/vitis_unified/myproject_bridge.cpp'))
        fout = open(f"{model.config.get_output_dir()}/{model.config.get_project_name()}_bridge.cpp", 'w')
        model_inputs = model.graphs[0].get_input_variables()
        model_outputs = model.graphs[-1].get_output_variables()
        model_brams = [var for var in model.graphs[0].get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        for line in f.readlines():
            newline = ''
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
            elif 'firmware/myproject' in line:
                for graph_idx, g in enumerate(model.graphs):
                    newline += '#undef DEFINES_H_\n'
                    if len(g.outputs) == 1:
                        newline += '#define result_t ' + 'result_graph' + str(graph_idx + 1) + '_t\n'
                    newline += line.replace('myproject', format(model.graphs[graph_idx].config.get_project_name()))
                    if len(g.outputs) == 1:
                        newline += (
                            'typedef result_graph' + str(graph_idx + 1) + '_t graph' + str(graph_idx + 1) + '_result_t;\n'
                        )
                        newline += '#undef result_t\n\n' if graph_idx < len(model.graphs) - 1 else '\n'
                newline += '\n'
            elif 'myproject' in line:
                newline = line.replace('myproject', format(model.config.get_project_name()))

            elif '// hls-fpga-machine-learning insert bram' in line:
                newline = line
                for bram in model_brams:
                    newline += f'#include \"firmware/weights/{bram.name}.h\"\n'

            elif '// hls-fpga-machine-learning insert header' in line:
                dtype = line.split('#', 1)[1].strip()
                inputs_str = ', '.join([f'{dtype} {i.name}[{i.size_cpp()}]' for i in model_inputs])
                outputs_str = ', '.join([f'{dtype} {o.name}[{o.size_cpp()}]' for o in model_outputs])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str + '\n'

            elif '// hls-fpga-machine-learning insert wrapper' in line:
                dtype = line.split('#', 1)[1].strip()
                newline = ''
                for inp in model_inputs:

                    if self.vitis_unified_config.isFreeInterimInput():
                        newline += indent + f"hls::stream<{inp.type.name}> {self.getWrapperPortName(inp, True)};\n"
                        newline += indent + "nnet::convert_data_pkt<{srcType}, {underlying_data_T}, {data_T}, {sz}>({inpRaw}, {inp_wrapper});\n".format(
                            srcType           =dtype,
                            underlying_data_T=inp.type.name,
                            data_T=self.get_axi_wrapper_type(inp),
                            sz=str(inp.size()),
                            inpRaw=inp.name,
                            inp_wrapper=self.getWrapperPortName(inp, True),
                        )
                    else:
                        newline += indent + f"hls::stream<{self.getDmaTypeName()}> " + self.getWrapperPortName(inp, True) + ";\n"
                        newline += indent + "nnet::convert_data<{dtype}, {dtype}, {sz}>({inpRaw}, {inp_wrapper});\n".format(
                            dtype=dtype,
                            sz=str(inp.size()),
                            inpRaw=inp.name,
                            inp_wrapper=self.getWrapperPortName(inp, True),
                        )

                    # newline += indent + '{var};\n'.format(var=i.definition_cpp(name_suffix='_ap'))
                    # newline += indent + 'nnet::convert_data<{}, {}, {}>({}, {}_ap);\n'.format(
                    #     dtype, i.type.name, i.size_cpp(), i.name, i.name
                    #)
                newline += '\n'


                for idx, g in enumerate(model.graphs):
                    for out in g.get_output_variables():
                        outStreamName = self.getWrapperPortName(out, False)
                        outStreamType = self.get_axi_wrapper_type(out) if self.vitis_unified_config.isFreeInterimOutput() else self.getDmaTypeName()
                        newline += indent + f"hls::stream<{outStreamType}> {outStreamName}(\"{outStreamName}\");\n"
                        # definition = o.definition_cpp(name_suffix='_ap')
                        # if len(g.outputs) == 1:
                        #     parts = definition.split(' ', 1)
                        #     datatype = 'graph' + str(idx + 1) + '_result_t'
                        #     if parts[0].startswith('hls::stream'):
                        #         modified_definition = 'hls::stream<' + datatype + '> ' + parts[1]
                        #     else:
                        #         modified_definition = datatype + ' ' + parts[1]
                        #     newline += indent + f"{modified_definition};\n"
                        # else:
                        #     newline += indent + f"{definition};\n"

                newline += '\n'

                top_level = ''
                myOutputNextInput = []
                #output_vars = ''
                for idx, g in enumerate(model.graphs):
                    # if idx == 0:
                    #     input_vars = ','.join([i.name + '_ap' for i in g.get_input_variables()])
                    # else:
                    #     input_vars = output_vars
                    # bram_vars = ','.join(
                    #     [b.name for b in [var for var in g.get_weight_variables() if var.storage.lower() == 'bram']]
                    # )
                    # output_vars = ','.join([o.name + '_ap' for o in g.get_output_variables()])
                    # # Concatenate the input, output, and bram variables. Filter out empty/null values
                    if idx == 0:
                        input_vars = [self.getWrapperPortName(inp, True) for inp in g.get_input_variables()]
                    else:
                        input_vars = myOutputNextInput.copy()

                    output_vars = [self.getWrapperPortName(out, False) for out in g.get_output_variables()]
                    myOutputNextInput = output_vars.copy()
                    bram_vars   = [b.name for b in [var for var in g.get_weight_variables() if var.storage.lower() == 'bram']]
                    allArgs = input_vars + output_vars + bram_vars
                    all_vars = ', '.join(allArgs)
                    top_level += indent + f"{g.config.get_project_name()}_axi({all_vars});\n"
                newline += top_level

                newline += '\n'

                for outIdx, o in enumerate(model_outputs):
                    # if len(model.graphs[-1].outputs) == 1:
                    #     newline += indent + 'nnet::convert_data<{}, {}, {}>({}_ap, {});\n'.format(
                    #         datatype, dtype, o.size_cpp(), o.name, o.name
                    #     )
                    # else:
                    #     newline += indent + 'nnet::convert_data<{}, {}, {}>({}_ap, {});\n'.format(
                    #         o.type.name, dtype, o.size_cpp(), o.name, o.name
                    #     )
                    if self.vitis_unified_config.isFreeInterimOutput():
                        newline += indent + (f"nnet::convert_data_pkt<{dtype}, {o.type.name}, "
                                             f"{self.get_outputSizeArrName(model)}[{str(outIdx)}]>"
                                             f"({self.getWrapperPortName(o, False)}, {o.name});\n")
                    else:
                        newline += indent + (f"nnet::convert_data<{dtype}, {dtype}, {str(o.size())}>"
                                             f"({self.getWrapperPortName(o, False)}, {o.name});\n")

            elif '// hls-fpga-machine-learning insert trace_outputs' in line:
                newline = ''
                for layer in model.get_layers():
                    func = layer.get_attr('function_cpp', None)
                    if func and model.config.trace_output and layer.get_attr('trace', False):
                        vars = layer.get_variables()
                        for var in vars:
                            newline += (
                                indent
                                + 'nnet::trace_outputs->insert(std::pair<std::string, void *>('
                                + f'"{layer.name}", (void *) malloc({var.size_cpp()} * element_size)));\n'
                            )

            elif '// hls-fpga-machine-learning insert namespace' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += indent + f'using namespace {namespace};\n'

            elif '// hls-fpga-machine-learning insert tb_input_writer' in line:
                funcs = [
                    ("float", "dump_tb_inputs_float"),
                    ("double", "dump_tb_inputs_double"),
                ]
                newline = ""
                for dtype, funcname in funcs:
                    newline += f'void {funcname}(\n'
                    newline += '    const char* output_path'
                    for inp in model_inputs:
                        newline += f',\n    {dtype} {inp.name}[{inp.size_cpp()}]'
                    newline += '\n) {\n\n'

                    for inp in model_inputs:
                        decl = inp.definition_cpp(name_suffix='_ap').strip()
                        ap = inp.name + "_ap"
                        if decl.startswith("hls::stream"):
                            newline += f'    {decl};\n'
                        else:
                            newline += f'    {inp.type.name} {ap}[{inp.size_cpp()}];\n'
                        newline += (
                            f'    nnet::convert_data<{dtype}, {inp.type.name}, {inp.size_cpp()}>' f'({inp.name}, {ap});\n'
                        )
                    newline += "\n"
                    newline += f'    std::ofstream fout(std::string(output_path) + "/{inp.name}_input_data.txt");\n'

                    for inp in model_inputs:
                        decl = inp.definition_cpp(name_suffix='_ap').strip()
                        dims = inp.shape

                        if decl.startswith("hls::stream"):
                            if len(dims) == 1:
                                N = dims[0]
                                newline += f'    for(int i = 0; i < {N}; i++) {{\n'
                                newline += f'        auto temp = {inp.name}_ap.read();\n'
                                newline += (
                                    f'        ap_uint<{inp.type.name}::value_type::width> bits = ' f'temp[0].range();\n'
                                )
                                newline += f'        fout << bits.to_uint()' f' << (i+1<{N} ? \' \' : \'\\n\');\n'
                                newline += '    }\n'
                            else:
                                inputs_list = model.nn_config['inputs']
                                fifo_depth = next((e['fifo_depth'] for e in inputs_list if e['name'] == inp.name), None)
                                batch_size = next((e['batch_size'] for e in inputs_list if e['name'] == inp.name), None)
                                newline += f'    for(int r = 0; r < {fifo_depth}; r++) {{\n'
                                newline += f'        auto temp = {inp.name}_ap.read();\n'
                                newline += f'        for(int c = 0; c < {batch_size}; c++) {{\n'
                                newline += (
                                    f'            ap_uint<{inp.type.name}::value_type::width> bits = ' f'temp[c].range();\n'
                                )
                                newline += (
                                    f'            fout << bits.to_uint()' f' << (c+1<{batch_size} ? \' \' : \'\\n\');\n'
                                )
                                newline += '        }\n'
                                newline += '    }\n'
                        else:
                            ap = inp.name + "_ap"
                            N = inp.size_cpp()
                            newline += f'    for(int i = 0; i < {N}; i++) {{\n'
                            newline += f'        ap_uint<{inp.type.name}::width> bits = ' f'{ap}[i].range();\n'
                            newline += f'        fout << bits.to_uint()' f' << (i+1<{N} ? \' \' : \'\\n\');\n'
                            newline += '    }\n'
                    newline += "    fout.close();\n"
                    newline += "}\n"
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_build_script(self, model):
        filedir = Path(__file__).parent
        ##### we require vitis unified not
        super().write_build_script(model)


        # build_prj.tcl (lagacy unused)
        srcpath = (filedir / '../templates/vitis_unified/build_prj.tcl').resolve()
        dstpath = f'{model.config.get_output_dir()}/build_prj.tcl'
        copyfile(srcpath, dstpath)

        #### we build 3 config file for hls_config.cfg/hls_config_cosim.cfg/hls_config_csim.cfg

        for configType in ["hls_config.cfg","hls_config_cosim.cfg","hls_config_csim.cfg"]:
            # hls_config.cfg
            srcpath = (filedir / '../templates/vitis_unified/hls_config.cfg').resolve()
            despath = f'{model.config.get_output_dir()}/{configType}'
            df      = open(despath, 'w')
            with open(srcpath, 'r') as sf:
                lines = sf.readlines()
            for line in lines:
                if "{PART}" in line:
                    line = line.replace("{PART}", self.vitis_unified_config.get_part())
                if "{CLK}" in line:
                    line = line.replace("{CLK}", model.config.get_config_value('ClockPeriod'))
                if "{OUTDIR}" in line:
                    line = line.replace("{OUTDIR}", model.config.get_output_dir())
                if "{CLK_UC}" in line:
                    line = line.replace("{CLK_UC}", model.config.get_config_value('ClockUncertainty', '12.5%'))
                if "{PRJ_NAME}" in line:
                    line = line.replace("{PRJ_NAME}", model.config.get_project_name())
                if "{TOP_NAME}" in line:
                    line = line.replace("{TOP_NAME}", model.config.get_project_name() + "_axi")
                if ("-DRTL_SIM" in line) and (configType != "hls_config_cosim.cfg"):
                    line = ""

                df.write(line)
            df.close()


    ##### override stitch multigraph
    def write_build_script_multigraph(self, model):
        """Write the build script (build_lib.sh) for stitched multigraph project
        Args:
            model (MultiModelGraph): the hls4ml multigraph model.
        """
        filedir = Path(__file__).parent
        os.makedirs(model.config.get_output_dir(), exist_ok=True)
        build_lib_src = (filedir / '../templates/vitis_unified/build_lib_multigraph.sh').resolve()
        build_lib_dst = Path(f'{model.config.get_output_dir()}/build_lib.sh').resolve()
        graph_project_names = ' '.join(f"\"{g.config.get_output_dir().split('/')[-1]}\"" for g in model.graphs)

        with open(build_lib_src) as src, open(build_lib_dst, 'w') as dst:
            for line in src.readlines():
                line = line.replace('myproject', model.config.config['OriginalProjectName'])
                line = line.replace('myproject_stitched', model.config.config['ProjectName'])
                line = line.replace('mystamp', model.config.config['Stamp'])
                line = line.replace('mygraph_name_list', graph_project_names)
                dst.write(line)
        os.chmod(build_lib_dst, os.stat(build_lib_dst).st_mode | stat.S_IEXEC)



    def write_hls(self, model, is_multigraph=False):

        from hls4ml.backends import VitisUnifiedConfig

        self.vitis_unified_config = VitisUnifiedConfig(
            model.config, model.get_input_variables(), model.get_output_variables()
        )
        super().write_hls(model, is_multigraph=is_multigraph)
        if not is_multigraph:
            self.write_board_script(model)
            self.write_driver(model)
            self.write_wrapper_test(model)
            self.write_axi_wrapper(model)
            self.modify_build_script(model)
            self.write_new_tar(model)
        else:
            self.write_bridge_multigraph(model)
            #self.modify_write_build_script_multigraph(model)



