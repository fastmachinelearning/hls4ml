import os

from .meta import VitisUnifiedWriterMeta


class VitisUnified_WrapperGen:

    @classmethod
    def gen_io_str(self, mg, indent, inp_gmem_t, out_gmem_t, inps, outs, meta=None):

        inputPtrList = []
        outputPtrList = []

        for inp_idx, inp in enumerate(inps):
            inputPtrList.append(f"{indent} {inp_gmem_t}* {mg.get_io_port_name(inp, True, inp_idx)}")

        for out_idx, out in enumerate(outs):
            outputPtrList.append(f"{indent} {out_gmem_t}* {mg.get_io_port_name(out, False, out_idx)}")

        line = ", ".join(inputPtrList) + ",\n"
        line += ", ".join(outputPtrList) + "\n"

        return line

    @classmethod
    def write_wrapper(self, meta: VitisUnifiedWriterMeta, model, mg):
        if mg.is_axi_master(meta):
            VitisUnified_WrapperGen.write_wrapper_axim(meta, model, mg)
        else:
            VitisUnified_WrapperGen.write_wrapper_axis(meta, model, mg)

    @classmethod
    def write_wrapper_axis(self, meta: VitisUnifiedWriterMeta, model, mg):

        inp_gmem_t, out_gmem_t, inps, outs = meta.vitis_unified_config.get_corrected_types()

        if len(inps) != 1 or len(outs) != 1:
            raise ValueError(
                f"AXIS wrapper requires exactly 1 input and 1 output port. Found {len(inps)} inputs and {len(outs)} outputs."
            )
        inp, out = inps[0], outs[0]
        indent = '      '

        # start write myproject_axis.cpp

        filedir = os.path.dirname(os.path.abspath(__file__))
        fin = open(os.path.join(filedir, '../../templates/vitis_unified/myproject_axis.cpp'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{mg.get_project_name(model)}_axis.cpp', 'w')

        for line in fin.readlines():
            if 'MY_PROJECT_TOP_FUNC' in line:
                newline = line.replace('MY_PROJECT_TOP_FUNC', mg.get_top_wrap_func_name(model, False))
            elif '// hls-fpga-machine-learning insert include' in line:
                newline = f'#include "{mg.get_project_name(model)}_axis.h"\n'
            elif '// hls-fpga-machine-learning insert interface' in line:
                newline = ''
                newline += indent + '#pragma HLS INTERFACE axis port=in\n'
                newline += indent + '#pragma HLS INTERFACE axis port=out\n'
                newline += indent + '#pragma HLS INTERFACE ap_ctrl_none port=return\n'
                newline += indent + '#pragma HLS DATAFLOW\n'
            elif '// hls-fpga-machine-learning insert local vars' in line:
                newline = ''
                newline += indent + 'bool is_last = false;\n'
                newline += indent + 'hls::stream<' + inp.type.name + '> in_local("input_1");\n'
                newline += indent + 'hls::stream<' + out.type.name + '> out_local("output_1");\n\n'
                newline += indent + '#pragma HLS STREAM variable=in_local depth={}\n'.format(
                    model.get_input_variables()[0].pragma[1]
                )
                newline += indent + '#pragma HLS STREAM variable=out_local depth={}\n'.format(
                    model.get_output_variables()[0].pragma[1]
                )
            elif '// hls-fpga-machine-learning insert enqueue' in line:
                newline = ''
                newline += indent + "/// enqueue input data\n"
                newline += indent + f'{mg.get_dma_type_name()} tmp;\n'
                newline += indent + 'for(unsigned i = 0; i < N_IN / {input_t}::size; ++i) {{\n'
                newline += indent + indent + '{input_t} ctype;\n'
                newline += indent + indent + 'for(unsigned j = 0; j < {input_t}::size; j++) {{\n'
                newline += indent + indent + indent + 'in.read(tmp);\n'
                newline += indent + indent + indent + 'ctype[j] = tmp.data;\n'
                newline += indent + indent + indent + 'is_last = tmp.last;\n'
                newline += indent + indent + '}}\n'
                newline += indent + indent + 'in_local.write(ctype);\n'
                newline += indent + '}}\n'
                newline += indent + 'tmp.last = 0;\n'
                newline = newline.format(input_t=inp.type.name)
            elif '// hls-fpga-machine-learning insert call' in line:
                newline = indent + f'{model.config.get_project_name()}(in_local, out_local);\n'
            elif '// hls-fpga-machine-learning insert dequeue' in line:

                newline = ''
                newline += indent + 'for(unsigned i = 0; i < N_OUT / {result_t}::size; ++i) {{\n'
                newline += indent + indent + '{result_t} ctype = out_local.read();\n'
                newline += indent + indent + 'for(unsigned j = 0; j < {result_t}::size; j++) {{\n'
                newline += indent + indent + indent + f'tmp.data = ({inp_gmem_t}) (ctype[j]);\n'
                newline += indent + indent + indent + 'if(is_last) {{tmp.last = (((i+1)*(j+1))==N_OUT);}}\n'
                newline += indent + indent + indent + 'out.write(tmp);\n'
                newline += indent + indent + '}}\n'
                newline += indent + '}}\n'
                newline = newline.format(result_t=out.type.name)
            else:
                newline = line
            fout.write(newline)

        # start write myproject_axis.h

        filedir = os.path.dirname(os.path.abspath(__file__))
        fin = open(os.path.join(filedir, '../../templates/vitis_unified/myproject_axis.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{mg.get_project_name(model)}_axis.h', 'w')

        for line in fin.readlines():

            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
            elif '// hls-fpga-machine-learning insert include' in line:
                newline = f'#include "{model.config.get_project_name()}.h"\n'
                newline += '#include "ap_axi_sdata.h"\n'
            elif 'MY_PROJECT_TOP_FUNC' in line:
                newline = line.replace('MY_PROJECT_TOP_FUNC', mg.get_top_wrap_func_name(model, False))
            elif '// hls-fpga-machine-learning insert definitions' in line:
                newline = ''
                newline += f'static const unsigned N_IN = {inp.size()};\n'
                newline += f'static const unsigned N_OUT = {out.size()};\n'
                newline += f'typedef hls::axis<{inp_gmem_t}, 0, 0, 0> {mg.get_dma_type_name()};\n'
            else:
                newline = line
            fout.write(newline)

        fin.close()
        fout.close()

    @classmethod
    def write_wrapper_axim(self, meta: VitisUnifiedWriterMeta, model, mg):

        inp_gmem_t, out_gmem_t, inps, outs = meta.vitis_unified_config.get_corrected_types()
        indent = '      '

        # start write myproject_axim.cpp

        filedir = os.path.dirname(os.path.abspath(__file__))
        fin = open(os.path.join(filedir, f'../../templates/vitis_unified/{mg.get_project_name(model)}_axim.cpp'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{mg.get_wrapper_file_name(model, True)}.cpp', 'w')

        for line in fin.readlines():

            if "MY_PROJECT_DM_INC" in line:
                line = line.replace("MY_PROJECT_DM_INC", mg.get_wrapper_file_name(model, True))
            elif "MY_PROJECT_TOP_FUNC" in line:
                line = line.replace("MY_PROJECT_TOP_FUNC", mg.get_top_wrap_func_name(model, True))
            elif "STREAM_BUF_IN_SZ" in line:
                line = line.replace("VAL", str(meta.vitis_unified_config.get_in_stream_bufferSz()))
            elif "STREAM_BUF_OUT_SZ" in line:
                line = line.replace("VAL", str(meta.vitis_unified_config.get_out_stream_bufferSz()))

            elif "// vitis-unified-wrapper-io" in line:
                line = self.gen_io_str(mg, indent, inp_gmem_t, out_gmem_t, inps, outs) + "\n"
            elif "// vitis-unified-wrapper-interface" in line:
                # This section will generate the pragma to specify interface type
                # --> axi master (memory read input)
                # --> axi master (memory write output)
                # BOTH IS MASTER
                # Please note that gmem_in/out depth size must match with the cosim array allocation
                # if the cosim allocation is larger than depth, the result will not correct
                # if the cosim allocation is lower than depth, the result is correct,
                # but the system will throw segment falut error
                # the depth size will not impact the resource usage in hls generation
                for inp_idx, inp in enumerate(inps):
                    line += (
                        f"#pragma HLS INTERFACE m_axi     port={mg.get_io_port_name(inp, True, inp_idx)} "
                        f"bundle = gmem_in{inp_idx} depth={str(inp.size())}\n"
                    )
                for out_idx, out in enumerate(outs):
                    line += (
                        f"#pragma HLS INTERFACE m_axi     port={mg.get_io_port_name(out, False, out_idx)} "
                        f"bundle = gmem_out{out_idx} depth={str(out.size())}\n"
                    )
            elif "// vitis-unified-wrapper-stream-dec" in line:
                # this declare the stream buffer that axi master read will store the input  and axi master write
                # will retrieve the output
                for inp_idx, inp in enumerate(inps):
                    line += f"{indent} static hls::stream<{inp.type.name}> {mg.get_local_stream_name(inp, True, inp_idx)};\n"
                for out_idx, out in enumerate(outs):
                    line += (
                        f"{indent} static hls::stream<{out.type.name}> {mg.get_local_stream_name(out, False, out_idx)};\n"
                    )

            elif "// vitis-unified-wrapper-stream-config" in line:
                for inp_idx, inp in enumerate(inps):
                    line += (
                        f"#pragma HLS STREAM variable={mg.get_local_stream_name(inp, True, inp_idx)} "
                        f"depth=STREAM_BUF_IN_SZ\n"
                    )
                for out_idx, out in enumerate(outs):
                    line += (
                        f"#pragma HLS STREAM variable={mg.get_local_stream_name(out, False, out_idx)} "
                        f"depth=STREAM_BUF_OUT_SZ\n"
                    )

            elif "// vitis-unified-wrapper-load" in line:
                # this call the load_input function to  convert axi_master read to axi stream (buffer)
                for inp_idx, inp in enumerate(inps):
                    line += (
                        f"load_input({mg.get_io_port_name(inp, True, inp_idx)}, "
                        f"{mg.get_local_stream_name(inp, True, inp_idx)}, amtQuery, {str(inp.size())});\n"
                    )
            elif "// vitis-unified-wrapper-compute" in line:
                #
                poolList = []
                for inp_idx, inp in enumerate(inps):
                    poolList.append(f"{mg.get_local_stream_name(inp, True, inp_idx)}")
                for out_idx, out in enumerate(outs):
                    poolList.append(f"{mg.get_local_stream_name(out, False, out_idx)}")
                joinedIo = f",\n{indent}{indent}{indent}".join(poolList)
                line += f"{indent} {mg.get_top_model_name(model)}({joinedIo});\n"

            elif "// vitis-unified-wrapper-store" in line:
                # this call the store_result function to convert axi_master read to axi stream (buffer)
                for out_idx, out in enumerate(outs):
                    line += (
                        f"store_result({mg.get_io_port_name(out, False, out_idx)}, "
                        f"{mg.get_local_stream_name(out, False, out_idx)}, amtQuery, {str(out.size())});\n"
                    )

            fout.write(line)

        fin.close()
        fout.close()

        # start write myproject_axim.h

        filedir = os.path.dirname(os.path.abspath(__file__))
        fin = open(os.path.join(filedir, '../../templates/vitis_unified/myproject_axim.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{mg.get_wrapper_file_name(model, True)}.h', 'w')

        for line in fin.readlines():

            if "FILENAME" in line:
                line = line.replace("FILENAME", mg.get_wrapper_file_name(model, True).upper())
            elif "MY_PROJECT_INC.h" in line:
                line = line.replace("MY_PROJECT_INC", mg.get_main_file_name(model))
            elif "MY_PROJECT_TOP_FUNC" in line:
                line = line.replace("MY_PROJECT_TOP_FUNC", mg.get_top_wrap_func_name(model, True))
            elif "// vitis-unified-wrapper-io" in line:
                line += self.gen_io_str(mg, indent, inp_gmem_t, out_gmem_t, inps, outs) + "\n"
            fout.write(line)

        fin.close()
        fout.close()
