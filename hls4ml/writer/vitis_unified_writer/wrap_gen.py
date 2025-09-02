import os

from .meta import VitisUnifiedWriterMeta

# main function


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

        inp_gmem_t, out_gmem_t, inps, outs = meta.vitis_unified_config.get_corrected_types()
        indent = '      '

        # start write myproject_dm.cpp ##

        filedir = os.path.dirname(os.path.abspath(__file__))
        fin = open(os.path.join(filedir, '../../templates/vitis_unified/myproject_dm.cpp'))
        fout = open(f'{model.config.get_output_dir()}/firmware/myproject_dm.cpp', 'w')

        for line in fin.readlines():

            if "MY_PROJECT_DM_INC" in line:
                line = line.replace("MY_PROJECT_DM_INC", mg.get_wrapper_file_name(model))
            elif "MY_PROJECT_TOP_FUNC" in line:
                line = line.replace("MY_PROJECT_TOP_FUNC", mg.get_top_wrap_func_name(model))
            elif "STREAM_BUF_IN_SZ" in line:
                line = line.replace("VAL", str(meta.vitis_unified_config.get_in_stream_bufferSz()))
            elif "STREAM_BUF_OUT_SZ" in line:
                line = line.replace("VAL", str(meta.vitis_unified_config.get_out_stream_bufferSz()))

            elif "// vitis-unified-wrapper-io" in line:
                line = self.gen_io_str(mg, indent, inp_gmem_t, out_gmem_t, inps, outs) + "\n"
            elif "// vitis-unified-wrapper-interface" in line:
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
                for inp_idx, inp in enumerate(inps):
                    line += (
                        f"load_input({mg.get_io_port_name(inp, True, inp_idx)}, "
                        f"{mg.get_local_stream_name(inp, True, inp_idx)}, amtQuery, {str(inp.size())});\n"
                    )
            elif "// vitis-unified-wrapper-compute" in line:
                poolList = []
                for inp_idx, inp in enumerate(inps):
                    poolList.append(f"{mg.get_local_stream_name(inp, True, inp_idx)}")
                for out_idx, out in enumerate(outs):
                    poolList.append(f"{mg.get_local_stream_name(out, False, out_idx)}")
                joinedIo = f",\n{indent}{indent}{indent}".join(poolList)
                line += f"{indent} {mg.get_top_model_name(model)}({joinedIo});\n"

            elif "// vitis-unified-wrapper-store" in line:
                for out_idx, out in enumerate(outs):
                    line += (
                        f"store_result({mg.get_io_port_name(out, False, out_idx)}, "
                        f"{mg.get_local_stream_name(out, False, out_idx)}, amtQuery, {str(out.size())});\n"
                    )

            fout.write(line)

        fin.close()
        fout.close()

        #
        # start write myproject_dm.h

        filedir = os.path.dirname(os.path.abspath(__file__))
        fin = open(os.path.join(filedir, '../../templates/vitis_unified/myproject_dm.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/myproject_dm.h', 'w')

        for line in fin.readlines():

            if "FILENAME" in line:
                line = line.replace("FILENAME", mg.get_wrapper_file_name(model).upper())
            elif "MY_PROJECT_INC.h" in line:
                line = line.replace("MY_PROJECT_INC", mg.get_main_file_name(model))
            elif "MY_PROJECT_TOP_FUNC" in line:
                line = line.replace("MY_PROJECT_TOP_FUNC", mg.get_top_wrap_func_name(model))
            elif "// vitis-unified-wrapper-io" in line:
                line += self.gen_io_str(mg, indent, inp_gmem_t, out_gmem_t, inps, outs) + "\n"
            fout.write(line)

        fin.close()
        fout.close()
