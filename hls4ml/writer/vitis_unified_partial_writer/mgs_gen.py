import os
import shutil
import stat
from pathlib import Path
import math


from hls4ml.writer.vitis_unified_writer.meta import VitisUnifiedWriterMeta

#### This class is allowed for multigraph only
class VitisUnifiedPartial_MagicArchGen():

    ########### allow for
    @classmethod
    def convert_idx_to_io_name(cls, meta, mg, list_of_io_idx, multi_graph, gid, is_input, default_name: str):
        inp_axis_t, out_axis_t, inps, outs = meta.vitis_unified_config.get_corrected_types()
        sub_graph = multi_graph.graphs[gid]
        io_tensor = sub_graph.get_input_variables() if is_input else sub_graph.get_output_variables()
        result_name_list = []
        for io_idx in list_of_io_idx:
            if io_idx is None:
                result_name_list.append(default_name)
            else:
                io_name = mg.get_io_port_name(io_tensor[io_idx], is_input, io_idx)
                result_name_list.append(io_name)
        return result_name_list

    @classmethod
    def gen_vivado_project(cls, meta: VitisUnifiedWriterMeta, model, mg):
        filedir = os.path.dirname(os.path.abspath(__file__))

        vivado_project_des_folder_path = f'{model.config.get_output_dir()}/vivado_project'

        if os.path.exists(vivado_project_des_folder_path):
            shutil.rmtree(vivado_project_des_folder_path)

        os.makedirs(vivado_project_des_folder_path)

        board_name = meta.vitis_unified_config.get_board()

        #### copy project building script
        vivado_project_src_script_path = os.path.join(filedir,
                                                      f'../../templates/vitis_unified_partial/board_support/{board_name}/project_builder.tcl')
        des_project_script_path        = f'{vivado_project_des_folder_path}/project_builder.tcl'
        shutil.copyfile(vivado_project_src_script_path, des_project_script_path)

        #### copy mgs argument
        vivado_project_src_meta_arg_path = os.path.join(filedir,
                                                        f'../../templates/vitis_unified_partial/board_support/mga_meta.tcl')
        des_projectscript_meta_arg_path = f'{vivado_project_des_folder_path}/mga_meta.tcl'

        fin = open(vivado_project_src_meta_arg_path, 'r')
        fout = open(des_projectscript_meta_arg_path, 'w')

        mgs_buffer_meta_list    = meta.vitis_unified_config.get_mgs_meta_list()
        amt_subGraph = meta.vitis_unified_config.get_amt_graph()

        graph_idx_width = str(max(0, int(math.ceil(math.log2(amt_subGraph)))))

        for line in fin.readlines():
            if "HLS_CFG_AMT_MGS" in line:
                line = line.replace("VAL", str(len(mgs_buffer_meta_list)))
            if "HLS_CFG_BANK_IDX_WIDTH" in line:
                line = line.replace("VAL", graph_idx_width)
            if "HLS_CFG_MGS_WRAP_WIDTH" in line:
                width_list = ([str(meta.vitis_unified_config.get_dma_size())])
                width_list.extend([str(magic_buffer_meta.data_width) for magic_buffer_meta in mgs_buffer_meta_list])
                line = line.replace("VAL", "{"+ " ".join(width_list) + "}")
            if "HLS_CFG_MGS_M" in line:
                #### output side
                all_output_connect = []  #### each element is for each subgraph
                for gid in range(meta.vitis_unified_config.get_amt_graph()):
                    mgs_buffer_con_idx = meta.vitis_unified_config.get_mgs_mng().get_io_idx_for_all_mgs_buffer_with_dma(
                        gid, False)
                    output_connect_names = cls.convert_idx_to_io_name(meta, mg, mgs_buffer_con_idx, model, gid, False, "DUMMY")
                    output_connect_str = "{" + " ".join(output_connect_names) + "}"
                    all_output_connect.append(output_connect_str)

                line = line.replace("VAL", "{" + " ".join(all_output_connect) + "}")

            if "HLS_CFG_MGS_S" in line:
                #### input side
                all_input_connect = []  #### each element is for each subgraph
                for gid in range(meta.vitis_unified_config.get_amt_graph()):
                    mgs_buffer_con_idx = meta.vitis_unified_config.get_mgs_mng().get_io_idx_for_all_mgs_buffer_with_dma(
                        gid, True)
                    input_connect_names = cls.convert_idx_to_io_name(meta, mg, mgs_buffer_con_idx, model, gid, True, "DUMMY")
                    input_connect_str = "{" + " ".join(input_connect_names) + "}"
                    all_input_connect.append(input_connect_str)

                line = line.replace("VAL", "{" + " ".join(all_input_connect) + "}")

            if "HLS_CFG_HLS_SRC" in line:
                kernel_paths = [sub_graph.config.get_output_dir() + "/unifiedWorkspace" for sub_graph in model.graphs]
                line = line.replace("VAL", "{" + " ".join(kernel_paths) + "}")
            if "HLS_CFG_HLS_TOP_NAME" in line:
                kernel_topNames = [ mg.get_top_wrap_func_name(sub_graph) for sub_graph in model.graphs]
                line = line.replace("VAL", "{" + " ".join(kernel_topNames) + "}")


            fout.write(line)

        fin .close()
        fout.close()






    @classmethod
    def copyMagicArchIp(self, meta: VitisUnifiedWriterMeta, model):

        filedir = os.path.dirname(os.path.abspath(__file__))
        magic_arch_src_folder_path = os.path.join(filedir, '../../templates/vitis_unified_partial/ips')
        magic_arch_des_folder_path = f'{model.config.get_output_dir()}/ips'

        if os.path.exists(magic_arch_des_folder_path):
            shutil.rmtree(magic_arch_des_folder_path)
        shutil.copytree(magic_arch_src_folder_path, magic_arch_des_folder_path, dirs_exist_ok=True)

        ##### delete the magic streamer grp generated ip first
        magic_stream_grp_ip_path = f'{model.config.get_output_dir()}/ips/magic_streamer_grp_ip'

        if  os.path.exists(magic_stream_grp_ip_path):
            shutil.rmtree(magic_stream_grp_ip_path)
        os.makedirs(magic_stream_grp_ip_path)




    @classmethod
    def write_mgs(self, meta: VitisUnifiedWriterMeta, model):

        filedir = os.path.dirname(os.path.abspath(__file__))
        fin = open(os.path.join(filedir, '../../templates/vitis_unified_partial/ips/magic_streamer_grp_src/streamGrp.v'), 'r')

        fout = open(f'{model.config.get_output_dir()}/ips/magic_streamer_grp_src/streamGrp.v', 'w')

        mgs_buffer_meta_list = meta.vitis_unified_config.get_mgs_meta_list()
        ####     v------ mgs0                       v--------- mgs1
        #### [(data width, indexWidth, ....), (data width, indexWidth, ....)]

        for line in fin.readlines():
            newline = line

            if "// hls4ml-streamGrp-gen-parameter" in line:
                parameterList = []
                for idx, magic_buffer_meta in enumerate(mgs_buffer_meta_list):
                    parameterList.append(f"parameter DATA_WIDTH_{idx+1} = {magic_buffer_meta.data_width}")
                    parameterList.append(f"parameter STORAGE_IDX_WIDTH_{idx+1} = {magic_buffer_meta.row_idx_width}")

                parameterStr = ",\n".join(parameterList)
                newline += parameterStr + "\n"

            elif "// hls4ml-streamGrp-gen-io" in line:
                ioList = []
                for idx in range(1, len(mgs_buffer_meta_list) + 1):
                    ioList.append(f"//io for MGS{str(idx)}")
                    ioList.append(f"input  wire [DATA_WIDTH_{str(idx)} -1:0]              S{str(idx)}_AXI_TDATA" )
                    ioList.append(f"input  wire                                           S{str(idx)}_AXI_TVALID")
                    ioList.append(f"output wire                                           S{str(idx)}_AXI_TREADY")
                    ioList.append(f"input  wire                                           S{str(idx)}_AXI_TLAST" )
                    ioList.append(f"output wire [DATA_WIDTH_{str(idx)}-1:0]               M{str(idx)}_AXI_TDATA" )
                    ioList.append(f"output wire                                           M{str(idx)}_AXI_TVALID")
                    ioList.append(f"input  wire                                           M{str(idx)}_AXI_TREADY")
                    ioList.append(f"output wire                                           M{str(idx)}_AXI_TLAST" )
                    ioList.append("//----------debugging -------------")
                    ioList.append(f"output wire [4-1:0]                                   dbg_state_{str(idx)}")
                    ioList.append(f"output wire [(STORAGE_IDX_WIDTH_{idx}+1)-1:0]         dbg_amt_store_bytes_{str(idx)}")
                    ioList.append(f"output wire [(STORAGE_IDX_WIDTH_{idx}+1)-1:0]         dbg_amt_load_bytes_{str(idx)}")

                ioList.append("//--------- Pool commanding ------------")
                ioList.append("///// for load  [Reset/Init] of dma will be ignore")
                ioList.append("///// for store [Reset/Init] of dma will be ignore")
                ioList.append(f"input  wire[{len(mgs_buffer_meta_list)}: 0]                           storeReset")
                ioList.append(f"input  wire[{len(mgs_buffer_meta_list)}: 0]                           loadReset")
                ioList.append(f"input  wire[{len(mgs_buffer_meta_list)}: 0]                           storeInit")
                ioList.append(f"input  wire[{len(mgs_buffer_meta_list)}: 0]                           loadInit")
                ioList.append(f"output wire[{len(mgs_buffer_meta_list)}: 0]                           finStore")
                ioList.append(f"input  wire                                               finStoreProxyDma")

                newline += ",\n".join(ioList) + ",\n"

            elif "// hls4ml-streamGrp-gen-logic-assign" in line:
                newline += "assign finStore[0] = finStoreProxyDma;\n"


            elif "// hls4ml-streamGrp-gen-create-module" in line:


                for idx in range(1, len(mgs_buffer_meta_list)+1):
                    newline += f"//create module for MGS{str(idx)}\n"
                    newline += f"MagicStreammerCore #(\n"
                    newline += f"  .DATA_WIDTH(DATA_WIDTH_{idx}),\n"
                    newline += f"  .STORAGE_IDX_WIDTH(STORAGE_IDX_WIDTH_{idx})\n"
                    newline += f")\n"
                    newline += f"MGS{str(idx)} (\n"
                    newline += f"  .S_AXI_TDATA(S{str(idx)}_AXI_TDATA),\n"
                    newline += f"  .S_AXI_TVALID(S{str(idx)}_AXI_TVALID),\n"
                    newline += f"  .S_AXI_TREADY(S{str(idx)}_AXI_TREADY),\n"
                    newline += f"  .S_AXI_TLAST(S{str(idx)}_AXI_TLAST),\n"
                    newline += f"  .M_AXI_TDATA(M{str(idx)}_AXI_TDATA),\n"
                    newline += f"  .M_AXI_TVALID(M{str(idx)}_AXI_TVALID),\n"
                    newline += f"  .M_AXI_TREADY(M{str(idx)}_AXI_TREADY),\n"
                    newline += f"  .M_AXI_TLAST(M{str(idx)}_AXI_TLAST),\n"
                    newline += f"//------ command --------\n"
                    newline += f"  .storeReset (storeReset[{str(idx)}]),\n"
                    newline += f"  .loadReset  (loadReset [{str(idx)}]),\n"
                    newline += f"  .storeInit  (storeInit [{str(idx)}]),\n"
                    newline += f"  .loadInit   (loadInit  [{str(idx)}]),\n"
                    newline += f"  .finStore   (finStore  [{str(idx)}]),\n"
                    newline += f"//------ debugging ------\n"
                    newline += f"  .dbg_state(dbg_state_{str(idx)}),\n"
                    newline += f"  .dbg_amt_store_bytes(dbg_amt_store_bytes_{str(idx)}),\n"
                    newline += f"  .dbg_amt_load_bytes(dbg_amt_load_bytes_{str(idx)}),\n"
                    newline += f"  .clk(clk),\n"
                    newline += f"  .reset(nreset)\n"
                    newline += f");\n"



            fout.write(newline)