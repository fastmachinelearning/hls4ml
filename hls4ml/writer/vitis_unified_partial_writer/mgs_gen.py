import os
import shutil
import stat
from pathlib import Path


from hls4ml.writer.vitis_unified_writer.meta import VitisUnifiedWriterMeta

class VitisUnifiedPartial_MagicArchGen():


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

        metaList = meta.vitis_unified_config.get_mgs_meta_list()
        ####     v------ mgs0                       v--------- mgs1
        #### [(data width, indexWidth, ....), (data width, indexWidth, ....)]

        for line in fin.readlines():
            newline = line

            if "// hls4ml-streamGrp-gen-parameter" in line:
                parameterList = []
                for idx, (data_width, index_width, *_) in enumerate(metaList):
                    parameterList.append(f"parameter DATA_WIDTH_{idx+1} = {data_width}")
                    parameterList.append(f"parameter STORAGE_IDX_WIDTH_{idx+1} = {index_width}")

                parameterStr = ",\n".join(parameterList)
                newline += parameterStr + "\n"

            elif "// hls4ml-streamGrp-gen-io" in line:
                ioList = []
                for idx in range(1, len(metaList) + 1):
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
                ioList.append(f"input  wire[{len(metaList)}: 0]                           storeReset")
                ioList.append(f"input  wire[{len(metaList)}: 0]                           loadReset")
                ioList.append(f"input  wire[{len(metaList)}: 0]                           storeInit")
                ioList.append(f"input  wire[{len(metaList)}: 0]                           loadInit")
                ioList.append(f"output wire[{len(metaList)}: 0]                           finStore")
                ioList.append(f"input  wire                                               finStoreProxyDma")

                newline += ",\n".join(ioList) + ",\n"

            elif "// hls4ml-streamGrp-gen-logic-assign" in line:
                newline += "assign finStore[0] = finStoreProxyDma;\n"


            elif "// hls4ml-streamGrp-gen-create-module" in line:


                for idx in range(1, len(metaList)+1):
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