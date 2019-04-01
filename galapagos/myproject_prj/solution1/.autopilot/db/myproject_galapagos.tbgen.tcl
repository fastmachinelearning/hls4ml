set C_TypeInfoList {{ 
"myproject_galapagos" : [[], { "return": [[], "void"]} , [{"ExternC" : 0}], [ {"in": [[],{ "pointer": "0"}] }, {"out": [[],{ "pointer": "0"}] }],[],""], 
"0": [ "stream<galapagos_packet>", {"hls_type": {"stream": [[[[],"1"]],"2"]}}], 
"1": [ "galapagos_packet", {"struct": [[],[],[{ "data": [[], "3"]},{ "dest": [[], "4"]},{ "last": [[], "5"]},{ "id": [[], "4"]},{ "user": [[], "6"]},{ "keep": [[], "4"]}],""]}], 
"3": [ "ap_uint<64>", {"hls_type": {"ap_uint": [[[[], {"scalar": { "int": 64}}]],""]}}], 
"4": [ "ap_uint<8>", {"hls_type": {"ap_uint": [[[[], {"scalar": { "int": 8}}]],""]}}], 
"5": [ "ap_uint<1>", {"hls_type": {"ap_uint": [[[[], {"scalar": { "int": 1}}]],""]}}], 
"6": [ "ap_uint<40>", {"hls_type": {"ap_uint": [[[[], {"scalar": { "int": 40}}]],""]}}],
"2": ["hls", ""]
}}
set moduleName myproject_galapagos
set isTaskLevelControl 1
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set C_modelName {myproject_galapagos}
set C_modelType { void 0 }
set C_modelArgList {
	{ in_V_data_V int 64 regular {fifo 0 volatile }  }
	{ in_V_dest_V int 8 regular {fifo 0 volatile }  }
	{ in_V_last_V int 1 regular {fifo 0 volatile }  }
	{ in_V_id_V int 8 regular {fifo 0 volatile }  }
	{ in_V_user_V int 40 regular {fifo 0 volatile }  }
	{ in_V_keep_V int 8 regular {fifo 0 volatile }  }
	{ out_V_data_V int 64 regular {fifo 1 volatile }  }
	{ out_V_dest_V int 8 regular {fifo 1 volatile }  }
	{ out_V_last_V int 1 regular {fifo 1 volatile }  }
	{ out_V_id_V int 8 regular {fifo 1 volatile }  }
	{ out_V_user_V int 40 regular {fifo 1 volatile }  }
	{ out_V_keep_V int 8 regular {fifo 1 volatile }  }
}
set C_modelArgMapList {[ 
	{ "Name" : "in_V_data_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "READONLY", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "in.V.data.V","cData": "uint64","bit_use": { "low": 0,"up": 63},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "in_V_dest_V", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "in.V.dest.V","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "in_V_last_V", "interface" : "fifo", "bitwidth" : 1, "direction" : "READONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "in.V.last.V","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "in_V_id_V", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "in.V.id.V","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "in_V_user_V", "interface" : "fifo", "bitwidth" : 40, "direction" : "READONLY", "bitSlice":[{"low":0,"up":39,"cElement": [{"cName": "in.V.user.V","cData": "uint40","bit_use": { "low": 0,"up": 39},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "in_V_keep_V", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "in.V.keep.V","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "out_V_data_V", "interface" : "fifo", "bitwidth" : 64, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":63,"cElement": [{"cName": "out.V.data.V","cData": "uint64","bit_use": { "low": 0,"up": 63},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "out_V_dest_V", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "out.V.dest.V","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "out_V_last_V", "interface" : "fifo", "bitwidth" : 1, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":0,"cElement": [{"cName": "out.V.last.V","cData": "uint1","bit_use": { "low": 0,"up": 0},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "out_V_id_V", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "out.V.id.V","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "out_V_user_V", "interface" : "fifo", "bitwidth" : 40, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":39,"cElement": [{"cName": "out.V.user.V","cData": "uint40","bit_use": { "low": 0,"up": 39},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "out_V_keep_V", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":7,"cElement": [{"cName": "out.V.keep.V","cData": "uint8","bit_use": { "low": 0,"up": 7},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} ]}
# RTL Port declarations: 
set portNum 42
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ in_V_data_V_dout sc_in sc_lv 64 signal 0 } 
	{ in_V_data_V_empty_n sc_in sc_logic 1 signal 0 } 
	{ in_V_data_V_read sc_out sc_logic 1 signal 0 } 
	{ in_V_dest_V_dout sc_in sc_lv 8 signal 1 } 
	{ in_V_dest_V_empty_n sc_in sc_logic 1 signal 1 } 
	{ in_V_dest_V_read sc_out sc_logic 1 signal 1 } 
	{ in_V_last_V_dout sc_in sc_lv 1 signal 2 } 
	{ in_V_last_V_empty_n sc_in sc_logic 1 signal 2 } 
	{ in_V_last_V_read sc_out sc_logic 1 signal 2 } 
	{ in_V_id_V_dout sc_in sc_lv 8 signal 3 } 
	{ in_V_id_V_empty_n sc_in sc_logic 1 signal 3 } 
	{ in_V_id_V_read sc_out sc_logic 1 signal 3 } 
	{ in_V_user_V_dout sc_in sc_lv 40 signal 4 } 
	{ in_V_user_V_empty_n sc_in sc_logic 1 signal 4 } 
	{ in_V_user_V_read sc_out sc_logic 1 signal 4 } 
	{ in_V_keep_V_dout sc_in sc_lv 8 signal 5 } 
	{ in_V_keep_V_empty_n sc_in sc_logic 1 signal 5 } 
	{ in_V_keep_V_read sc_out sc_logic 1 signal 5 } 
	{ out_V_data_V_din sc_out sc_lv 64 signal 6 } 
	{ out_V_data_V_full_n sc_in sc_logic 1 signal 6 } 
	{ out_V_data_V_write sc_out sc_logic 1 signal 6 } 
	{ out_V_dest_V_din sc_out sc_lv 8 signal 7 } 
	{ out_V_dest_V_full_n sc_in sc_logic 1 signal 7 } 
	{ out_V_dest_V_write sc_out sc_logic 1 signal 7 } 
	{ out_V_last_V_din sc_out sc_lv 1 signal 8 } 
	{ out_V_last_V_full_n sc_in sc_logic 1 signal 8 } 
	{ out_V_last_V_write sc_out sc_logic 1 signal 8 } 
	{ out_V_id_V_din sc_out sc_lv 8 signal 9 } 
	{ out_V_id_V_full_n sc_in sc_logic 1 signal 9 } 
	{ out_V_id_V_write sc_out sc_logic 1 signal 9 } 
	{ out_V_user_V_din sc_out sc_lv 40 signal 10 } 
	{ out_V_user_V_full_n sc_in sc_logic 1 signal 10 } 
	{ out_V_user_V_write sc_out sc_logic 1 signal 10 } 
	{ out_V_keep_V_din sc_out sc_lv 8 signal 11 } 
	{ out_V_keep_V_full_n sc_in sc_logic 1 signal 11 } 
	{ out_V_keep_V_write sc_out sc_logic 1 signal 11 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "in_V_data_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "in_V_data_V", "role": "dout" }} , 
 	{ "name": "in_V_data_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "in_V_data_V", "role": "empty_n" }} , 
 	{ "name": "in_V_data_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "in_V_data_V", "role": "read" }} , 
 	{ "name": "in_V_dest_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "in_V_dest_V", "role": "dout" }} , 
 	{ "name": "in_V_dest_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "in_V_dest_V", "role": "empty_n" }} , 
 	{ "name": "in_V_dest_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "in_V_dest_V", "role": "read" }} , 
 	{ "name": "in_V_last_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "in_V_last_V", "role": "dout" }} , 
 	{ "name": "in_V_last_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "in_V_last_V", "role": "empty_n" }} , 
 	{ "name": "in_V_last_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "in_V_last_V", "role": "read" }} , 
 	{ "name": "in_V_id_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "in_V_id_V", "role": "dout" }} , 
 	{ "name": "in_V_id_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "in_V_id_V", "role": "empty_n" }} , 
 	{ "name": "in_V_id_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "in_V_id_V", "role": "read" }} , 
 	{ "name": "in_V_user_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":40, "type": "signal", "bundle":{"name": "in_V_user_V", "role": "dout" }} , 
 	{ "name": "in_V_user_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "in_V_user_V", "role": "empty_n" }} , 
 	{ "name": "in_V_user_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "in_V_user_V", "role": "read" }} , 
 	{ "name": "in_V_keep_V_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "in_V_keep_V", "role": "dout" }} , 
 	{ "name": "in_V_keep_V_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "in_V_keep_V", "role": "empty_n" }} , 
 	{ "name": "in_V_keep_V_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "in_V_keep_V", "role": "read" }} , 
 	{ "name": "out_V_data_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "out_V_data_V", "role": "din" }} , 
 	{ "name": "out_V_data_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "out_V_data_V", "role": "full_n" }} , 
 	{ "name": "out_V_data_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "out_V_data_V", "role": "write" }} , 
 	{ "name": "out_V_dest_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "out_V_dest_V", "role": "din" }} , 
 	{ "name": "out_V_dest_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "out_V_dest_V", "role": "full_n" }} , 
 	{ "name": "out_V_dest_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "out_V_dest_V", "role": "write" }} , 
 	{ "name": "out_V_last_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "out_V_last_V", "role": "din" }} , 
 	{ "name": "out_V_last_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "out_V_last_V", "role": "full_n" }} , 
 	{ "name": "out_V_last_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "out_V_last_V", "role": "write" }} , 
 	{ "name": "out_V_id_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "out_V_id_V", "role": "din" }} , 
 	{ "name": "out_V_id_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "out_V_id_V", "role": "full_n" }} , 
 	{ "name": "out_V_id_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "out_V_id_V", "role": "write" }} , 
 	{ "name": "out_V_user_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":40, "type": "signal", "bundle":{"name": "out_V_user_V", "role": "din" }} , 
 	{ "name": "out_V_user_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "out_V_user_V", "role": "full_n" }} , 
 	{ "name": "out_V_user_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "out_V_user_V", "role": "write" }} , 
 	{ "name": "out_V_keep_V_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "out_V_keep_V", "role": "din" }} , 
 	{ "name": "out_V_keep_V_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "out_V_keep_V", "role": "full_n" }} , 
 	{ "name": "out_V_keep_V_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "out_V_keep_V", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1"],
		"CDFG" : "myproject_galapagos",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "115", "EstimateLatencyMax" : "115",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "in_V_data_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "in_V_data_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "in_V_dest_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "in_V_dest_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "in_V_last_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "in_V_last_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "in_V_id_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "in_V_id_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "in_V_user_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "in_V_user_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "in_V_keep_V", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "in_V_keep_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "out_V_data_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "out_V_data_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "out_V_dest_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "out_V_dest_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "out_V_last_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "out_V_last_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "out_V_id_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "out_V_id_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "out_V_user_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "out_V_user_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "out_V_keep_V", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "out_V_keep_V_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "exp_table2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_myproject_fu_216", "Port" : "exp_table2"}]},
			{"Name" : "invert_table3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_myproject_fu_216", "Port" : "invert_table3"}]}],
		"SubInstanceBlock" : [
			{"SubInstance" : "grp_myproject_fu_216", "SubBlockPort" : ["data_V_blk_n"]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216", "Parent" : "0", "Child" : ["2", "129", "202", "205"],
		"CDFG" : "myproject",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "14", "EstimateLatencyMin" : "14", "EstimateLatencyMax" : "14",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "data_V", "Type" : "Vld", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "data_V_blk_n", "Type" : "RtlPort"}]},
			{"Name" : "exp_table2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "202", "SubInstance" : "grp_softmax_fu_58", "Port" : "exp_table2"}]},
			{"Name" : "invert_table3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "202", "SubInstance" : "grp_softmax_fu_58", "Port" : "invert_table3"}]}]},
	{"ID" : "2", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36", "Parent" : "1", "Child" : ["3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122", "123", "124", "125", "126", "127", "128"],
		"CDFG" : "conv_1d_0_0_0_0",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "3", "EstimateLatencyMin" : "3", "EstimateLatencyMax" : "3",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "1",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "data_V_read", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "3", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U1", "Parent" : "2"},
	{"ID" : "4", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U2", "Parent" : "2"},
	{"ID" : "5", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U3", "Parent" : "2"},
	{"ID" : "6", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U4", "Parent" : "2"},
	{"ID" : "7", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U5", "Parent" : "2"},
	{"ID" : "8", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapageOg_U6", "Parent" : "2"},
	{"ID" : "9", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U7", "Parent" : "2"},
	{"ID" : "10", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagfYi_U8", "Parent" : "2"},
	{"ID" : "11", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagg8j_U9", "Parent" : "2"},
	{"ID" : "12", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U10", "Parent" : "2"},
	{"ID" : "13", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U11", "Parent" : "2"},
	{"ID" : "14", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U12", "Parent" : "2"},
	{"ID" : "15", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U13", "Parent" : "2"},
	{"ID" : "16", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U14", "Parent" : "2"},
	{"ID" : "17", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U15", "Parent" : "2"},
	{"ID" : "18", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U16", "Parent" : "2"},
	{"ID" : "19", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U17", "Parent" : "2"},
	{"ID" : "20", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagfYi_U18", "Parent" : "2"},
	{"ID" : "21", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagfYi_U19", "Parent" : "2"},
	{"ID" : "22", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U20", "Parent" : "2"},
	{"ID" : "23", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagfYi_U21", "Parent" : "2"},
	{"ID" : "24", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U22", "Parent" : "2"},
	{"ID" : "25", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U23", "Parent" : "2"},
	{"ID" : "26", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U24", "Parent" : "2"},
	{"ID" : "27", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U25", "Parent" : "2"},
	{"ID" : "28", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U26", "Parent" : "2"},
	{"ID" : "29", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagjbC_U27", "Parent" : "2"},
	{"ID" : "30", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U28", "Parent" : "2"},
	{"ID" : "31", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U29", "Parent" : "2"},
	{"ID" : "32", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U30", "Parent" : "2"},
	{"ID" : "33", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagjbC_U31", "Parent" : "2"},
	{"ID" : "34", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagfYi_U32", "Parent" : "2"},
	{"ID" : "35", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U33", "Parent" : "2"},
	{"ID" : "36", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagkbM_U34", "Parent" : "2"},
	{"ID" : "37", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U35", "Parent" : "2"},
	{"ID" : "38", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapageOg_U36", "Parent" : "2"},
	{"ID" : "39", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapageOg_U37", "Parent" : "2"},
	{"ID" : "40", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U38", "Parent" : "2"},
	{"ID" : "41", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U39", "Parent" : "2"},
	{"ID" : "42", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagfYi_U40", "Parent" : "2"},
	{"ID" : "43", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U41", "Parent" : "2"},
	{"ID" : "44", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U42", "Parent" : "2"},
	{"ID" : "45", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U43", "Parent" : "2"},
	{"ID" : "46", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagjbC_U44", "Parent" : "2"},
	{"ID" : "47", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U45", "Parent" : "2"},
	{"ID" : "48", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U46", "Parent" : "2"},
	{"ID" : "49", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U47", "Parent" : "2"},
	{"ID" : "50", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U48", "Parent" : "2"},
	{"ID" : "51", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U49", "Parent" : "2"},
	{"ID" : "52", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U50", "Parent" : "2"},
	{"ID" : "53", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U51", "Parent" : "2"},
	{"ID" : "54", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U52", "Parent" : "2"},
	{"ID" : "55", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U53", "Parent" : "2"},
	{"ID" : "56", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U54", "Parent" : "2"},
	{"ID" : "57", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapageOg_U55", "Parent" : "2"},
	{"ID" : "58", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U56", "Parent" : "2"},
	{"ID" : "59", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U57", "Parent" : "2"},
	{"ID" : "60", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U58", "Parent" : "2"},
	{"ID" : "61", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagkbM_U59", "Parent" : "2"},
	{"ID" : "62", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U60", "Parent" : "2"},
	{"ID" : "63", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U61", "Parent" : "2"},
	{"ID" : "64", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U62", "Parent" : "2"},
	{"ID" : "65", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U63", "Parent" : "2"},
	{"ID" : "66", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U64", "Parent" : "2"},
	{"ID" : "67", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagjbC_U65", "Parent" : "2"},
	{"ID" : "68", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U66", "Parent" : "2"},
	{"ID" : "69", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U67", "Parent" : "2"},
	{"ID" : "70", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U68", "Parent" : "2"},
	{"ID" : "71", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagfYi_U69", "Parent" : "2"},
	{"ID" : "72", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U70", "Parent" : "2"},
	{"ID" : "73", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U71", "Parent" : "2"},
	{"ID" : "74", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U72", "Parent" : "2"},
	{"ID" : "75", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U73", "Parent" : "2"},
	{"ID" : "76", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U74", "Parent" : "2"},
	{"ID" : "77", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U75", "Parent" : "2"},
	{"ID" : "78", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U76", "Parent" : "2"},
	{"ID" : "79", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U77", "Parent" : "2"},
	{"ID" : "80", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U78", "Parent" : "2"},
	{"ID" : "81", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U79", "Parent" : "2"},
	{"ID" : "82", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U80", "Parent" : "2"},
	{"ID" : "83", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U81", "Parent" : "2"},
	{"ID" : "84", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagfYi_U82", "Parent" : "2"},
	{"ID" : "85", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagg8j_U83", "Parent" : "2"},
	{"ID" : "86", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U84", "Parent" : "2"},
	{"ID" : "87", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U85", "Parent" : "2"},
	{"ID" : "88", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U86", "Parent" : "2"},
	{"ID" : "89", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U87", "Parent" : "2"},
	{"ID" : "90", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U88", "Parent" : "2"},
	{"ID" : "91", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U89", "Parent" : "2"},
	{"ID" : "92", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U90", "Parent" : "2"},
	{"ID" : "93", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U91", "Parent" : "2"},
	{"ID" : "94", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagfYi_U92", "Parent" : "2"},
	{"ID" : "95", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U93", "Parent" : "2"},
	{"ID" : "96", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U94", "Parent" : "2"},
	{"ID" : "97", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U95", "Parent" : "2"},
	{"ID" : "98", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U96", "Parent" : "2"},
	{"ID" : "99", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U97", "Parent" : "2"},
	{"ID" : "100", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U98", "Parent" : "2"},
	{"ID" : "101", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagkbM_U99", "Parent" : "2"},
	{"ID" : "102", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U100", "Parent" : "2"},
	{"ID" : "103", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U101", "Parent" : "2"},
	{"ID" : "104", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U102", "Parent" : "2"},
	{"ID" : "105", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U103", "Parent" : "2"},
	{"ID" : "106", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapageOg_U104", "Parent" : "2"},
	{"ID" : "107", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U105", "Parent" : "2"},
	{"ID" : "108", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U106", "Parent" : "2"},
	{"ID" : "109", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U107", "Parent" : "2"},
	{"ID" : "110", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U108", "Parent" : "2"},
	{"ID" : "111", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U109", "Parent" : "2"},
	{"ID" : "112", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagjbC_U110", "Parent" : "2"},
	{"ID" : "113", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U111", "Parent" : "2"},
	{"ID" : "114", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U112", "Parent" : "2"},
	{"ID" : "115", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapageOg_U113", "Parent" : "2"},
	{"ID" : "116", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U114", "Parent" : "2"},
	{"ID" : "117", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapageOg_U115", "Parent" : "2"},
	{"ID" : "118", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U116", "Parent" : "2"},
	{"ID" : "119", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U117", "Parent" : "2"},
	{"ID" : "120", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U118", "Parent" : "2"},
	{"ID" : "121", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U119", "Parent" : "2"},
	{"ID" : "122", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U120", "Parent" : "2"},
	{"ID" : "123", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U121", "Parent" : "2"},
	{"ID" : "124", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U122", "Parent" : "2"},
	{"ID" : "125", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagjbC_U123", "Parent" : "2"},
	{"ID" : "126", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U124", "Parent" : "2"},
	{"ID" : "127", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U125", "Parent" : "2"},
	{"ID" : "128", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagg8j_U126", "Parent" : "2"},
	{"ID" : "129", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42", "Parent" : "1", "Child" : ["130", "131", "132", "133", "134", "135", "136", "137", "138", "139", "140", "141", "142", "143", "144", "145", "146", "147", "148", "149", "150", "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165", "166", "167", "168", "169", "170", "171", "172", "173", "174", "175", "176", "177", "178", "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189", "190", "191", "192", "193", "194", "195", "196", "197", "198", "199", "200", "201"],
		"CDFG" : "compute_layer_0_0_0_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "0", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "2", "EstimateLatencyMin" : "2", "EstimateLatencyMax" : "2",
		"Combinational" : "0",
		"Datapath" : "1",
		"ClockEnable" : "1",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "data_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_4_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_5_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_6_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_7_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_8_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_9_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_10_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_11_V_read", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "130", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagmb6_U151", "Parent" : "129"},
	{"ID" : "131", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagncg_U152", "Parent" : "129"},
	{"ID" : "132", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U153", "Parent" : "129"},
	{"ID" : "133", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U154", "Parent" : "129"},
	{"ID" : "134", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U155", "Parent" : "129"},
	{"ID" : "135", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U156", "Parent" : "129"},
	{"ID" : "136", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagqcK_U157", "Parent" : "129"},
	{"ID" : "137", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagqcK_U158", "Parent" : "129"},
	{"ID" : "138", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U159", "Parent" : "129"},
	{"ID" : "139", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U160", "Parent" : "129"},
	{"ID" : "140", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U161", "Parent" : "129"},
	{"ID" : "141", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagncg_U162", "Parent" : "129"},
	{"ID" : "142", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U163", "Parent" : "129"},
	{"ID" : "143", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U164", "Parent" : "129"},
	{"ID" : "144", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagsc4_U165", "Parent" : "129"},
	{"ID" : "145", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagtde_U166", "Parent" : "129"},
	{"ID" : "146", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U167", "Parent" : "129"},
	{"ID" : "147", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U168", "Parent" : "129"},
	{"ID" : "148", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagmb6_U169", "Parent" : "129"},
	{"ID" : "149", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagtde_U170", "Parent" : "129"},
	{"ID" : "150", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagncg_U171", "Parent" : "129"},
	{"ID" : "151", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U172", "Parent" : "129"},
	{"ID" : "152", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U173", "Parent" : "129"},
	{"ID" : "153", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U174", "Parent" : "129"},
	{"ID" : "154", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagncg_U175", "Parent" : "129"},
	{"ID" : "155", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagudo_U176", "Parent" : "129"},
	{"ID" : "156", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U177", "Parent" : "129"},
	{"ID" : "157", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagudo_U178", "Parent" : "129"},
	{"ID" : "158", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagudo_U179", "Parent" : "129"},
	{"ID" : "159", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U180", "Parent" : "129"},
	{"ID" : "160", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U181", "Parent" : "129"},
	{"ID" : "161", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagncg_U182", "Parent" : "129"},
	{"ID" : "162", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagncg_U183", "Parent" : "129"},
	{"ID" : "163", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U184", "Parent" : "129"},
	{"ID" : "164", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagvdy_U185", "Parent" : "129"},
	{"ID" : "165", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U186", "Parent" : "129"},
	{"ID" : "166", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U187", "Parent" : "129"},
	{"ID" : "167", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagwdI_U188", "Parent" : "129"},
	{"ID" : "168", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U189", "Parent" : "129"},
	{"ID" : "169", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagtde_U190", "Parent" : "129"},
	{"ID" : "170", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U191", "Parent" : "129"},
	{"ID" : "171", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U192", "Parent" : "129"},
	{"ID" : "172", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U193", "Parent" : "129"},
	{"ID" : "173", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U194", "Parent" : "129"},
	{"ID" : "174", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagudo_U195", "Parent" : "129"},
	{"ID" : "175", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagvdy_U196", "Parent" : "129"},
	{"ID" : "176", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U197", "Parent" : "129"},
	{"ID" : "177", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U198", "Parent" : "129"},
	{"ID" : "178", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U199", "Parent" : "129"},
	{"ID" : "179", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagqcK_U200", "Parent" : "129"},
	{"ID" : "180", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagvdy_U201", "Parent" : "129"},
	{"ID" : "181", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U202", "Parent" : "129"},
	{"ID" : "182", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagudo_U203", "Parent" : "129"},
	{"ID" : "183", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U204", "Parent" : "129"},
	{"ID" : "184", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagwdI_U205", "Parent" : "129"},
	{"ID" : "185", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U206", "Parent" : "129"},
	{"ID" : "186", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U207", "Parent" : "129"},
	{"ID" : "187", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagvdy_U208", "Parent" : "129"},
	{"ID" : "188", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagtde_U209", "Parent" : "129"},
	{"ID" : "189", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U210", "Parent" : "129"},
	{"ID" : "190", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U211", "Parent" : "129"},
	{"ID" : "191", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagxdS_U212", "Parent" : "129"},
	{"ID" : "192", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagudo_U213", "Parent" : "129"},
	{"ID" : "193", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U214", "Parent" : "129"},
	{"ID" : "194", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagyd2_U215", "Parent" : "129"},
	{"ID" : "195", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagudo_U216", "Parent" : "129"},
	{"ID" : "196", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U217", "Parent" : "129"},
	{"ID" : "197", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U218", "Parent" : "129"},
	{"ID" : "198", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagzec_U219", "Parent" : "129"},
	{"ID" : "199", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U220", "Parent" : "129"},
	{"ID" : "200", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U221", "Parent" : "129"},
	{"ID" : "201", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagqcK_U222", "Parent" : "129"},
	{"ID" : "202", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_softmax_fu_58", "Parent" : "1", "Child" : ["203", "204"],
		"CDFG" : "softmax",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1",
		"Pipeline" : "Aligned", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "6", "EstimateLatencyMin" : "6", "EstimateLatencyMax" : "6",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "1",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "data_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_4_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_5_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "exp_table2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "invert_table3", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "203", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_softmax_fu_58.exp_table2_U", "Parent" : "202"},
	{"ID" : "204", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.grp_softmax_fu_58.invert_table3_U", "Parent" : "202"},
	{"ID" : "205", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_myproject_fu_216.call_ret2_relu_fu_72", "Parent" : "1",
		"CDFG" : "relu",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "1", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "0", "EstimateLatencyMin" : "0", "EstimateLatencyMax" : "0",
		"Combinational" : "1",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"Port" : [
			{"Name" : "data_0_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_1_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_2_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_3_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_4_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_5_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_6_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_7_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_8_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_9_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_10_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "data_11_V_read", "Type" : "None", "Direction" : "I"}]}]}


set ArgLastReadFirstWriteLatency {
	myproject_galapagos {
		in_V_data_V {Type I LastRead 3 FirstWrite -1}
		in_V_dest_V {Type I LastRead 3 FirstWrite -1}
		in_V_last_V {Type I LastRead 3 FirstWrite -1}
		in_V_id_V {Type I LastRead 3 FirstWrite -1}
		in_V_user_V {Type I LastRead 3 FirstWrite -1}
		in_V_keep_V {Type I LastRead 3 FirstWrite -1}
		out_V_data_V {Type O LastRead -1 FirstWrite 19}
		out_V_dest_V {Type O LastRead -1 FirstWrite 19}
		out_V_last_V {Type O LastRead -1 FirstWrite 19}
		out_V_id_V {Type O LastRead -1 FirstWrite 19}
		out_V_user_V {Type O LastRead -1 FirstWrite 19}
		out_V_keep_V {Type O LastRead -1 FirstWrite 19}
		exp_table2 {Type I LastRead -1 FirstWrite -1}
		invert_table3 {Type I LastRead -1 FirstWrite -1}}
	myproject {
		data_V {Type I LastRead 0 FirstWrite -1}
		exp_table2 {Type I LastRead -1 FirstWrite -1}
		invert_table3 {Type I LastRead -1 FirstWrite -1}}
	conv_1d_0_0_0_0 {
		data_V_read {Type I LastRead 0 FirstWrite -1}}
	compute_layer_0_0_0_s {
		data_0_V_read {Type I LastRead 0 FirstWrite -1}
		data_1_V_read {Type I LastRead 0 FirstWrite -1}
		data_2_V_read {Type I LastRead 0 FirstWrite -1}
		data_3_V_read {Type I LastRead 0 FirstWrite -1}
		data_4_V_read {Type I LastRead 0 FirstWrite -1}
		data_5_V_read {Type I LastRead 0 FirstWrite -1}
		data_6_V_read {Type I LastRead 0 FirstWrite -1}
		data_7_V_read {Type I LastRead 0 FirstWrite -1}
		data_8_V_read {Type I LastRead 0 FirstWrite -1}
		data_9_V_read {Type I LastRead 0 FirstWrite -1}
		data_10_V_read {Type I LastRead 0 FirstWrite -1}
		data_11_V_read {Type I LastRead 0 FirstWrite -1}}
	softmax {
		data_0_V_read {Type I LastRead 0 FirstWrite -1}
		data_1_V_read {Type I LastRead 0 FirstWrite -1}
		data_2_V_read {Type I LastRead 0 FirstWrite -1}
		data_3_V_read {Type I LastRead 0 FirstWrite -1}
		data_4_V_read {Type I LastRead 0 FirstWrite -1}
		data_5_V_read {Type I LastRead 0 FirstWrite -1}
		exp_table2 {Type I LastRead -1 FirstWrite -1}
		invert_table3 {Type I LastRead -1 FirstWrite -1}}
	relu {
		data_0_V_read {Type I LastRead 0 FirstWrite -1}
		data_1_V_read {Type I LastRead 0 FirstWrite -1}
		data_2_V_read {Type I LastRead 0 FirstWrite -1}
		data_3_V_read {Type I LastRead 0 FirstWrite -1}
		data_4_V_read {Type I LastRead 0 FirstWrite -1}
		data_5_V_read {Type I LastRead 0 FirstWrite -1}
		data_6_V_read {Type I LastRead 0 FirstWrite -1}
		data_7_V_read {Type I LastRead 0 FirstWrite -1}
		data_8_V_read {Type I LastRead 0 FirstWrite -1}
		data_9_V_read {Type I LastRead 0 FirstWrite -1}
		data_10_V_read {Type I LastRead 0 FirstWrite -1}
		data_11_V_read {Type I LastRead 0 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "115", "Max" : "115"}
	, {"Name" : "Interval", "Min" : "116", "Max" : "116"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	in_V_data_V { ap_fifo {  { in_V_data_V_dout fifo_data 0 64 }  { in_V_data_V_empty_n fifo_status 0 1 }  { in_V_data_V_read fifo_update 1 1 } } }
	in_V_dest_V { ap_fifo {  { in_V_dest_V_dout fifo_data 0 8 }  { in_V_dest_V_empty_n fifo_status 0 1 }  { in_V_dest_V_read fifo_update 1 1 } } }
	in_V_last_V { ap_fifo {  { in_V_last_V_dout fifo_data 0 1 }  { in_V_last_V_empty_n fifo_status 0 1 }  { in_V_last_V_read fifo_update 1 1 } } }
	in_V_id_V { ap_fifo {  { in_V_id_V_dout fifo_data 0 8 }  { in_V_id_V_empty_n fifo_status 0 1 }  { in_V_id_V_read fifo_update 1 1 } } }
	in_V_user_V { ap_fifo {  { in_V_user_V_dout fifo_data 0 40 }  { in_V_user_V_empty_n fifo_status 0 1 }  { in_V_user_V_read fifo_update 1 1 } } }
	in_V_keep_V { ap_fifo {  { in_V_keep_V_dout fifo_data 0 8 }  { in_V_keep_V_empty_n fifo_status 0 1 }  { in_V_keep_V_read fifo_update 1 1 } } }
	out_V_data_V { ap_fifo {  { out_V_data_V_din fifo_data 1 64 }  { out_V_data_V_full_n fifo_status 0 1 }  { out_V_data_V_write fifo_update 1 1 } } }
	out_V_dest_V { ap_fifo {  { out_V_dest_V_din fifo_data 1 8 }  { out_V_dest_V_full_n fifo_status 0 1 }  { out_V_dest_V_write fifo_update 1 1 } } }
	out_V_last_V { ap_fifo {  { out_V_last_V_din fifo_data 1 1 }  { out_V_last_V_full_n fifo_status 0 1 }  { out_V_last_V_write fifo_update 1 1 } } }
	out_V_id_V { ap_fifo {  { out_V_id_V_din fifo_data 1 8 }  { out_V_id_V_full_n fifo_status 0 1 }  { out_V_id_V_write fifo_update 1 1 } } }
	out_V_user_V { ap_fifo {  { out_V_user_V_din fifo_data 1 40 }  { out_V_user_V_full_n fifo_status 0 1 }  { out_V_user_V_write fifo_update 1 1 } } }
	out_V_keep_V { ap_fifo {  { out_V_keep_V_din fifo_data 1 8 }  { out_V_keep_V_full_n fifo_status 0 1 }  { out_V_keep_V_write fifo_update 1 1 } } }
}

set busDeadlockParameterList { 
}

# RTL port scheduling information:
set fifoSchedulingInfoList { 
	in_V_data_V { fifo_read 6 no_conditional }
	in_V_dest_V { fifo_read 6 no_conditional }
	in_V_last_V { fifo_read 6 no_conditional }
	in_V_id_V { fifo_read 6 no_conditional }
	in_V_user_V { fifo_read 6 no_conditional }
	in_V_keep_V { fifo_read 6 no_conditional }
	out_V_data_V { fifo_write 6 no_conditional }
	out_V_dest_V { fifo_write 6 no_conditional }
	out_V_last_V { fifo_write 6 no_conditional }
	out_V_id_V { fifo_write 6 no_conditional }
	out_V_user_V { fifo_write 6 no_conditional }
	out_V_keep_V { fifo_write 6 no_conditional }
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
