set C_TypeInfoList {{ 
"myproject" : [[], { "return": [[], "void"]} , [{"ExternC" : 0}], [ {"data": [[], {"array": ["0", [10]]}] }, {"res": [[], {"array": ["1", [1]]}] }, {"const_size_in": [[], {"reference":  {"scalar": "unsigned short"}}] }, {"const_size_out": [[], {"reference":  {"scalar": "unsigned short"}}] }],[],""], 
"0": [ "input_t", {"typedef": [[[],"2"],""]}], 
"2": [ "ap_fixed<18, 8, 5, 3, 0>", {"hls_type": {"ap_fixed": [[[[], {"scalar": { "int": 18}}],[[], {"scalar": { "int": 8}}],[[], {"scalar": { "3": 5}}],[[], {"scalar": { "4": 3}}],[[], {"scalar": { "int": 0}}]],""]}}], 
"3": [ "ap_q_mode", {"enum": [[],[],[{"SC_RND":  {"scalar": "__integer__"}},{"SC_RND_ZERO":  {"scalar": "__integer__"}},{"SC_RND_MIN_INF":  {"scalar": "__integer__"}},{"SC_RND_INF":  {"scalar": "__integer__"}},{"SC_RND_CONV":  {"scalar": "__integer__"}},{"SC_TRN":  {"scalar": "__integer__"}},{"SC_TRN_ZERO":  {"scalar": "__integer__"}}],""]}], 
"4": [ "ap_o_mode", {"enum": [[],[],[{"SC_SAT":  {"scalar": "__integer__"}},{"SC_SAT_ZERO":  {"scalar": "__integer__"}},{"SC_SAT_SYM":  {"scalar": "__integer__"}},{"SC_WRAP":  {"scalar": "__integer__"}},{"SC_WRAP_SM":  {"scalar": "__integer__"}}],""]}], 
"1": [ "result_t", {"typedef": [[[],"2"],""]}]
}}
set moduleName myproject
set isCombinational 0
set isDatapathOnly 0
set isPipelined 1
set pipeline_type function
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set C_modelName {myproject}
set C_modelType { void 0 }
set C_modelArgList {
	{ data_0_V int 18 regular {pointer 0}  }
	{ data_1_V int 18 regular {pointer 0}  }
	{ data_2_V int 18 regular {pointer 0}  }
	{ data_3_V int 18 regular {pointer 0}  }
	{ data_4_V int 18 regular {pointer 0}  }
	{ data_5_V int 18 regular {pointer 0}  }
	{ data_6_V int 18 regular {pointer 0}  }
	{ data_7_V int 18 regular {pointer 0}  }
	{ data_8_V int 18 regular {pointer 0}  }
	{ data_9_V int 18 regular {pointer 0}  }
	{ res_0_V int 18 regular {pointer 1}  }
	{ const_size_in int 16 regular {pointer 1}  }
	{ const_size_out int 16 regular {pointer 1}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "data_0_V", "interface" : "wire", "bitwidth" : 18, "direction" : "READONLY", "bitSlice":[{"low":0,"up":17,"cElement": [{"cName": "data.V","cData": "int18","bit_use": { "low": 0,"up": 17},"cArray": [{"low" : 0,"up" : 0,"step" : 2}]}]}]} , 
 	{ "Name" : "data_1_V", "interface" : "wire", "bitwidth" : 18, "direction" : "READONLY", "bitSlice":[{"low":0,"up":17,"cElement": [{"cName": "data.V","cData": "int18","bit_use": { "low": 0,"up": 17},"cArray": [{"low" : 1,"up" : 1,"step" : 2}]}]}]} , 
 	{ "Name" : "data_2_V", "interface" : "wire", "bitwidth" : 18, "direction" : "READONLY", "bitSlice":[{"low":0,"up":17,"cElement": [{"cName": "data.V","cData": "int18","bit_use": { "low": 0,"up": 17},"cArray": [{"low" : 2,"up" : 2,"step" : 2}]}]}]} , 
 	{ "Name" : "data_3_V", "interface" : "wire", "bitwidth" : 18, "direction" : "READONLY", "bitSlice":[{"low":0,"up":17,"cElement": [{"cName": "data.V","cData": "int18","bit_use": { "low": 0,"up": 17},"cArray": [{"low" : 3,"up" : 3,"step" : 2}]}]}]} , 
 	{ "Name" : "data_4_V", "interface" : "wire", "bitwidth" : 18, "direction" : "READONLY", "bitSlice":[{"low":0,"up":17,"cElement": [{"cName": "data.V","cData": "int18","bit_use": { "low": 0,"up": 17},"cArray": [{"low" : 4,"up" : 4,"step" : 2}]}]}]} , 
 	{ "Name" : "data_5_V", "interface" : "wire", "bitwidth" : 18, "direction" : "READONLY", "bitSlice":[{"low":0,"up":17,"cElement": [{"cName": "data.V","cData": "int18","bit_use": { "low": 0,"up": 17},"cArray": [{"low" : 5,"up" : 5,"step" : 2}]}]}]} , 
 	{ "Name" : "data_6_V", "interface" : "wire", "bitwidth" : 18, "direction" : "READONLY", "bitSlice":[{"low":0,"up":17,"cElement": [{"cName": "data.V","cData": "int18","bit_use": { "low": 0,"up": 17},"cArray": [{"low" : 6,"up" : 6,"step" : 2}]}]}]} , 
 	{ "Name" : "data_7_V", "interface" : "wire", "bitwidth" : 18, "direction" : "READONLY", "bitSlice":[{"low":0,"up":17,"cElement": [{"cName": "data.V","cData": "int18","bit_use": { "low": 0,"up": 17},"cArray": [{"low" : 7,"up" : 7,"step" : 2}]}]}]} , 
 	{ "Name" : "data_8_V", "interface" : "wire", "bitwidth" : 18, "direction" : "READONLY", "bitSlice":[{"low":0,"up":17,"cElement": [{"cName": "data.V","cData": "int18","bit_use": { "low": 0,"up": 17},"cArray": [{"low" : 8,"up" : 8,"step" : 2}]}]}]} , 
 	{ "Name" : "data_9_V", "interface" : "wire", "bitwidth" : 18, "direction" : "READONLY", "bitSlice":[{"low":0,"up":17,"cElement": [{"cName": "data.V","cData": "int18","bit_use": { "low": 0,"up": 17},"cArray": [{"low" : 9,"up" : 9,"step" : 2}]}]}]} , 
 	{ "Name" : "res_0_V", "interface" : "wire", "bitwidth" : 18, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":17,"cElement": [{"cName": "res.V","cData": "int18","bit_use": { "low": 0,"up": 17},"cArray": [{"low" : 0,"up" : 0,"step" : 2}]}]}]} , 
 	{ "Name" : "const_size_in", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":15,"cElement": [{"cName": "const_size_in","cData": "unsigned short","bit_use": { "low": 0,"up": 15},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} , 
 	{ "Name" : "const_size_out", "interface" : "wire", "bitwidth" : 16, "direction" : "WRITEONLY", "bitSlice":[{"low":0,"up":15,"cElement": [{"cName": "const_size_out","cData": "unsigned short","bit_use": { "low": 0,"up": 15},"cArray": [{"low" : 0,"up" : 0,"step" : 1}]}]}]} ]}
# RTL Port declarations: 
set portNum 22
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ data_0_V sc_in sc_lv 18 signal 0 } 
	{ data_1_V sc_in sc_lv 18 signal 1 } 
	{ data_2_V sc_in sc_lv 18 signal 2 } 
	{ data_3_V sc_in sc_lv 18 signal 3 } 
	{ data_4_V sc_in sc_lv 18 signal 4 } 
	{ data_5_V sc_in sc_lv 18 signal 5 } 
	{ data_6_V sc_in sc_lv 18 signal 6 } 
	{ data_7_V sc_in sc_lv 18 signal 7 } 
	{ data_8_V sc_in sc_lv 18 signal 8 } 
	{ data_9_V sc_in sc_lv 18 signal 9 } 
	{ res_0_V sc_out sc_lv 18 signal 10 } 
	{ res_0_V_ap_vld sc_out sc_logic 1 outvld 10 } 
	{ const_size_in sc_out sc_lv 16 signal 11 } 
	{ const_size_in_ap_vld sc_out sc_logic 1 outvld 11 } 
	{ const_size_out sc_out sc_lv 16 signal 12 } 
	{ const_size_out_ap_vld sc_out sc_logic 1 outvld 12 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "data_0_V", "direction": "in", "datatype": "sc_lv", "bitwidth":18, "type": "signal", "bundle":{"name": "data_0_V", "role": "default" }} , 
 	{ "name": "data_1_V", "direction": "in", "datatype": "sc_lv", "bitwidth":18, "type": "signal", "bundle":{"name": "data_1_V", "role": "default" }} , 
 	{ "name": "data_2_V", "direction": "in", "datatype": "sc_lv", "bitwidth":18, "type": "signal", "bundle":{"name": "data_2_V", "role": "default" }} , 
 	{ "name": "data_3_V", "direction": "in", "datatype": "sc_lv", "bitwidth":18, "type": "signal", "bundle":{"name": "data_3_V", "role": "default" }} , 
 	{ "name": "data_4_V", "direction": "in", "datatype": "sc_lv", "bitwidth":18, "type": "signal", "bundle":{"name": "data_4_V", "role": "default" }} , 
 	{ "name": "data_5_V", "direction": "in", "datatype": "sc_lv", "bitwidth":18, "type": "signal", "bundle":{"name": "data_5_V", "role": "default" }} , 
 	{ "name": "data_6_V", "direction": "in", "datatype": "sc_lv", "bitwidth":18, "type": "signal", "bundle":{"name": "data_6_V", "role": "default" }} , 
 	{ "name": "data_7_V", "direction": "in", "datatype": "sc_lv", "bitwidth":18, "type": "signal", "bundle":{"name": "data_7_V", "role": "default" }} , 
 	{ "name": "data_8_V", "direction": "in", "datatype": "sc_lv", "bitwidth":18, "type": "signal", "bundle":{"name": "data_8_V", "role": "default" }} , 
 	{ "name": "data_9_V", "direction": "in", "datatype": "sc_lv", "bitwidth":18, "type": "signal", "bundle":{"name": "data_9_V", "role": "default" }} , 
 	{ "name": "res_0_V", "direction": "out", "datatype": "sc_lv", "bitwidth":18, "type": "signal", "bundle":{"name": "res_0_V", "role": "default" }} , 
 	{ "name": "res_0_V_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "res_0_V", "role": "ap_vld" }} , 
 	{ "name": "const_size_in", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "const_size_in", "role": "default" }} , 
 	{ "name": "const_size_in_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "const_size_in", "role": "ap_vld" }} , 
 	{ "name": "const_size_out", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "const_size_out", "role": "default" }} , 
 	{ "name": "const_size_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "const_size_out", "role": "ap_vld" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "42", "51", "52"],
		"CDFG" : "myproject",
		"VariableLatency" : "0",
		"AlignedPipeline" : "1",
		"UnalignedPipeline" : "0",
		"ProcessNetwork" : "0",
		"Combinational" : "0",
		"ControlExist" : "1",
		"Port" : [
		{"Name" : "data_0_V", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_1_V", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_2_V", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_3_V", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_4_V", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_5_V", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_6_V", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_7_V", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_8_V", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_9_V", "Type" : "None", "Direction" : "I"},
		{"Name" : "res_0_V", "Type" : "Vld", "Direction" : "O"},
		{"Name" : "const_size_in", "Type" : "Vld", "Direction" : "O"},
		{"Name" : "const_size_out", "Type" : "Vld", "Direction" : "O"},
		{"Name" : "sigmoid_table2", "Type" : "Memory", "Direction" : "I",
			"SubConnect" : [
			{"ID" : "52", "SubInstance" : "grp_sigmoid_fu_241", "Port" : "sigmoid_table2"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145", "Parent" : "0", "Child" : ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41"],
		"CDFG" : "compute_layer_0_0_0_s",
		"VariableLatency" : "0",
		"AlignedPipeline" : "1",
		"UnalignedPipeline" : "0",
		"ProcessNetwork" : "0",
		"Combinational" : "0",
		"ControlExist" : "1",
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
		{"Name" : "data_9_V_read", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "2", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sbkb_U1", "Parent" : "1"},
	{"ID" : "3", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18scud_U2", "Parent" : "1"},
	{"ID" : "4", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sdEe_U3", "Parent" : "1"},
	{"ID" : "5", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18seOg_U4", "Parent" : "1"},
	{"ID" : "6", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sdEe_U5", "Parent" : "1"},
	{"ID" : "7", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sdEe_U6", "Parent" : "1"},
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sdEe_U7", "Parent" : "1"},
	{"ID" : "9", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sfYi_U8", "Parent" : "1"},
	{"ID" : "10", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sdEe_U9", "Parent" : "1"},
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sg8j_U10", "Parent" : "1"},
	{"ID" : "12", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18shbi_U11", "Parent" : "1"},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sdEe_U12", "Parent" : "1"},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sfYi_U13", "Parent" : "1"},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18seOg_U14", "Parent" : "1"},
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sfYi_U15", "Parent" : "1"},
	{"ID" : "17", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sg8j_U16", "Parent" : "1"},
	{"ID" : "18", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sdEe_U17", "Parent" : "1"},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18scud_U18", "Parent" : "1"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sdEe_U19", "Parent" : "1"},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sfYi_U20", "Parent" : "1"},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sg8j_U21", "Parent" : "1"},
	{"ID" : "23", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sibs_U22", "Parent" : "1"},
	{"ID" : "24", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sdEe_U23", "Parent" : "1"},
	{"ID" : "25", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18shbi_U24", "Parent" : "1"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sjbC_U25", "Parent" : "1"},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sdEe_U26", "Parent" : "1"},
	{"ID" : "28", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sdEe_U27", "Parent" : "1"},
	{"ID" : "29", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sdEe_U28", "Parent" : "1"},
	{"ID" : "30", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18seOg_U29", "Parent" : "1"},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18seOg_U30", "Parent" : "1"},
	{"ID" : "32", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18seOg_U31", "Parent" : "1"},
	{"ID" : "33", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18scud_U32", "Parent" : "1"},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sdEe_U33", "Parent" : "1"},
	{"ID" : "35", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sdEe_U34", "Parent" : "1"},
	{"ID" : "36", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sdEe_U35", "Parent" : "1"},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18skbM_U36", "Parent" : "1"},
	{"ID" : "38", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sg8j_U37", "Parent" : "1"},
	{"ID" : "39", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sdEe_U38", "Parent" : "1"},
	{"ID" : "40", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sdEe_U39", "Parent" : "1"},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_145.myproject_mul_18sdEe_U40", "Parent" : "1"},
	{"ID" : "42", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_169", "Parent" : "0", "Child" : ["43", "44", "45", "46", "47", "48", "49", "50"],
		"CDFG" : "compute_layer_0_0_0_1",
		"VariableLatency" : "0",
		"AlignedPipeline" : "1",
		"UnalignedPipeline" : "0",
		"ProcessNetwork" : "0",
		"Combinational" : "0",
		"ControlExist" : "1",
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
		{"Name" : "data_11_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_12_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_13_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_14_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_15_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_16_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_17_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_18_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_19_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_20_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_21_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_22_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_23_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_24_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_25_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_26_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_27_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_28_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_29_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_30_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_31_V_read", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "43", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_169.myproject_mul_18scud_x_U83", "Parent" : "42"},
	{"ID" : "44", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_169.myproject_mul_18scud_x_U84", "Parent" : "42"},
	{"ID" : "45", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_169.myproject_mul_18scud_x_U85", "Parent" : "42"},
	{"ID" : "46", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_169.myproject_mul_18sfYi_x_U86", "Parent" : "42"},
	{"ID" : "47", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_169.myproject_mul_18shbi_x_U87", "Parent" : "42"},
	{"ID" : "48", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_169.myproject_mul_18shbi_x_U88", "Parent" : "42"},
	{"ID" : "49", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_169.myproject_mul_18skbM_x_U89", "Parent" : "42"},
	{"ID" : "50", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_169.myproject_mul_18shbi_x_U90", "Parent" : "42"},
	{"ID" : "51", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.call_ret2_relu_fu_205", "Parent" : "0",
		"CDFG" : "relu",
		"VariableLatency" : "0",
		"AlignedPipeline" : "0",
		"UnalignedPipeline" : "0",
		"ProcessNetwork" : "0",
		"Combinational" : "1",
		"ControlExist" : "0",
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
		{"Name" : "data_11_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_12_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_13_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_14_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_15_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_16_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_17_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_18_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_19_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_20_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_21_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_22_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_23_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_24_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_25_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_26_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_27_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_28_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_29_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_30_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "data_31_V_read", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "52", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_sigmoid_fu_241", "Parent" : "0", "Child" : ["53"],
		"CDFG" : "sigmoid",
		"VariableLatency" : "0",
		"AlignedPipeline" : "1",
		"UnalignedPipeline" : "0",
		"ProcessNetwork" : "0",
		"Combinational" : "0",
		"ControlExist" : "1",
		"Port" : [
		{"Name" : "data_V_read", "Type" : "None", "Direction" : "I"},
		{"Name" : "sigmoid_table2", "Type" : "Memory", "Direction" : "I"}]},
	{"ID" : "53", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sigmoid_fu_241.sigmoid_table2_U", "Parent" : "52"}]}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "17", "Max" : "17"}
	, {"Name" : "Interval", "Min" : "7", "Max" : "7"}
]}

set Spec2ImplPortList { 
	data_0_V { ap_none {  { data_0_V in_data 0 18 } } }
	data_1_V { ap_none {  { data_1_V in_data 0 18 } } }
	data_2_V { ap_none {  { data_2_V in_data 0 18 } } }
	data_3_V { ap_none {  { data_3_V in_data 0 18 } } }
	data_4_V { ap_none {  { data_4_V in_data 0 18 } } }
	data_5_V { ap_none {  { data_5_V in_data 0 18 } } }
	data_6_V { ap_none {  { data_6_V in_data 0 18 } } }
	data_7_V { ap_none {  { data_7_V in_data 0 18 } } }
	data_8_V { ap_none {  { data_8_V in_data 0 18 } } }
	data_9_V { ap_none {  { data_9_V in_data 0 18 } } }
	res_0_V { ap_vld {  { res_0_V out_data 1 18 }  { res_0_V_ap_vld out_vld 1 1 } } }
	const_size_in { ap_vld {  { const_size_in out_data 1 16 }  { const_size_in_ap_vld out_vld 1 1 } } }
	const_size_out { ap_vld {  { const_size_out out_data 1 16 }  { const_size_out_ap_vld out_vld 1 1 } } }
}

set busDeadlockParameterList { 
}

# RTL port scheduling information:
set fifoSchedulingInfoList { 
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
