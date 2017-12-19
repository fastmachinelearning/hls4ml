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
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "142", "166", "167"],
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
			{"ID" : "167", "SubInstance" : "grp_sigmoid_fu_241", "Port" : "sigmoid_table2"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145", "Parent" : "0", "Child" : ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122", "123", "124", "125", "126", "127", "128", "129", "130", "131", "132", "133", "134", "135", "136", "137", "138", "139", "140", "141"],
		"CDFG" : "compute_layer_0_0_0_1",
		"VariableLatency" : "0",
		"AlignedPipeline" : "1",
		"UnalignedPipeline" : "0",
		"ProcessNetwork" : "0",
		"Combinational" : "0",
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
		{"Name" : "data_9_V_read", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "2", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sbkb_U1", "Parent" : "1"},
	{"ID" : "3", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18scud_U2", "Parent" : "1"},
	{"ID" : "4", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sdEe_U3", "Parent" : "1"},
	{"ID" : "5", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U4", "Parent" : "1"},
	{"ID" : "6", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U5", "Parent" : "1"},
	{"ID" : "7", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sdEe_U6", "Parent" : "1"},
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U7", "Parent" : "1"},
	{"ID" : "9", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U8", "Parent" : "1"},
	{"ID" : "10", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U9", "Parent" : "1"},
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U10", "Parent" : "1"},
	{"ID" : "12", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U11", "Parent" : "1"},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sg8j_U12", "Parent" : "1"},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U13", "Parent" : "1"},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U14", "Parent" : "1"},
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18shbi_U15", "Parent" : "1"},
	{"ID" : "17", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sibs_U16", "Parent" : "1"},
	{"ID" : "18", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sdEe_U17", "Parent" : "1"},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sbkb_U18", "Parent" : "1"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U19", "Parent" : "1"},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sg8j_U20", "Parent" : "1"},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sdEe_U21", "Parent" : "1"},
	{"ID" : "23", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U22", "Parent" : "1"},
	{"ID" : "24", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U23", "Parent" : "1"},
	{"ID" : "25", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U24", "Parent" : "1"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sdEe_U25", "Parent" : "1"},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sjbC_U26", "Parent" : "1"},
	{"ID" : "28", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U27", "Parent" : "1"},
	{"ID" : "29", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sibs_U28", "Parent" : "1"},
	{"ID" : "30", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U29", "Parent" : "1"},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U30", "Parent" : "1"},
	{"ID" : "32", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18shbi_U31", "Parent" : "1"},
	{"ID" : "33", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sg8j_U32", "Parent" : "1"},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sbkb_U33", "Parent" : "1"},
	{"ID" : "35", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U34", "Parent" : "1"},
	{"ID" : "36", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U35", "Parent" : "1"},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U36", "Parent" : "1"},
	{"ID" : "38", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U37", "Parent" : "1"},
	{"ID" : "39", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U38", "Parent" : "1"},
	{"ID" : "40", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sg8j_U39", "Parent" : "1"},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sdEe_U40", "Parent" : "1"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18skbM_U41", "Parent" : "1"},
	{"ID" : "43", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U42", "Parent" : "1"},
	{"ID" : "44", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sg8j_U43", "Parent" : "1"},
	{"ID" : "45", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sbkb_U44", "Parent" : "1"},
	{"ID" : "46", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U45", "Parent" : "1"},
	{"ID" : "47", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sg8j_U46", "Parent" : "1"},
	{"ID" : "48", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U47", "Parent" : "1"},
	{"ID" : "49", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U48", "Parent" : "1"},
	{"ID" : "50", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U49", "Parent" : "1"},
	{"ID" : "51", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U50", "Parent" : "1"},
	{"ID" : "52", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U51", "Parent" : "1"},
	{"ID" : "53", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U52", "Parent" : "1"},
	{"ID" : "54", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U53", "Parent" : "1"},
	{"ID" : "55", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18scud_U54", "Parent" : "1"},
	{"ID" : "56", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U55", "Parent" : "1"},
	{"ID" : "57", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U56", "Parent" : "1"},
	{"ID" : "58", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sjbC_U57", "Parent" : "1"},
	{"ID" : "59", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18scud_U58", "Parent" : "1"},
	{"ID" : "60", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sg8j_U59", "Parent" : "1"},
	{"ID" : "61", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U60", "Parent" : "1"},
	{"ID" : "62", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U61", "Parent" : "1"},
	{"ID" : "63", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U62", "Parent" : "1"},
	{"ID" : "64", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sg8j_U63", "Parent" : "1"},
	{"ID" : "65", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U64", "Parent" : "1"},
	{"ID" : "66", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U65", "Parent" : "1"},
	{"ID" : "67", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sibs_U66", "Parent" : "1"},
	{"ID" : "68", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sjbC_U67", "Parent" : "1"},
	{"ID" : "69", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U68", "Parent" : "1"},
	{"ID" : "70", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U69", "Parent" : "1"},
	{"ID" : "71", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sbkb_U70", "Parent" : "1"},
	{"ID" : "72", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U71", "Parent" : "1"},
	{"ID" : "73", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U72", "Parent" : "1"},
	{"ID" : "74", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sdEe_U73", "Parent" : "1"},
	{"ID" : "75", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U74", "Parent" : "1"},
	{"ID" : "76", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U75", "Parent" : "1"},
	{"ID" : "77", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U76", "Parent" : "1"},
	{"ID" : "78", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U77", "Parent" : "1"},
	{"ID" : "79", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sdEe_U78", "Parent" : "1"},
	{"ID" : "80", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U79", "Parent" : "1"},
	{"ID" : "81", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U80", "Parent" : "1"},
	{"ID" : "82", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U81", "Parent" : "1"},
	{"ID" : "83", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sibs_U82", "Parent" : "1"},
	{"ID" : "84", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U83", "Parent" : "1"},
	{"ID" : "85", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U84", "Parent" : "1"},
	{"ID" : "86", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18shbi_U85", "Parent" : "1"},
	{"ID" : "87", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18scud_U86", "Parent" : "1"},
	{"ID" : "88", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U87", "Parent" : "1"},
	{"ID" : "89", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sdEe_U88", "Parent" : "1"},
	{"ID" : "90", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U89", "Parent" : "1"},
	{"ID" : "91", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18scud_U90", "Parent" : "1"},
	{"ID" : "92", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18skbM_U91", "Parent" : "1"},
	{"ID" : "93", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U92", "Parent" : "1"},
	{"ID" : "94", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sbkb_U93", "Parent" : "1"},
	{"ID" : "95", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U94", "Parent" : "1"},
	{"ID" : "96", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sdEe_U95", "Parent" : "1"},
	{"ID" : "97", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18shbi_U96", "Parent" : "1"},
	{"ID" : "98", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U97", "Parent" : "1"},
	{"ID" : "99", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U98", "Parent" : "1"},
	{"ID" : "100", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sg8j_U99", "Parent" : "1"},
	{"ID" : "101", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18scud_U100", "Parent" : "1"},
	{"ID" : "102", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sdEe_U101", "Parent" : "1"},
	{"ID" : "103", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18scud_U102", "Parent" : "1"},
	{"ID" : "104", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U103", "Parent" : "1"},
	{"ID" : "105", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U104", "Parent" : "1"},
	{"ID" : "106", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U105", "Parent" : "1"},
	{"ID" : "107", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sbkb_U106", "Parent" : "1"},
	{"ID" : "108", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U107", "Parent" : "1"},
	{"ID" : "109", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U108", "Parent" : "1"},
	{"ID" : "110", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sbkb_U109", "Parent" : "1"},
	{"ID" : "111", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U110", "Parent" : "1"},
	{"ID" : "112", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sdEe_U111", "Parent" : "1"},
	{"ID" : "113", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U112", "Parent" : "1"},
	{"ID" : "114", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sdEe_U113", "Parent" : "1"},
	{"ID" : "115", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sg8j_U114", "Parent" : "1"},
	{"ID" : "116", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18shbi_U115", "Parent" : "1"},
	{"ID" : "117", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sg8j_U116", "Parent" : "1"},
	{"ID" : "118", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U117", "Parent" : "1"},
	{"ID" : "119", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sbkb_U118", "Parent" : "1"},
	{"ID" : "120", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sibs_U119", "Parent" : "1"},
	{"ID" : "121", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sdEe_U120", "Parent" : "1"},
	{"ID" : "122", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U121", "Parent" : "1"},
	{"ID" : "123", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sibs_U122", "Parent" : "1"},
	{"ID" : "124", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U123", "Parent" : "1"},
	{"ID" : "125", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sdEe_U124", "Parent" : "1"},
	{"ID" : "126", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18shbi_U125", "Parent" : "1"},
	{"ID" : "127", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sibs_U126", "Parent" : "1"},
	{"ID" : "128", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U127", "Parent" : "1"},
	{"ID" : "129", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sdEe_U128", "Parent" : "1"},
	{"ID" : "130", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sbkb_U129", "Parent" : "1"},
	{"ID" : "131", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18scud_U130", "Parent" : "1"},
	{"ID" : "132", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sbkb_U131", "Parent" : "1"},
	{"ID" : "133", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sbkb_U132", "Parent" : "1"},
	{"ID" : "134", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U133", "Parent" : "1"},
	{"ID" : "135", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18scud_U134", "Parent" : "1"},
	{"ID" : "136", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sfYi_U135", "Parent" : "1"},
	{"ID" : "137", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U136", "Parent" : "1"},
	{"ID" : "138", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sibs_U137", "Parent" : "1"},
	{"ID" : "139", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18seOg_U138", "Parent" : "1"},
	{"ID" : "140", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18sdEe_U139", "Parent" : "1"},
	{"ID" : "141", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_1_fu_145.myproject_mul_18scud_U140", "Parent" : "1"},
	{"ID" : "142", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169", "Parent" : "0", "Child" : ["143", "144", "145", "146", "147", "148", "149", "150", "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165"],
		"CDFG" : "compute_layer_0_0_0_s",
		"VariableLatency" : "0",
		"AlignedPipeline" : "1",
		"UnalignedPipeline" : "0",
		"ProcessNetwork" : "0",
		"Combinational" : "0",
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
	{"ID" : "143", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18sfYi_x_U183", "Parent" : "142"},
	{"ID" : "144", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18sfYi_x_U184", "Parent" : "142"},
	{"ID" : "145", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18sfYi_x_U185", "Parent" : "142"},
	{"ID" : "146", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18sdEe_x_U186", "Parent" : "142"},
	{"ID" : "147", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18sdEe_x_U187", "Parent" : "142"},
	{"ID" : "148", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18sg8j_x_U188", "Parent" : "142"},
	{"ID" : "149", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18sbkb_x_U189", "Parent" : "142"},
	{"ID" : "150", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18sdEe_x_U190", "Parent" : "142"},
	{"ID" : "151", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18shbi_x_U191", "Parent" : "142"},
	{"ID" : "152", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18shbi_x_U192", "Parent" : "142"},
	{"ID" : "153", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18skbM_x_U193", "Parent" : "142"},
	{"ID" : "154", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18sbkb_x_U194", "Parent" : "142"},
	{"ID" : "155", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18slbW_U195", "Parent" : "142"},
	{"ID" : "156", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18sg8j_x_U196", "Parent" : "142"},
	{"ID" : "157", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18sg8j_x_U197", "Parent" : "142"},
	{"ID" : "158", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18sg8j_x_U198", "Parent" : "142"},
	{"ID" : "159", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18sfYi_x_U199", "Parent" : "142"},
	{"ID" : "160", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18sfYi_x_U200", "Parent" : "142"},
	{"ID" : "161", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18sg8j_x_U201", "Parent" : "142"},
	{"ID" : "162", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18sfYi_x_U202", "Parent" : "142"},
	{"ID" : "163", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18slbW_U203", "Parent" : "142"},
	{"ID" : "164", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18sfYi_x_U204", "Parent" : "142"},
	{"ID" : "165", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_169.myproject_mul_18sg8j_x_U205", "Parent" : "142"},
	{"ID" : "166", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.call_ret2_relu_fu_205", "Parent" : "0",
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
	{"ID" : "167", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_sigmoid_fu_241", "Parent" : "0", "Child" : ["168"],
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
	{"ID" : "168", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sigmoid_fu_241.sigmoid_table2_U", "Parent" : "167"}]}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "12", "Max" : "12"}
	, {"Name" : "Interval", "Min" : "1", "Max" : "1"}
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
