set moduleName myproject
set isTaskLevelControl 1
set isCombinational 0
set isDatapathOnly 0
set isPipelined 1
set pipeline_type function
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set C_modelName {myproject}
set C_modelType { int 189 }
set C_modelArgList {
	{ data_V int 768 regular {pointer 0}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "data_V", "interface" : "wire", "bitwidth" : 768, "direction" : "READONLY"} , 
 	{ "Name" : "ap_return", "interface" : "wire", "bitwidth" : 189} ]}
# RTL Port declarations: 
set portNum 10
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ data_V_ap_vld sc_in sc_logic 1 invld 0 } 
	{ data_V sc_in sc_lv 768 signal 0 } 
	{ ap_return sc_out sc_lv 189 signal -1 } 
	{ data_V_blk_n sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "data_V_ap_vld", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "data_V", "role": "ap_vld" }} , 
 	{ "name": "data_V", "direction": "in", "datatype": "sc_lv", "bitwidth":768, "type": "signal", "bundle":{"name": "data_V", "role": "default" }} , 
 	{ "name": "ap_return", "direction": "out", "datatype": "sc_lv", "bitwidth":189, "type": "signal", "bundle":{"name": "ap_return", "role": "default" }} , 
 	{ "name": "data_V_blk_n", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "data_V_blk_n", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "128", "201", "204"],
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
					{"ID" : "201", "SubInstance" : "grp_softmax_fu_58", "Port" : "exp_table2"}]},
			{"Name" : "invert_table3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "201", "SubInstance" : "grp_softmax_fu_58", "Port" : "invert_table3"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36", "Parent" : "0", "Child" : ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122", "123", "124", "125", "126", "127"],
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
	{"ID" : "2", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U1", "Parent" : "1"},
	{"ID" : "3", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U2", "Parent" : "1"},
	{"ID" : "4", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U3", "Parent" : "1"},
	{"ID" : "5", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U4", "Parent" : "1"},
	{"ID" : "6", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U5", "Parent" : "1"},
	{"ID" : "7", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapageOg_U6", "Parent" : "1"},
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U7", "Parent" : "1"},
	{"ID" : "9", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagfYi_U8", "Parent" : "1"},
	{"ID" : "10", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagg8j_U9", "Parent" : "1"},
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U10", "Parent" : "1"},
	{"ID" : "12", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U11", "Parent" : "1"},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U12", "Parent" : "1"},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U13", "Parent" : "1"},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U14", "Parent" : "1"},
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U15", "Parent" : "1"},
	{"ID" : "17", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U16", "Parent" : "1"},
	{"ID" : "18", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U17", "Parent" : "1"},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagfYi_U18", "Parent" : "1"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagfYi_U19", "Parent" : "1"},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U20", "Parent" : "1"},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagfYi_U21", "Parent" : "1"},
	{"ID" : "23", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U22", "Parent" : "1"},
	{"ID" : "24", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U23", "Parent" : "1"},
	{"ID" : "25", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U24", "Parent" : "1"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U25", "Parent" : "1"},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U26", "Parent" : "1"},
	{"ID" : "28", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagjbC_U27", "Parent" : "1"},
	{"ID" : "29", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U28", "Parent" : "1"},
	{"ID" : "30", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U29", "Parent" : "1"},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U30", "Parent" : "1"},
	{"ID" : "32", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagjbC_U31", "Parent" : "1"},
	{"ID" : "33", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagfYi_U32", "Parent" : "1"},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U33", "Parent" : "1"},
	{"ID" : "35", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagkbM_U34", "Parent" : "1"},
	{"ID" : "36", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U35", "Parent" : "1"},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapageOg_U36", "Parent" : "1"},
	{"ID" : "38", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapageOg_U37", "Parent" : "1"},
	{"ID" : "39", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U38", "Parent" : "1"},
	{"ID" : "40", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U39", "Parent" : "1"},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagfYi_U40", "Parent" : "1"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U41", "Parent" : "1"},
	{"ID" : "43", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U42", "Parent" : "1"},
	{"ID" : "44", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U43", "Parent" : "1"},
	{"ID" : "45", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagjbC_U44", "Parent" : "1"},
	{"ID" : "46", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U45", "Parent" : "1"},
	{"ID" : "47", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U46", "Parent" : "1"},
	{"ID" : "48", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U47", "Parent" : "1"},
	{"ID" : "49", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U48", "Parent" : "1"},
	{"ID" : "50", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U49", "Parent" : "1"},
	{"ID" : "51", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U50", "Parent" : "1"},
	{"ID" : "52", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U51", "Parent" : "1"},
	{"ID" : "53", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U52", "Parent" : "1"},
	{"ID" : "54", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U53", "Parent" : "1"},
	{"ID" : "55", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U54", "Parent" : "1"},
	{"ID" : "56", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapageOg_U55", "Parent" : "1"},
	{"ID" : "57", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U56", "Parent" : "1"},
	{"ID" : "58", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U57", "Parent" : "1"},
	{"ID" : "59", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U58", "Parent" : "1"},
	{"ID" : "60", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagkbM_U59", "Parent" : "1"},
	{"ID" : "61", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U60", "Parent" : "1"},
	{"ID" : "62", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U61", "Parent" : "1"},
	{"ID" : "63", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U62", "Parent" : "1"},
	{"ID" : "64", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U63", "Parent" : "1"},
	{"ID" : "65", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U64", "Parent" : "1"},
	{"ID" : "66", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagjbC_U65", "Parent" : "1"},
	{"ID" : "67", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U66", "Parent" : "1"},
	{"ID" : "68", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U67", "Parent" : "1"},
	{"ID" : "69", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U68", "Parent" : "1"},
	{"ID" : "70", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagfYi_U69", "Parent" : "1"},
	{"ID" : "71", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U70", "Parent" : "1"},
	{"ID" : "72", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U71", "Parent" : "1"},
	{"ID" : "73", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U72", "Parent" : "1"},
	{"ID" : "74", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U73", "Parent" : "1"},
	{"ID" : "75", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U74", "Parent" : "1"},
	{"ID" : "76", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U75", "Parent" : "1"},
	{"ID" : "77", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U76", "Parent" : "1"},
	{"ID" : "78", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U77", "Parent" : "1"},
	{"ID" : "79", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U78", "Parent" : "1"},
	{"ID" : "80", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U79", "Parent" : "1"},
	{"ID" : "81", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U80", "Parent" : "1"},
	{"ID" : "82", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U81", "Parent" : "1"},
	{"ID" : "83", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagfYi_U82", "Parent" : "1"},
	{"ID" : "84", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagg8j_U83", "Parent" : "1"},
	{"ID" : "85", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U84", "Parent" : "1"},
	{"ID" : "86", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U85", "Parent" : "1"},
	{"ID" : "87", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U86", "Parent" : "1"},
	{"ID" : "88", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U87", "Parent" : "1"},
	{"ID" : "89", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U88", "Parent" : "1"},
	{"ID" : "90", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U89", "Parent" : "1"},
	{"ID" : "91", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U90", "Parent" : "1"},
	{"ID" : "92", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U91", "Parent" : "1"},
	{"ID" : "93", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagfYi_U92", "Parent" : "1"},
	{"ID" : "94", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U93", "Parent" : "1"},
	{"ID" : "95", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U94", "Parent" : "1"},
	{"ID" : "96", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U95", "Parent" : "1"},
	{"ID" : "97", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U96", "Parent" : "1"},
	{"ID" : "98", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U97", "Parent" : "1"},
	{"ID" : "99", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U98", "Parent" : "1"},
	{"ID" : "100", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagkbM_U99", "Parent" : "1"},
	{"ID" : "101", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U100", "Parent" : "1"},
	{"ID" : "102", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U101", "Parent" : "1"},
	{"ID" : "103", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U102", "Parent" : "1"},
	{"ID" : "104", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U103", "Parent" : "1"},
	{"ID" : "105", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapageOg_U104", "Parent" : "1"},
	{"ID" : "106", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U105", "Parent" : "1"},
	{"ID" : "107", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U106", "Parent" : "1"},
	{"ID" : "108", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagdEe_U107", "Parent" : "1"},
	{"ID" : "109", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U108", "Parent" : "1"},
	{"ID" : "110", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U109", "Parent" : "1"},
	{"ID" : "111", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagjbC_U110", "Parent" : "1"},
	{"ID" : "112", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U111", "Parent" : "1"},
	{"ID" : "113", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U112", "Parent" : "1"},
	{"ID" : "114", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapageOg_U113", "Parent" : "1"},
	{"ID" : "115", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U114", "Parent" : "1"},
	{"ID" : "116", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapageOg_U115", "Parent" : "1"},
	{"ID" : "117", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U116", "Parent" : "1"},
	{"ID" : "118", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U117", "Parent" : "1"},
	{"ID" : "119", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaghbi_U118", "Parent" : "1"},
	{"ID" : "120", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U119", "Parent" : "1"},
	{"ID" : "121", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagcud_U120", "Parent" : "1"},
	{"ID" : "122", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U121", "Parent" : "1"},
	{"ID" : "123", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapaglbW_U122", "Parent" : "1"},
	{"ID" : "124", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagjbC_U123", "Parent" : "1"},
	{"ID" : "125", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagbkb_U124", "Parent" : "1"},
	{"ID" : "126", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagibs_U125", "Parent" : "1"},
	{"ID" : "127", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_1d_0_0_0_0_fu_36.myproject_galapagg8j_U126", "Parent" : "1"},
	{"ID" : "128", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42", "Parent" : "0", "Child" : ["129", "130", "131", "132", "133", "134", "135", "136", "137", "138", "139", "140", "141", "142", "143", "144", "145", "146", "147", "148", "149", "150", "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165", "166", "167", "168", "169", "170", "171", "172", "173", "174", "175", "176", "177", "178", "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189", "190", "191", "192", "193", "194", "195", "196", "197", "198", "199", "200"],
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
	{"ID" : "129", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagmb6_U151", "Parent" : "128"},
	{"ID" : "130", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagncg_U152", "Parent" : "128"},
	{"ID" : "131", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U153", "Parent" : "128"},
	{"ID" : "132", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U154", "Parent" : "128"},
	{"ID" : "133", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U155", "Parent" : "128"},
	{"ID" : "134", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U156", "Parent" : "128"},
	{"ID" : "135", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagqcK_U157", "Parent" : "128"},
	{"ID" : "136", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagqcK_U158", "Parent" : "128"},
	{"ID" : "137", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U159", "Parent" : "128"},
	{"ID" : "138", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U160", "Parent" : "128"},
	{"ID" : "139", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U161", "Parent" : "128"},
	{"ID" : "140", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagncg_U162", "Parent" : "128"},
	{"ID" : "141", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U163", "Parent" : "128"},
	{"ID" : "142", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U164", "Parent" : "128"},
	{"ID" : "143", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagsc4_U165", "Parent" : "128"},
	{"ID" : "144", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagtde_U166", "Parent" : "128"},
	{"ID" : "145", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U167", "Parent" : "128"},
	{"ID" : "146", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U168", "Parent" : "128"},
	{"ID" : "147", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagmb6_U169", "Parent" : "128"},
	{"ID" : "148", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagtde_U170", "Parent" : "128"},
	{"ID" : "149", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagncg_U171", "Parent" : "128"},
	{"ID" : "150", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U172", "Parent" : "128"},
	{"ID" : "151", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U173", "Parent" : "128"},
	{"ID" : "152", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U174", "Parent" : "128"},
	{"ID" : "153", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagncg_U175", "Parent" : "128"},
	{"ID" : "154", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagudo_U176", "Parent" : "128"},
	{"ID" : "155", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U177", "Parent" : "128"},
	{"ID" : "156", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagudo_U178", "Parent" : "128"},
	{"ID" : "157", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagudo_U179", "Parent" : "128"},
	{"ID" : "158", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U180", "Parent" : "128"},
	{"ID" : "159", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U181", "Parent" : "128"},
	{"ID" : "160", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagncg_U182", "Parent" : "128"},
	{"ID" : "161", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagncg_U183", "Parent" : "128"},
	{"ID" : "162", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U184", "Parent" : "128"},
	{"ID" : "163", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagvdy_U185", "Parent" : "128"},
	{"ID" : "164", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U186", "Parent" : "128"},
	{"ID" : "165", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U187", "Parent" : "128"},
	{"ID" : "166", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagwdI_U188", "Parent" : "128"},
	{"ID" : "167", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U189", "Parent" : "128"},
	{"ID" : "168", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagtde_U190", "Parent" : "128"},
	{"ID" : "169", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U191", "Parent" : "128"},
	{"ID" : "170", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U192", "Parent" : "128"},
	{"ID" : "171", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U193", "Parent" : "128"},
	{"ID" : "172", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U194", "Parent" : "128"},
	{"ID" : "173", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagudo_U195", "Parent" : "128"},
	{"ID" : "174", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagvdy_U196", "Parent" : "128"},
	{"ID" : "175", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U197", "Parent" : "128"},
	{"ID" : "176", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U198", "Parent" : "128"},
	{"ID" : "177", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U199", "Parent" : "128"},
	{"ID" : "178", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagqcK_U200", "Parent" : "128"},
	{"ID" : "179", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagvdy_U201", "Parent" : "128"},
	{"ID" : "180", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U202", "Parent" : "128"},
	{"ID" : "181", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagudo_U203", "Parent" : "128"},
	{"ID" : "182", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U204", "Parent" : "128"},
	{"ID" : "183", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagwdI_U205", "Parent" : "128"},
	{"ID" : "184", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U206", "Parent" : "128"},
	{"ID" : "185", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U207", "Parent" : "128"},
	{"ID" : "186", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagvdy_U208", "Parent" : "128"},
	{"ID" : "187", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagtde_U209", "Parent" : "128"},
	{"ID" : "188", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U210", "Parent" : "128"},
	{"ID" : "189", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U211", "Parent" : "128"},
	{"ID" : "190", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagxdS_U212", "Parent" : "128"},
	{"ID" : "191", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagudo_U213", "Parent" : "128"},
	{"ID" : "192", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U214", "Parent" : "128"},
	{"ID" : "193", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagyd2_U215", "Parent" : "128"},
	{"ID" : "194", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagudo_U216", "Parent" : "128"},
	{"ID" : "195", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U217", "Parent" : "128"},
	{"ID" : "196", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagrcU_U218", "Parent" : "128"},
	{"ID" : "197", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagzec_U219", "Parent" : "128"},
	{"ID" : "198", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagocq_U220", "Parent" : "128"},
	{"ID" : "199", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagpcA_U221", "Parent" : "128"},
	{"ID" : "200", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_compute_layer_0_0_0_s_fu_42.myproject_galapagqcK_U222", "Parent" : "128"},
	{"ID" : "201", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_softmax_fu_58", "Parent" : "0", "Child" : ["202", "203"],
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
	{"ID" : "202", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_softmax_fu_58.exp_table2_U", "Parent" : "201"},
	{"ID" : "203", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_softmax_fu_58.invert_table3_U", "Parent" : "201"},
	{"ID" : "204", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.call_ret2_relu_fu_72", "Parent" : "0",
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
	{"Name" : "Latency", "Min" : "14", "Max" : "14"}
	, {"Name" : "Interval", "Min" : "1", "Max" : "1"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	data_V { ap_vld {  { data_V_ap_vld in_vld 0 1 }  { data_V in_data 0 768 } } }
}
