 

onerror {resume}
quietly WaveActivateNextPane {} 0

add wave -noupdate -divider input1_rsc 
add wave -noupdate /uvm_root/uvm_test_top/environment/agent_inst_name/agent_inst_name_monitor/txn_stream
add wave -noupdate -group input1_rsc_bus /hdl_top/input1_rsc_bus/*
add wave -noupdate -divider output1_rsc 
add wave -noupdate /uvm_root/uvm_test_top/environment/agent_inst_name/agent_inst_name_monitor/txn_stream
add wave -noupdate -group output1_rsc_bus /hdl_top/output1_rsc_bus/*
add wave -noupdate -divider const_size_in_1_rsc 
add wave -noupdate /uvm_root/uvm_test_top/environment/agent_inst_name/agent_inst_name_monitor/txn_stream
add wave -noupdate -group const_size_in_1_rsc_bus /hdl_top/const_size_in_1_rsc_bus/*
add wave -noupdate -divider const_size_out_1_rsc 
add wave -noupdate /uvm_root/uvm_test_top/environment/agent_inst_name/agent_inst_name_monitor/txn_stream
add wave -noupdate -group const_size_out_1_rsc_bus /hdl_top/const_size_out_1_rsc_bus/*

TreeUpdate [SetDefaultTree]
quietly wave cursor active 0
configure wave -namecolwidth 472
configure wave -valuecolwidth 100
configure wave -justifyvalue left
configure wave -signalnamewidth 0
configure wave -snapdistance 10
configure wave -datasetprefix 0
configure wave -rowmargin 4
configure wave -childrowmargin 2
configure wave -gridoffset 0
configure wave -gridperiod 1
configure wave -griddelta 40
configure wave -timeline 0
configure wave -timelineunits ns
update
WaveRestoreZoom {27 ns} {168 ns}

