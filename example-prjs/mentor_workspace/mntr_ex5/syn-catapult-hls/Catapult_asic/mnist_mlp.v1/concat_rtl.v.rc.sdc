# written for flow package RTLCompiler 
set sdc_version 1.7 

set_operating_conditions typical
set_load 2.0 [all_outputs]
## driver/slew constraints on inputs
set data_inputs [list input1_rsc_vld {input1_rsc_dat[*]} output1_rsc_rdy {w2_rsc_Q3[*]} {w2_rsc_Q2[*]} b2_rsc_vld {b2_rsc_dat[*]} {w4_rsc_Q3[*]} {w4_rsc_Q2[*]} b4_rsc_vld {b4_rsc_dat[*]} {w6_rsc_0_0_dat[*]} {w6_rsc_1_0_dat[*]} {w6_rsc_2_0_dat[*]} {w6_rsc_3_0_dat[*]} {w6_rsc_4_0_dat[*]} {w6_rsc_5_0_dat[*]} {w6_rsc_6_0_dat[*]} {w6_rsc_7_0_dat[*]} {w6_rsc_8_0_dat[*]} {w6_rsc_9_0_dat[*]} b6_rsc_vld {b6_rsc_dat[*]}]
set_driving_cell -no_design_rule -library NangateOpenCellLibrary -lib_cell BUF_X2 -pin Z $data_inputs
create_clock -name clk -period 10.0 -waveform { 0.0 5.0 } [get_ports {clk}]
set_clock_uncertainty 0.0 [get_clocks {clk}]

create_clock -name virtual_io_clk -period 10.0
## IO TIMING CONSTRAINTS
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {rst}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {input1_rsc_dat[*]}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {input1_rsc_vld}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {input1_rsc_rdy}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {output1_rsc_dat[*]}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {output1_rsc_vld}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {output1_rsc_rdy}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {const_size_in_1_rsc_dat[*]}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {const_size_in_1_rsc_vld}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {const_size_out_1_rsc_dat[*]}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {const_size_out_1_rsc_vld}]
set_max_delay 10.0 -from [all_inputs] -to [all_outputs]
set_output_delay -clock [get_clocks {clk}] 0.01 [get_ports {w2_rsc_CE2}]
set_output_delay -clock [get_clocks {clk}] 0.01 [get_ports {w2_rsc_A2[*]}]
set_input_delay -clock [get_clocks {clk}] 0.1 [get_ports {w2_rsc_Q2[*]}]
set_output_delay -clock [get_clocks {clk}] 0.01 [get_ports {w2_rsc_CE3}]
set_output_delay -clock [get_clocks {clk}] 0.01 [get_ports {w2_rsc_A3[*]}]
set_input_delay -clock [get_clocks {clk}] 0.1 [get_ports {w2_rsc_Q3[*]}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {b2_rsc_dat[*]}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {b2_rsc_vld}]
set_output_delay -clock [get_clocks {clk}] 0.01 [get_ports {w4_rsc_CE2}]
set_output_delay -clock [get_clocks {clk}] 0.01 [get_ports {w4_rsc_A2[*]}]
set_input_delay -clock [get_clocks {clk}] 0.1 [get_ports {w4_rsc_Q2[*]}]
set_output_delay -clock [get_clocks {clk}] 0.01 [get_ports {w4_rsc_CE3}]
set_output_delay -clock [get_clocks {clk}] 0.01 [get_ports {w4_rsc_A3[*]}]
set_input_delay -clock [get_clocks {clk}] 0.1 [get_ports {w4_rsc_Q3[*]}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {b4_rsc_dat[*]}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {b4_rsc_vld}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {w6_rsc_0_0_dat[*]}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {w6_rsc_1_0_dat[*]}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {w6_rsc_2_0_dat[*]}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {w6_rsc_3_0_dat[*]}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {w6_rsc_4_0_dat[*]}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {w6_rsc_5_0_dat[*]}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {w6_rsc_6_0_dat[*]}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {w6_rsc_7_0_dat[*]}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {w6_rsc_8_0_dat[*]}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {w6_rsc_9_0_dat[*]}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {b6_rsc_dat[*]}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {b6_rsc_vld}]

