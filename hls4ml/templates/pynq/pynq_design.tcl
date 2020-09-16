set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]

create_project project_1 ${myproject}_pynq -part xc7z020clg400-1 -force

set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]
set_property  ip_repo_paths  ${myproject}_prj [current_project]
update_ip_catalog

# Create Block Designer design
create_bd_design "design_1"
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system7_0]
create_bd_cell -type ip -vlnv xilinx.com:hls:${myproject}_axi:1.0 ${myproject}_axi_0
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/processing_system7_0/M_AXI_GP0} Slave {/${myproject}_axi_0/s_axi_AXILiteS} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins ${myproject}_axi_0/s_axi_AXILiteS]

make_wrapper -files [get_files ./${myproject}_pynq/project_1.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse ./${myproject}_pynq/project_1.srcs/sources_1/bd/design_1/hdl/design_1_wrapper.v

launch_runs synth_1
wait_on_run -timeout 360 synth_1
launch_runs impl_1 -to_step write_bitstream
