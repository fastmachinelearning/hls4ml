set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]

create_project project_1 ${project_name}_vivado_accelerator -part xc7z020clg400-1 -force

set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]
set_property  ip_repo_paths  ${project_name}_prj [current_project]
update_ip_catalog

# Create Block Designer design
create_bd_design "design_1"
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system7_0]
create_bd_cell -type ip -vlnv xilinx.com:hls:${project_name}_axi:1.0 ${project_name}_axi_0
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/processing_system7_0/M_AXI_GP0} Slave {/${project_name}_axi_0/s_axi_AXILiteS} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins ${project_name}_axi_0/s_axi_AXILiteS]

make_wrapper -files [get_files ./${project_name}_vivado_accelerator/project_1.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse ./${project_name}_vivado_accelerator/project_1.srcs/sources_1/bd/design_1/hdl/design_1_wrapper.v

reset_run impl_1
reset_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 6
wait_on_run -timeout 360 impl_1

open_run impl_1
report_utilization -file util.rpt -hierarchical -hierarchical_percentages
