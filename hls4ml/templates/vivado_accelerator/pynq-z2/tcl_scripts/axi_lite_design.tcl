set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]
source [file join $tcldir statistics.tcl]

set outputDir vivado_reports
set reportBase ${project_name}_report
set implJobs 4
if {![catch {set implJobs [exec nproc]}]} {
  if {$implJobs < 1} { set implJobs 1 }
}
file mkdir $outputDir

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
launch_runs impl_1 -to_step write_bitstream -jobs $implJobs
wait_on_run -timeout 360 impl_1

open_run impl_1
write_checkpoint -force $outputDir/post_route_system.dcp
report_route_status -file $outputDir/post_route_status_system.rpt
report_timing_summary -file $outputDir/post_route_timing_summary_system.rpt
report_power -file $outputDir/post_route_power_system.rpt
report_drc -file $outputDir/post_imp_drc_system.rpt
report_utilization -file $outputDir/post_route_util_system.rpt
report_utilization -hierarchical -hierarchical_percentages -file $outputDir/post_route_util_hier_system.rpt
dump_statistics $outputDir $reportBase "post_route_system"
close_design
close_project
