set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]

# Project names
set project_name "project_1"
set design_name "design_1"
set hls_solution_name "solution1"
set ps_name "zynq_ultra_ps_e_0"
set acc_name "${myproject}_axi_0"

# Board and chip part names
create_project ${project_name} ${myproject}_vivado_accelerator -part xczu9eg-ffvb1156-2-e -force
set_property board_part avnet.com:ultra96v2:part0:1.2 [current_project]

# Create block design
create_bd_design ${design_name}

# Setup IP repo
#set_property  ip_repo_paths ${myproject}_prj [current_project]
set_property  ip_repo_paths ${myproject}_prj/${hls_solution_name}/impl/ip [current_project]
update_ip_catalog

# Create and setup PS
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.3 ${ps_name}
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1" }  [get_bd_cells ${ps_name}]
set_property -dict [list CONFIG.PSU__USE__S_AXI_GP0 {1} CONFIG.PSU__SAXIGP0__DATA_WIDTH {32}] [get_bd_cells ${ps_name}]

# Create accelerator
create_bd_cell -type ip -vlnv xilinx.com:hls:myproject_axi:1.0 ${acc_name}

# Wiring
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {Auto} \
    Clk_slave {Auto} \
    Clk_xbar {Auto} \
    Master "/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD" \
    Slave "/myproject_axi_0/s_axi_CTRL_BUS" \
    intc_ip {New AXI Interconnect} \
    master_apm {0}} [get_bd_intf_pins ${acc_name}/s_axi_CTRL_BUS]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {Auto} \
    Clk_slave "/zynq_ultra_ps_e_0/pl_clk0 (100 MHz)" \
    Clk_xbar "/zynq_ultra_ps_e_0/pl_clk0 (100 MHz)" \
    Master "/zynq_ultra_ps_e_0/M_AXI_HPM1_FPD" \
    Slave "/myproject_axi_0/s_axi_CTRL_BUS" \
    intc_ip {/ps8_0_axi_periph} \
    master_apm {0}} [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM1_FPD]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
   Clk_master "/zynq_ultra_ps_e_0/pl_clk0 (100 MHz)" \
   Clk_slave {Auto} \
   Clk_xbar {Auto} \
   Master "/myproject_axi_0/m_axi_IN_BUS" \
   Slave "/zynq_ultra_ps_e_0/S_AXI_HPC0_FPD" \
   intc_ip {Auto} \
   master_apm {0}} [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC0_FPD]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
   Clk_master "/zynq_ultra_ps_e_0/pl_clk0 (100 MHz)" \
   Clk_slave "/zynq_ultra_ps_e_0/pl_clk0 (100 MHz)" \
   Clk_xbar "/zynq_ultra_ps_e_0/pl_clk0 (100 MHz)" \
   Master "/myproject_axi_0/m_axi_OUT_BUS" \
   Slave "/zynq_ultra_ps_e_0/S_AXI_HPC0_FPD" \
   intc_ip {/axi_smc} \
   master_apm {0}} [get_bd_intf_pins ${acc_name}/m_axi_OUT_BUS]

# Wiring interrupt signal
connect_bd_net [get_bd_pins ${acc_name}/interrupt] [get_bd_pins ${ps_name}/pl_ps_irq0]

# Top level wrapper
make_wrapper -files [get_files ./${myproject}_vivado_accelerator/${project_name}.srcs/sources_1/bd/${design_name}/${design_name}.bd] -top
add_files -norecurse ./${myproject}_vivado_accelerator/${project_name}.srcs/sources_1/bd/${design_name}/hdl/${design_name}_wrapper.v

# Memory mapping
delete_bd_objs [get_bd_addr_segs -excluded ${acc_name}/Data_m_axi_IN_BUS/SEG_${ps_name}_HPC0_LPS_OCM]
delete_bd_objs [get_bd_addr_segs -excluded ${acc_name}/Data_m_axi_OUT_BUS/SEG_${ps_name}_HPC0_LPS_OCM]

# Run synthesis and implementation
reset_run impl_1
reset_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 6
wait_on_run -timeout 360 impl_1

# Reporting
open_run impl_1
report_utilization -file util.rpt -hierarchical -hierarchical_percentages

# Export HDF file for SDK flow
file mkdir ./hdf
file copy -force ${myproject}_vivado_accelerator/${project_name}.runs/impl_1/${design_name}_wrapper.sysdef ./hdf/${design_name}_wrapper.hdf
