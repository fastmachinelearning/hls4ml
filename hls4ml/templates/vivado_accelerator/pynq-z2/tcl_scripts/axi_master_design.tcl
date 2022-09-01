set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]

# Project names
set design_name "design_1"
set hls_solution_name "solution1"
set ps_name "processing_system7_0"
set acc_name "${project_name}_axi_0"

# Board and chip part names
create_project ${project_name} ${project_name}_vivado_accelerator -part xc7z020clg400-1 -force
set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]

# Create block design
create_bd_design ${design_name}

# Setup IP repo
#set_property  ip_repo_paths ${project_name}_prj [current_project]
set_property  ip_repo_paths ${project_name}_prj/${hls_solution_name}/impl/ip [current_project]
update_ip_catalog

# Create and setup PS
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ${ps_name}
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config " \
    make_external {FIXED_IO, DDR} \
    apply_board_preset {1} \
    Master {Disable} \
    Slave {Disable} " [get_bd_cells ${ps_name}]
set_property -dict [list \
    CONFIG.PCW_USE_S_AXI_GP0 {1} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
    CONFIG.PCW_IRQ_F2P_INTR {1}\
    ] [get_bd_cells ${ps_name}]

# Create accelerator
create_bd_cell -type ip -vlnv xilinx.com:hls:myproject_axi:1.0 ${acc_name}

# Wiring
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config " \
    Clk_master {Auto} \
    Clk_slave {Auto} \
    Clk_xbar {Auto} \
    Master /${ps_name}/M_AXI_GP0 \
    Slave /${acc_name}/s_axi_CTRL_BUS \
    intc_ip {New AXI Interconnect} \
    master_apm {0}" [get_bd_intf_pins ${acc_name}/s_axi_CTRL_BUS]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config " \
    Clk_master {Auto} \
    Clk_slave {Auto} \
    Clk_xbar {Auto} \
    Master /${acc_name}/m_axi_IN_BUS \
    Slave /${ps_name}/S_AXI_GP0 \
    intc_ip {Auto} \
    master_apm {0}" [get_bd_intf_pins ${ps_name}/S_AXI_GP0]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config " \
    Clk_master /${ps_name}/FCLK_CLK0 (100 MHz) \
    Clk_slave /${ps_name}/FCLK_CLK0 (100 MHz) \
    Clk_xbar /${ps_name}/FCLK_CLK0 (100 MHz) \
    Master /${acc_name}/m_axi_OUT_BUS \
    Slave /${ps_name}/S_AXI_GP0 \
    intc_ip {/axi_smc} \
    master_apm {0}" [get_bd_intf_pins ${acc_name}/m_axi_OUT_BUS]

# Wiring interrupt signal
connect_bd_net [get_bd_pins ${acc_name}/interrupt] [get_bd_pins ${ps_name}/IRQ_F2P]

# Top level wrapper
make_wrapper -files [get_files ./${project_name}_vivado_accelerator/${project_name}.srcs/sources_1/bd/${design_name}/${design_name}.bd] -top
add_files -norecurse ./${project_name}_vivado_accelerator/${project_name}.srcs/sources_1/bd/${design_name}/hdl/${design_name}_wrapper.v

# Memory mapping
delete_bd_objs [get_bd_addr_segs ${project_name}/Data_m_axi_IN_BUS/SEG_${ps_name}_GP0_QSPI_LINEAR]
delete_bd_objs [get_bd_addr_segs -excluded ${acc_name}/Data_m_axi_IN_BUS/SEG_${ps_name}_GP0_IOP]
delete_bd_objs [get_bd_addr_segs -excluded ${acc_name}/Data_m_axi_IN_BUS/SEG_${ps_name}_GP0_M_AXI_GP0]
delete_bd_objs [get_bd_addr_segs ${acc_name}/Data_m_axi_OUT_BUS/SEG_${ps_name}_GP0_QSPI_LINEAR]
delete_bd_objs [get_bd_addr_segs -excluded ${acc_name}/Data_m_axi_OUT_BUS/SEG_${ps_name}_GP0_IOP]
delete_bd_objs [get_bd_addr_segs -excluded ${acc_name}/Data_m_axi_OUT_BUS/SEG_${ps_name}_GP0_M_AXI_GP0]

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
file copy -force ${project_name}_vivado_accelerator/${project_name}.runs/impl_1/${design_name}_wrapper.sysdef ./hdf/${design_name}_wrapper.hdf
