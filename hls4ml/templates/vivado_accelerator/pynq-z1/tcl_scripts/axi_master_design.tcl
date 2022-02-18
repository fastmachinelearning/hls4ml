set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]

# Project names
set project_name "project_1"
set design_name "design_1"
set hls_solution_name "solution1"
set ps_name "processing_system7_0"
set acc_name "${myproject}_axi_0"
set part_name "xc7z020clg400-1"
set board_name "www.digilentinc.com:pynq-z1:part0:1.0"

# Set board and chip part names
create_project ${project_name} ${myproject}_vivado_accelerator -part ${part_name} -force
set_property board_part ${board_name} [current_project]

# Create block design
create_bd_design ${design_name}

# Setup IP repo
#set_property  ip_repo_paths ${myproject}_prj [current_project]
set_property  ip_repo_paths ${myproject}_prj/${hls_solution_name}/impl/ip [current_project]
update_ip_catalog

# Create and setup PS
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ${ps_name}
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells ${ps_name}]
set_property -dict [list CONFIG.PCW_USE_S_AXI_GP0 {1} CONFIG.PCW_USE_FABRIC_INTERRUPT {1} CONFIG.PCW_IRQ_F2P_INTR {1}] [get_bd_cells ${ps_name}]

# Create accelerator
create_bd_cell -type ip -vlnv xilinx.com:hls:myproject_axi:1.0 ${acc_name}

# Wiring
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {Auto} \
    Clk_slave {Auto} \
    Clk_xbar {Auto} \
    Master {/myproject_axi_0/m_axi_IN_BUS} \
    Slave {/processing_system7_0/S_AXI_GP0} \
    intc_ip {Auto} \
    master_apm {0}} [get_bd_intf_pins processing_system7_0/S_AXI_GP0]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {Auto} \
    Clk_slave {Auto} \
    Clk_xbar {Auto} \
    Master {/processing_system7_0/M_AXI_GP0} \
    Slave {/myproject_axi_0/s_axi_CTRL_BUS} \
    intc_ip {New AXI Interconnect} \
    master_apm {0}} [get_bd_intf_pins myproject_axi_0/s_axi_CTRL_BUS]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {/processing_system7_0/FCLK_CLK0 (100 MHz)} \
    Clk_slave {/processing_system7_0/FCLK_CLK0 (100 MHz)} \
    Clk_xbar {/processing_system7_0/FCLK_CLK0 (100 MHz)} \
    Master {/myproject_axi_0/m_axi_OUT_BUS} \
    Slave {/processing_system7_0/S_AXI_GP0} \
    intc_ip {/axi_smc} \
    master_apm {0}} [get_bd_intf_pins myproject_axi_0/m_axi_OUT_BUS]

# Wiring interrupt signal
connect_bd_net [get_bd_pins myproject_axi_0/interrupt] [get_bd_pins processing_system7_0/IRQ_F2P]

# Top level wrapper
make_wrapper -files [get_files ./${myproject}_vivado_accelerator/${project_name}.srcs/sources_1/bd/${design_name}/${design_name}.bd] -top
add_files -norecurse ./${myproject}_vivado_accelerator/${project_name}.srcs/sources_1/bd/${design_name}/hdl/${design_name}_wrapper.v

# Memory mapping
delete_bd_objs [get_bd_addr_segs myproject_axi_0/Data_m_axi_IN_BUS/SEG_processing_system7_0_GP0_QSPI_LINEAR]
delete_bd_objs [get_bd_addr_segs -excluded myproject_axi_0/Data_m_axi_IN_BUS/SEG_processing_system7_0_GP0_IOP]
delete_bd_objs [get_bd_addr_segs -excluded myproject_axi_0/Data_m_axi_IN_BUS/SEG_processing_system7_0_GP0_M_AXI_GP0]
delete_bd_objs [get_bd_addr_segs myproject_axi_0/Data_m_axi_OUT_BUS/SEG_processing_system7_0_GP0_QSPI_LINEAR]
delete_bd_objs [get_bd_addr_segs -excluded myproject_axi_0/Data_m_axi_OUT_BUS/SEG_processing_system7_0_GP0_IOP]
delete_bd_objs [get_bd_addr_segs -excluded myproject_axi_0/Data_m_axi_OUT_BUS/SEG_processing_system7_0_GP0_M_AXI_GP0]

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
