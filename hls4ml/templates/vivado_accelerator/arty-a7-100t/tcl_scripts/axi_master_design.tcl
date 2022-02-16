set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]

set eembc_power 1

# Project names
set project_name "project_1"
set design_name "design_1"
set hls_solution_name "solution1"
set acc_name "${myproject}_axi"
set part_name "xc7a100tcsg324-1"
set board_name "digilentinc.com:arty-a7-100:part0:1.0"

# Set board and chip part names
create_project ${project_name} ${myproject}_vivado_accelerator -part ${part_name} -force
set_property board_part ${board_name} [current_project]

# Create block design
create_bd_design ${design_name}

# Setup IP repo
#set_property  ip_repo_paths ${myproject}_prj [current_project]
set_property  ip_repo_paths ${myproject}_prj/${hls_solution_name}/impl/ip [current_project]
update_ip_catalog

# Create clock wizard
create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz_0
apply_board_connection -board_interface "sys_clock" -ip_intf "clk_wiz_0/clock_CLK_IN1" -diagram ${design_name}
set_property name clk_wizard [get_bd_cells clk_wiz_0]
set_property -dict [list CONFIG.CLKOUT2_USED {true} CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {166.667} CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {200.00} CONFIG.MMCM_CLKOUT0_DIVIDE_F {6.000} CONFIG.MMCM_CLKOUT1_DIVIDE {5} CONFIG.NUM_OUT_CLKS {2} CONFIG.CLKOUT1_JITTER {118.758} CONFIG.CLKOUT2_JITTER {114.829} CONFIG.CLKOUT2_PHASE_ERROR {98.575}] [get_bd_cells clk_wizard]
#set_property -dict [list CONFIG.RESET_TYPE {ACTIVE_LOW} CONFIG.RESET_PORT {resetn}] [get_bd_cells clk_wizard]

# Create MIG
create_bd_cell -type ip -vlnv xilinx.com:ip:mig_7series:4.2 mig_7series_0
apply_board_connection -board_interface "ddr3_sdram" -ip_intf "mig_7series_0/mig_ddr_interface" -diagram ${design_name}

# Wire MIG and clock wizard
delete_bd_objs [get_bd_nets clk_ref_i_1] [get_bd_ports clk_ref_i]
delete_bd_objs [get_bd_nets sys_clk_i_1] [get_bd_ports sys_clk_i]
connect_bd_net [get_bd_pins clk_wizard/clk_out2] [get_bd_pins mig_7series_0/clk_ref_i]
connect_bd_net [get_bd_pins clk_wizard/clk_out1] [get_bd_pins mig_7series_0/sys_clk_i]

# Setup reset
#set_property -dict [list CONFIG.RESET_BOARD_INTERFACE {reset}] [get_bd_cells clk_wizard]
apply_bd_automation -rule xilinx.com:bd_rule:board -config { Board_Interface {reset ( System Reset ) } Manual_Source {New External Port (ACTIVE_LOW)}}  [get_bd_pins mig_7series_0/sys_rst]

# Create instance of MicroBlaze
create_bd_cell -type ip -vlnv xilinx.com:ip:microblaze:11.0 microblaze_mcu
apply_bd_automation -rule xilinx.com:bd_rule:microblaze -config { \
    axi_intc {0} \
    axi_periph {Enabled} \
    cache {16KB} \
    clk {/mig_7series_0/ui_clk (83 MHz)} \
    debug_module {Debug Only} \
    ecc {None} \
    local_mem {128KB} \
    preset {None} } [get_bd_cells microblaze_mcu]

# Enable full FPU
set_property -dict [list CONFIG.C_USE_FPU {2}] [get_bd_cells microblaze_mcu]

# Create UART interface
#create_bd_cell -type ip -vlnv xilinx.com:ip:axi_uart16550:2.0 axi_uart
#apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/mig_7series_0/ui_clk (83 MHz)} Clk_slave {Auto} Clk_xbar {Auto} Master {/microblaze_mcu (Periph)} Slave {/axi_uart/S_AXI} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins axi_uart/S_AXI]
#apply_bd_automation -rule xilinx.com:bd_rule:board -config { Board_Interface {usb_uart ( USB UART ) } Manual_Source {Auto}}  [get_bd_intf_pins axi_uart/UART]

# Create UART-lite interface
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_uartlite:2.0 axi_uart
if { ${eembc_power} } {
    set_property -dict [list CONFIG.C_BAUDRATE {9600}] [get_bd_cells axi_uart]
} else {
    apply_board_connection -board_interface "usb_uart" -ip_intf "axi_uart/UART" -diagram ${design_name}
    set_property -dict [list CONFIG.C_BAUDRATE {115200}] [get_bd_cells axi_uart]
}
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {/mig_7series_0/ui_clk (83 MHz)} \
    Clk_slave {Auto} \
    Clk_xbar {Auto} \
    Master {/microblaze_mcu (Periph)} \
    Slave {/axi_uart/S_AXI} \
    intc_ip {New AXI Interconnect} \
    master_apm {0}} [get_bd_intf_pins axi_uart/S_AXI]

# Forward UART interface to PMOD pins
if { ${eembc_power} } {
    create_bd_port -dir O pmod_uart_txd
    create_bd_port -dir I pmod_uart_rxd
    connect_bd_net [get_bd_pins /axi_uart/tx] [get_bd_ports pmod_uart_txd]
    connect_bd_net [get_bd_pins /axi_uart/rx] [get_bd_ports pmod_uart_rxd]
    add_files -fileset constrs_1 -norecurse uart_pmod.xdc
}

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {/mig_7series_0/ui_clk (83 MHz)} \
    Clk_slave {/mig_7series_0/ui_clk (83 MHz)} \
    Clk_xbar {/mig_7series_0/ui_clk (83 MHz)} \
    Master {/microblaze_mcu (Cached)} \
    Slave {/mig_7series_0/S_AXI} \
    intc_ip {Auto} master_apm {0} } [get_bd_intf_pins mig_7series_0/S_AXI]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {/mig_7series_0/ui_clk (83 MHz)} \
    Clk_slave {Auto} \
    Clk_xbar {Auto} \
    Master {/microblaze_mcu (Periph)} \
    Slave {/axi_uart/S_AXI} \
    intc_ip {New AXI Interconnect} \
    master_apm {0} } [get_bd_intf_pins axi_uart/S_AXI]

# Add accelerator and connect s-axi interface
create_bd_cell -type ip -vlnv xilinx.com:hls:${acc_name}:1.0 ${acc_name}
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/mig_7series_0/ui_clk (83 MHz)} Clk_slave {Auto} Clk_xbar {/mig_7series_0/ui_clk (83 MHz)} Master {/microblaze_mcu (Periph)} Slave {/${acc_name}/s_axi_CTRL_BUS} intc_ip {/microblaze_mcu_axi_periph} master_apm {0}}  [get_bd_intf_pins ${acc_name}/s_axi_CTRL_BUS]

# Connect m-axi interfaces
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/mig_7series_0/ui_clk (83 MHz)} Clk_slave {/mig_7series_0/ui_clk (83 MHz)} Clk_xbar {/mig_7series_0/ui_clk (83 MHz)} Master {/${acc_name}/m_axi_IN_BUS} Slave {/mig_7series_0/S_AXI} intc_ip {/axi_smc} master_apm {0}}  [get_bd_intf_pins ${acc_name}/m_axi_IN_BUS]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/mig_7series_0/ui_clk (83 MHz)} Clk_slave {/mig_7series_0/ui_clk (83 MHz)} Clk_xbar {/mig_7series_0/ui_clk (83 MHz)} Master {/${acc_name}/m_axi_OUT_BUS} Slave {/mig_7series_0/S_AXI} intc_ip {/axi_smc} master_apm {0}}  [get_bd_intf_pins ${acc_name}/m_axi_OUT_BUS]

# Reset
apply_bd_automation -rule xilinx.com:bd_rule:board -config { Board_Interface {reset ( System Reset ) } Manual_Source {Auto}}  [get_bd_pins clk_wizard/reset]

# Add timer
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_timer:2.0 axi_timer_mcu

# Wire timer
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/mig_7series_0/ui_clk (83 MHz)} Clk_slave {Auto} Clk_xbar {/mig_7series_0/ui_clk (83 MHz)} Master {/microblaze_mcu (Periph)} Slave {/axi_timer_mcu/S_AXI} intc_ip {/microblaze_mcu_axi_periph} master_apm {0}}  [get_bd_intf_pins axi_timer_mcu/S_AXI]

# Add AXI GPIO controlled pin
if { ${eembc_power} } {
    # Add AXI GPIO IP
    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_0
    # Wire it up to a single output pin (to a PMOD)
    set_property -dict [list CONFIG.C_GPIO_WIDTH {1} CONFIG.C_ALL_OUTPUTS {1}] [get_bd_cells axi_gpio_0]
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
        Clk_master {/mig_7series_0/ui_clk (83 MHz)} \
        Clk_slave {Auto} \
        Clk_xbar {/mig_7series_0/ui_clk (83 MHz)} \
        Master {/microblaze_mcu (Periph)} \
        Slave {/axi_gpio_0/S_AXI} \
        intc_ip {/microblaze_mcu_axi_periph} \
        master_apm {0}} [get_bd_intf_pins axi_gpio_0/S_AXI]
    create_bd_port -dir O pmod_pin
    connect_bd_net [get_bd_ports pmod_pin] [get_bd_pins axi_gpio_0/gpio_io_o]

    add_files -fileset constrs_1 -norecurse pin_pmod.xdc
}

# Add Quad SPI for cold boot
if { ${eembc_power} } {
    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_quad_spi:3.2 axi_quad_spi_0
    set_property -dict [list CONFIG.C_SPI_MEMORY {3} CONFIG.C_SPI_MODE {2} CONFIG.C_SCK_RATIO {2}] [get_bd_cells axi_quad_spi_0]
    apply_bd_automation -rule xilinx.com:bd_rule:board -config { Board_Interface {qspi_flash ( Quad SPI Flash ) } Manual_Source {Auto}}  [get_bd_intf_pins axi_quad_spi_0/SPI_0]
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/mig_7series_0/ui_clk (83 MHz)} Clk_slave {Auto} Clk_xbar {/mig_7series_0/ui_clk (83 MHz)} Master {/microblaze_mcu (Periph)} Slave {/axi_quad_spi_0/AXI_LITE} intc_ip {/microblaze_mcu_axi_periph} master_apm {0}}  [get_bd_intf_pins axi_quad_spi_0/AXI_LITE]
    set_property -dict [list CONFIG.CLKOUT3_USED {true} CONFIG.CLKOUT3_REQUESTED_OUT_FREQ {50} CONFIG.MMCM_CLKOUT2_DIVIDE {20} CONFIG.NUM_OUT_CLKS {3} CONFIG.CLKOUT3_JITTER {151.636} CONFIG.CLKOUT3_PHASE_ERROR {98.575}] [get_bd_cells clk_wizard]
    connect_bd_net [get_bd_pins clk_wizard/clk_out3] [get_bd_pins axi_quad_spi_0/ext_spi_clk]
    set_property -dict [list CONFIG.C_SPI_MEMORY {3}] [get_bd_cells axi_quad_spi_0]
    add_files -fileset constrs_1 -norecurse qspi.xdc
}

# Validate the design block we created
validate_bd_design

# Save design
save_bd_design

# Top level wrapper
#make_wrapper -files [get_files ./${myproject}_vivado_accelerator/${project_name}.srcs/sources_1/bd/${design_name}/${design_name}.bd] -top
#add_files -norecurse ./${myproject}_vivado_accelerator/${project_name}.srcs/sources_1/bd/${design_name}/hdl/${design_name}_wrapper.v
add_files -norecurse $design_name\_wrapper.v

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
