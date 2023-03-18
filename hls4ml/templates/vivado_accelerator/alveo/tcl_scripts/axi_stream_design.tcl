set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]

create_project project_1 ${project_name}_vivado_accelerator -part ${part} -force

set_property  ip_repo_paths  ${project_name}_prj [current_project]
update_ip_catalog


add_files -scan_for_includes [list src/krnl_rtl_int.sv src/krnl_rtl_axi_read_master.sv src/krnl_rtl_counter.sv src/${project_name}_kernel.v src/krnl_rtl_axi_write_master.sv src/krnl_rtl_control_s_axi.v]
import_files [list src/krnl_rtl_int.sv src/krnl_rtl_axi_read_master.sv src/krnl_rtl_counter.sv src/${project_name}_kernel.v src/krnl_rtl_axi_write_master.sv src/krnl_rtl_control_s_axi.v]



create_ip -vlnv xilinx.com:hls:${project_name}_axi:1.0 -module_name ${project_name}_axi_0


ipx::package_project -root_dir hls4ml_IP -vendor fastmachinelearning.org -library hls4ml -taxonomy /UserIP -import_files -set_current false
ipx::unload_core hls4ml_IP/component.xml
ipx::edit_ip_in_project -upgrade true -name tmp_edit_project -directory hls4ml_IP hls4ml_IP/component.xml
ipx::associate_bus_interfaces -busif m_axi_gmem -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif s_axi_control -clock ap_clk [ipx::current_core]
ipx::add_bus_parameter FREQ_HZ [ipx::get_bus_interfaces ap_clk -of_objects [ipx::current_core]]



set_property value_resolve_type user [ipx::get_bus_parameters -of [::ipx::get_bus_interfaces -of [ipx::current_core] *clk*] "FREQ_HZ"]



ipx::add_register CTRL [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]
ipx::add_register GIER [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]
ipx::add_register IP_IER [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]
ipx::add_register IP_ISR [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]
ipx::add_register fifo_in [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]
ipx::add_register fifo_out [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]
ipx::add_register length_r_in [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]
ipx::add_register length_r_out [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]


# Commands to set the descrtiprion, address offset and size

# CTRL register properties
set_property Description "Control Signals" [ipx::get_registers CTRL -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property Address_Offset 0x000 [ipx::get_registers CTRL -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property Size 32 [ipx::get_registers CTRL -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]

# GIER register properties
set_property Description "Global Interrupt Enable Register" [ipx::get_registers GIER -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property Address_Offset 0x004 [ipx::get_registers GIER -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property Size 32 [ipx::get_registers GIER -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]

# IP_IER register properties
set_property Description "IP Interrupt Enable Register" [ipx::get_registers IP_IER -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property Address_Offset 0x008 [ipx::get_registers IP_IER -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property Size 32 [ipx::get_registers IP_IER -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]

# IP_ISR register properties
set_property Description "IP Interrupt Status Register" [ipx::get_registers IP_ISR -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property Address_Offset 0x00C [ipx::get_registers IP_ISR -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property Size 32 [ipx::get_registers IP_ISR -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]

# fifo_in register properties
set_property Description "fifo_in pointer argument" [ipx::get_registers fifo_in -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property Address_Offset 0x010 [ipx::get_registers fifo_in -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property Size 64 [ipx::get_registers fifo_in -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]

# fifo_out register properties
set_property Description "fifo_out pointer argument" [ipx::get_registers fifo_out -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property Address_Offset 0x01C [ipx::get_registers fifo_out -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property Size 64 [ipx::get_registers fifo_out -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]

# length_r_in register properties
set_property Description "length_r_in value" [ipx::get_registers length_r_in -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property Address_Offset 0x028 [ipx::get_registers length_r_in -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property Size 32 [ipx::get_registers length_r_in -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]

# length_r_out register properties
set_property Description "length_r_out value" [ipx::get_registers length_r_out -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property Address_Offset 0x030 [ipx::get_registers length_r_out -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property Size 32 [ipx::get_registers length_r_out -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]

ipx::add_register_parameter ASSOCIATED_BUSIF [ipx::get_registers fifo_in -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
ipx::add_register_parameter ASSOCIATED_BUSIF [ipx::get_registers fifo_out -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]

# Commands to set m_axi_gmem as value in the register ASSOCIATED_BUSIF parameters
set_property Value m_axi_gmem [ipx::get_register_parameters ASSOCIATED_BUSIF -of_objects [ipx::get_registers fifo_in -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]]
set_property Value m_axi_gmem [ipx::get_register_parameters ASSOCIATED_BUSIF -of_objects [ipx::get_registers fifo_out -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]]

set core [ipx::current_core]


set_property xpm_libraries {XPM_CDC XPM_MEMORY XPM_FIFO} $core
set_property sdx_kernel true $core
set_property sdx_kernel_type rtl $core



set_property core_revision 2 [ipx::current_core]
ipx::update_source_project_archive -component [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]
ipx::check_integrity -quiet [ipx::current_core]
ipx::archive_core hls4ml_IP/fastmachinelearning.org_hls4ml_krnl_rtl_1.0.zip [ipx::current_core]
current_project project_1


package_xo  -force -xo_path xo_files/${project_name}_kernel.xo -kernel_name krnl_rtl -ip_directory hls4ml_IP
