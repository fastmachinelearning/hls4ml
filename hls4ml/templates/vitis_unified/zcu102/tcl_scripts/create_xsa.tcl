################################################################
# create_xsa.tcl
#
# Sources zcu102_axi_stream_platform.tcl to build the block design, then creates
# the HDL wrapper and exports the XSA hardware description file.
#
# Usage:
#   cd <tcl_scripts_dir>
#   vivado -mode batch -source create_xsa.tcl
#
# Output:
#   output/zcu102_axi_stream_platform.xsa
#
# Requirements:
#   Vivado 2023.2 with ZCU102 board files installed
#
# Note: For a clean run, remove the myproj/ directory first if it
#       exists from a previous run.
################################################################

set script_dir [file dirname [file normalize [info script]]]
set output_dir [file join $script_dir "output"]
set xsa_path   [file join $output_dir "zcu102_axi_stream_platform.xsa"]

file mkdir $output_dir

# Change to script directory so zcu102_axi_stream_platform.tcl is found
cd $script_dir

# ----------------------------------------------------------------
# Build block design (creates project + BD via zcu102_axi_stream_platform.tcl)
# ----------------------------------------------------------------
source [file join $script_dir "zcu102_axi_stream_platform.tcl"]

# ----------------------------------------------------------------
# Create HDL wrapper
# ----------------------------------------------------------------
set bd_file [get_files -norecurse "vitis_design.bd"]
make_wrapper -files $bd_file -top

# make_wrapper creates {bd_base_name}_wrapper.v for vitis_design.bd
set wrapper_name "vitis_design_wrapper"
set proj_dir [get_property directory [current_project]]
add_files -norecurse [file join $proj_dir "project_1.gen" "sources_1" "bd" "vitis_design" "hdl" "${wrapper_name}.v"]
set_property top $wrapper_name [current_fileset]
update_compile_order -fileset sources_1

# ----------------------------------------------------------------
# Generate block design outputs (required before write_hw_platform)
# ----------------------------------------------------------------
generate_target all [get_files vitis_design.bd]

# ----------------------------------------------------------------
# Platform design intent (required for v++ accelerated platform)
# ----------------------------------------------------------------
set_property platform.default_output_type "sd_card" [current_project]
set_property platform.design_intent.embedded "true" [current_project]
set_property platform.design_intent.server_managed "false" [current_project]
set_property platform.design_intent.external_host "false" [current_project]
set_property platform.design_intent.datacenter "false" [current_project]

# ----------------------------------------------------------------
# Export XSA as extensible pre-synthesis platform (for v++ link)
# ----------------------------------------------------------------
write_hw_platform -hw -force -file $xsa_path

puts "INFO: XSA written to $xsa_path"
