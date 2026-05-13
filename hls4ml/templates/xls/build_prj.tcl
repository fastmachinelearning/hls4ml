# build_prj.tcl
# Usage:
#   vivado -mode batch -nolog -nojournal -source build_prj.tcl --tclargs <project_name> <board_part> <clock_period> <clock_uncertainty> [--pr]

if {[llength $argv] < 4} {
    puts stderr "ERROR: missing arguments\nUsage: vivado -mode batch -nolog -nojournal  -source build_prj.tcl -tclargs <project_name> <board_part> <clock_period> <clock_uncertainty> [--pr]"
    exit 1
}

# get arguments
set project_name      [lindex $argv 0]
set board             [lindex $argv 1]
set clock_period      [lindex $argv 2]
set clock_uncertainty [lindex $argv 3]
set do_pr             0
if {[llength $argv] > 4 && [lindex $argv 4] eq "--pr"} {
    set do_pr 1
}

set prj_root [file normalize [file dirname [info script]]]
set prj_files [glob -nocomplain "${prj_root}/firmware/*.sv"]
set output_dir "${prj_root}/output_${project_name}"
set top_module "__${project_name}__${project_name}"

# Parameters used in xdc
set xdc_path "${prj_root}/constraints.xdc"
set uncertainty_hold_r $clock_uncertainty
set uncertainty_setup_r $clock_uncertainty
set delay_max_r 0.4
set delay_min_r 0.2


set source_type "verilog"

create_project $project_name "${output_dir}/$project_name" -force -part $board

set_property DEFAULT_LIB work [current_project]
set_property TARGET_LANGUAGE Verilog [current_project]

read_verilog $prj_files
read_xdc "${xdc_path}" -mode out_of_context

set_property top $top_module [current_fileset]

file mkdir $output_dir
file mkdir "${output_dir}/reports"

# synth
synth_design -top $top_module -mode out_of_context -global_retiming on \
    -flatten_hierarchy full -resource_sharing auto -directive AreaOptimized_High

write_checkpoint -force "${output_dir}/${project_name}_post_synth.dcp"

report_timing_summary -file "${output_dir}/reports/${project_name}_post_synth_timing.rpt"
report_power -file "${output_dir}/reports/${project_name}_post_synth_power.rpt"
report_utilization -file "${output_dir}/reports/${project_name}_post_synth_util.rpt"
