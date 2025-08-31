# synth_pr.tcl
# Usage:
#   vivado -mode batch -nolog -nojournal -source synth_pr.tcl --tclargs <sv_file> <board_part> [--pr]

if {![llength $argv] >= 3} {
    puts stderr "ERROR: missing arguments\nUsage: vivado -mode batch -source synth_pr.tcl -tclargs <sv_file> <board_part> <clock_period> [--pr]"
    exit 1
}

# get arguments
set sv_file     [lindex $argv 0]
set board       [lindex $argv 1]
set clk_period  [lindex $argv 2]
set do_pr     0
if {[llength $argv] > 3 && [lindex $argv 3] eq "--pr"} {
    set do_pr 1
}

# infer top name from the file (strip path and extension)
set proj_name  [file rootname [file tail $sv_file]]
set top_name   $proj_name
file delete -force "./${proj_name}_prj"
file mkdir "./${proj_name}_prj"
set rpt_dir "./reports"
file mkdir $rpt_dir

# create project
create_project $proj_name "./${proj_name}_prj" -part $board


# add clock
create_clock -name sys_clk -period $clk_period [get_ports clk]

# add the SV files
add_files $sv_file
set_property top $top_name [current_fileset]
update_compile_order -fileset sources_1

# launch synth (as you already do)
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# report timing
report_clocks -file [file join $rpt_dir "clocks_post_synth.rpt"] 
report_timing_summary -delay_type min_max -check_timing -warn_on_violation \
  -max_paths 10 -file [file join $rpt_dir "timing_post_synth.rpt"] 

# set common opt/physopt/route switches for impl_1
set_property STEPS.OPT_DESIGN.ARGS  {-retarget -propconst -sweep -bram_power_opt -shift_register_opt} [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true                                                    [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS {-directive Explore}                                          [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS {-directive Explore}                                             [get_runs impl_1]

# launch implementation
launch_runs impl_1 -to_step route_design -jobs 4
wait_on_run impl_1

# report resource & timing after synthesis
report_utilization -file [file join $rpt_dir "synth_util.rpt"]