# synth_pr.tcl
# Usage:
#   vivado -mode batch -nolog -nojournal -source synth_pr.tcl --tclargs <sv_file> <board_part> [--pr]

if {![llength $argv] >= 2} {
    puts stderr "ERROR: missing arguments\nUsage: vivado -mode batch -source synth_pr.tcl -tclargs <sv_file> <board_part> [--pr]"
    exit 1
}

# get arguments
set sv_file   [lindex $argv 0]
set board     [lindex $argv 1]
set do_pr     0
if {[llength $argv] > 2 && [lindex $argv 2] eq "--pr"} {
    set do_pr 1
}

# infer top name from the file (strip path and extension)
set proj_name  [file rootname [file tail $sv_file]]
set top_name   $proj_name
file delete -force "./${proj_name}_prj"
file mkdir "./${proj_name}_prj"

# create project
create_project $proj_name "./${proj_name}_prj" -part $board

# add the SV files
add_files $sv_file
set_property top $top_name [current_fileset]
update_compile_order -fileset sources_1

#synthesize
launch_runs synth_1 -jobs 4
wait_on_run synth_1
open_run synth_1

# report resource & timing after synthesis
set rpt_dir "./reports"
file mkdir $rpt_dir
report_utilization -file [file join $rpt_dir "synth_util.rpt"]
# report_timing_summary  -file synth_timing.rpt