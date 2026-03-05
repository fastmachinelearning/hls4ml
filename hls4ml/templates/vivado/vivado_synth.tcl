set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]
source [file join $tcldir statistics.tcl]

set outputDir vivado_reports
set reportBase ${project_name}_report

set_param general.maxThreads 1
file mkdir $outputDir
create_project ${project_name}_vsynth -part $part -force
add_files ${project_name}_prj/solution1/syn/verilog
update_compile_order -fileset sources_1

set sdcFile ${outputDir}/${project_name}.xdc
set ofile_xdc [open $sdcFile w]
puts $ofile_xdc "create_clock -period $clock_period -name default \[get_ports ap_clk\]"
puts $ofile_xdc "set_property HD.CLK_SRC BUFGCTRL_X0Y0 \[get_ports ap_clk\]"
close $ofile_xdc
read_xdc $sdcFile

synth_design -mode out_of_context -no_iobuf -top ${project_name} -part $part
write_checkpoint -force $outputDir/post_synth.dcp
report_timing_summary -file $outputDir/post_synth_timing_summary.rpt
report_utilization -file $outputDir/post_synth_util.rpt
report_utilization -hierarchical -hierarchical_percentages -file $outputDir/post_synth_util_hier.rpt
dump_statistics $outputDir $reportBase "post_synth"
opt_design
dump_statistics $outputDir $reportBase "post_opt_design"
report_utilization -file $outputDir/post_opt_design_util.rpt
report_utilization -hierarchical -hierarchical_percentages -file $outputDir/post_opt_design_util_hier.rpt
place_design -directive Explore
report_clock_utilization -file $outputDir/clock_util.rpt
set timing_paths [get_timing_paths -max_paths 1 -nworst 1 -setup]
if {$timing_paths != "" && [get_property SLACK $timing_paths] < 0.5} {
  puts "Found setup timing violations => running physical optimization"
  phys_opt_design
}
write_checkpoint -force $outputDir/post_place.dcp
report_utilization -file $outputDir/post_place_util.rpt
report_utilization -hierarchical -hierarchical_percentages -file $outputDir/post_place_util_hier.rpt
report_timing_summary -file $outputDir/post_place_timing_summary.rpt
dump_statistics $outputDir $reportBase "post_place"
route_design -directive Explore
write_checkpoint -force $outputDir/post_route.dcp
report_route_status -file $outputDir/post_route_status.rpt
report_timing_summary -file $outputDir/post_route_timing_summary.rpt
report_power -file $outputDir/post_route_power.rpt
report_drc -file $outputDir/post_imp_drc.rpt
report_utilization -file $outputDir/post_route_util.rpt
report_utilization -hierarchical -hierarchical_percentages -file $outputDir/post_route_util_hier.rpt
dump_statistics $outputDir $reportBase "post_route"
close_design
close_project
