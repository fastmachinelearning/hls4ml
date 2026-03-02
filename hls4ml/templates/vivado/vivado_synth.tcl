set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]

add_files ${project_name}_prj/solution1/syn/verilog
synth_design -top ${project_name} -part $part
opt_design -retarget -propconst -sweep -bram_power_opt -shift_register_opt
report_utilization -file vivado_synth.rpt

set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]

set outputDir vivado_reports
set reportBase ${project_name}_report

proc dump_statistics { outputDir reportBase stage_name} {
  set reportJson [file join $outputDir ${stage_name}_${reportBase}.json]
  set util_rpt [report_utilization -return_string]
  set LUTFFPairs 0
  set SliceRegisters 0
  set Slice 0
  set SliceLUTs 0
  set SliceLUTs1 0
  set BRAMFIFO36 0
  set BRAMFIFO18 0
  set BRAMFIFO36_star 0
  set BRAMFIFO18_star 0
  set BRAM18 0
  set BRAMFIFO 0
  set RAMS32 0
  set RAMD32 0
  set RAMD64E 0
  set DRAM 0
  set BIOB 0
  set DSPs 0
  set TotPower 0
  set design_slack 0
  set design_req 0
  set design_delay 0
  regexp --  {\s*Slice Registers\s*\|\s*([^[:blank:]]+)} $util_rpt ignore SliceRegisters
  regexp --  {\s*Slice\s*\|\s*([^[:blank:]]+)} $util_rpt ignore Slice
  regexp --  {\s*LUT as Logic\s*\|\s*([^[:blank:]]+)} $util_rpt ignore SliceLUTs
  regexp --  {\s*RAMB36/FIFO36\s*\|\s*([^[:blank:]]+)} $util_rpt ignore BRAMFIFO36
  regexp --  {\s*RAMB18/FIFO18\s*\|\s*([^[:blank:]]+)} $util_rpt ignore BRAMFIFO18
  regexp --  {\s*RAMB36/FIFO\*\s*\|\s*([^[:blank:]]+)} $util_rpt ignore BRAMFIFO36_star
  regexp --  {\s*RAMB18/FIFO\*\s*\|\s*([^[:blank:]]+)} $util_rpt ignore BRAMFIFO18_star
  regexp --  {\s*RAMB18\s*\|\s*([^[:blank:]]+)} $util_rpt ignore BRAM18
  set BRAMFIFO [expr {(2 *$BRAMFIFO36) + $BRAMFIFO18 + (2*$BRAMFIFO36_star) + $BRAMFIFO18_star + $BRAM18}]
  regexp --  {\s*LUT as Memory\s*\|\s*([^[:blank:]]+)} $util_rpt ignore DRAM
  regexp --  {\s*Bonded IOB\s*\|\s*([^[:blank:]]+)} $util_rpt ignore BIOB
  regexp --  {\s*DSPs\s*\|\s*([^[:blank:]]+)} $util_rpt ignore DSPs
  set power_rpt [report_power -return_string]
  regexp --  {\s*Total On-Chip Power \(W\)\s*\|\s*([^[:blank:]]+)} $power_rpt ignore TotPower
  set Timing_Paths [get_timing_paths -max_paths 1 -nworst 1 -setup]
  if { [expr {$Timing_Paths == ""}] } {
    set design_slack 0
    set design_req 0
  } else {
    set design_slack [get_property SLACK $Timing_Paths]
    set design_req [get_property REQUIREMENT  $Timing_Paths]
  }
  if { [expr {$design_slack == ""}] } {
    set design_slack 0
  }
  if { [expr {$design_req == ""}] } {
    set design_req 0
  }
#   set design_delay [expr {$design_req - $design_slack}]
#   file delete -force $reportXml
#   set ofile_report [open $reportXml w]
#   puts $ofile_report "<?xml version=\"1.0\"?>"
#   puts $ofile_report "<document>"
#   puts $ofile_report "  <application>"
#   puts $ofile_report "    <section stringID=\"XILINX_SYNTHESIS_SUMMARY\">"
#   puts $ofile_report "      <item stringID=\"XILINX_SLICE\" value=\"$Slice\"/>"
#   puts $ofile_report "      <item stringID=\"XILINX_SLICE_REGISTERS\" value=\"$SliceRegisters\"/>"
#   puts $ofile_report "      <item stringID=\"XILINX_SLICE_LUTS\" value=\"$SliceLUTs\"/>"
#   puts $ofile_report "      <item stringID=\"XILINX_BLOCK_RAMFIFO\" value=\"$BRAMFIFO\"/>"
#   puts $ofile_report "      <item stringID=\"XILINX_DRAM\" value=\"$DRAM\"/>"
#   puts $ofile_report "      <item stringID=\"XILINX_IOPIN\" value=\"$BIOB\"/>"
#   puts $ofile_report "      <item stringID=\"XILINX_DSPS\" value=\"$DSPs\"/>"
#   puts $ofile_report "      <item stringID=\"XILINX_POWER\" value=\"$TotPower\"/>"
#   puts $ofile_report "      <item stringID=\"XILINX_DESIGN_DELAY\" value=\"$design_delay\"/>"
#   puts $ofile_report "      <item stringID=\"XILINX_CLOCK_SLACK\" value=\"$design_slack\"/>"
#   puts $ofile_report "    </section>"
#   puts $ofile_report "  </application>"
#   puts $ofile_report "</document>"
#   close $ofile_report
  set ofile_json [open $reportJson w]
  puts $ofile_json "\{"
  puts $ofile_json "  \"XILINX_SYNTHESIS_SUMMARY\": \{"
  puts $ofile_json "    \"XILINX_SLICE\": \"$Slice\","
  puts $ofile_json "    \"XILINX_SLICE_REGISTERS\": \"$SliceRegisters\","
  puts $ofile_json "    \"XILINX_SLICE_LUTS\": \"$SliceLUTs\","
  puts $ofile_json "    \"XILINX_BLOCK_RAMFIFO\": \"$BRAMFIFO\","
  puts $ofile_json "    \"XILINX_DRAM\": \"$DRAM\","
  puts $ofile_json "    \"XILINX_IOPIN\": \"$BIOB\","
  puts $ofile_json "    \"XILINX_DSPS\": \"$DSPs\","
  puts $ofile_json "    \"XILINX_POWER\": \"$TotPower\","
  puts $ofile_json "    \"XILINX_CLOCK_SLACK\": \"$design_slack\""
  puts $ofile_json "  \}"
  puts $ofile_json "\}"
  flush $ofile_json
  close $ofile_json
}; #END PROC

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
