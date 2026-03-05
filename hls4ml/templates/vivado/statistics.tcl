# Shared dump_statistics procedure for Vivado accelerator and regular Vivado design scripts.
# Parses utilization, power, and timing reports into JSON.

proc dump_statistics { outputDir reportBase stage_name} {
  set reportJson [file join $outputDir ${stage_name}_${reportBase}.json]
  set util_rpt [report_utilization -return_string]
  set SliceRegisters 0
  set Slice 0
  set SliceLUTs 0
  set BRAMFIFO36 0
  set BRAMFIFO18 0
  set BRAMFIFO36_star 0
  set BRAMFIFO18_star 0
  set BRAM18 0
  set BRAMFIFO 0
  set DRAM 0
  set BIOB 0
  set DSPs 0
  set TotPower 0
  set DynamicPower 0
  set StaticPower 0
  set design_slack 0
  set design_req 0
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
  regexp --  {\s*Dynamic \(W\)\s*\|\s*([^[:blank:]]+)} $power_rpt ignore DynamicPower
  regexp --  {\s*Device Static \(W\)\s*\|\s*([^[:blank:]]+)} $power_rpt ignore StaticPower
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
  puts $ofile_json "    \"XILINX_POWER_DYNAMIC\": \"$DynamicPower\","
  puts $ofile_json "    \"XILINX_POWER_STATIC\": \"$StaticPower\","
  puts $ofile_json "    \"XILINX_CLOCK_SLACK\": \"$design_slack\""
  puts $ofile_json "  \}"
  puts $ofile_json "\}"
  flush $ofile_json
  close $ofile_json
}
