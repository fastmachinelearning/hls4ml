#######################################################
#
#                                                     
#######################################################

################################################
# Incremental Synthesis
################################################
set incremental_opto 1

synthesize -to_mapped -eff $ec::INCR_EFFORT -incr   
puts "Runtime & Memory after incremental synthesis"
timestat INCREMENTAL

foreach cg [find / -cost_group -null_ok *] {
  report timing -cost_group [list $cg] > $ec::reportDir/[basename $cg]_post_incr.rpt
}


# report time and memory
puts "\nEC INFO: Total cpu-time and memory after SYN2GEN: [get_attr runtime /] sec., [get_attr memory_usage /] MBytes.\n"

# report syn-to-generic design
report design > $ec::reportDir/syn2gen.design

# report syn-to-generic gates
report gates > $ec::reportDir/syn2gen.gate

# report syn-to-generic area
report area > $ec::reportDir/syn2gen.area

# report syn-to-generic timing
report timing -full > $ec::reportDir/syn2gen.timing

# report syn-to-generic timing groups
report timing -end -slack 0 > $ec::reportDir/syn2gen.timing.ep
report timing -from [dc::all_inputs] > $ec::reportDir/syn2gen.timing.in
report timing -to   [dc::all_outputs] > $ec::reportDir/syn2gen.timing.out
set ec::CNT 1
foreach ec::CLK [find /designs* -clock *] {
  exec echo "####################" > $ec::reportDir/syn2gen.timing.clk$ec::CNT
  exec echo "# from clock: $ec::CLK" >> $ec::reportDir/syn2gen.timing.clk$ec::CNT
  exec echo "# to clock: $ec::CLK" >> $ec::reportDir/syn2gen.timing.clk$ec::CNT
  exec echo "####################" >> $ec::reportDir/syn2gen.timing.clk$ec::CNT
  report timing -from $ec::CLK -to $ec::CLK >> $ec::reportDir/syn2gen.timing.clk$ec::CNT
  incr ec::CNT
}

### Create reports
report clock_gating > $ec::reportDir/clockgating.rpt
report power -depth 0 > $ec::reportDir/power.rpt
report gates -power > $ec::reportDir/gates_power.rpt
##report operand_isolation > $ec::reportDir/op_isolation.rpt
report area > $ec::reportDir/area.rpt
report gates > $ec::reportDir/gates.rpt
