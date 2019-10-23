#######################################################
#                                                     
# 
#                                                     
#######################################################
################################################
# Leakage/Dynamic power/Clock Gating setup
################################################

set_attribute max_leakage_power 0.0 "$ec::DESIGN"

#####################################################################
# synthesize -to_generic -effort $ec::SYN_EFFORT
#####################################################################
set_attribute remove_assigns true /
set_remove_assign_options -verbose

synthesize -to_generic -effort $ec::SYN_EFFORT
report datapath > $ec::reportDir/datapath_generic.rpt

################################################
# Synthesizing to gates
################################################

synthesize -to_mapped -eff $ec::MAP_EFFORT -no_incr
puts "Runtime & Memory after 'synthesize -to_map -no_incr'"
report datapath > $ec::reportDir/datapath_mapped.rpt
