

# Calculate actual uncertainty values
set uncertainty_setup [expr {$clock_period * $uncertainty_setup_r}]
set uncertainty_hold [expr {$clock_period * $uncertainty_hold_r}]
set delay_max [expr {$clock_period * $delay_max_r}]
set delay_min [expr {$clock_period * $delay_min_r}]

# Create clock with variable period
create_clock -period $clock_period -name sys_clk [get_ports {clk}]

# Input/Output constraints
# NB: read_xdc supports only a restricted XDC command subset (UG903);
# remove_from_collection [all_inputs] [get_ports clk] is not allowed here, so filter instead.
set_input_delay -clock sys_clk -max $delay_max [get_ports -filter {DIRECTION == IN && NAME != clk}]
set_input_delay -clock sys_clk -min $delay_min [get_ports -filter {DIRECTION == IN && NAME != clk}]

set_output_delay -clock sys_clk -max $delay_max [all_outputs]
set_output_delay -clock sys_clk -min $delay_min [all_outputs]

# Apply calculated uncertainty values
set_clock_uncertainty -setup $uncertainty_setup [get_clocks sys_clk]
set_clock_uncertainty -hold $uncertainty_hold [get_clocks sys_clk]
