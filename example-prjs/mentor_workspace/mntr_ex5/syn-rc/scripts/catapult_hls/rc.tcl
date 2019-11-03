## HLS RC synthesis script

puts "-- Note: RTL Compiler Started"

set hls_status 0

proc run_cmd { cmd errstr } {
  upvar hls_status hls_status
  puts $cmd
  set retVal {}
  if { !$hls_status } {
    if { [catch { set retVal [uplevel 1 [list eval $cmd] ] } ] } {
      puts "Error: Unable to $errstr."
      set hls_status 1
    }
  } else {
    puts "Error: $errstr skipped due to previous errors."
  }
  set retVal
}

# Source custom RTLCompiler script for specified stage
# stage is one of: initial analyze synthesis reports final
proc source_custom_script { stage } {
   global env
   if { [info exists env(RTLCompiler_CustomScriptDirPath)] } {
      set dir_path $env(RTLCompiler_CustomScriptDirPath)
      if { $dir_path ne "" } {
         set script [file join $dir_path rc_${stage}.tcl]
         if { [file exists $script] } {
            set cmd "source $script"
            set msg [list run custom script $script]
            uplevel 1 [list run_cmd $cmd $msg]
         }
      }
   }
}

## Set the variable for file path prefixing 
#set RTL_TOOL_SCRIPT_DIR /home/giuseppe/research/projects/fastml/hls4ml-mentor.git/example-prjs/mentor_workspace/mntr_ex5/syn-catapult-hls/Catapult_asic/mnist_mlp.v1/.
#set RTL_TOOL_SCRIPT_DIR [file dirname [file normalize [info script]]]
#puts "-- RTL_TOOL_SCRIPT_DIR is set to '$RTL_TOOL_SCRIPT_DIR' "
set MGC_HOME /opt/cad/catapult

puts "Note: Removing old directory gate_synthesis_rc"
if { [file isdirectory "./output/catapult_hls/gate_synthesis_rc"] } {
  file delete -force -- "./output/catapult_hls/gate_synthesis_rc"
}
puts "Note: Creating directory ./output/catapult_hls/gate_synthesis_rc"
file mkdir "./output/catapult_hls/gate_synthesis_rc"
lcd "./output/catapult_hls/gate_synthesis_rc"

## Initialize RC-HLS variables
set_attr hdl_max_loop_limit 1000
set hls_status 0
set_attr delete_unloaded_insts false

# Source potential custom script
source_custom_script initial

## Configure technology settings
set_attr library /opt/cad/catapult/pkgs/siflibs/nangate/nangate45nm_nldm.lib
set_attr lef_library /opt/cad/catapult/pkgs/siflibs/nangate/nangate45nm.lef

## Exclude cells from synthesis
set_attr avoid true [find /lib*/NangateOpenCellLibrary -libcell CLKBUF_X1]
set_attr avoid true [find /lib*/NangateOpenCellLibrary -libcell CLKBUF_X2]
set_attr avoid true [find /lib*/NangateOpenCellLibrary -libcell CLKBUF_X3]

# Source potential custom script
source_custom_script analyze

## Analyze concat_rtl.v 
#run_cmd {read_hdl -v2001   $RTL_TOOL_SCRIPT_DIR/concat_rtl.v} {analyze file concat_rtl.v}
run_cmd {read_hdl -v2001   ../../../input/concat_rtl.v} {analyze file concat_rtl.v}

## Elaborate design mnist_mlp 
run_cmd {elaborate  "mnist_mlp"} {elaborate mnist_mlp {}}

# INFO: in catapult interconnect_mode has been set to none - interconnect_mode in RTL compiler will be set to wireloads but no wireload settings will be written into the constraints file 
set_attr interconnect_mode wireload

## Include SDC file
cd /designs/mnist_mlp
#read_sdc -stop_on_errors $RTL_TOOL_SCRIPT_DIR/concat_rtl.v.rc.sdc
read_sdc -stop_on_errors ../../../scripts/constraint.sdc
cd /

puts "[clock format [clock seconds] -format {%a %b %d %H:%M:%S %Z %Y}]"

set digns_with_mul [find /designs* -subdesign mul*]
if { [llength $digns_with_mul] > 0 } {
  set_attribute user_sub_arch {non_booth} [find /designs* -subdesign mul*]
}
puts "-- Starting synthesis for design 'mnist_mlp'"
# Source potential custom script
source_custom_script synthesis
uniquify mnist_mlp
synthesize -to_mapped

# Source potential custom script
source_custom_script reports
puts "-- Requested 3 fractional digits for design 'mnist_mlp' timing"
puts "-- Requested  fractional digits for design 'mnist_mlp' capacitance"
puts "-- Tool output delay factor to nS: 0.001"
puts "-- Library delay factor to nS: 1.0"
puts "-- Characterization mode: p2p "
puts "-- Synthesis area report for design 'mnist_mlp'"
report area /designs/mnist_mlp
report gates /designs/mnist_mlp
puts "-- END Synthesis area report for design 'mnist_mlp'"

cd /designs/mnist_mlp
ungroup -all
  if { [llength [find / -clock {*/clk}] ] > 0 } {
    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '0' 'INOUT' CLOCK 'clk'"

  }
  if { [llength [find / -clock {*/input1_rsc_vld}] ] > 0 } {
    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

  }
  if { [llength [find / -clock {*/layer7_out_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -clock {*/layer7_out_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -clock {*/layer7_out_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -clock {*/layer7_out_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -clock {*/layer7_out_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -clock {*/layer7_out_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -clock {*/layer7_out_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -clock {*/layer7_out_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -clock {*/layer7_out_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -clock {*/layer7_out_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -clock {*/layer7_out_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -clock {*/layer7_out_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -clock {*/layer7_out_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -clock {*/layer7_out_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -clock {*/layer7_out_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"

  }
  if { [llength [find / -clock {*/const_size_in_1_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -clock {*/const_size_in_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -clock {*/const_size_in_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -clock {*/const_size_in_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -clock {*/const_size_in_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -clock {*/const_size_in_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -clock {*/const_size_in_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -clock {*/const_size_in_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -clock {*/const_size_in_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -clock {*/const_size_in_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -clock {*/const_size_in_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -clock {*/const_size_in_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -clock {*/const_size_in_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -clock {*/const_size_in_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -clock {*/const_size_in_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"

  }
  if { [llength [find / -clock {*/const_size_out_1_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -clock {*/const_size_out_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -clock {*/const_size_out_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -clock {*/const_size_out_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -clock {*/const_size_out_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -clock {*/const_size_out_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -clock {*/const_size_out_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -clock {*/const_size_out_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -clock {*/const_size_out_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -clock {*/const_size_out_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -clock {*/const_size_out_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -clock {*/const_size_out_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -clock {*/const_size_out_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -clock {*/const_size_out_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -clock {*/const_size_out_1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"

  }
  if { [llength [find / -clock {*/clk}] ] > 0 && [llength [find / -clock {*/clk}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '0' 'INOUT' CLOCK 'clk'"
  }

  if { [llength [find / -clock {*/clk}] ] > 0 && [llength [find / -clock {*/input1_rsc_vld}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '1' 'IN' CLOCK 'input1_rsc_vld'"
  }

  if { [llength [find / -clock {*/clk}] ] > 0 && [llength [find / -clock {*/layer7_out_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -clock {*/layer7_out_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
  }

  if { [llength [find / -clock {*/clk}] ] > 0 && [llength [find / -clock {*/const_size_in_1_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -clock {*/const_size_in_1_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
  }

  if { [llength [find / -clock {*/clk}] ] > 0 && [llength [find / -clock {*/const_size_out_1_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -clock {*/const_size_out_1_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
  }

  if { [llength [find / -clock {*/input1_rsc_vld}] ] > 0 && [llength [find / -clock {*/clk}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '0' 'INOUT' CLOCK 'clk'"
  }

  if { [llength [find / -clock {*/input1_rsc_vld}] ] > 0 && [llength [find / -clock {*/input1_rsc_vld}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '1' 'IN' CLOCK 'input1_rsc_vld'"
  }

  if { [llength [find / -clock {*/input1_rsc_vld}] ] > 0 && [llength [find / -clock {*/layer7_out_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -clock {*/layer7_out_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
  }

  if { [llength [find / -clock {*/input1_rsc_vld}] ] > 0 && [llength [find / -clock {*/const_size_in_1_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -clock {*/const_size_in_1_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
  }

  if { [llength [find / -clock {*/input1_rsc_vld}] ] > 0 && [llength [find / -clock {*/const_size_out_1_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -clock {*/const_size_out_1_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
  }

  if { [llength [find / -clock {*/layer7_out_rsc_rdy}] ] > 0 && [llength [find / -clock {*/clk}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '0' 'INOUT' CLOCK 'clk'"
  }

  if { [llength [find / -clock {*/layer7_out_rsc_rdy}] ] > 0 && [llength [find / -clock {*/input1_rsc_vld}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '1' 'IN' CLOCK 'input1_rsc_vld'"
  }

  if { [llength [find / -clock {*/layer7_out_rsc_rdy}] ] > 0 && [llength [find / -clock {*/layer7_out_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -clock {*/layer7_out_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
  }

  if { [llength [find / -clock {*/layer7_out_rsc_rdy}] ] > 0 && [llength [find / -clock {*/const_size_in_1_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -clock {*/const_size_in_1_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
  }

  if { [llength [find / -clock {*/layer7_out_rsc_rdy}] ] > 0 && [llength [find / -clock {*/const_size_out_1_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -clock {*/const_size_out_1_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
  }

  if { [llength [find / -clock {*/const_size_in_1_rsc_rdy}] ] > 0 && [llength [find / -clock {*/clk}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '0' 'INOUT' CLOCK 'clk'"
  }

  if { [llength [find / -clock {*/const_size_in_1_rsc_rdy}] ] > 0 && [llength [find / -clock {*/input1_rsc_vld}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '1' 'IN' CLOCK 'input1_rsc_vld'"
  }

  if { [llength [find / -clock {*/const_size_in_1_rsc_rdy}] ] > 0 && [llength [find / -clock {*/layer7_out_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -clock {*/layer7_out_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
  }

  if { [llength [find / -clock {*/const_size_in_1_rsc_rdy}] ] > 0 && [llength [find / -clock {*/const_size_in_1_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -clock {*/const_size_in_1_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
  }

  if { [llength [find / -clock {*/const_size_in_1_rsc_rdy}] ] > 0 && [llength [find / -clock {*/const_size_out_1_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -clock {*/const_size_out_1_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
  }

  if { [llength [find / -clock {*/const_size_out_1_rsc_rdy}] ] > 0 && [llength [find / -clock {*/clk}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '0' 'INOUT' CLOCK 'clk'"
  }

  if { [llength [find / -clock {*/const_size_out_1_rsc_rdy}] ] > 0 && [llength [find / -clock {*/input1_rsc_vld}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '1' 'IN' CLOCK 'input1_rsc_vld'"
  }

  if { [llength [find / -clock {*/const_size_out_1_rsc_rdy}] ] > 0 && [llength [find / -clock {*/layer7_out_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -clock {*/layer7_out_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy'"
  }

  if { [llength [find / -clock {*/const_size_out_1_rsc_rdy}] ] > 0 && [llength [find / -clock {*/const_size_in_1_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -clock {*/const_size_in_1_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy'"
  }

  if { [llength [find / -clock {*/const_size_out_1_rsc_rdy}] ] > 0 && [llength [find / -clock {*/const_size_out_1_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -clock {*/const_size_out_1_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy'"
  }

  if { [llength [find / -clock {*/clk}] ] > 0 } {
    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '2' 'OUT' port 'layer7_out_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/layer7_out_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '2' 'OUT' port 'layer7_out_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '2' 'OUT' port 'layer7_out_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/layer7_out_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '2' 'OUT' port 'layer7_out_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '5' 'IN' port 'w2_rsc_web'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w2_rsc_web}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '5' 'IN' port 'w2_rsc_web'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '5' 'IN' port 'w2_rsc_enb'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w2_rsc_enb}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '5' 'IN' port 'w2_rsc_enb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '5' 'IN' port 'w2_rsc_db'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w2_rsc_db[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '5' 'IN' port 'w2_rsc_db'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '5' 'IN' port 'w2_rsc_adrb'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w2_rsc_adrb[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '5' 'IN' port 'w2_rsc_adrb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '5' 'IN' port 'w2_rsc_wea'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w2_rsc_wea}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '5' 'IN' port 'w2_rsc_wea'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '5' 'IN' port 'w2_rsc_ena'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w2_rsc_ena}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '5' 'IN' port 'w2_rsc_ena'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '5' 'IN' port 'w2_rsc_da'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w2_rsc_da[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '5' 'IN' port 'w2_rsc_da'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '5' 'IN' port 'w2_rsc_adra'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w2_rsc_adra[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '5' 'IN' port 'w2_rsc_adra'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '7' 'IN' port 'w4_rsc_web'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w4_rsc_web}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '7' 'IN' port 'w4_rsc_web'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '7' 'IN' port 'w4_rsc_enb'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w4_rsc_enb}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '7' 'IN' port 'w4_rsc_enb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '7' 'IN' port 'w4_rsc_db'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w4_rsc_db[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '7' 'IN' port 'w4_rsc_db'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '7' 'IN' port 'w4_rsc_adrb'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w4_rsc_adrb[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '7' 'IN' port 'w4_rsc_adrb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '7' 'IN' port 'w4_rsc_wea'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w4_rsc_wea}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '7' 'IN' port 'w4_rsc_wea'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '7' 'IN' port 'w4_rsc_ena'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w4_rsc_ena}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '7' 'IN' port 'w4_rsc_ena'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '7' 'IN' port 'w4_rsc_da'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w4_rsc_da[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '7' 'IN' port 'w4_rsc_da'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '7' 'IN' port 'w4_rsc_adra'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w4_rsc_adra[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '7' 'IN' port 'w4_rsc_adra'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '9' 'IN' port 'w6_rsc_web'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w6_rsc_web}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '9' 'IN' port 'w6_rsc_web'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '9' 'IN' port 'w6_rsc_enb'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w6_rsc_enb}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '9' 'IN' port 'w6_rsc_enb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '9' 'IN' port 'w6_rsc_db'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w6_rsc_db[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '9' 'IN' port 'w6_rsc_db'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '9' 'IN' port 'w6_rsc_adrb'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w6_rsc_adrb[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '9' 'IN' port 'w6_rsc_adrb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '9' 'IN' port 'w6_rsc_wea'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w6_rsc_wea}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '9' 'IN' port 'w6_rsc_wea'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '9' 'IN' port 'w6_rsc_ena'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w6_rsc_ena}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '9' 'IN' port 'w6_rsc_ena'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '9' 'IN' port 'w6_rsc_da'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w6_rsc_da[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '9' 'IN' port 'w6_rsc_da'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '9' 'IN' port 'w6_rsc_adra'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/w6_rsc_adra[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '9' 'IN' port 'w6_rsc_adra'"

  }

  if { [llength [find / -clock {*/input1_rsc_vld}] ] > 0 } {
    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '2' 'OUT' port 'layer7_out_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/layer7_out_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '2' 'OUT' port 'layer7_out_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '2' 'OUT' port 'layer7_out_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/layer7_out_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '2' 'OUT' port 'layer7_out_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_web'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w2_rsc_web}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_web'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_enb'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w2_rsc_enb}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_enb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_db'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w2_rsc_db[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_db'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_adrb'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w2_rsc_adrb[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_adrb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_wea'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w2_rsc_wea}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_wea'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_ena'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w2_rsc_ena}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_ena'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_da'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w2_rsc_da[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_da'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_adra'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w2_rsc_adra[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_adra'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_web'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w4_rsc_web}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_web'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_enb'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w4_rsc_enb}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_enb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_db'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w4_rsc_db[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_db'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_adrb'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w4_rsc_adrb[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_adrb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_wea'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w4_rsc_wea}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_wea'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_ena'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w4_rsc_ena}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_ena'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_da'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w4_rsc_da[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_da'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_adra'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w4_rsc_adra[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_adra'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_web'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w6_rsc_web}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_web'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_enb'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w6_rsc_enb}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_enb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_db'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w6_rsc_db[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_db'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_adrb'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w6_rsc_adrb[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_adrb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_wea'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w6_rsc_wea}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_wea'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_ena'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w6_rsc_ena}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_ena'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_da'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w6_rsc_da[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_da'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_adra'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/w6_rsc_adra[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_adra'"

  }

  if { [llength [find / -clock {*/layer7_out_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/layer7_out_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/layer7_out_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_web'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w2_rsc_web}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_web'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_enb'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w2_rsc_enb}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_enb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_db'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w2_rsc_db[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_db'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_adrb'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w2_rsc_adrb[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_adrb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_wea'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w2_rsc_wea}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_wea'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_ena'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w2_rsc_ena}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_ena'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_da'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w2_rsc_da[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_da'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_adra'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w2_rsc_adra[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_adra'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_web'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w4_rsc_web}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_web'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_enb'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w4_rsc_enb}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_enb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_db'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w4_rsc_db[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_db'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_adrb'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w4_rsc_adrb[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_adrb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_wea'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w4_rsc_wea}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_wea'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_ena'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w4_rsc_ena}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_ena'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_da'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w4_rsc_da[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_da'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_adra'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w4_rsc_adra[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_adra'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_web'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w6_rsc_web}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_web'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_enb'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w6_rsc_enb}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_enb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_db'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w6_rsc_db[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_db'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_adrb'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w6_rsc_adrb[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_adrb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_wea'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w6_rsc_wea}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_wea'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_ena'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w6_rsc_ena}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_ena'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_da'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w6_rsc_da[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_da'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_adra'"
    report timing -full_pin_names -from [find / -clock {*/layer7_out_rsc_rdy}] -to [find / -port {*/w6_rsc_adra[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_adra'"

  }

  if { [llength [find / -clock {*/const_size_in_1_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/layer7_out_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/layer7_out_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_web'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w2_rsc_web}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_web'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_enb'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w2_rsc_enb}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_enb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_db'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w2_rsc_db[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_db'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_adrb'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w2_rsc_adrb[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_adrb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_wea'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w2_rsc_wea}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_wea'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_ena'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w2_rsc_ena}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_ena'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_da'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w2_rsc_da[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_da'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_adra'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w2_rsc_adra[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_adra'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_web'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w4_rsc_web}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_web'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_enb'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w4_rsc_enb}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_enb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_db'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w4_rsc_db[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_db'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_adrb'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w4_rsc_adrb[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_adrb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_wea'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w4_rsc_wea}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_wea'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_ena'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w4_rsc_ena}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_ena'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_da'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w4_rsc_da[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_da'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_adra'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w4_rsc_adra[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_adra'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_web'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w6_rsc_web}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_web'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_enb'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w6_rsc_enb}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_enb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_db'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w6_rsc_db[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_db'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_adrb'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w6_rsc_adrb[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_adrb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_wea'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w6_rsc_wea}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_wea'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_ena'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w6_rsc_ena}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_ena'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_da'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w6_rsc_da[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_da'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_adra'"
    report timing -full_pin_names -from [find / -clock {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w6_rsc_adra[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '3' 'OUT' CLOCK 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_adra'"

  }

  if { [llength [find / -clock {*/const_size_out_1_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/layer7_out_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/layer7_out_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_web'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w2_rsc_web}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_web'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_enb'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w2_rsc_enb}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_enb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_db'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w2_rsc_db[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_db'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_adrb'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w2_rsc_adrb[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_adrb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_wea'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w2_rsc_wea}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_wea'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_ena'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w2_rsc_ena}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_ena'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_da'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w2_rsc_da[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_da'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_adra'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w2_rsc_adra[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_adra'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_web'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w4_rsc_web}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_web'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_enb'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w4_rsc_enb}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_enb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_db'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w4_rsc_db[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_db'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_adrb'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w4_rsc_adrb[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_adrb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_wea'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w4_rsc_wea}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_wea'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_ena'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w4_rsc_ena}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_ena'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_da'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w4_rsc_da[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_da'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_adra'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w4_rsc_adra[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_adra'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_web'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w6_rsc_web}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_web'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_enb'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w6_rsc_enb}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_enb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_db'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w6_rsc_db[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_db'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_adrb'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w6_rsc_adrb[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_adrb'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_wea'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w6_rsc_wea}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_wea'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_ena'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w6_rsc_ena}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_ena'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_da'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w6_rsc_da[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_da'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_adra'"
    report timing -full_pin_names -from [find / -clock {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w6_rsc_adra[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '4' 'OUT' CLOCK 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_adra'"

  }

  if { [llength [all_inputs -design mnist_mlp]] != 0 && [llength [all_outputs -design mnist_mlp]] != 0 } {
    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '2' 'OUT' port 'layer7_out_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/layer7_out_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '2' 'OUT' port 'layer7_out_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '2' 'OUT' port 'layer7_out_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/layer7_out_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '2' 'OUT' port 'layer7_out_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w2_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w2_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w2_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w2_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w2_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w2_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w2_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w2_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '5' 'IN' port 'w2_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w4_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w4_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w4_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w4_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w4_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w4_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w4_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w4_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '7' 'IN' port 'w4_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w6_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w6_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w6_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w6_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w6_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w6_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w6_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/w6_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '9' 'IN' port 'w6_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '2' 'OUT' port 'layer7_out_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/layer7_out_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '2' 'OUT' port 'layer7_out_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '2' 'OUT' port 'layer7_out_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/layer7_out_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '2' 'OUT' port 'layer7_out_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '5' 'IN' port 'w2_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w2_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '5' 'IN' port 'w2_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '5' 'IN' port 'w2_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w2_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '5' 'IN' port 'w2_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '5' 'IN' port 'w2_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w2_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '5' 'IN' port 'w2_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '5' 'IN' port 'w2_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w2_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '5' 'IN' port 'w2_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '5' 'IN' port 'w2_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w2_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '5' 'IN' port 'w2_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '5' 'IN' port 'w2_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w2_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '5' 'IN' port 'w2_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '5' 'IN' port 'w2_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w2_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '5' 'IN' port 'w2_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '5' 'IN' port 'w2_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w2_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '5' 'IN' port 'w2_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '7' 'IN' port 'w4_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w4_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '7' 'IN' port 'w4_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '7' 'IN' port 'w4_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w4_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '7' 'IN' port 'w4_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '7' 'IN' port 'w4_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w4_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '7' 'IN' port 'w4_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '7' 'IN' port 'w4_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w4_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '7' 'IN' port 'w4_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '7' 'IN' port 'w4_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w4_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '7' 'IN' port 'w4_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '7' 'IN' port 'w4_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w4_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '7' 'IN' port 'w4_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '7' 'IN' port 'w4_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w4_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '7' 'IN' port 'w4_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '7' 'IN' port 'w4_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w4_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '7' 'IN' port 'w4_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '9' 'IN' port 'w6_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w6_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '9' 'IN' port 'w6_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '9' 'IN' port 'w6_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w6_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '9' 'IN' port 'w6_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '9' 'IN' port 'w6_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w6_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '9' 'IN' port 'w6_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '9' 'IN' port 'w6_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w6_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '9' 'IN' port 'w6_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '9' 'IN' port 'w6_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w6_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '9' 'IN' port 'w6_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '9' 'IN' port 'w6_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w6_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '9' 'IN' port 'w6_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '9' 'IN' port 'w6_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w6_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '9' 'IN' port 'w6_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '9' 'IN' port 'w6_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/w6_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '9' 'IN' port 'w6_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/layer7_out_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/layer7_out_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w2_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w2_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w2_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w2_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w2_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w2_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w2_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w2_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '5' 'IN' port 'w2_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w4_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w4_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w4_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w4_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w4_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w4_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w4_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w4_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '7' 'IN' port 'w4_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w6_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w6_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w6_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w6_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w6_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w6_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w6_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/layer7_out_rsc_rdy}] -to [find / -port {*/w6_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'layer7_out_rsc_rdy' '9' 'IN' port 'w6_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/layer7_out_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/layer7_out_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w2_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w2_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w2_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w2_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w2_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w2_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w2_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w2_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '5' 'IN' port 'w2_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w4_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w4_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w4_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w4_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w4_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w4_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w4_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w4_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '7' 'IN' port 'w4_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w6_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w6_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w6_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w6_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w6_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w6_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w6_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/const_size_in_1_rsc_rdy}] -to [find / -port {*/w6_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '3' 'OUT' port 'const_size_in_1_rsc_rdy' '9' 'IN' port 'w6_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/layer7_out_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/layer7_out_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '2' 'OUT' port 'layer7_out_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w2_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w2_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w2_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w2_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w2_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w2_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w2_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w2_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '5' 'IN' port 'w2_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w4_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w4_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w4_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w4_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w4_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w4_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w4_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w4_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '7' 'IN' port 'w4_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w6_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w6_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w6_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w6_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w6_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w6_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w6_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/const_size_out_1_rsc_rdy}] -to [find / -port {*/w6_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '4' 'OUT' port 'const_size_out_1_rsc_rdy' '9' 'IN' port 'w6_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '2' 'OUT' port 'layer7_out_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/layer7_out_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '2' 'OUT' port 'layer7_out_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '2' 'OUT' port 'layer7_out_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/layer7_out_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '2' 'OUT' port 'layer7_out_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '5' 'IN' port 'w2_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w2_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '5' 'IN' port 'w2_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '5' 'IN' port 'w2_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w2_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '5' 'IN' port 'w2_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '5' 'IN' port 'w2_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w2_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '5' 'IN' port 'w2_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '5' 'IN' port 'w2_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w2_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '5' 'IN' port 'w2_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '5' 'IN' port 'w2_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w2_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '5' 'IN' port 'w2_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '5' 'IN' port 'w2_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w2_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '5' 'IN' port 'w2_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '5' 'IN' port 'w2_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w2_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '5' 'IN' port 'w2_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '5' 'IN' port 'w2_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w2_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '5' 'IN' port 'w2_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '7' 'IN' port 'w4_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w4_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '7' 'IN' port 'w4_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '7' 'IN' port 'w4_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w4_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '7' 'IN' port 'w4_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '7' 'IN' port 'w4_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w4_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '7' 'IN' port 'w4_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '7' 'IN' port 'w4_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w4_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '7' 'IN' port 'w4_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '7' 'IN' port 'w4_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w4_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '7' 'IN' port 'w4_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '7' 'IN' port 'w4_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w4_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '7' 'IN' port 'w4_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '7' 'IN' port 'w4_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w4_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '7' 'IN' port 'w4_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '7' 'IN' port 'w4_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w4_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '7' 'IN' port 'w4_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '9' 'IN' port 'w6_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w6_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '9' 'IN' port 'w6_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '9' 'IN' port 'w6_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w6_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '9' 'IN' port 'w6_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '9' 'IN' port 'w6_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w6_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '9' 'IN' port 'w6_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '9' 'IN' port 'w6_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w6_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '9' 'IN' port 'w6_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '9' 'IN' port 'w6_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w6_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '9' 'IN' port 'w6_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '9' 'IN' port 'w6_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w6_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '9' 'IN' port 'w6_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '9' 'IN' port 'w6_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w6_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '9' 'IN' port 'w6_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '9' 'IN' port 'w6_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qb[*]}] -to [find / -port {*/w6_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qb' '9' 'IN' port 'w6_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '2' 'OUT' port 'layer7_out_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/layer7_out_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '2' 'OUT' port 'layer7_out_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '2' 'OUT' port 'layer7_out_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/layer7_out_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '2' 'OUT' port 'layer7_out_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '5' 'IN' port 'w2_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w2_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '5' 'IN' port 'w2_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '5' 'IN' port 'w2_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w2_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '5' 'IN' port 'w2_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '5' 'IN' port 'w2_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w2_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '5' 'IN' port 'w2_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '5' 'IN' port 'w2_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w2_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '5' 'IN' port 'w2_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '5' 'IN' port 'w2_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w2_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '5' 'IN' port 'w2_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '5' 'IN' port 'w2_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w2_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '5' 'IN' port 'w2_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '5' 'IN' port 'w2_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w2_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '5' 'IN' port 'w2_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '5' 'IN' port 'w2_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w2_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '5' 'IN' port 'w2_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '7' 'IN' port 'w4_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w4_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '7' 'IN' port 'w4_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '7' 'IN' port 'w4_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w4_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '7' 'IN' port 'w4_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '7' 'IN' port 'w4_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w4_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '7' 'IN' port 'w4_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '7' 'IN' port 'w4_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w4_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '7' 'IN' port 'w4_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '7' 'IN' port 'w4_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w4_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '7' 'IN' port 'w4_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '7' 'IN' port 'w4_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w4_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '7' 'IN' port 'w4_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '7' 'IN' port 'w4_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w4_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '7' 'IN' port 'w4_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '7' 'IN' port 'w4_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w4_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '7' 'IN' port 'w4_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '9' 'IN' port 'w6_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w6_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '9' 'IN' port 'w6_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '9' 'IN' port 'w6_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w6_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '9' 'IN' port 'w6_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '9' 'IN' port 'w6_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w6_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '9' 'IN' port 'w6_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '9' 'IN' port 'w6_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w6_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '9' 'IN' port 'w6_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '9' 'IN' port 'w6_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w6_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '9' 'IN' port 'w6_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '9' 'IN' port 'w6_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w6_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '9' 'IN' port 'w6_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '9' 'IN' port 'w6_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w6_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '9' 'IN' port 'w6_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '9' 'IN' port 'w6_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_qa[*]}] -to [find / -port {*/w6_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '5' 'IN' port 'w2_rsc_qa' '9' 'IN' port 'w6_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '2' 'OUT' port 'layer7_out_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/layer7_out_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '2' 'OUT' port 'layer7_out_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '2' 'OUT' port 'layer7_out_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/layer7_out_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '2' 'OUT' port 'layer7_out_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '5' 'IN' port 'w2_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w2_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '5' 'IN' port 'w2_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '5' 'IN' port 'w2_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w2_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '5' 'IN' port 'w2_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '5' 'IN' port 'w2_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w2_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '5' 'IN' port 'w2_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '5' 'IN' port 'w2_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w2_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '5' 'IN' port 'w2_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '5' 'IN' port 'w2_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w2_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '5' 'IN' port 'w2_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '5' 'IN' port 'w2_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w2_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '5' 'IN' port 'w2_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '5' 'IN' port 'w2_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w2_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '5' 'IN' port 'w2_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '5' 'IN' port 'w2_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w2_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '5' 'IN' port 'w2_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '7' 'IN' port 'w4_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w4_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '7' 'IN' port 'w4_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '7' 'IN' port 'w4_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w4_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '7' 'IN' port 'w4_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '7' 'IN' port 'w4_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w4_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '7' 'IN' port 'w4_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '7' 'IN' port 'w4_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w4_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '7' 'IN' port 'w4_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '7' 'IN' port 'w4_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w4_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '7' 'IN' port 'w4_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '7' 'IN' port 'w4_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w4_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '7' 'IN' port 'w4_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '7' 'IN' port 'w4_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w4_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '7' 'IN' port 'w4_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '7' 'IN' port 'w4_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w4_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '7' 'IN' port 'w4_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '9' 'IN' port 'w6_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w6_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '9' 'IN' port 'w6_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '9' 'IN' port 'w6_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w6_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '9' 'IN' port 'w6_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '9' 'IN' port 'w6_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w6_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '9' 'IN' port 'w6_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '9' 'IN' port 'w6_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w6_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '9' 'IN' port 'w6_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '9' 'IN' port 'w6_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w6_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '9' 'IN' port 'w6_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '9' 'IN' port 'w6_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w6_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '9' 'IN' port 'w6_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '9' 'IN' port 'w6_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w6_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '9' 'IN' port 'w6_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '9' 'IN' port 'w6_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/w6_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '9' 'IN' port 'w6_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '2' 'OUT' port 'layer7_out_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/layer7_out_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '2' 'OUT' port 'layer7_out_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '2' 'OUT' port 'layer7_out_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/layer7_out_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '2' 'OUT' port 'layer7_out_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '5' 'IN' port 'w2_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w2_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '5' 'IN' port 'w2_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '5' 'IN' port 'w2_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w2_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '5' 'IN' port 'w2_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '5' 'IN' port 'w2_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w2_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '5' 'IN' port 'w2_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '5' 'IN' port 'w2_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w2_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '5' 'IN' port 'w2_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '5' 'IN' port 'w2_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w2_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '5' 'IN' port 'w2_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '5' 'IN' port 'w2_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w2_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '5' 'IN' port 'w2_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '5' 'IN' port 'w2_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w2_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '5' 'IN' port 'w2_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '5' 'IN' port 'w2_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w2_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '5' 'IN' port 'w2_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '7' 'IN' port 'w4_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w4_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '7' 'IN' port 'w4_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '7' 'IN' port 'w4_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w4_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '7' 'IN' port 'w4_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '7' 'IN' port 'w4_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w4_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '7' 'IN' port 'w4_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '7' 'IN' port 'w4_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w4_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '7' 'IN' port 'w4_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '7' 'IN' port 'w4_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w4_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '7' 'IN' port 'w4_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '7' 'IN' port 'w4_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w4_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '7' 'IN' port 'w4_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '7' 'IN' port 'w4_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w4_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '7' 'IN' port 'w4_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '7' 'IN' port 'w4_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w4_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '7' 'IN' port 'w4_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '9' 'IN' port 'w6_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w6_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '9' 'IN' port 'w6_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '9' 'IN' port 'w6_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w6_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '9' 'IN' port 'w6_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '9' 'IN' port 'w6_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w6_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '9' 'IN' port 'w6_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '9' 'IN' port 'w6_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w6_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '9' 'IN' port 'w6_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '9' 'IN' port 'w6_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w6_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '9' 'IN' port 'w6_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '9' 'IN' port 'w6_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w6_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '9' 'IN' port 'w6_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '9' 'IN' port 'w6_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w6_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '9' 'IN' port 'w6_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '9' 'IN' port 'w6_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qb[*]}] -to [find / -port {*/w6_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qb' '9' 'IN' port 'w6_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '2' 'OUT' port 'layer7_out_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/layer7_out_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '2' 'OUT' port 'layer7_out_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '2' 'OUT' port 'layer7_out_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/layer7_out_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '2' 'OUT' port 'layer7_out_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '5' 'IN' port 'w2_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w2_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '5' 'IN' port 'w2_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '5' 'IN' port 'w2_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w2_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '5' 'IN' port 'w2_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '5' 'IN' port 'w2_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w2_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '5' 'IN' port 'w2_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '5' 'IN' port 'w2_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w2_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '5' 'IN' port 'w2_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '5' 'IN' port 'w2_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w2_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '5' 'IN' port 'w2_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '5' 'IN' port 'w2_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w2_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '5' 'IN' port 'w2_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '5' 'IN' port 'w2_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w2_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '5' 'IN' port 'w2_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '5' 'IN' port 'w2_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w2_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '5' 'IN' port 'w2_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '7' 'IN' port 'w4_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w4_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '7' 'IN' port 'w4_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '7' 'IN' port 'w4_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w4_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '7' 'IN' port 'w4_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '7' 'IN' port 'w4_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w4_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '7' 'IN' port 'w4_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '7' 'IN' port 'w4_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w4_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '7' 'IN' port 'w4_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '7' 'IN' port 'w4_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w4_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '7' 'IN' port 'w4_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '7' 'IN' port 'w4_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w4_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '7' 'IN' port 'w4_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '7' 'IN' port 'w4_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w4_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '7' 'IN' port 'w4_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '7' 'IN' port 'w4_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w4_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '7' 'IN' port 'w4_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '9' 'IN' port 'w6_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w6_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '9' 'IN' port 'w6_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '9' 'IN' port 'w6_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w6_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '9' 'IN' port 'w6_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '9' 'IN' port 'w6_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w6_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '9' 'IN' port 'w6_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '9' 'IN' port 'w6_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w6_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '9' 'IN' port 'w6_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '9' 'IN' port 'w6_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w6_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '9' 'IN' port 'w6_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '9' 'IN' port 'w6_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w6_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '9' 'IN' port 'w6_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '9' 'IN' port 'w6_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w6_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '9' 'IN' port 'w6_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '9' 'IN' port 'w6_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_qa[*]}] -to [find / -port {*/w6_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '7' 'IN' port 'w4_rsc_qa' '9' 'IN' port 'w6_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '2' 'OUT' port 'layer7_out_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/layer7_out_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '2' 'OUT' port 'layer7_out_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '2' 'OUT' port 'layer7_out_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/layer7_out_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '2' 'OUT' port 'layer7_out_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '5' 'IN' port 'w2_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w2_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '5' 'IN' port 'w2_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '5' 'IN' port 'w2_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w2_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '5' 'IN' port 'w2_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '5' 'IN' port 'w2_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w2_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '5' 'IN' port 'w2_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '5' 'IN' port 'w2_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w2_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '5' 'IN' port 'w2_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '5' 'IN' port 'w2_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w2_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '5' 'IN' port 'w2_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '5' 'IN' port 'w2_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w2_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '5' 'IN' port 'w2_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '5' 'IN' port 'w2_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w2_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '5' 'IN' port 'w2_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '5' 'IN' port 'w2_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w2_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '5' 'IN' port 'w2_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '7' 'IN' port 'w4_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w4_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '7' 'IN' port 'w4_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '7' 'IN' port 'w4_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w4_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '7' 'IN' port 'w4_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '7' 'IN' port 'w4_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w4_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '7' 'IN' port 'w4_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '7' 'IN' port 'w4_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w4_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '7' 'IN' port 'w4_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '7' 'IN' port 'w4_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w4_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '7' 'IN' port 'w4_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '7' 'IN' port 'w4_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w4_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '7' 'IN' port 'w4_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '7' 'IN' port 'w4_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w4_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '7' 'IN' port 'w4_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '7' 'IN' port 'w4_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w4_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '7' 'IN' port 'w4_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '9' 'IN' port 'w6_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w6_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '9' 'IN' port 'w6_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '9' 'IN' port 'w6_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w6_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '9' 'IN' port 'w6_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '9' 'IN' port 'w6_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w6_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '9' 'IN' port 'w6_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '9' 'IN' port 'w6_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w6_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '9' 'IN' port 'w6_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '9' 'IN' port 'w6_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w6_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '9' 'IN' port 'w6_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '9' 'IN' port 'w6_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w6_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '9' 'IN' port 'w6_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '9' 'IN' port 'w6_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w6_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '9' 'IN' port 'w6_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '9' 'IN' port 'w6_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/w6_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '9' 'IN' port 'w6_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '2' 'OUT' port 'layer7_out_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/layer7_out_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '2' 'OUT' port 'layer7_out_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '2' 'OUT' port 'layer7_out_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/layer7_out_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '2' 'OUT' port 'layer7_out_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '5' 'IN' port 'w2_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w2_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '5' 'IN' port 'w2_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '5' 'IN' port 'w2_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w2_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '5' 'IN' port 'w2_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '5' 'IN' port 'w2_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w2_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '5' 'IN' port 'w2_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '5' 'IN' port 'w2_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w2_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '5' 'IN' port 'w2_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '5' 'IN' port 'w2_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w2_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '5' 'IN' port 'w2_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '5' 'IN' port 'w2_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w2_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '5' 'IN' port 'w2_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '5' 'IN' port 'w2_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w2_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '5' 'IN' port 'w2_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '5' 'IN' port 'w2_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w2_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '5' 'IN' port 'w2_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '7' 'IN' port 'w4_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w4_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '7' 'IN' port 'w4_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '7' 'IN' port 'w4_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w4_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '7' 'IN' port 'w4_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '7' 'IN' port 'w4_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w4_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '7' 'IN' port 'w4_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '7' 'IN' port 'w4_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w4_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '7' 'IN' port 'w4_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '7' 'IN' port 'w4_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w4_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '7' 'IN' port 'w4_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '7' 'IN' port 'w4_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w4_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '7' 'IN' port 'w4_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '7' 'IN' port 'w4_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w4_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '7' 'IN' port 'w4_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '7' 'IN' port 'w4_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w4_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '7' 'IN' port 'w4_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '9' 'IN' port 'w6_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w6_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '9' 'IN' port 'w6_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '9' 'IN' port 'w6_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w6_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '9' 'IN' port 'w6_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '9' 'IN' port 'w6_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w6_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '9' 'IN' port 'w6_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '9' 'IN' port 'w6_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w6_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '9' 'IN' port 'w6_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '9' 'IN' port 'w6_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w6_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '9' 'IN' port 'w6_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '9' 'IN' port 'w6_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w6_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '9' 'IN' port 'w6_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '9' 'IN' port 'w6_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w6_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '9' 'IN' port 'w6_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '9' 'IN' port 'w6_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qb[*]}] -to [find / -port {*/w6_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qb' '9' 'IN' port 'w6_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '2' 'OUT' port 'layer7_out_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/layer7_out_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '2' 'OUT' port 'layer7_out_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '2' 'OUT' port 'layer7_out_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/layer7_out_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '2' 'OUT' port 'layer7_out_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '5' 'IN' port 'w2_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w2_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '5' 'IN' port 'w2_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '5' 'IN' port 'w2_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w2_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '5' 'IN' port 'w2_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '5' 'IN' port 'w2_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w2_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '5' 'IN' port 'w2_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '5' 'IN' port 'w2_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w2_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '5' 'IN' port 'w2_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '5' 'IN' port 'w2_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w2_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '5' 'IN' port 'w2_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '5' 'IN' port 'w2_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w2_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '5' 'IN' port 'w2_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '5' 'IN' port 'w2_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w2_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '5' 'IN' port 'w2_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '5' 'IN' port 'w2_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w2_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '5' 'IN' port 'w2_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '7' 'IN' port 'w4_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w4_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '7' 'IN' port 'w4_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '7' 'IN' port 'w4_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w4_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '7' 'IN' port 'w4_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '7' 'IN' port 'w4_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w4_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '7' 'IN' port 'w4_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '7' 'IN' port 'w4_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w4_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '7' 'IN' port 'w4_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '7' 'IN' port 'w4_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w4_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '7' 'IN' port 'w4_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '7' 'IN' port 'w4_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w4_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '7' 'IN' port 'w4_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '7' 'IN' port 'w4_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w4_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '7' 'IN' port 'w4_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '7' 'IN' port 'w4_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w4_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '7' 'IN' port 'w4_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '9' 'IN' port 'w6_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w6_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '9' 'IN' port 'w6_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '9' 'IN' port 'w6_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w6_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '9' 'IN' port 'w6_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '9' 'IN' port 'w6_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w6_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '9' 'IN' port 'w6_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '9' 'IN' port 'w6_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w6_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '9' 'IN' port 'w6_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '9' 'IN' port 'w6_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w6_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '9' 'IN' port 'w6_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '9' 'IN' port 'w6_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w6_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '9' 'IN' port 'w6_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '9' 'IN' port 'w6_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w6_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '9' 'IN' port 'w6_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '9' 'IN' port 'w6_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_qa[*]}] -to [find / -port {*/w6_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '9' 'IN' port 'w6_rsc_qa' '9' 'IN' port 'w6_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '2' 'OUT' port 'layer7_out_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/layer7_out_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '2' 'OUT' port 'layer7_out_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '2' 'OUT' port 'layer7_out_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/layer7_out_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '2' 'OUT' port 'layer7_out_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '5' 'IN' port 'w2_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w2_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '5' 'IN' port 'w2_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '5' 'IN' port 'w2_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w2_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '5' 'IN' port 'w2_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '5' 'IN' port 'w2_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w2_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '5' 'IN' port 'w2_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '5' 'IN' port 'w2_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w2_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '5' 'IN' port 'w2_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '5' 'IN' port 'w2_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w2_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '5' 'IN' port 'w2_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '5' 'IN' port 'w2_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w2_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '5' 'IN' port 'w2_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '5' 'IN' port 'w2_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w2_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '5' 'IN' port 'w2_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '5' 'IN' port 'w2_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w2_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '5' 'IN' port 'w2_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '7' 'IN' port 'w4_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w4_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '7' 'IN' port 'w4_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '7' 'IN' port 'w4_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w4_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '7' 'IN' port 'w4_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '7' 'IN' port 'w4_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w4_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '7' 'IN' port 'w4_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '7' 'IN' port 'w4_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w4_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '7' 'IN' port 'w4_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '7' 'IN' port 'w4_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w4_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '7' 'IN' port 'w4_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '7' 'IN' port 'w4_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w4_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '7' 'IN' port 'w4_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '7' 'IN' port 'w4_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w4_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '7' 'IN' port 'w4_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '7' 'IN' port 'w4_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w4_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '7' 'IN' port 'w4_rsc_adra'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '9' 'IN' port 'w6_rsc_web'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w6_rsc_web}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '9' 'IN' port 'w6_rsc_web'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '9' 'IN' port 'w6_rsc_enb'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w6_rsc_enb}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '9' 'IN' port 'w6_rsc_enb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '9' 'IN' port 'w6_rsc_db'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w6_rsc_db[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '9' 'IN' port 'w6_rsc_db'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '9' 'IN' port 'w6_rsc_adrb'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w6_rsc_adrb[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '9' 'IN' port 'w6_rsc_adrb'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '9' 'IN' port 'w6_rsc_wea'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w6_rsc_wea}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '9' 'IN' port 'w6_rsc_wea'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '9' 'IN' port 'w6_rsc_ena'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w6_rsc_ena}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '9' 'IN' port 'w6_rsc_ena'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '9' 'IN' port 'w6_rsc_da'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w6_rsc_da[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '9' 'IN' port 'w6_rsc_da'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '9' 'IN' port 'w6_rsc_adra'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/w6_rsc_adra[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '9' 'IN' port 'w6_rsc_adra'"

  }

if {$hls_status} {
  puts "Warning: Check transcript for errors hls_status=$hls_status"
}
puts "[clock format [clock seconds] -format {%a %b %d %H:%M:%S %Z %Y}]"
change_names -verilog
#write_hdl mnist_mlp > $RTL_TOOL_SCRIPT_DIR/gate.rc.v.v
#puts "-- Netlist for design 'mnist_mlp' written to $RTL_TOOL_SCRIPT_DIR/gate.rc.v.v"
write_hdl mnist_mlp > gate.rc.v.v
puts "-- Netlist for design 'mnist_mlp' written to gate.rc.v.v"
#write_sdc > $RTL_TOOL_SCRIPT_DIR/gate.rc.v.sdc
#puts "-- SDC for design 'mnist_mlp' written to $RTL_TOOL_SCRIPT_DIR/gate.rc.v.sdc"
write_sdc > gate.rc.v.sdc
puts "-- SDC for design 'mnist_mlp' written to gate.rc.v.sdc"
#write_sdf > $RTL_TOOL_SCRIPT_DIR/gate.rc.v.sdf
#puts "-- SDF for design 'mnist_mlp' written to $RTL_TOOL_SCRIPT_DIR/gate.rc.v.sdf"
write_sdf > gate.rc.v.sdf
puts "-- SDF for design 'mnist_mlp' written to gate.rc.v.sdf"

# Source potential custom script
source_custom_script final
puts "-- Synthesis finished for design 'mnist_mlp'"

