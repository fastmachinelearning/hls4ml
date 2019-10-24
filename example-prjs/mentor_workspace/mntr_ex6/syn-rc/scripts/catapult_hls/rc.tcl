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
#set RTL_TOOL_SCRIPT_DIR /home/giuseppe/research/projects/fastml/hls4ml-mentor.git/example-prjs/mentor_workspace/mntr_ex6/syn-catapult-hls/Catapult_asic/keras1layer.v1/.
#set RTL_TOOL_SCRIPT_DIR ./input/.
#set RTL_TOOL_SCRIPT_DIR [file dirname [file normalize [info script]]]
#puts "-- RTL_TOOL_SCRIPT_DIR is set to '$RTL_TOOL_SCRIPT_DIR' "
set MGC_HOME /opt/cad/catapult

#puts "Note: Removing old directory gate_synthesis_rc"
#if { [file isdirectory "gate_synthesis_rc"] } {
#  file delete -force -- "gate_synthesis_rc"
#}
#puts "Note: Creating directory gate_synthesis_rc"
#file mkdir "gate_synthesis_rc"
#lcd "gate_synthesis_rc"
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

## Elaborate design keras1layer 
run_cmd {elaborate  "keras1layer"} {elaborate keras1layer {}}

# INFO: in catapult interconnect_mode has been set to none - interconnect_mode in RTL compiler will be set to wireloads but no wireload settings will be written into the constraints file 
set_attr interconnect_mode wireload

## Include SDC file
cd /designs/keras1layer
#read_sdc -stop_on_errors $RTL_TOOL_SCRIPT_DIR/concat_rtl.v.rc.sdc
read_sdc -stop_on_errors ../../../scripts/constraint.sdc
cd /

puts "[clock format [clock seconds] -format {%a %b %d %H:%M:%S %Z %Y}]"

set digns_with_mul [find /designs* -subdesign mul*]
if { [llength $digns_with_mul] > 0 } {
  set_attribute user_sub_arch {non_booth} [find /designs* -subdesign mul*]
}
puts "-- Starting synthesis for design 'keras1layer'"
# Source potential custom script
source_custom_script synthesis
uniquify keras1layer
synthesize -to_mapped

# Source potential custom script
source_custom_script reports
puts "-- Requested 3 fractional digits for design 'keras1layer' timing"
puts "-- Requested  fractional digits for design 'keras1layer' capacitance"
puts "-- Tool output delay factor to nS: 0.001"
puts "-- Library delay factor to nS: 1.0"
puts "-- Characterization mode: p2p "
puts "-- Synthesis area report for design 'keras1layer'"
report area /designs/keras1layer
report gates /designs/keras1layer
puts "-- END Synthesis area report for design 'keras1layer'"

cd /designs/keras1layer
ungroup -all
  if { [llength [find / -clock {*/clk}] ] > 0 } {
    puts "-- Synthesis input_to_register:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_vld' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/input_1_rsc_vld}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_vld' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/input_1_rsc_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'keras1layer' '19' 'OUT' port 'layer5_out_rsc_rdy' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/layer5_out_rsc_rdy}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'keras1layer' '19' 'OUT' port 'layer5_out_rsc_rdy' '0' 'INOUT' CLOCK 'clk'"

  }
  if { [llength [find / -clock {*/input_1_rsc_vld}] ] > 0 } {
    puts "-- Synthesis input_to_register:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_vld' '18' 'IN' CLOCK 'input_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/input_1_rsc_vld}] -to [find / -clock {*/input_1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_vld' '18' 'IN' CLOCK 'input_1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_dat' '18' 'IN' CLOCK 'input_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/input_1_rsc_dat[*]}] -to [find / -clock {*/input_1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_dat' '18' 'IN' CLOCK 'input_1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'keras1layer' '19' 'OUT' port 'layer5_out_rsc_rdy' '18' 'IN' CLOCK 'input_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/layer5_out_rsc_rdy}] -to [find / -clock {*/input_1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'keras1layer' '19' 'OUT' port 'layer5_out_rsc_rdy' '18' 'IN' CLOCK 'input_1_rsc_vld'"

  }
  if { [llength [find / -clock {*/layer5_out_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis input_to_register:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_vld' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/input_1_rsc_vld}] -to [find / -clock {*/layer5_out_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_vld' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_dat' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/input_1_rsc_dat[*]}] -to [find / -clock {*/layer5_out_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_dat' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'keras1layer' '19' 'OUT' port 'layer5_out_rsc_rdy' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/layer5_out_rsc_rdy}] -to [find / -clock {*/layer5_out_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'keras1layer' '19' 'OUT' port 'layer5_out_rsc_rdy' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy'"

  }
  if { [llength [find / -clock {*/clk}] ] > 0 && [llength [find / -clock {*/clk}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'keras1layer' '0' 'INOUT' CLOCK 'clk' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis register_to_register:timing report for design 'keras1layer' '0' 'INOUT' CLOCK 'clk' '0' 'INOUT' CLOCK 'clk'"
  }

  if { [llength [find / -clock {*/clk}] ] > 0 && [llength [find / -clock {*/input_1_rsc_vld}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'keras1layer' '0' 'INOUT' CLOCK 'clk' '18' 'IN' CLOCK 'input_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -clock {*/input_1_rsc_vld}]
    puts "-- END Synthesis register_to_register:timing report for design 'keras1layer' '0' 'INOUT' CLOCK 'clk' '18' 'IN' CLOCK 'input_1_rsc_vld'"
  }

  if { [llength [find / -clock {*/clk}] ] > 0 && [llength [find / -clock {*/layer5_out_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'keras1layer' '0' 'INOUT' CLOCK 'clk' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -clock {*/layer5_out_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'keras1layer' '0' 'INOUT' CLOCK 'clk' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy'"
  }

  if { [llength [find / -clock {*/input_1_rsc_vld}] ] > 0 && [llength [find / -clock {*/clk}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'keras1layer' '18' 'IN' CLOCK 'input_1_rsc_vld' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -clock {*/input_1_rsc_vld}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis register_to_register:timing report for design 'keras1layer' '18' 'IN' CLOCK 'input_1_rsc_vld' '0' 'INOUT' CLOCK 'clk'"
  }

  if { [llength [find / -clock {*/input_1_rsc_vld}] ] > 0 && [llength [find / -clock {*/input_1_rsc_vld}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'keras1layer' '18' 'IN' CLOCK 'input_1_rsc_vld' '18' 'IN' CLOCK 'input_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/input_1_rsc_vld}] -to [find / -clock {*/input_1_rsc_vld}]
    puts "-- END Synthesis register_to_register:timing report for design 'keras1layer' '18' 'IN' CLOCK 'input_1_rsc_vld' '18' 'IN' CLOCK 'input_1_rsc_vld'"
  }

  if { [llength [find / -clock {*/input_1_rsc_vld}] ] > 0 && [llength [find / -clock {*/layer5_out_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'keras1layer' '18' 'IN' CLOCK 'input_1_rsc_vld' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/input_1_rsc_vld}] -to [find / -clock {*/layer5_out_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'keras1layer' '18' 'IN' CLOCK 'input_1_rsc_vld' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy'"
  }

  if { [llength [find / -clock {*/layer5_out_rsc_rdy}] ] > 0 && [llength [find / -clock {*/clk}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'keras1layer' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -clock {*/layer5_out_rsc_rdy}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis register_to_register:timing report for design 'keras1layer' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy' '0' 'INOUT' CLOCK 'clk'"
  }

  if { [llength [find / -clock {*/layer5_out_rsc_rdy}] ] > 0 && [llength [find / -clock {*/input_1_rsc_vld}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'keras1layer' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy' '18' 'IN' CLOCK 'input_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/layer5_out_rsc_rdy}] -to [find / -clock {*/input_1_rsc_vld}]
    puts "-- END Synthesis register_to_register:timing report for design 'keras1layer' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy' '18' 'IN' CLOCK 'input_1_rsc_vld'"
  }

  if { [llength [find / -clock {*/layer5_out_rsc_rdy}] ] > 0 && [llength [find / -clock {*/layer5_out_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'keras1layer' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/layer5_out_rsc_rdy}] -to [find / -clock {*/layer5_out_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'keras1layer' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy'"
  }

  if { [llength [find / -clock {*/clk}] ] > 0 } {
    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '0' 'INOUT' CLOCK 'clk' '18' 'IN' port 'input_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/input_1_rsc_rdy}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '0' 'INOUT' CLOCK 'clk' '18' 'IN' port 'input_1_rsc_rdy'"

    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '0' 'INOUT' CLOCK 'clk' '19' 'OUT' port 'layer5_out_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/layer5_out_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '0' 'INOUT' CLOCK 'clk' '19' 'OUT' port 'layer5_out_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '0' 'INOUT' CLOCK 'clk' '19' 'OUT' port 'layer5_out_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/layer5_out_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '0' 'INOUT' CLOCK 'clk' '19' 'OUT' port 'layer5_out_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '0' 'INOUT' CLOCK 'clk' '20' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '0' 'INOUT' CLOCK 'clk' '20' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '0' 'INOUT' CLOCK 'clk' '20' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '0' 'INOUT' CLOCK 'clk' '20' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '0' 'INOUT' CLOCK 'clk' '21' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '0' 'INOUT' CLOCK 'clk' '21' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '0' 'INOUT' CLOCK 'clk' '21' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '0' 'INOUT' CLOCK 'clk' '21' 'OUT' port 'const_size_out_1_rsc_dat'"

  }

  if { [llength [find / -clock {*/input_1_rsc_vld}] ] > 0 } {
    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '18' 'IN' CLOCK 'input_1_rsc_vld' '18' 'IN' port 'input_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/input_1_rsc_vld}] -to [find / -port {*/input_1_rsc_rdy}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '18' 'IN' CLOCK 'input_1_rsc_vld' '18' 'IN' port 'input_1_rsc_rdy'"

    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '18' 'IN' CLOCK 'input_1_rsc_vld' '19' 'OUT' port 'layer5_out_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/input_1_rsc_vld}] -to [find / -port {*/layer5_out_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '18' 'IN' CLOCK 'input_1_rsc_vld' '19' 'OUT' port 'layer5_out_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '18' 'IN' CLOCK 'input_1_rsc_vld' '19' 'OUT' port 'layer5_out_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/input_1_rsc_vld}] -to [find / -port {*/layer5_out_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '18' 'IN' CLOCK 'input_1_rsc_vld' '19' 'OUT' port 'layer5_out_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '18' 'IN' CLOCK 'input_1_rsc_vld' '20' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/input_1_rsc_vld}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '18' 'IN' CLOCK 'input_1_rsc_vld' '20' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '18' 'IN' CLOCK 'input_1_rsc_vld' '20' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/input_1_rsc_vld}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '18' 'IN' CLOCK 'input_1_rsc_vld' '20' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '18' 'IN' CLOCK 'input_1_rsc_vld' '21' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/input_1_rsc_vld}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '18' 'IN' CLOCK 'input_1_rsc_vld' '21' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '18' 'IN' CLOCK 'input_1_rsc_vld' '21' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/input_1_rsc_vld}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '18' 'IN' CLOCK 'input_1_rsc_vld' '21' 'OUT' port 'const_size_out_1_rsc_dat'"

  }

  if { [llength [find / -clock {*/layer5_out_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy' '18' 'IN' port 'input_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/layer5_out_rsc_rdy}] -to [find / -port {*/input_1_rsc_rdy}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy' '18' 'IN' port 'input_1_rsc_rdy'"

    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy' '19' 'OUT' port 'layer5_out_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/layer5_out_rsc_rdy}] -to [find / -port {*/layer5_out_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy' '19' 'OUT' port 'layer5_out_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy' '19' 'OUT' port 'layer5_out_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/layer5_out_rsc_rdy}] -to [find / -port {*/layer5_out_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy' '19' 'OUT' port 'layer5_out_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy' '20' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/layer5_out_rsc_rdy}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy' '20' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy' '20' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/layer5_out_rsc_rdy}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy' '20' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy' '21' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/layer5_out_rsc_rdy}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy' '21' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'keras1layer' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy' '21' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/layer5_out_rsc_rdy}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'keras1layer' '19' 'OUT' CLOCK 'layer5_out_rsc_rdy' '21' 'OUT' port 'const_size_out_1_rsc_dat'"

  }

  if { [llength [all_inputs -design keras1layer]] != 0 && [llength [all_outputs -design keras1layer]] != 0 } {
    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_vld' '18' 'IN' port 'input_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/input_1_rsc_vld}] -to [find / -port {*/input_1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_vld' '18' 'IN' port 'input_1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_vld' '19' 'OUT' port 'layer5_out_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/input_1_rsc_vld}] -to [find / -port {*/layer5_out_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_vld' '19' 'OUT' port 'layer5_out_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_vld' '19' 'OUT' port 'layer5_out_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/input_1_rsc_vld}] -to [find / -port {*/layer5_out_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_vld' '19' 'OUT' port 'layer5_out_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_vld' '20' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/input_1_rsc_vld}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_vld' '20' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_vld' '20' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/input_1_rsc_vld}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_vld' '20' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_vld' '21' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/input_1_rsc_vld}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_vld' '21' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_vld' '21' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/input_1_rsc_vld}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_vld' '21' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_dat' '18' 'IN' port 'input_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/input_1_rsc_dat[*]}] -to [find / -port {*/input_1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_dat' '18' 'IN' port 'input_1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_dat' '19' 'OUT' port 'layer5_out_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/input_1_rsc_dat[*]}] -to [find / -port {*/layer5_out_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_dat' '19' 'OUT' port 'layer5_out_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_dat' '19' 'OUT' port 'layer5_out_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/input_1_rsc_dat[*]}] -to [find / -port {*/layer5_out_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_dat' '19' 'OUT' port 'layer5_out_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_dat' '20' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/input_1_rsc_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_dat' '20' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_dat' '20' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/input_1_rsc_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_dat' '20' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_dat' '21' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/input_1_rsc_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_dat' '21' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_dat' '21' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/input_1_rsc_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '18' 'IN' port 'input_1_rsc_dat' '21' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '19' 'OUT' port 'layer5_out_rsc_rdy' '18' 'IN' port 'input_1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/layer5_out_rsc_rdy}] -to [find / -port {*/input_1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '19' 'OUT' port 'layer5_out_rsc_rdy' '18' 'IN' port 'input_1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '19' 'OUT' port 'layer5_out_rsc_rdy' '19' 'OUT' port 'layer5_out_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/layer5_out_rsc_rdy}] -to [find / -port {*/layer5_out_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '19' 'OUT' port 'layer5_out_rsc_rdy' '19' 'OUT' port 'layer5_out_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '19' 'OUT' port 'layer5_out_rsc_rdy' '19' 'OUT' port 'layer5_out_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/layer5_out_rsc_rdy}] -to [find / -port {*/layer5_out_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '19' 'OUT' port 'layer5_out_rsc_rdy' '19' 'OUT' port 'layer5_out_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '19' 'OUT' port 'layer5_out_rsc_rdy' '20' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/layer5_out_rsc_rdy}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '19' 'OUT' port 'layer5_out_rsc_rdy' '20' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '19' 'OUT' port 'layer5_out_rsc_rdy' '20' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/layer5_out_rsc_rdy}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '19' 'OUT' port 'layer5_out_rsc_rdy' '20' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '19' 'OUT' port 'layer5_out_rsc_rdy' '21' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/layer5_out_rsc_rdy}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '19' 'OUT' port 'layer5_out_rsc_rdy' '21' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'keras1layer' '19' 'OUT' port 'layer5_out_rsc_rdy' '21' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/layer5_out_rsc_rdy}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'keras1layer' '19' 'OUT' port 'layer5_out_rsc_rdy' '21' 'OUT' port 'const_size_out_1_rsc_dat'"

  }

if {$hls_status} {
  puts "Warning: Check transcript for errors hls_status=$hls_status"
}
puts "[clock format [clock seconds] -format {%a %b %d %H:%M:%S %Z %Y}]"
change_names -verilog
#write_hdl keras1layer > $RTL_TOOL_SCRIPT_DIR/gate.rc.v.v
#puts "-- Netlist for design 'keras1layer' written to $RTL_TOOL_SCRIPT_DIR/gate.rc.v.v"
write_hdl keras1layer > gate.rc.v.v
puts "-- Netlist for design 'keras1layer' written to gate.rc.v.v"
#write_sdc > $RTL_TOOL_SCRIPT_DIR/gate.rc.v.sdc
#puts "-- SDC for design 'keras1layer' written to $RTL_TOOL_SCRIPT_DIR/gate.rc.v.sdc"
write_sdc > gate.rc.v.sdc
puts "-- SDC for design 'keras1layer' written to gate.rc.v.sdc"
#write_sdf > $RTL_TOOL_SCRIPT_DIR/gate.rc.v.sdf
#puts "-- SDF for design 'keras1layer' written to $RTL_TOOL_SCRIPT_DIR/gate.rc.v.sdf"
write_sdf > gate.rc.v.sdf
puts "-- SDF for design 'keras1layer' written to gate.rc.v.sdf"

report messages -all > messages.rpt
report timing > timing.rpt

# Source potential custom script
source_custom_script final
puts "-- Synthesis finished for design 'keras1layer'"

