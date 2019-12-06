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
#if { [file isdirectory "gate_synthesis_rc"] } {
#  file delete -force -- "gate_synthesis_rc"
#}
#puts "Note: Creating directory gate_synthesis_rc"
#file mkdir "gate_synthesis_rc"
#lcd "gate_synthesis_rc"
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
#run_cmd {read_hdl -v2001   $RTL_TOOL_SCRIPT_DIR/concat_rtl.v} {analyze file 'concat_rtl.v'}
run_cmd {read_hdl -v2001   ../../../input-rtl/concat_rtl.v} {analyze file concat_rtl.v}

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

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '2' 'OUT' port 'output1_rsc_rdy' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/output1_rsc_rdy}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '2' 'OUT' port 'output1_rsc_rdy' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '27' 'IN' port 'w2_rsc_0_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_0_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '27' 'IN' port 'w2_rsc_0_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '28' 'IN' port 'w2_rsc_1_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_1_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '28' 'IN' port 'w2_rsc_1_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '29' 'IN' port 'w2_rsc_2_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_2_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '29' 'IN' port 'w2_rsc_2_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '30' 'IN' port 'w2_rsc_3_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_3_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '30' 'IN' port 'w2_rsc_3_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '31' 'IN' port 'w2_rsc_4_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_4_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '31' 'IN' port 'w2_rsc_4_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '32' 'IN' port 'w2_rsc_5_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_5_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '32' 'IN' port 'w2_rsc_5_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '33' 'IN' port 'w2_rsc_6_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_6_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '33' 'IN' port 'w2_rsc_6_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '34' 'IN' port 'w2_rsc_7_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_7_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '34' 'IN' port 'w2_rsc_7_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '35' 'IN' port 'w2_rsc_8_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_8_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '35' 'IN' port 'w2_rsc_8_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '36' 'IN' port 'w2_rsc_9_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_9_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '36' 'IN' port 'w2_rsc_9_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '37' 'IN' port 'w2_rsc_10_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_10_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '37' 'IN' port 'w2_rsc_10_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '38' 'IN' port 'w2_rsc_11_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_11_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '38' 'IN' port 'w2_rsc_11_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '39' 'IN' port 'w2_rsc_12_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_12_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '39' 'IN' port 'w2_rsc_12_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '40' 'IN' port 'w2_rsc_13_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_13_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '40' 'IN' port 'w2_rsc_13_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '41' 'IN' port 'w2_rsc_14_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_14_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '41' 'IN' port 'w2_rsc_14_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '42' 'IN' port 'w2_rsc_15_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_15_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '42' 'IN' port 'w2_rsc_15_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '43' 'IN' port 'w2_rsc_16_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_16_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '43' 'IN' port 'w2_rsc_16_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '44' 'IN' port 'w2_rsc_17_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_17_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '44' 'IN' port 'w2_rsc_17_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '45' 'IN' port 'w2_rsc_18_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_18_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '45' 'IN' port 'w2_rsc_18_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '46' 'IN' port 'w2_rsc_19_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_19_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '46' 'IN' port 'w2_rsc_19_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '47' 'IN' port 'w2_rsc_20_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_20_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '47' 'IN' port 'w2_rsc_20_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '48' 'IN' port 'w2_rsc_21_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_21_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '48' 'IN' port 'w2_rsc_21_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '49' 'IN' port 'w2_rsc_22_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_22_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '49' 'IN' port 'w2_rsc_22_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '50' 'IN' port 'w2_rsc_23_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_23_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '50' 'IN' port 'w2_rsc_23_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '51' 'IN' port 'w2_rsc_24_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_24_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '51' 'IN' port 'w2_rsc_24_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '52' 'IN' port 'w2_rsc_25_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_25_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '52' 'IN' port 'w2_rsc_25_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '53' 'IN' port 'w2_rsc_26_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_26_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '53' 'IN' port 'w2_rsc_26_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '54' 'IN' port 'w2_rsc_27_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_27_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '54' 'IN' port 'w2_rsc_27_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '55' 'IN' port 'w2_rsc_28_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_28_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '55' 'IN' port 'w2_rsc_28_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '56' 'IN' port 'w2_rsc_29_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_29_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '56' 'IN' port 'w2_rsc_29_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '57' 'IN' port 'w2_rsc_30_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_30_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '57' 'IN' port 'w2_rsc_30_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '58' 'IN' port 'w2_rsc_31_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_31_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '58' 'IN' port 'w2_rsc_31_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '59' 'IN' port 'w2_rsc_32_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_32_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '59' 'IN' port 'w2_rsc_32_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '60' 'IN' port 'w2_rsc_33_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_33_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '60' 'IN' port 'w2_rsc_33_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '61' 'IN' port 'w2_rsc_34_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_34_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '61' 'IN' port 'w2_rsc_34_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '62' 'IN' port 'w2_rsc_35_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_35_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '62' 'IN' port 'w2_rsc_35_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '63' 'IN' port 'w2_rsc_36_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_36_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '63' 'IN' port 'w2_rsc_36_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '64' 'IN' port 'w2_rsc_37_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_37_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '64' 'IN' port 'w2_rsc_37_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '65' 'IN' port 'w2_rsc_38_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_38_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '65' 'IN' port 'w2_rsc_38_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '66' 'IN' port 'w2_rsc_39_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_39_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '66' 'IN' port 'w2_rsc_39_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '67' 'IN' port 'w2_rsc_40_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_40_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '67' 'IN' port 'w2_rsc_40_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '68' 'IN' port 'w2_rsc_41_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_41_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '68' 'IN' port 'w2_rsc_41_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '69' 'IN' port 'w2_rsc_42_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_42_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '69' 'IN' port 'w2_rsc_42_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '70' 'IN' port 'w2_rsc_43_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_43_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '70' 'IN' port 'w2_rsc_43_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '71' 'IN' port 'w2_rsc_44_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_44_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '71' 'IN' port 'w2_rsc_44_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '72' 'IN' port 'w2_rsc_45_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_45_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '72' 'IN' port 'w2_rsc_45_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '73' 'IN' port 'w2_rsc_46_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_46_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '73' 'IN' port 'w2_rsc_46_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '74' 'IN' port 'w2_rsc_47_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_47_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '74' 'IN' port 'w2_rsc_47_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '75' 'IN' port 'w2_rsc_48_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_48_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '75' 'IN' port 'w2_rsc_48_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '76' 'IN' port 'w2_rsc_49_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_49_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '76' 'IN' port 'w2_rsc_49_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '77' 'IN' port 'w2_rsc_50_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_50_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '77' 'IN' port 'w2_rsc_50_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '78' 'IN' port 'w2_rsc_51_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_51_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '78' 'IN' port 'w2_rsc_51_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '79' 'IN' port 'w2_rsc_52_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_52_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '79' 'IN' port 'w2_rsc_52_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '80' 'IN' port 'w2_rsc_53_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_53_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '80' 'IN' port 'w2_rsc_53_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '81' 'IN' port 'w2_rsc_54_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_54_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '81' 'IN' port 'w2_rsc_54_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '82' 'IN' port 'w2_rsc_55_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_55_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '82' 'IN' port 'w2_rsc_55_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '83' 'IN' port 'w2_rsc_56_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_56_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '83' 'IN' port 'w2_rsc_56_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '84' 'IN' port 'w2_rsc_57_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_57_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '84' 'IN' port 'w2_rsc_57_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '85' 'IN' port 'w2_rsc_58_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_58_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '85' 'IN' port 'w2_rsc_58_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '86' 'IN' port 'w2_rsc_59_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_59_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '86' 'IN' port 'w2_rsc_59_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '87' 'IN' port 'w2_rsc_60_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_60_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '87' 'IN' port 'w2_rsc_60_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '88' 'IN' port 'w2_rsc_61_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_61_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '88' 'IN' port 'w2_rsc_61_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '89' 'IN' port 'w2_rsc_62_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_62_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '89' 'IN' port 'w2_rsc_62_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '90' 'IN' port 'w2_rsc_63_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_63_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '90' 'IN' port 'w2_rsc_63_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '91' 'IN' port 'w4_rsc_0_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_0_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '91' 'IN' port 'w4_rsc_0_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '92' 'IN' port 'w4_rsc_1_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_1_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '92' 'IN' port 'w4_rsc_1_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '93' 'IN' port 'w4_rsc_2_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_2_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '93' 'IN' port 'w4_rsc_2_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '94' 'IN' port 'w4_rsc_3_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_3_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '94' 'IN' port 'w4_rsc_3_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '95' 'IN' port 'w4_rsc_4_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_4_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '95' 'IN' port 'w4_rsc_4_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '96' 'IN' port 'w4_rsc_5_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_5_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '96' 'IN' port 'w4_rsc_5_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '97' 'IN' port 'w4_rsc_6_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_6_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '97' 'IN' port 'w4_rsc_6_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '98' 'IN' port 'w4_rsc_7_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_7_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '98' 'IN' port 'w4_rsc_7_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '99' 'IN' port 'w4_rsc_8_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_8_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '99' 'IN' port 'w4_rsc_8_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '100' 'IN' port 'w4_rsc_9_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_9_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '100' 'IN' port 'w4_rsc_9_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '101' 'IN' port 'w4_rsc_10_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_10_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '101' 'IN' port 'w4_rsc_10_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '102' 'IN' port 'w4_rsc_11_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_11_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '102' 'IN' port 'w4_rsc_11_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '103' 'IN' port 'w4_rsc_12_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_12_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '103' 'IN' port 'w4_rsc_12_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '104' 'IN' port 'w4_rsc_13_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_13_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '104' 'IN' port 'w4_rsc_13_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '105' 'IN' port 'w4_rsc_14_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_14_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '105' 'IN' port 'w4_rsc_14_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '106' 'IN' port 'w4_rsc_15_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_15_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '106' 'IN' port 'w4_rsc_15_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '107' 'IN' port 'w4_rsc_16_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_16_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '107' 'IN' port 'w4_rsc_16_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '108' 'IN' port 'w4_rsc_17_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_17_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '108' 'IN' port 'w4_rsc_17_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '109' 'IN' port 'w4_rsc_18_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_18_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '109' 'IN' port 'w4_rsc_18_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '110' 'IN' port 'w4_rsc_19_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_19_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '110' 'IN' port 'w4_rsc_19_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '111' 'IN' port 'w4_rsc_20_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_20_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '111' 'IN' port 'w4_rsc_20_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '112' 'IN' port 'w4_rsc_21_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_21_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '112' 'IN' port 'w4_rsc_21_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '113' 'IN' port 'w4_rsc_22_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_22_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '113' 'IN' port 'w4_rsc_22_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '114' 'IN' port 'w4_rsc_23_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_23_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '114' 'IN' port 'w4_rsc_23_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '115' 'IN' port 'w4_rsc_24_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_24_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '115' 'IN' port 'w4_rsc_24_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '116' 'IN' port 'w4_rsc_25_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_25_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '116' 'IN' port 'w4_rsc_25_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '117' 'IN' port 'w4_rsc_26_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_26_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '117' 'IN' port 'w4_rsc_26_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '118' 'IN' port 'w4_rsc_27_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_27_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '118' 'IN' port 'w4_rsc_27_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '119' 'IN' port 'w4_rsc_28_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_28_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '119' 'IN' port 'w4_rsc_28_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '120' 'IN' port 'w4_rsc_29_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_29_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '120' 'IN' port 'w4_rsc_29_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '121' 'IN' port 'w4_rsc_30_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_30_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '121' 'IN' port 'w4_rsc_30_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '122' 'IN' port 'w4_rsc_31_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_31_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '122' 'IN' port 'w4_rsc_31_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '123' 'IN' port 'w4_rsc_32_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_32_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '123' 'IN' port 'w4_rsc_32_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '124' 'IN' port 'w4_rsc_33_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_33_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '124' 'IN' port 'w4_rsc_33_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '125' 'IN' port 'w4_rsc_34_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_34_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '125' 'IN' port 'w4_rsc_34_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '126' 'IN' port 'w4_rsc_35_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_35_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '126' 'IN' port 'w4_rsc_35_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '127' 'IN' port 'w4_rsc_36_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_36_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '127' 'IN' port 'w4_rsc_36_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '128' 'IN' port 'w4_rsc_37_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_37_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '128' 'IN' port 'w4_rsc_37_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '129' 'IN' port 'w4_rsc_38_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_38_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '129' 'IN' port 'w4_rsc_38_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '130' 'IN' port 'w4_rsc_39_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_39_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '130' 'IN' port 'w4_rsc_39_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '131' 'IN' port 'w4_rsc_40_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_40_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '131' 'IN' port 'w4_rsc_40_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '132' 'IN' port 'w4_rsc_41_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_41_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '132' 'IN' port 'w4_rsc_41_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '133' 'IN' port 'w4_rsc_42_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_42_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '133' 'IN' port 'w4_rsc_42_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '134' 'IN' port 'w4_rsc_43_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_43_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '134' 'IN' port 'w4_rsc_43_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '135' 'IN' port 'w4_rsc_44_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_44_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '135' 'IN' port 'w4_rsc_44_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '136' 'IN' port 'w4_rsc_45_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_45_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '136' 'IN' port 'w4_rsc_45_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '137' 'IN' port 'w4_rsc_46_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_46_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '137' 'IN' port 'w4_rsc_46_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '138' 'IN' port 'w4_rsc_47_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_47_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '138' 'IN' port 'w4_rsc_47_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '139' 'IN' port 'w4_rsc_48_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_48_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '139' 'IN' port 'w4_rsc_48_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '140' 'IN' port 'w4_rsc_49_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_49_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '140' 'IN' port 'w4_rsc_49_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '141' 'IN' port 'w4_rsc_50_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_50_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '141' 'IN' port 'w4_rsc_50_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '142' 'IN' port 'w4_rsc_51_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_51_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '142' 'IN' port 'w4_rsc_51_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '143' 'IN' port 'w4_rsc_52_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_52_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '143' 'IN' port 'w4_rsc_52_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '144' 'IN' port 'w4_rsc_53_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_53_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '144' 'IN' port 'w4_rsc_53_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '145' 'IN' port 'w4_rsc_54_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_54_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '145' 'IN' port 'w4_rsc_54_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '146' 'IN' port 'w4_rsc_55_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_55_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '146' 'IN' port 'w4_rsc_55_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '147' 'IN' port 'w4_rsc_56_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_56_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '147' 'IN' port 'w4_rsc_56_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '148' 'IN' port 'w4_rsc_57_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_57_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '148' 'IN' port 'w4_rsc_57_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '149' 'IN' port 'w4_rsc_58_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_58_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '149' 'IN' port 'w4_rsc_58_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '150' 'IN' port 'w4_rsc_59_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_59_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '150' 'IN' port 'w4_rsc_59_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '151' 'IN' port 'w4_rsc_60_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_60_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '151' 'IN' port 'w4_rsc_60_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '152' 'IN' port 'w4_rsc_61_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_61_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '152' 'IN' port 'w4_rsc_61_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '153' 'IN' port 'w4_rsc_62_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_62_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '153' 'IN' port 'w4_rsc_62_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '154' 'IN' port 'w4_rsc_63_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_63_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '154' 'IN' port 'w4_rsc_63_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '155' 'IN' port 'w6_rsc_0_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_0_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '155' 'IN' port 'w6_rsc_0_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '156' 'IN' port 'w6_rsc_1_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_1_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '156' 'IN' port 'w6_rsc_1_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '157' 'IN' port 'w6_rsc_2_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_2_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '157' 'IN' port 'w6_rsc_2_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '158' 'IN' port 'w6_rsc_3_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_3_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '158' 'IN' port 'w6_rsc_3_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '159' 'IN' port 'w6_rsc_4_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_4_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '159' 'IN' port 'w6_rsc_4_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '160' 'IN' port 'w6_rsc_5_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_5_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '160' 'IN' port 'w6_rsc_5_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '161' 'IN' port 'w6_rsc_6_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_6_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '161' 'IN' port 'w6_rsc_6_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '162' 'IN' port 'w6_rsc_7_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_7_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '162' 'IN' port 'w6_rsc_7_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '163' 'IN' port 'w6_rsc_8_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_8_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '163' 'IN' port 'w6_rsc_8_0_dat' '0' 'INOUT' CLOCK 'clk'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '164' 'IN' port 'w6_rsc_9_0_dat' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_9_0_dat[*]}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '164' 'IN' port 'w6_rsc_9_0_dat' '0' 'INOUT' CLOCK 'clk'"

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

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '2' 'OUT' port 'output1_rsc_rdy' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/output1_rsc_rdy}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '2' 'OUT' port 'output1_rsc_rdy' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '27' 'IN' port 'w2_rsc_0_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_0_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '27' 'IN' port 'w2_rsc_0_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '28' 'IN' port 'w2_rsc_1_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_1_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '28' 'IN' port 'w2_rsc_1_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '29' 'IN' port 'w2_rsc_2_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_2_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '29' 'IN' port 'w2_rsc_2_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '30' 'IN' port 'w2_rsc_3_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_3_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '30' 'IN' port 'w2_rsc_3_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '31' 'IN' port 'w2_rsc_4_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_4_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '31' 'IN' port 'w2_rsc_4_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '32' 'IN' port 'w2_rsc_5_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_5_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '32' 'IN' port 'w2_rsc_5_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '33' 'IN' port 'w2_rsc_6_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_6_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '33' 'IN' port 'w2_rsc_6_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '34' 'IN' port 'w2_rsc_7_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_7_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '34' 'IN' port 'w2_rsc_7_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '35' 'IN' port 'w2_rsc_8_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_8_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '35' 'IN' port 'w2_rsc_8_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '36' 'IN' port 'w2_rsc_9_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_9_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '36' 'IN' port 'w2_rsc_9_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '37' 'IN' port 'w2_rsc_10_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_10_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '37' 'IN' port 'w2_rsc_10_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '38' 'IN' port 'w2_rsc_11_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_11_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '38' 'IN' port 'w2_rsc_11_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '39' 'IN' port 'w2_rsc_12_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_12_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '39' 'IN' port 'w2_rsc_12_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '40' 'IN' port 'w2_rsc_13_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_13_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '40' 'IN' port 'w2_rsc_13_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '41' 'IN' port 'w2_rsc_14_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_14_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '41' 'IN' port 'w2_rsc_14_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '42' 'IN' port 'w2_rsc_15_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_15_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '42' 'IN' port 'w2_rsc_15_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '43' 'IN' port 'w2_rsc_16_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_16_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '43' 'IN' port 'w2_rsc_16_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '44' 'IN' port 'w2_rsc_17_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_17_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '44' 'IN' port 'w2_rsc_17_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '45' 'IN' port 'w2_rsc_18_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_18_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '45' 'IN' port 'w2_rsc_18_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '46' 'IN' port 'w2_rsc_19_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_19_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '46' 'IN' port 'w2_rsc_19_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '47' 'IN' port 'w2_rsc_20_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_20_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '47' 'IN' port 'w2_rsc_20_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '48' 'IN' port 'w2_rsc_21_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_21_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '48' 'IN' port 'w2_rsc_21_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '49' 'IN' port 'w2_rsc_22_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_22_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '49' 'IN' port 'w2_rsc_22_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '50' 'IN' port 'w2_rsc_23_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_23_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '50' 'IN' port 'w2_rsc_23_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '51' 'IN' port 'w2_rsc_24_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_24_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '51' 'IN' port 'w2_rsc_24_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '52' 'IN' port 'w2_rsc_25_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_25_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '52' 'IN' port 'w2_rsc_25_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '53' 'IN' port 'w2_rsc_26_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_26_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '53' 'IN' port 'w2_rsc_26_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '54' 'IN' port 'w2_rsc_27_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_27_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '54' 'IN' port 'w2_rsc_27_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '55' 'IN' port 'w2_rsc_28_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_28_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '55' 'IN' port 'w2_rsc_28_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '56' 'IN' port 'w2_rsc_29_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_29_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '56' 'IN' port 'w2_rsc_29_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '57' 'IN' port 'w2_rsc_30_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_30_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '57' 'IN' port 'w2_rsc_30_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '58' 'IN' port 'w2_rsc_31_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_31_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '58' 'IN' port 'w2_rsc_31_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '59' 'IN' port 'w2_rsc_32_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_32_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '59' 'IN' port 'w2_rsc_32_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '60' 'IN' port 'w2_rsc_33_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_33_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '60' 'IN' port 'w2_rsc_33_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '61' 'IN' port 'w2_rsc_34_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_34_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '61' 'IN' port 'w2_rsc_34_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '62' 'IN' port 'w2_rsc_35_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_35_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '62' 'IN' port 'w2_rsc_35_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '63' 'IN' port 'w2_rsc_36_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_36_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '63' 'IN' port 'w2_rsc_36_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '64' 'IN' port 'w2_rsc_37_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_37_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '64' 'IN' port 'w2_rsc_37_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '65' 'IN' port 'w2_rsc_38_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_38_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '65' 'IN' port 'w2_rsc_38_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '66' 'IN' port 'w2_rsc_39_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_39_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '66' 'IN' port 'w2_rsc_39_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '67' 'IN' port 'w2_rsc_40_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_40_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '67' 'IN' port 'w2_rsc_40_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '68' 'IN' port 'w2_rsc_41_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_41_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '68' 'IN' port 'w2_rsc_41_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '69' 'IN' port 'w2_rsc_42_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_42_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '69' 'IN' port 'w2_rsc_42_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '70' 'IN' port 'w2_rsc_43_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_43_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '70' 'IN' port 'w2_rsc_43_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '71' 'IN' port 'w2_rsc_44_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_44_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '71' 'IN' port 'w2_rsc_44_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '72' 'IN' port 'w2_rsc_45_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_45_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '72' 'IN' port 'w2_rsc_45_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '73' 'IN' port 'w2_rsc_46_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_46_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '73' 'IN' port 'w2_rsc_46_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '74' 'IN' port 'w2_rsc_47_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_47_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '74' 'IN' port 'w2_rsc_47_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '75' 'IN' port 'w2_rsc_48_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_48_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '75' 'IN' port 'w2_rsc_48_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '76' 'IN' port 'w2_rsc_49_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_49_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '76' 'IN' port 'w2_rsc_49_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '77' 'IN' port 'w2_rsc_50_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_50_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '77' 'IN' port 'w2_rsc_50_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '78' 'IN' port 'w2_rsc_51_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_51_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '78' 'IN' port 'w2_rsc_51_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '79' 'IN' port 'w2_rsc_52_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_52_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '79' 'IN' port 'w2_rsc_52_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '80' 'IN' port 'w2_rsc_53_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_53_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '80' 'IN' port 'w2_rsc_53_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '81' 'IN' port 'w2_rsc_54_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_54_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '81' 'IN' port 'w2_rsc_54_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '82' 'IN' port 'w2_rsc_55_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_55_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '82' 'IN' port 'w2_rsc_55_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '83' 'IN' port 'w2_rsc_56_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_56_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '83' 'IN' port 'w2_rsc_56_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '84' 'IN' port 'w2_rsc_57_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_57_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '84' 'IN' port 'w2_rsc_57_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '85' 'IN' port 'w2_rsc_58_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_58_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '85' 'IN' port 'w2_rsc_58_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '86' 'IN' port 'w2_rsc_59_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_59_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '86' 'IN' port 'w2_rsc_59_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '87' 'IN' port 'w2_rsc_60_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_60_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '87' 'IN' port 'w2_rsc_60_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '88' 'IN' port 'w2_rsc_61_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_61_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '88' 'IN' port 'w2_rsc_61_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '89' 'IN' port 'w2_rsc_62_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_62_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '89' 'IN' port 'w2_rsc_62_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '90' 'IN' port 'w2_rsc_63_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_63_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '90' 'IN' port 'w2_rsc_63_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '91' 'IN' port 'w4_rsc_0_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_0_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '91' 'IN' port 'w4_rsc_0_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '92' 'IN' port 'w4_rsc_1_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_1_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '92' 'IN' port 'w4_rsc_1_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '93' 'IN' port 'w4_rsc_2_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_2_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '93' 'IN' port 'w4_rsc_2_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '94' 'IN' port 'w4_rsc_3_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_3_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '94' 'IN' port 'w4_rsc_3_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '95' 'IN' port 'w4_rsc_4_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_4_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '95' 'IN' port 'w4_rsc_4_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '96' 'IN' port 'w4_rsc_5_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_5_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '96' 'IN' port 'w4_rsc_5_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '97' 'IN' port 'w4_rsc_6_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_6_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '97' 'IN' port 'w4_rsc_6_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '98' 'IN' port 'w4_rsc_7_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_7_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '98' 'IN' port 'w4_rsc_7_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '99' 'IN' port 'w4_rsc_8_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_8_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '99' 'IN' port 'w4_rsc_8_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '100' 'IN' port 'w4_rsc_9_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_9_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '100' 'IN' port 'w4_rsc_9_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '101' 'IN' port 'w4_rsc_10_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_10_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '101' 'IN' port 'w4_rsc_10_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '102' 'IN' port 'w4_rsc_11_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_11_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '102' 'IN' port 'w4_rsc_11_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '103' 'IN' port 'w4_rsc_12_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_12_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '103' 'IN' port 'w4_rsc_12_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '104' 'IN' port 'w4_rsc_13_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_13_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '104' 'IN' port 'w4_rsc_13_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '105' 'IN' port 'w4_rsc_14_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_14_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '105' 'IN' port 'w4_rsc_14_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '106' 'IN' port 'w4_rsc_15_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_15_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '106' 'IN' port 'w4_rsc_15_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '107' 'IN' port 'w4_rsc_16_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_16_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '107' 'IN' port 'w4_rsc_16_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '108' 'IN' port 'w4_rsc_17_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_17_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '108' 'IN' port 'w4_rsc_17_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '109' 'IN' port 'w4_rsc_18_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_18_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '109' 'IN' port 'w4_rsc_18_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '110' 'IN' port 'w4_rsc_19_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_19_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '110' 'IN' port 'w4_rsc_19_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '111' 'IN' port 'w4_rsc_20_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_20_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '111' 'IN' port 'w4_rsc_20_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '112' 'IN' port 'w4_rsc_21_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_21_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '112' 'IN' port 'w4_rsc_21_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '113' 'IN' port 'w4_rsc_22_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_22_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '113' 'IN' port 'w4_rsc_22_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '114' 'IN' port 'w4_rsc_23_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_23_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '114' 'IN' port 'w4_rsc_23_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '115' 'IN' port 'w4_rsc_24_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_24_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '115' 'IN' port 'w4_rsc_24_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '116' 'IN' port 'w4_rsc_25_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_25_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '116' 'IN' port 'w4_rsc_25_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '117' 'IN' port 'w4_rsc_26_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_26_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '117' 'IN' port 'w4_rsc_26_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '118' 'IN' port 'w4_rsc_27_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_27_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '118' 'IN' port 'w4_rsc_27_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '119' 'IN' port 'w4_rsc_28_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_28_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '119' 'IN' port 'w4_rsc_28_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '120' 'IN' port 'w4_rsc_29_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_29_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '120' 'IN' port 'w4_rsc_29_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '121' 'IN' port 'w4_rsc_30_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_30_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '121' 'IN' port 'w4_rsc_30_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '122' 'IN' port 'w4_rsc_31_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_31_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '122' 'IN' port 'w4_rsc_31_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '123' 'IN' port 'w4_rsc_32_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_32_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '123' 'IN' port 'w4_rsc_32_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '124' 'IN' port 'w4_rsc_33_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_33_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '124' 'IN' port 'w4_rsc_33_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '125' 'IN' port 'w4_rsc_34_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_34_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '125' 'IN' port 'w4_rsc_34_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '126' 'IN' port 'w4_rsc_35_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_35_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '126' 'IN' port 'w4_rsc_35_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '127' 'IN' port 'w4_rsc_36_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_36_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '127' 'IN' port 'w4_rsc_36_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '128' 'IN' port 'w4_rsc_37_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_37_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '128' 'IN' port 'w4_rsc_37_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '129' 'IN' port 'w4_rsc_38_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_38_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '129' 'IN' port 'w4_rsc_38_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '130' 'IN' port 'w4_rsc_39_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_39_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '130' 'IN' port 'w4_rsc_39_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '131' 'IN' port 'w4_rsc_40_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_40_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '131' 'IN' port 'w4_rsc_40_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '132' 'IN' port 'w4_rsc_41_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_41_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '132' 'IN' port 'w4_rsc_41_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '133' 'IN' port 'w4_rsc_42_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_42_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '133' 'IN' port 'w4_rsc_42_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '134' 'IN' port 'w4_rsc_43_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_43_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '134' 'IN' port 'w4_rsc_43_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '135' 'IN' port 'w4_rsc_44_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_44_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '135' 'IN' port 'w4_rsc_44_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '136' 'IN' port 'w4_rsc_45_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_45_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '136' 'IN' port 'w4_rsc_45_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '137' 'IN' port 'w4_rsc_46_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_46_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '137' 'IN' port 'w4_rsc_46_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '138' 'IN' port 'w4_rsc_47_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_47_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '138' 'IN' port 'w4_rsc_47_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '139' 'IN' port 'w4_rsc_48_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_48_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '139' 'IN' port 'w4_rsc_48_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '140' 'IN' port 'w4_rsc_49_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_49_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '140' 'IN' port 'w4_rsc_49_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '141' 'IN' port 'w4_rsc_50_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_50_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '141' 'IN' port 'w4_rsc_50_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '142' 'IN' port 'w4_rsc_51_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_51_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '142' 'IN' port 'w4_rsc_51_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '143' 'IN' port 'w4_rsc_52_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_52_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '143' 'IN' port 'w4_rsc_52_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '144' 'IN' port 'w4_rsc_53_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_53_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '144' 'IN' port 'w4_rsc_53_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '145' 'IN' port 'w4_rsc_54_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_54_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '145' 'IN' port 'w4_rsc_54_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '146' 'IN' port 'w4_rsc_55_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_55_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '146' 'IN' port 'w4_rsc_55_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '147' 'IN' port 'w4_rsc_56_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_56_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '147' 'IN' port 'w4_rsc_56_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '148' 'IN' port 'w4_rsc_57_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_57_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '148' 'IN' port 'w4_rsc_57_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '149' 'IN' port 'w4_rsc_58_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_58_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '149' 'IN' port 'w4_rsc_58_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '150' 'IN' port 'w4_rsc_59_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_59_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '150' 'IN' port 'w4_rsc_59_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '151' 'IN' port 'w4_rsc_60_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_60_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '151' 'IN' port 'w4_rsc_60_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '152' 'IN' port 'w4_rsc_61_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_61_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '152' 'IN' port 'w4_rsc_61_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '153' 'IN' port 'w4_rsc_62_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_62_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '153' 'IN' port 'w4_rsc_62_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '154' 'IN' port 'w4_rsc_63_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_63_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '154' 'IN' port 'w4_rsc_63_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '155' 'IN' port 'w6_rsc_0_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_0_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '155' 'IN' port 'w6_rsc_0_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '156' 'IN' port 'w6_rsc_1_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_1_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '156' 'IN' port 'w6_rsc_1_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '157' 'IN' port 'w6_rsc_2_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_2_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '157' 'IN' port 'w6_rsc_2_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '158' 'IN' port 'w6_rsc_3_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_3_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '158' 'IN' port 'w6_rsc_3_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '159' 'IN' port 'w6_rsc_4_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_4_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '159' 'IN' port 'w6_rsc_4_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '160' 'IN' port 'w6_rsc_5_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_5_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '160' 'IN' port 'w6_rsc_5_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '161' 'IN' port 'w6_rsc_6_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_6_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '161' 'IN' port 'w6_rsc_6_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '162' 'IN' port 'w6_rsc_7_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_7_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '162' 'IN' port 'w6_rsc_7_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '163' 'IN' port 'w6_rsc_8_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_8_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '163' 'IN' port 'w6_rsc_8_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '164' 'IN' port 'w6_rsc_9_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_9_0_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '164' 'IN' port 'w6_rsc_9_0_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '1' 'IN' CLOCK 'input1_rsc_vld'"

  }
  if { [llength [find / -clock {*/output1_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '2' 'OUT' port 'output1_rsc_rdy' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/output1_rsc_rdy}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '2' 'OUT' port 'output1_rsc_rdy' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '27' 'IN' port 'w2_rsc_0_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_0_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '27' 'IN' port 'w2_rsc_0_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '28' 'IN' port 'w2_rsc_1_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_1_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '28' 'IN' port 'w2_rsc_1_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '29' 'IN' port 'w2_rsc_2_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_2_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '29' 'IN' port 'w2_rsc_2_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '30' 'IN' port 'w2_rsc_3_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_3_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '30' 'IN' port 'w2_rsc_3_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '31' 'IN' port 'w2_rsc_4_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_4_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '31' 'IN' port 'w2_rsc_4_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '32' 'IN' port 'w2_rsc_5_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_5_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '32' 'IN' port 'w2_rsc_5_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '33' 'IN' port 'w2_rsc_6_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_6_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '33' 'IN' port 'w2_rsc_6_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '34' 'IN' port 'w2_rsc_7_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_7_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '34' 'IN' port 'w2_rsc_7_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '35' 'IN' port 'w2_rsc_8_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_8_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '35' 'IN' port 'w2_rsc_8_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '36' 'IN' port 'w2_rsc_9_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_9_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '36' 'IN' port 'w2_rsc_9_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '37' 'IN' port 'w2_rsc_10_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_10_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '37' 'IN' port 'w2_rsc_10_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '38' 'IN' port 'w2_rsc_11_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_11_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '38' 'IN' port 'w2_rsc_11_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '39' 'IN' port 'w2_rsc_12_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_12_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '39' 'IN' port 'w2_rsc_12_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '40' 'IN' port 'w2_rsc_13_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_13_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '40' 'IN' port 'w2_rsc_13_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '41' 'IN' port 'w2_rsc_14_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_14_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '41' 'IN' port 'w2_rsc_14_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '42' 'IN' port 'w2_rsc_15_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_15_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '42' 'IN' port 'w2_rsc_15_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '43' 'IN' port 'w2_rsc_16_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_16_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '43' 'IN' port 'w2_rsc_16_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '44' 'IN' port 'w2_rsc_17_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_17_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '44' 'IN' port 'w2_rsc_17_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '45' 'IN' port 'w2_rsc_18_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_18_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '45' 'IN' port 'w2_rsc_18_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '46' 'IN' port 'w2_rsc_19_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_19_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '46' 'IN' port 'w2_rsc_19_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '47' 'IN' port 'w2_rsc_20_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_20_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '47' 'IN' port 'w2_rsc_20_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '48' 'IN' port 'w2_rsc_21_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_21_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '48' 'IN' port 'w2_rsc_21_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '49' 'IN' port 'w2_rsc_22_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_22_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '49' 'IN' port 'w2_rsc_22_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '50' 'IN' port 'w2_rsc_23_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_23_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '50' 'IN' port 'w2_rsc_23_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '51' 'IN' port 'w2_rsc_24_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_24_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '51' 'IN' port 'w2_rsc_24_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '52' 'IN' port 'w2_rsc_25_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_25_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '52' 'IN' port 'w2_rsc_25_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '53' 'IN' port 'w2_rsc_26_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_26_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '53' 'IN' port 'w2_rsc_26_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '54' 'IN' port 'w2_rsc_27_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_27_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '54' 'IN' port 'w2_rsc_27_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '55' 'IN' port 'w2_rsc_28_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_28_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '55' 'IN' port 'w2_rsc_28_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '56' 'IN' port 'w2_rsc_29_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_29_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '56' 'IN' port 'w2_rsc_29_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '57' 'IN' port 'w2_rsc_30_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_30_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '57' 'IN' port 'w2_rsc_30_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '58' 'IN' port 'w2_rsc_31_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_31_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '58' 'IN' port 'w2_rsc_31_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '59' 'IN' port 'w2_rsc_32_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_32_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '59' 'IN' port 'w2_rsc_32_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '60' 'IN' port 'w2_rsc_33_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_33_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '60' 'IN' port 'w2_rsc_33_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '61' 'IN' port 'w2_rsc_34_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_34_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '61' 'IN' port 'w2_rsc_34_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '62' 'IN' port 'w2_rsc_35_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_35_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '62' 'IN' port 'w2_rsc_35_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '63' 'IN' port 'w2_rsc_36_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_36_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '63' 'IN' port 'w2_rsc_36_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '64' 'IN' port 'w2_rsc_37_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_37_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '64' 'IN' port 'w2_rsc_37_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '65' 'IN' port 'w2_rsc_38_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_38_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '65' 'IN' port 'w2_rsc_38_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '66' 'IN' port 'w2_rsc_39_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_39_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '66' 'IN' port 'w2_rsc_39_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '67' 'IN' port 'w2_rsc_40_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_40_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '67' 'IN' port 'w2_rsc_40_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '68' 'IN' port 'w2_rsc_41_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_41_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '68' 'IN' port 'w2_rsc_41_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '69' 'IN' port 'w2_rsc_42_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_42_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '69' 'IN' port 'w2_rsc_42_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '70' 'IN' port 'w2_rsc_43_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_43_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '70' 'IN' port 'w2_rsc_43_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '71' 'IN' port 'w2_rsc_44_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_44_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '71' 'IN' port 'w2_rsc_44_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '72' 'IN' port 'w2_rsc_45_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_45_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '72' 'IN' port 'w2_rsc_45_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '73' 'IN' port 'w2_rsc_46_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_46_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '73' 'IN' port 'w2_rsc_46_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '74' 'IN' port 'w2_rsc_47_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_47_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '74' 'IN' port 'w2_rsc_47_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '75' 'IN' port 'w2_rsc_48_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_48_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '75' 'IN' port 'w2_rsc_48_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '76' 'IN' port 'w2_rsc_49_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_49_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '76' 'IN' port 'w2_rsc_49_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '77' 'IN' port 'w2_rsc_50_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_50_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '77' 'IN' port 'w2_rsc_50_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '78' 'IN' port 'w2_rsc_51_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_51_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '78' 'IN' port 'w2_rsc_51_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '79' 'IN' port 'w2_rsc_52_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_52_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '79' 'IN' port 'w2_rsc_52_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '80' 'IN' port 'w2_rsc_53_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_53_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '80' 'IN' port 'w2_rsc_53_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '81' 'IN' port 'w2_rsc_54_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_54_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '81' 'IN' port 'w2_rsc_54_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '82' 'IN' port 'w2_rsc_55_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_55_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '82' 'IN' port 'w2_rsc_55_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '83' 'IN' port 'w2_rsc_56_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_56_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '83' 'IN' port 'w2_rsc_56_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '84' 'IN' port 'w2_rsc_57_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_57_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '84' 'IN' port 'w2_rsc_57_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '85' 'IN' port 'w2_rsc_58_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_58_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '85' 'IN' port 'w2_rsc_58_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '86' 'IN' port 'w2_rsc_59_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_59_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '86' 'IN' port 'w2_rsc_59_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '87' 'IN' port 'w2_rsc_60_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_60_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '87' 'IN' port 'w2_rsc_60_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '88' 'IN' port 'w2_rsc_61_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_61_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '88' 'IN' port 'w2_rsc_61_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '89' 'IN' port 'w2_rsc_62_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_62_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '89' 'IN' port 'w2_rsc_62_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '90' 'IN' port 'w2_rsc_63_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_63_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '90' 'IN' port 'w2_rsc_63_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '91' 'IN' port 'w4_rsc_0_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_0_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '91' 'IN' port 'w4_rsc_0_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '92' 'IN' port 'w4_rsc_1_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_1_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '92' 'IN' port 'w4_rsc_1_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '93' 'IN' port 'w4_rsc_2_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_2_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '93' 'IN' port 'w4_rsc_2_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '94' 'IN' port 'w4_rsc_3_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_3_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '94' 'IN' port 'w4_rsc_3_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '95' 'IN' port 'w4_rsc_4_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_4_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '95' 'IN' port 'w4_rsc_4_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '96' 'IN' port 'w4_rsc_5_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_5_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '96' 'IN' port 'w4_rsc_5_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '97' 'IN' port 'w4_rsc_6_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_6_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '97' 'IN' port 'w4_rsc_6_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '98' 'IN' port 'w4_rsc_7_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_7_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '98' 'IN' port 'w4_rsc_7_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '99' 'IN' port 'w4_rsc_8_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_8_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '99' 'IN' port 'w4_rsc_8_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '100' 'IN' port 'w4_rsc_9_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_9_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '100' 'IN' port 'w4_rsc_9_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '101' 'IN' port 'w4_rsc_10_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_10_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '101' 'IN' port 'w4_rsc_10_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '102' 'IN' port 'w4_rsc_11_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_11_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '102' 'IN' port 'w4_rsc_11_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '103' 'IN' port 'w4_rsc_12_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_12_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '103' 'IN' port 'w4_rsc_12_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '104' 'IN' port 'w4_rsc_13_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_13_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '104' 'IN' port 'w4_rsc_13_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '105' 'IN' port 'w4_rsc_14_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_14_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '105' 'IN' port 'w4_rsc_14_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '106' 'IN' port 'w4_rsc_15_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_15_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '106' 'IN' port 'w4_rsc_15_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '107' 'IN' port 'w4_rsc_16_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_16_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '107' 'IN' port 'w4_rsc_16_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '108' 'IN' port 'w4_rsc_17_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_17_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '108' 'IN' port 'w4_rsc_17_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '109' 'IN' port 'w4_rsc_18_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_18_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '109' 'IN' port 'w4_rsc_18_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '110' 'IN' port 'w4_rsc_19_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_19_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '110' 'IN' port 'w4_rsc_19_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '111' 'IN' port 'w4_rsc_20_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_20_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '111' 'IN' port 'w4_rsc_20_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '112' 'IN' port 'w4_rsc_21_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_21_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '112' 'IN' port 'w4_rsc_21_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '113' 'IN' port 'w4_rsc_22_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_22_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '113' 'IN' port 'w4_rsc_22_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '114' 'IN' port 'w4_rsc_23_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_23_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '114' 'IN' port 'w4_rsc_23_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '115' 'IN' port 'w4_rsc_24_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_24_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '115' 'IN' port 'w4_rsc_24_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '116' 'IN' port 'w4_rsc_25_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_25_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '116' 'IN' port 'w4_rsc_25_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '117' 'IN' port 'w4_rsc_26_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_26_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '117' 'IN' port 'w4_rsc_26_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '118' 'IN' port 'w4_rsc_27_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_27_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '118' 'IN' port 'w4_rsc_27_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '119' 'IN' port 'w4_rsc_28_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_28_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '119' 'IN' port 'w4_rsc_28_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '120' 'IN' port 'w4_rsc_29_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_29_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '120' 'IN' port 'w4_rsc_29_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '121' 'IN' port 'w4_rsc_30_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_30_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '121' 'IN' port 'w4_rsc_30_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '122' 'IN' port 'w4_rsc_31_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_31_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '122' 'IN' port 'w4_rsc_31_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '123' 'IN' port 'w4_rsc_32_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_32_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '123' 'IN' port 'w4_rsc_32_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '124' 'IN' port 'w4_rsc_33_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_33_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '124' 'IN' port 'w4_rsc_33_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '125' 'IN' port 'w4_rsc_34_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_34_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '125' 'IN' port 'w4_rsc_34_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '126' 'IN' port 'w4_rsc_35_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_35_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '126' 'IN' port 'w4_rsc_35_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '127' 'IN' port 'w4_rsc_36_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_36_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '127' 'IN' port 'w4_rsc_36_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '128' 'IN' port 'w4_rsc_37_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_37_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '128' 'IN' port 'w4_rsc_37_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '129' 'IN' port 'w4_rsc_38_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_38_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '129' 'IN' port 'w4_rsc_38_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '130' 'IN' port 'w4_rsc_39_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_39_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '130' 'IN' port 'w4_rsc_39_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '131' 'IN' port 'w4_rsc_40_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_40_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '131' 'IN' port 'w4_rsc_40_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '132' 'IN' port 'w4_rsc_41_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_41_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '132' 'IN' port 'w4_rsc_41_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '133' 'IN' port 'w4_rsc_42_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_42_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '133' 'IN' port 'w4_rsc_42_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '134' 'IN' port 'w4_rsc_43_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_43_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '134' 'IN' port 'w4_rsc_43_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '135' 'IN' port 'w4_rsc_44_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_44_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '135' 'IN' port 'w4_rsc_44_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '136' 'IN' port 'w4_rsc_45_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_45_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '136' 'IN' port 'w4_rsc_45_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '137' 'IN' port 'w4_rsc_46_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_46_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '137' 'IN' port 'w4_rsc_46_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '138' 'IN' port 'w4_rsc_47_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_47_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '138' 'IN' port 'w4_rsc_47_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '139' 'IN' port 'w4_rsc_48_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_48_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '139' 'IN' port 'w4_rsc_48_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '140' 'IN' port 'w4_rsc_49_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_49_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '140' 'IN' port 'w4_rsc_49_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '141' 'IN' port 'w4_rsc_50_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_50_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '141' 'IN' port 'w4_rsc_50_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '142' 'IN' port 'w4_rsc_51_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_51_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '142' 'IN' port 'w4_rsc_51_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '143' 'IN' port 'w4_rsc_52_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_52_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '143' 'IN' port 'w4_rsc_52_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '144' 'IN' port 'w4_rsc_53_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_53_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '144' 'IN' port 'w4_rsc_53_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '145' 'IN' port 'w4_rsc_54_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_54_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '145' 'IN' port 'w4_rsc_54_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '146' 'IN' port 'w4_rsc_55_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_55_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '146' 'IN' port 'w4_rsc_55_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '147' 'IN' port 'w4_rsc_56_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_56_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '147' 'IN' port 'w4_rsc_56_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '148' 'IN' port 'w4_rsc_57_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_57_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '148' 'IN' port 'w4_rsc_57_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '149' 'IN' port 'w4_rsc_58_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_58_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '149' 'IN' port 'w4_rsc_58_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '150' 'IN' port 'w4_rsc_59_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_59_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '150' 'IN' port 'w4_rsc_59_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '151' 'IN' port 'w4_rsc_60_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_60_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '151' 'IN' port 'w4_rsc_60_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '152' 'IN' port 'w4_rsc_61_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_61_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '152' 'IN' port 'w4_rsc_61_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '153' 'IN' port 'w4_rsc_62_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_62_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '153' 'IN' port 'w4_rsc_62_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '154' 'IN' port 'w4_rsc_63_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_63_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '154' 'IN' port 'w4_rsc_63_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '155' 'IN' port 'w6_rsc_0_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_0_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '155' 'IN' port 'w6_rsc_0_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '156' 'IN' port 'w6_rsc_1_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_1_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '156' 'IN' port 'w6_rsc_1_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '157' 'IN' port 'w6_rsc_2_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_2_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '157' 'IN' port 'w6_rsc_2_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '158' 'IN' port 'w6_rsc_3_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_3_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '158' 'IN' port 'w6_rsc_3_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '159' 'IN' port 'w6_rsc_4_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_4_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '159' 'IN' port 'w6_rsc_4_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '160' 'IN' port 'w6_rsc_5_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_5_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '160' 'IN' port 'w6_rsc_5_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '161' 'IN' port 'w6_rsc_6_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_6_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '161' 'IN' port 'w6_rsc_6_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '162' 'IN' port 'w6_rsc_7_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_7_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '162' 'IN' port 'w6_rsc_7_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '163' 'IN' port 'w6_rsc_8_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_8_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '163' 'IN' port 'w6_rsc_8_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '164' 'IN' port 'w6_rsc_9_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_9_0_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '164' 'IN' port 'w6_rsc_9_0_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

    puts "-- Synthesis input_to_register:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis input_to_register:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '2' 'OUT' CLOCK 'output1_rsc_rdy'"

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

  if { [llength [find / -clock {*/clk}] ] > 0 && [llength [find / -clock {*/output1_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
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

  if { [llength [find / -clock {*/input1_rsc_vld}] ] > 0 && [llength [find / -clock {*/output1_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
  }

  if { [llength [find / -clock {*/output1_rsc_rdy}] ] > 0 && [llength [find / -clock {*/clk}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'output1_rsc_rdy' '0' 'INOUT' CLOCK 'clk'"
    report timing -full_pin_names -from [find / -clock {*/output1_rsc_rdy}] -to [find / -clock {*/clk}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'output1_rsc_rdy' '0' 'INOUT' CLOCK 'clk'"
  }

  if { [llength [find / -clock {*/output1_rsc_rdy}] ] > 0 && [llength [find / -clock {*/input1_rsc_vld}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'output1_rsc_rdy' '1' 'IN' CLOCK 'input1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/output1_rsc_rdy}] -to [find / -clock {*/input1_rsc_vld}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'output1_rsc_rdy' '1' 'IN' CLOCK 'input1_rsc_vld'"
  }

  if { [llength [find / -clock {*/output1_rsc_rdy}] ] > 0 && [llength [find / -clock {*/output1_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_register:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'output1_rsc_rdy' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/output1_rsc_rdy}] -to [find / -clock {*/output1_rsc_rdy}]
    puts "-- END Synthesis register_to_register:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'output1_rsc_rdy' '2' 'OUT' CLOCK 'output1_rsc_rdy'"
  }

  if { [llength [find / -clock {*/clk}] ] > 0 } {
    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/clk}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '0' 'INOUT' CLOCK 'clk' '2' 'OUT' port 'output1_rsc_dat'"

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

  }

  if { [llength [find / -clock {*/input1_rsc_vld}] ] > 0 } {
    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/input1_rsc_vld}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '1' 'IN' CLOCK 'input1_rsc_vld' '2' 'OUT' port 'output1_rsc_dat'"

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

  }

  if { [llength [find / -clock {*/output1_rsc_rdy}] ] > 0 } {
    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'output1_rsc_rdy' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -clock {*/output1_rsc_rdy}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'output1_rsc_rdy' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'output1_rsc_rdy' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/output1_rsc_rdy}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'output1_rsc_rdy' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'output1_rsc_rdy' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/output1_rsc_rdy}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'output1_rsc_rdy' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'output1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/output1_rsc_rdy}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'output1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'output1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/output1_rsc_rdy}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'output1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'output1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -clock {*/output1_rsc_rdy}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'output1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'output1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -clock {*/output1_rsc_rdy}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis register_to_output:timing report for design 'mnist_mlp' '2' 'OUT' CLOCK 'output1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

  }

  if { [llength [all_inputs -design mnist_mlp]] != 0 && [llength [all_outputs -design mnist_mlp]] != 0 } {
    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_vld}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_vld' '2' 'OUT' port 'output1_rsc_dat'"

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

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/input1_rsc_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '1' 'IN' port 'input1_rsc_dat' '2' 'OUT' port 'output1_rsc_dat'"

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

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'output1_rsc_rdy' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/output1_rsc_rdy}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'output1_rsc_rdy' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'output1_rsc_rdy' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/output1_rsc_rdy}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'output1_rsc_rdy' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'output1_rsc_rdy' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/output1_rsc_rdy}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'output1_rsc_rdy' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'output1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/output1_rsc_rdy}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'output1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'output1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/output1_rsc_rdy}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'output1_rsc_rdy' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'output1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/output1_rsc_rdy}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'output1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'output1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/output1_rsc_rdy}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '2' 'OUT' port 'output1_rsc_rdy' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '27' 'IN' port 'w2_rsc_0_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_0_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '27' 'IN' port 'w2_rsc_0_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '27' 'IN' port 'w2_rsc_0_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_0_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '27' 'IN' port 'w2_rsc_0_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '27' 'IN' port 'w2_rsc_0_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_0_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '27' 'IN' port 'w2_rsc_0_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '27' 'IN' port 'w2_rsc_0_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_0_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '27' 'IN' port 'w2_rsc_0_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '27' 'IN' port 'w2_rsc_0_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_0_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '27' 'IN' port 'w2_rsc_0_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '27' 'IN' port 'w2_rsc_0_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_0_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '27' 'IN' port 'w2_rsc_0_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '27' 'IN' port 'w2_rsc_0_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_0_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '27' 'IN' port 'w2_rsc_0_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '28' 'IN' port 'w2_rsc_1_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_1_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '28' 'IN' port 'w2_rsc_1_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '28' 'IN' port 'w2_rsc_1_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_1_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '28' 'IN' port 'w2_rsc_1_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '28' 'IN' port 'w2_rsc_1_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_1_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '28' 'IN' port 'w2_rsc_1_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '28' 'IN' port 'w2_rsc_1_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_1_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '28' 'IN' port 'w2_rsc_1_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '28' 'IN' port 'w2_rsc_1_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_1_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '28' 'IN' port 'w2_rsc_1_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '28' 'IN' port 'w2_rsc_1_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_1_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '28' 'IN' port 'w2_rsc_1_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '28' 'IN' port 'w2_rsc_1_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_1_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '28' 'IN' port 'w2_rsc_1_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '29' 'IN' port 'w2_rsc_2_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_2_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '29' 'IN' port 'w2_rsc_2_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '29' 'IN' port 'w2_rsc_2_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_2_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '29' 'IN' port 'w2_rsc_2_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '29' 'IN' port 'w2_rsc_2_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_2_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '29' 'IN' port 'w2_rsc_2_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '29' 'IN' port 'w2_rsc_2_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_2_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '29' 'IN' port 'w2_rsc_2_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '29' 'IN' port 'w2_rsc_2_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_2_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '29' 'IN' port 'w2_rsc_2_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '29' 'IN' port 'w2_rsc_2_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_2_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '29' 'IN' port 'w2_rsc_2_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '29' 'IN' port 'w2_rsc_2_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_2_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '29' 'IN' port 'w2_rsc_2_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '30' 'IN' port 'w2_rsc_3_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_3_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '30' 'IN' port 'w2_rsc_3_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '30' 'IN' port 'w2_rsc_3_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_3_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '30' 'IN' port 'w2_rsc_3_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '30' 'IN' port 'w2_rsc_3_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_3_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '30' 'IN' port 'w2_rsc_3_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '30' 'IN' port 'w2_rsc_3_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_3_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '30' 'IN' port 'w2_rsc_3_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '30' 'IN' port 'w2_rsc_3_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_3_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '30' 'IN' port 'w2_rsc_3_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '30' 'IN' port 'w2_rsc_3_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_3_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '30' 'IN' port 'w2_rsc_3_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '30' 'IN' port 'w2_rsc_3_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_3_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '30' 'IN' port 'w2_rsc_3_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '31' 'IN' port 'w2_rsc_4_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_4_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '31' 'IN' port 'w2_rsc_4_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '31' 'IN' port 'w2_rsc_4_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_4_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '31' 'IN' port 'w2_rsc_4_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '31' 'IN' port 'w2_rsc_4_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_4_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '31' 'IN' port 'w2_rsc_4_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '31' 'IN' port 'w2_rsc_4_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_4_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '31' 'IN' port 'w2_rsc_4_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '31' 'IN' port 'w2_rsc_4_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_4_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '31' 'IN' port 'w2_rsc_4_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '31' 'IN' port 'w2_rsc_4_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_4_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '31' 'IN' port 'w2_rsc_4_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '31' 'IN' port 'w2_rsc_4_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_4_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '31' 'IN' port 'w2_rsc_4_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '32' 'IN' port 'w2_rsc_5_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_5_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '32' 'IN' port 'w2_rsc_5_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '32' 'IN' port 'w2_rsc_5_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_5_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '32' 'IN' port 'w2_rsc_5_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '32' 'IN' port 'w2_rsc_5_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_5_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '32' 'IN' port 'w2_rsc_5_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '32' 'IN' port 'w2_rsc_5_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_5_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '32' 'IN' port 'w2_rsc_5_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '32' 'IN' port 'w2_rsc_5_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_5_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '32' 'IN' port 'w2_rsc_5_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '32' 'IN' port 'w2_rsc_5_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_5_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '32' 'IN' port 'w2_rsc_5_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '32' 'IN' port 'w2_rsc_5_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_5_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '32' 'IN' port 'w2_rsc_5_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '33' 'IN' port 'w2_rsc_6_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_6_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '33' 'IN' port 'w2_rsc_6_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '33' 'IN' port 'w2_rsc_6_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_6_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '33' 'IN' port 'w2_rsc_6_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '33' 'IN' port 'w2_rsc_6_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_6_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '33' 'IN' port 'w2_rsc_6_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '33' 'IN' port 'w2_rsc_6_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_6_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '33' 'IN' port 'w2_rsc_6_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '33' 'IN' port 'w2_rsc_6_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_6_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '33' 'IN' port 'w2_rsc_6_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '33' 'IN' port 'w2_rsc_6_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_6_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '33' 'IN' port 'w2_rsc_6_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '33' 'IN' port 'w2_rsc_6_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_6_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '33' 'IN' port 'w2_rsc_6_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '34' 'IN' port 'w2_rsc_7_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_7_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '34' 'IN' port 'w2_rsc_7_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '34' 'IN' port 'w2_rsc_7_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_7_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '34' 'IN' port 'w2_rsc_7_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '34' 'IN' port 'w2_rsc_7_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_7_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '34' 'IN' port 'w2_rsc_7_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '34' 'IN' port 'w2_rsc_7_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_7_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '34' 'IN' port 'w2_rsc_7_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '34' 'IN' port 'w2_rsc_7_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_7_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '34' 'IN' port 'w2_rsc_7_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '34' 'IN' port 'w2_rsc_7_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_7_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '34' 'IN' port 'w2_rsc_7_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '34' 'IN' port 'w2_rsc_7_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_7_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '34' 'IN' port 'w2_rsc_7_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '35' 'IN' port 'w2_rsc_8_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_8_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '35' 'IN' port 'w2_rsc_8_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '35' 'IN' port 'w2_rsc_8_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_8_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '35' 'IN' port 'w2_rsc_8_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '35' 'IN' port 'w2_rsc_8_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_8_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '35' 'IN' port 'w2_rsc_8_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '35' 'IN' port 'w2_rsc_8_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_8_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '35' 'IN' port 'w2_rsc_8_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '35' 'IN' port 'w2_rsc_8_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_8_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '35' 'IN' port 'w2_rsc_8_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '35' 'IN' port 'w2_rsc_8_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_8_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '35' 'IN' port 'w2_rsc_8_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '35' 'IN' port 'w2_rsc_8_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_8_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '35' 'IN' port 'w2_rsc_8_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '36' 'IN' port 'w2_rsc_9_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_9_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '36' 'IN' port 'w2_rsc_9_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '36' 'IN' port 'w2_rsc_9_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_9_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '36' 'IN' port 'w2_rsc_9_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '36' 'IN' port 'w2_rsc_9_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_9_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '36' 'IN' port 'w2_rsc_9_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '36' 'IN' port 'w2_rsc_9_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_9_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '36' 'IN' port 'w2_rsc_9_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '36' 'IN' port 'w2_rsc_9_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_9_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '36' 'IN' port 'w2_rsc_9_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '36' 'IN' port 'w2_rsc_9_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_9_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '36' 'IN' port 'w2_rsc_9_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '36' 'IN' port 'w2_rsc_9_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_9_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '36' 'IN' port 'w2_rsc_9_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '37' 'IN' port 'w2_rsc_10_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_10_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '37' 'IN' port 'w2_rsc_10_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '37' 'IN' port 'w2_rsc_10_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_10_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '37' 'IN' port 'w2_rsc_10_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '37' 'IN' port 'w2_rsc_10_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_10_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '37' 'IN' port 'w2_rsc_10_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '37' 'IN' port 'w2_rsc_10_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_10_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '37' 'IN' port 'w2_rsc_10_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '37' 'IN' port 'w2_rsc_10_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_10_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '37' 'IN' port 'w2_rsc_10_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '37' 'IN' port 'w2_rsc_10_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_10_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '37' 'IN' port 'w2_rsc_10_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '37' 'IN' port 'w2_rsc_10_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_10_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '37' 'IN' port 'w2_rsc_10_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '38' 'IN' port 'w2_rsc_11_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_11_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '38' 'IN' port 'w2_rsc_11_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '38' 'IN' port 'w2_rsc_11_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_11_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '38' 'IN' port 'w2_rsc_11_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '38' 'IN' port 'w2_rsc_11_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_11_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '38' 'IN' port 'w2_rsc_11_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '38' 'IN' port 'w2_rsc_11_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_11_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '38' 'IN' port 'w2_rsc_11_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '38' 'IN' port 'w2_rsc_11_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_11_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '38' 'IN' port 'w2_rsc_11_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '38' 'IN' port 'w2_rsc_11_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_11_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '38' 'IN' port 'w2_rsc_11_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '38' 'IN' port 'w2_rsc_11_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_11_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '38' 'IN' port 'w2_rsc_11_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '39' 'IN' port 'w2_rsc_12_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_12_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '39' 'IN' port 'w2_rsc_12_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '39' 'IN' port 'w2_rsc_12_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_12_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '39' 'IN' port 'w2_rsc_12_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '39' 'IN' port 'w2_rsc_12_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_12_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '39' 'IN' port 'w2_rsc_12_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '39' 'IN' port 'w2_rsc_12_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_12_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '39' 'IN' port 'w2_rsc_12_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '39' 'IN' port 'w2_rsc_12_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_12_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '39' 'IN' port 'w2_rsc_12_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '39' 'IN' port 'w2_rsc_12_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_12_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '39' 'IN' port 'w2_rsc_12_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '39' 'IN' port 'w2_rsc_12_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_12_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '39' 'IN' port 'w2_rsc_12_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '40' 'IN' port 'w2_rsc_13_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_13_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '40' 'IN' port 'w2_rsc_13_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '40' 'IN' port 'w2_rsc_13_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_13_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '40' 'IN' port 'w2_rsc_13_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '40' 'IN' port 'w2_rsc_13_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_13_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '40' 'IN' port 'w2_rsc_13_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '40' 'IN' port 'w2_rsc_13_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_13_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '40' 'IN' port 'w2_rsc_13_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '40' 'IN' port 'w2_rsc_13_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_13_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '40' 'IN' port 'w2_rsc_13_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '40' 'IN' port 'w2_rsc_13_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_13_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '40' 'IN' port 'w2_rsc_13_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '40' 'IN' port 'w2_rsc_13_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_13_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '40' 'IN' port 'w2_rsc_13_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '41' 'IN' port 'w2_rsc_14_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_14_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '41' 'IN' port 'w2_rsc_14_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '41' 'IN' port 'w2_rsc_14_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_14_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '41' 'IN' port 'w2_rsc_14_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '41' 'IN' port 'w2_rsc_14_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_14_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '41' 'IN' port 'w2_rsc_14_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '41' 'IN' port 'w2_rsc_14_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_14_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '41' 'IN' port 'w2_rsc_14_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '41' 'IN' port 'w2_rsc_14_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_14_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '41' 'IN' port 'w2_rsc_14_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '41' 'IN' port 'w2_rsc_14_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_14_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '41' 'IN' port 'w2_rsc_14_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '41' 'IN' port 'w2_rsc_14_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_14_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '41' 'IN' port 'w2_rsc_14_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '42' 'IN' port 'w2_rsc_15_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_15_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '42' 'IN' port 'w2_rsc_15_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '42' 'IN' port 'w2_rsc_15_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_15_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '42' 'IN' port 'w2_rsc_15_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '42' 'IN' port 'w2_rsc_15_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_15_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '42' 'IN' port 'w2_rsc_15_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '42' 'IN' port 'w2_rsc_15_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_15_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '42' 'IN' port 'w2_rsc_15_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '42' 'IN' port 'w2_rsc_15_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_15_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '42' 'IN' port 'w2_rsc_15_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '42' 'IN' port 'w2_rsc_15_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_15_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '42' 'IN' port 'w2_rsc_15_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '42' 'IN' port 'w2_rsc_15_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_15_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '42' 'IN' port 'w2_rsc_15_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '43' 'IN' port 'w2_rsc_16_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_16_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '43' 'IN' port 'w2_rsc_16_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '43' 'IN' port 'w2_rsc_16_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_16_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '43' 'IN' port 'w2_rsc_16_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '43' 'IN' port 'w2_rsc_16_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_16_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '43' 'IN' port 'w2_rsc_16_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '43' 'IN' port 'w2_rsc_16_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_16_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '43' 'IN' port 'w2_rsc_16_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '43' 'IN' port 'w2_rsc_16_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_16_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '43' 'IN' port 'w2_rsc_16_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '43' 'IN' port 'w2_rsc_16_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_16_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '43' 'IN' port 'w2_rsc_16_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '43' 'IN' port 'w2_rsc_16_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_16_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '43' 'IN' port 'w2_rsc_16_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '44' 'IN' port 'w2_rsc_17_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_17_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '44' 'IN' port 'w2_rsc_17_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '44' 'IN' port 'w2_rsc_17_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_17_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '44' 'IN' port 'w2_rsc_17_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '44' 'IN' port 'w2_rsc_17_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_17_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '44' 'IN' port 'w2_rsc_17_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '44' 'IN' port 'w2_rsc_17_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_17_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '44' 'IN' port 'w2_rsc_17_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '44' 'IN' port 'w2_rsc_17_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_17_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '44' 'IN' port 'w2_rsc_17_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '44' 'IN' port 'w2_rsc_17_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_17_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '44' 'IN' port 'w2_rsc_17_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '44' 'IN' port 'w2_rsc_17_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_17_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '44' 'IN' port 'w2_rsc_17_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '45' 'IN' port 'w2_rsc_18_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_18_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '45' 'IN' port 'w2_rsc_18_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '45' 'IN' port 'w2_rsc_18_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_18_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '45' 'IN' port 'w2_rsc_18_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '45' 'IN' port 'w2_rsc_18_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_18_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '45' 'IN' port 'w2_rsc_18_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '45' 'IN' port 'w2_rsc_18_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_18_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '45' 'IN' port 'w2_rsc_18_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '45' 'IN' port 'w2_rsc_18_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_18_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '45' 'IN' port 'w2_rsc_18_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '45' 'IN' port 'w2_rsc_18_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_18_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '45' 'IN' port 'w2_rsc_18_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '45' 'IN' port 'w2_rsc_18_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_18_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '45' 'IN' port 'w2_rsc_18_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '46' 'IN' port 'w2_rsc_19_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_19_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '46' 'IN' port 'w2_rsc_19_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '46' 'IN' port 'w2_rsc_19_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_19_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '46' 'IN' port 'w2_rsc_19_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '46' 'IN' port 'w2_rsc_19_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_19_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '46' 'IN' port 'w2_rsc_19_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '46' 'IN' port 'w2_rsc_19_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_19_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '46' 'IN' port 'w2_rsc_19_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '46' 'IN' port 'w2_rsc_19_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_19_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '46' 'IN' port 'w2_rsc_19_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '46' 'IN' port 'w2_rsc_19_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_19_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '46' 'IN' port 'w2_rsc_19_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '46' 'IN' port 'w2_rsc_19_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_19_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '46' 'IN' port 'w2_rsc_19_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '47' 'IN' port 'w2_rsc_20_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_20_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '47' 'IN' port 'w2_rsc_20_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '47' 'IN' port 'w2_rsc_20_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_20_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '47' 'IN' port 'w2_rsc_20_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '47' 'IN' port 'w2_rsc_20_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_20_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '47' 'IN' port 'w2_rsc_20_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '47' 'IN' port 'w2_rsc_20_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_20_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '47' 'IN' port 'w2_rsc_20_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '47' 'IN' port 'w2_rsc_20_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_20_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '47' 'IN' port 'w2_rsc_20_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '47' 'IN' port 'w2_rsc_20_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_20_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '47' 'IN' port 'w2_rsc_20_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '47' 'IN' port 'w2_rsc_20_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_20_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '47' 'IN' port 'w2_rsc_20_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '48' 'IN' port 'w2_rsc_21_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_21_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '48' 'IN' port 'w2_rsc_21_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '48' 'IN' port 'w2_rsc_21_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_21_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '48' 'IN' port 'w2_rsc_21_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '48' 'IN' port 'w2_rsc_21_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_21_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '48' 'IN' port 'w2_rsc_21_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '48' 'IN' port 'w2_rsc_21_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_21_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '48' 'IN' port 'w2_rsc_21_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '48' 'IN' port 'w2_rsc_21_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_21_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '48' 'IN' port 'w2_rsc_21_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '48' 'IN' port 'w2_rsc_21_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_21_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '48' 'IN' port 'w2_rsc_21_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '48' 'IN' port 'w2_rsc_21_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_21_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '48' 'IN' port 'w2_rsc_21_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '49' 'IN' port 'w2_rsc_22_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_22_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '49' 'IN' port 'w2_rsc_22_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '49' 'IN' port 'w2_rsc_22_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_22_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '49' 'IN' port 'w2_rsc_22_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '49' 'IN' port 'w2_rsc_22_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_22_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '49' 'IN' port 'w2_rsc_22_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '49' 'IN' port 'w2_rsc_22_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_22_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '49' 'IN' port 'w2_rsc_22_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '49' 'IN' port 'w2_rsc_22_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_22_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '49' 'IN' port 'w2_rsc_22_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '49' 'IN' port 'w2_rsc_22_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_22_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '49' 'IN' port 'w2_rsc_22_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '49' 'IN' port 'w2_rsc_22_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_22_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '49' 'IN' port 'w2_rsc_22_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '50' 'IN' port 'w2_rsc_23_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_23_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '50' 'IN' port 'w2_rsc_23_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '50' 'IN' port 'w2_rsc_23_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_23_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '50' 'IN' port 'w2_rsc_23_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '50' 'IN' port 'w2_rsc_23_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_23_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '50' 'IN' port 'w2_rsc_23_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '50' 'IN' port 'w2_rsc_23_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_23_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '50' 'IN' port 'w2_rsc_23_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '50' 'IN' port 'w2_rsc_23_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_23_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '50' 'IN' port 'w2_rsc_23_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '50' 'IN' port 'w2_rsc_23_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_23_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '50' 'IN' port 'w2_rsc_23_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '50' 'IN' port 'w2_rsc_23_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_23_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '50' 'IN' port 'w2_rsc_23_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '51' 'IN' port 'w2_rsc_24_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_24_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '51' 'IN' port 'w2_rsc_24_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '51' 'IN' port 'w2_rsc_24_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_24_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '51' 'IN' port 'w2_rsc_24_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '51' 'IN' port 'w2_rsc_24_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_24_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '51' 'IN' port 'w2_rsc_24_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '51' 'IN' port 'w2_rsc_24_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_24_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '51' 'IN' port 'w2_rsc_24_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '51' 'IN' port 'w2_rsc_24_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_24_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '51' 'IN' port 'w2_rsc_24_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '51' 'IN' port 'w2_rsc_24_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_24_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '51' 'IN' port 'w2_rsc_24_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '51' 'IN' port 'w2_rsc_24_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_24_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '51' 'IN' port 'w2_rsc_24_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '52' 'IN' port 'w2_rsc_25_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_25_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '52' 'IN' port 'w2_rsc_25_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '52' 'IN' port 'w2_rsc_25_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_25_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '52' 'IN' port 'w2_rsc_25_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '52' 'IN' port 'w2_rsc_25_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_25_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '52' 'IN' port 'w2_rsc_25_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '52' 'IN' port 'w2_rsc_25_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_25_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '52' 'IN' port 'w2_rsc_25_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '52' 'IN' port 'w2_rsc_25_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_25_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '52' 'IN' port 'w2_rsc_25_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '52' 'IN' port 'w2_rsc_25_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_25_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '52' 'IN' port 'w2_rsc_25_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '52' 'IN' port 'w2_rsc_25_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_25_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '52' 'IN' port 'w2_rsc_25_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '53' 'IN' port 'w2_rsc_26_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_26_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '53' 'IN' port 'w2_rsc_26_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '53' 'IN' port 'w2_rsc_26_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_26_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '53' 'IN' port 'w2_rsc_26_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '53' 'IN' port 'w2_rsc_26_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_26_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '53' 'IN' port 'w2_rsc_26_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '53' 'IN' port 'w2_rsc_26_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_26_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '53' 'IN' port 'w2_rsc_26_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '53' 'IN' port 'w2_rsc_26_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_26_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '53' 'IN' port 'w2_rsc_26_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '53' 'IN' port 'w2_rsc_26_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_26_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '53' 'IN' port 'w2_rsc_26_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '53' 'IN' port 'w2_rsc_26_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_26_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '53' 'IN' port 'w2_rsc_26_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '54' 'IN' port 'w2_rsc_27_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_27_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '54' 'IN' port 'w2_rsc_27_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '54' 'IN' port 'w2_rsc_27_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_27_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '54' 'IN' port 'w2_rsc_27_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '54' 'IN' port 'w2_rsc_27_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_27_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '54' 'IN' port 'w2_rsc_27_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '54' 'IN' port 'w2_rsc_27_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_27_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '54' 'IN' port 'w2_rsc_27_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '54' 'IN' port 'w2_rsc_27_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_27_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '54' 'IN' port 'w2_rsc_27_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '54' 'IN' port 'w2_rsc_27_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_27_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '54' 'IN' port 'w2_rsc_27_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '54' 'IN' port 'w2_rsc_27_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_27_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '54' 'IN' port 'w2_rsc_27_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '55' 'IN' port 'w2_rsc_28_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_28_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '55' 'IN' port 'w2_rsc_28_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '55' 'IN' port 'w2_rsc_28_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_28_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '55' 'IN' port 'w2_rsc_28_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '55' 'IN' port 'w2_rsc_28_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_28_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '55' 'IN' port 'w2_rsc_28_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '55' 'IN' port 'w2_rsc_28_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_28_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '55' 'IN' port 'w2_rsc_28_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '55' 'IN' port 'w2_rsc_28_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_28_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '55' 'IN' port 'w2_rsc_28_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '55' 'IN' port 'w2_rsc_28_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_28_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '55' 'IN' port 'w2_rsc_28_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '55' 'IN' port 'w2_rsc_28_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_28_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '55' 'IN' port 'w2_rsc_28_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '56' 'IN' port 'w2_rsc_29_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_29_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '56' 'IN' port 'w2_rsc_29_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '56' 'IN' port 'w2_rsc_29_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_29_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '56' 'IN' port 'w2_rsc_29_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '56' 'IN' port 'w2_rsc_29_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_29_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '56' 'IN' port 'w2_rsc_29_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '56' 'IN' port 'w2_rsc_29_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_29_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '56' 'IN' port 'w2_rsc_29_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '56' 'IN' port 'w2_rsc_29_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_29_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '56' 'IN' port 'w2_rsc_29_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '56' 'IN' port 'w2_rsc_29_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_29_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '56' 'IN' port 'w2_rsc_29_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '56' 'IN' port 'w2_rsc_29_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_29_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '56' 'IN' port 'w2_rsc_29_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '57' 'IN' port 'w2_rsc_30_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_30_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '57' 'IN' port 'w2_rsc_30_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '57' 'IN' port 'w2_rsc_30_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_30_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '57' 'IN' port 'w2_rsc_30_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '57' 'IN' port 'w2_rsc_30_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_30_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '57' 'IN' port 'w2_rsc_30_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '57' 'IN' port 'w2_rsc_30_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_30_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '57' 'IN' port 'w2_rsc_30_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '57' 'IN' port 'w2_rsc_30_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_30_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '57' 'IN' port 'w2_rsc_30_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '57' 'IN' port 'w2_rsc_30_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_30_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '57' 'IN' port 'w2_rsc_30_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '57' 'IN' port 'w2_rsc_30_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_30_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '57' 'IN' port 'w2_rsc_30_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '58' 'IN' port 'w2_rsc_31_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_31_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '58' 'IN' port 'w2_rsc_31_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '58' 'IN' port 'w2_rsc_31_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_31_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '58' 'IN' port 'w2_rsc_31_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '58' 'IN' port 'w2_rsc_31_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_31_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '58' 'IN' port 'w2_rsc_31_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '58' 'IN' port 'w2_rsc_31_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_31_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '58' 'IN' port 'w2_rsc_31_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '58' 'IN' port 'w2_rsc_31_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_31_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '58' 'IN' port 'w2_rsc_31_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '58' 'IN' port 'w2_rsc_31_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_31_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '58' 'IN' port 'w2_rsc_31_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '58' 'IN' port 'w2_rsc_31_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_31_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '58' 'IN' port 'w2_rsc_31_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '59' 'IN' port 'w2_rsc_32_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_32_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '59' 'IN' port 'w2_rsc_32_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '59' 'IN' port 'w2_rsc_32_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_32_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '59' 'IN' port 'w2_rsc_32_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '59' 'IN' port 'w2_rsc_32_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_32_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '59' 'IN' port 'w2_rsc_32_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '59' 'IN' port 'w2_rsc_32_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_32_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '59' 'IN' port 'w2_rsc_32_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '59' 'IN' port 'w2_rsc_32_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_32_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '59' 'IN' port 'w2_rsc_32_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '59' 'IN' port 'w2_rsc_32_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_32_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '59' 'IN' port 'w2_rsc_32_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '59' 'IN' port 'w2_rsc_32_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_32_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '59' 'IN' port 'w2_rsc_32_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '60' 'IN' port 'w2_rsc_33_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_33_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '60' 'IN' port 'w2_rsc_33_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '60' 'IN' port 'w2_rsc_33_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_33_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '60' 'IN' port 'w2_rsc_33_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '60' 'IN' port 'w2_rsc_33_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_33_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '60' 'IN' port 'w2_rsc_33_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '60' 'IN' port 'w2_rsc_33_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_33_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '60' 'IN' port 'w2_rsc_33_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '60' 'IN' port 'w2_rsc_33_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_33_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '60' 'IN' port 'w2_rsc_33_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '60' 'IN' port 'w2_rsc_33_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_33_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '60' 'IN' port 'w2_rsc_33_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '60' 'IN' port 'w2_rsc_33_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_33_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '60' 'IN' port 'w2_rsc_33_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '61' 'IN' port 'w2_rsc_34_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_34_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '61' 'IN' port 'w2_rsc_34_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '61' 'IN' port 'w2_rsc_34_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_34_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '61' 'IN' port 'w2_rsc_34_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '61' 'IN' port 'w2_rsc_34_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_34_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '61' 'IN' port 'w2_rsc_34_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '61' 'IN' port 'w2_rsc_34_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_34_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '61' 'IN' port 'w2_rsc_34_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '61' 'IN' port 'w2_rsc_34_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_34_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '61' 'IN' port 'w2_rsc_34_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '61' 'IN' port 'w2_rsc_34_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_34_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '61' 'IN' port 'w2_rsc_34_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '61' 'IN' port 'w2_rsc_34_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_34_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '61' 'IN' port 'w2_rsc_34_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '62' 'IN' port 'w2_rsc_35_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_35_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '62' 'IN' port 'w2_rsc_35_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '62' 'IN' port 'w2_rsc_35_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_35_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '62' 'IN' port 'w2_rsc_35_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '62' 'IN' port 'w2_rsc_35_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_35_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '62' 'IN' port 'w2_rsc_35_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '62' 'IN' port 'w2_rsc_35_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_35_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '62' 'IN' port 'w2_rsc_35_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '62' 'IN' port 'w2_rsc_35_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_35_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '62' 'IN' port 'w2_rsc_35_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '62' 'IN' port 'w2_rsc_35_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_35_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '62' 'IN' port 'w2_rsc_35_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '62' 'IN' port 'w2_rsc_35_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_35_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '62' 'IN' port 'w2_rsc_35_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '63' 'IN' port 'w2_rsc_36_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_36_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '63' 'IN' port 'w2_rsc_36_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '63' 'IN' port 'w2_rsc_36_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_36_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '63' 'IN' port 'w2_rsc_36_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '63' 'IN' port 'w2_rsc_36_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_36_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '63' 'IN' port 'w2_rsc_36_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '63' 'IN' port 'w2_rsc_36_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_36_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '63' 'IN' port 'w2_rsc_36_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '63' 'IN' port 'w2_rsc_36_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_36_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '63' 'IN' port 'w2_rsc_36_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '63' 'IN' port 'w2_rsc_36_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_36_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '63' 'IN' port 'w2_rsc_36_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '63' 'IN' port 'w2_rsc_36_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_36_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '63' 'IN' port 'w2_rsc_36_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '64' 'IN' port 'w2_rsc_37_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_37_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '64' 'IN' port 'w2_rsc_37_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '64' 'IN' port 'w2_rsc_37_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_37_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '64' 'IN' port 'w2_rsc_37_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '64' 'IN' port 'w2_rsc_37_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_37_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '64' 'IN' port 'w2_rsc_37_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '64' 'IN' port 'w2_rsc_37_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_37_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '64' 'IN' port 'w2_rsc_37_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '64' 'IN' port 'w2_rsc_37_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_37_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '64' 'IN' port 'w2_rsc_37_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '64' 'IN' port 'w2_rsc_37_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_37_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '64' 'IN' port 'w2_rsc_37_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '64' 'IN' port 'w2_rsc_37_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_37_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '64' 'IN' port 'w2_rsc_37_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '65' 'IN' port 'w2_rsc_38_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_38_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '65' 'IN' port 'w2_rsc_38_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '65' 'IN' port 'w2_rsc_38_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_38_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '65' 'IN' port 'w2_rsc_38_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '65' 'IN' port 'w2_rsc_38_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_38_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '65' 'IN' port 'w2_rsc_38_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '65' 'IN' port 'w2_rsc_38_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_38_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '65' 'IN' port 'w2_rsc_38_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '65' 'IN' port 'w2_rsc_38_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_38_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '65' 'IN' port 'w2_rsc_38_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '65' 'IN' port 'w2_rsc_38_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_38_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '65' 'IN' port 'w2_rsc_38_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '65' 'IN' port 'w2_rsc_38_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_38_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '65' 'IN' port 'w2_rsc_38_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '66' 'IN' port 'w2_rsc_39_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_39_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '66' 'IN' port 'w2_rsc_39_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '66' 'IN' port 'w2_rsc_39_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_39_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '66' 'IN' port 'w2_rsc_39_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '66' 'IN' port 'w2_rsc_39_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_39_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '66' 'IN' port 'w2_rsc_39_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '66' 'IN' port 'w2_rsc_39_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_39_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '66' 'IN' port 'w2_rsc_39_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '66' 'IN' port 'w2_rsc_39_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_39_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '66' 'IN' port 'w2_rsc_39_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '66' 'IN' port 'w2_rsc_39_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_39_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '66' 'IN' port 'w2_rsc_39_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '66' 'IN' port 'w2_rsc_39_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_39_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '66' 'IN' port 'w2_rsc_39_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '67' 'IN' port 'w2_rsc_40_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_40_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '67' 'IN' port 'w2_rsc_40_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '67' 'IN' port 'w2_rsc_40_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_40_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '67' 'IN' port 'w2_rsc_40_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '67' 'IN' port 'w2_rsc_40_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_40_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '67' 'IN' port 'w2_rsc_40_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '67' 'IN' port 'w2_rsc_40_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_40_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '67' 'IN' port 'w2_rsc_40_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '67' 'IN' port 'w2_rsc_40_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_40_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '67' 'IN' port 'w2_rsc_40_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '67' 'IN' port 'w2_rsc_40_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_40_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '67' 'IN' port 'w2_rsc_40_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '67' 'IN' port 'w2_rsc_40_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_40_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '67' 'IN' port 'w2_rsc_40_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '68' 'IN' port 'w2_rsc_41_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_41_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '68' 'IN' port 'w2_rsc_41_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '68' 'IN' port 'w2_rsc_41_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_41_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '68' 'IN' port 'w2_rsc_41_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '68' 'IN' port 'w2_rsc_41_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_41_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '68' 'IN' port 'w2_rsc_41_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '68' 'IN' port 'w2_rsc_41_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_41_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '68' 'IN' port 'w2_rsc_41_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '68' 'IN' port 'w2_rsc_41_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_41_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '68' 'IN' port 'w2_rsc_41_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '68' 'IN' port 'w2_rsc_41_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_41_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '68' 'IN' port 'w2_rsc_41_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '68' 'IN' port 'w2_rsc_41_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_41_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '68' 'IN' port 'w2_rsc_41_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '69' 'IN' port 'w2_rsc_42_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_42_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '69' 'IN' port 'w2_rsc_42_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '69' 'IN' port 'w2_rsc_42_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_42_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '69' 'IN' port 'w2_rsc_42_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '69' 'IN' port 'w2_rsc_42_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_42_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '69' 'IN' port 'w2_rsc_42_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '69' 'IN' port 'w2_rsc_42_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_42_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '69' 'IN' port 'w2_rsc_42_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '69' 'IN' port 'w2_rsc_42_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_42_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '69' 'IN' port 'w2_rsc_42_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '69' 'IN' port 'w2_rsc_42_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_42_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '69' 'IN' port 'w2_rsc_42_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '69' 'IN' port 'w2_rsc_42_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_42_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '69' 'IN' port 'w2_rsc_42_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '70' 'IN' port 'w2_rsc_43_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_43_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '70' 'IN' port 'w2_rsc_43_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '70' 'IN' port 'w2_rsc_43_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_43_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '70' 'IN' port 'w2_rsc_43_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '70' 'IN' port 'w2_rsc_43_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_43_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '70' 'IN' port 'w2_rsc_43_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '70' 'IN' port 'w2_rsc_43_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_43_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '70' 'IN' port 'w2_rsc_43_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '70' 'IN' port 'w2_rsc_43_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_43_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '70' 'IN' port 'w2_rsc_43_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '70' 'IN' port 'w2_rsc_43_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_43_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '70' 'IN' port 'w2_rsc_43_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '70' 'IN' port 'w2_rsc_43_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_43_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '70' 'IN' port 'w2_rsc_43_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '71' 'IN' port 'w2_rsc_44_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_44_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '71' 'IN' port 'w2_rsc_44_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '71' 'IN' port 'w2_rsc_44_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_44_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '71' 'IN' port 'w2_rsc_44_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '71' 'IN' port 'w2_rsc_44_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_44_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '71' 'IN' port 'w2_rsc_44_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '71' 'IN' port 'w2_rsc_44_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_44_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '71' 'IN' port 'w2_rsc_44_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '71' 'IN' port 'w2_rsc_44_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_44_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '71' 'IN' port 'w2_rsc_44_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '71' 'IN' port 'w2_rsc_44_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_44_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '71' 'IN' port 'w2_rsc_44_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '71' 'IN' port 'w2_rsc_44_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_44_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '71' 'IN' port 'w2_rsc_44_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '72' 'IN' port 'w2_rsc_45_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_45_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '72' 'IN' port 'w2_rsc_45_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '72' 'IN' port 'w2_rsc_45_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_45_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '72' 'IN' port 'w2_rsc_45_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '72' 'IN' port 'w2_rsc_45_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_45_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '72' 'IN' port 'w2_rsc_45_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '72' 'IN' port 'w2_rsc_45_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_45_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '72' 'IN' port 'w2_rsc_45_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '72' 'IN' port 'w2_rsc_45_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_45_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '72' 'IN' port 'w2_rsc_45_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '72' 'IN' port 'w2_rsc_45_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_45_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '72' 'IN' port 'w2_rsc_45_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '72' 'IN' port 'w2_rsc_45_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_45_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '72' 'IN' port 'w2_rsc_45_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '73' 'IN' port 'w2_rsc_46_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_46_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '73' 'IN' port 'w2_rsc_46_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '73' 'IN' port 'w2_rsc_46_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_46_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '73' 'IN' port 'w2_rsc_46_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '73' 'IN' port 'w2_rsc_46_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_46_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '73' 'IN' port 'w2_rsc_46_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '73' 'IN' port 'w2_rsc_46_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_46_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '73' 'IN' port 'w2_rsc_46_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '73' 'IN' port 'w2_rsc_46_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_46_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '73' 'IN' port 'w2_rsc_46_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '73' 'IN' port 'w2_rsc_46_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_46_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '73' 'IN' port 'w2_rsc_46_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '73' 'IN' port 'w2_rsc_46_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_46_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '73' 'IN' port 'w2_rsc_46_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '74' 'IN' port 'w2_rsc_47_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_47_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '74' 'IN' port 'w2_rsc_47_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '74' 'IN' port 'w2_rsc_47_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_47_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '74' 'IN' port 'w2_rsc_47_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '74' 'IN' port 'w2_rsc_47_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_47_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '74' 'IN' port 'w2_rsc_47_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '74' 'IN' port 'w2_rsc_47_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_47_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '74' 'IN' port 'w2_rsc_47_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '74' 'IN' port 'w2_rsc_47_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_47_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '74' 'IN' port 'w2_rsc_47_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '74' 'IN' port 'w2_rsc_47_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_47_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '74' 'IN' port 'w2_rsc_47_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '74' 'IN' port 'w2_rsc_47_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_47_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '74' 'IN' port 'w2_rsc_47_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '75' 'IN' port 'w2_rsc_48_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_48_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '75' 'IN' port 'w2_rsc_48_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '75' 'IN' port 'w2_rsc_48_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_48_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '75' 'IN' port 'w2_rsc_48_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '75' 'IN' port 'w2_rsc_48_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_48_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '75' 'IN' port 'w2_rsc_48_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '75' 'IN' port 'w2_rsc_48_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_48_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '75' 'IN' port 'w2_rsc_48_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '75' 'IN' port 'w2_rsc_48_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_48_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '75' 'IN' port 'w2_rsc_48_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '75' 'IN' port 'w2_rsc_48_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_48_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '75' 'IN' port 'w2_rsc_48_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '75' 'IN' port 'w2_rsc_48_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_48_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '75' 'IN' port 'w2_rsc_48_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '76' 'IN' port 'w2_rsc_49_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_49_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '76' 'IN' port 'w2_rsc_49_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '76' 'IN' port 'w2_rsc_49_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_49_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '76' 'IN' port 'w2_rsc_49_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '76' 'IN' port 'w2_rsc_49_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_49_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '76' 'IN' port 'w2_rsc_49_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '76' 'IN' port 'w2_rsc_49_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_49_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '76' 'IN' port 'w2_rsc_49_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '76' 'IN' port 'w2_rsc_49_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_49_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '76' 'IN' port 'w2_rsc_49_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '76' 'IN' port 'w2_rsc_49_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_49_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '76' 'IN' port 'w2_rsc_49_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '76' 'IN' port 'w2_rsc_49_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_49_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '76' 'IN' port 'w2_rsc_49_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '77' 'IN' port 'w2_rsc_50_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_50_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '77' 'IN' port 'w2_rsc_50_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '77' 'IN' port 'w2_rsc_50_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_50_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '77' 'IN' port 'w2_rsc_50_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '77' 'IN' port 'w2_rsc_50_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_50_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '77' 'IN' port 'w2_rsc_50_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '77' 'IN' port 'w2_rsc_50_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_50_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '77' 'IN' port 'w2_rsc_50_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '77' 'IN' port 'w2_rsc_50_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_50_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '77' 'IN' port 'w2_rsc_50_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '77' 'IN' port 'w2_rsc_50_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_50_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '77' 'IN' port 'w2_rsc_50_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '77' 'IN' port 'w2_rsc_50_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_50_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '77' 'IN' port 'w2_rsc_50_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '78' 'IN' port 'w2_rsc_51_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_51_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '78' 'IN' port 'w2_rsc_51_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '78' 'IN' port 'w2_rsc_51_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_51_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '78' 'IN' port 'w2_rsc_51_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '78' 'IN' port 'w2_rsc_51_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_51_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '78' 'IN' port 'w2_rsc_51_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '78' 'IN' port 'w2_rsc_51_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_51_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '78' 'IN' port 'w2_rsc_51_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '78' 'IN' port 'w2_rsc_51_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_51_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '78' 'IN' port 'w2_rsc_51_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '78' 'IN' port 'w2_rsc_51_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_51_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '78' 'IN' port 'w2_rsc_51_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '78' 'IN' port 'w2_rsc_51_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_51_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '78' 'IN' port 'w2_rsc_51_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '79' 'IN' port 'w2_rsc_52_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_52_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '79' 'IN' port 'w2_rsc_52_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '79' 'IN' port 'w2_rsc_52_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_52_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '79' 'IN' port 'w2_rsc_52_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '79' 'IN' port 'w2_rsc_52_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_52_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '79' 'IN' port 'w2_rsc_52_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '79' 'IN' port 'w2_rsc_52_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_52_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '79' 'IN' port 'w2_rsc_52_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '79' 'IN' port 'w2_rsc_52_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_52_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '79' 'IN' port 'w2_rsc_52_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '79' 'IN' port 'w2_rsc_52_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_52_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '79' 'IN' port 'w2_rsc_52_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '79' 'IN' port 'w2_rsc_52_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_52_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '79' 'IN' port 'w2_rsc_52_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '80' 'IN' port 'w2_rsc_53_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_53_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '80' 'IN' port 'w2_rsc_53_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '80' 'IN' port 'w2_rsc_53_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_53_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '80' 'IN' port 'w2_rsc_53_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '80' 'IN' port 'w2_rsc_53_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_53_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '80' 'IN' port 'w2_rsc_53_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '80' 'IN' port 'w2_rsc_53_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_53_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '80' 'IN' port 'w2_rsc_53_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '80' 'IN' port 'w2_rsc_53_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_53_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '80' 'IN' port 'w2_rsc_53_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '80' 'IN' port 'w2_rsc_53_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_53_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '80' 'IN' port 'w2_rsc_53_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '80' 'IN' port 'w2_rsc_53_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_53_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '80' 'IN' port 'w2_rsc_53_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '81' 'IN' port 'w2_rsc_54_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_54_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '81' 'IN' port 'w2_rsc_54_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '81' 'IN' port 'w2_rsc_54_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_54_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '81' 'IN' port 'w2_rsc_54_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '81' 'IN' port 'w2_rsc_54_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_54_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '81' 'IN' port 'w2_rsc_54_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '81' 'IN' port 'w2_rsc_54_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_54_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '81' 'IN' port 'w2_rsc_54_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '81' 'IN' port 'w2_rsc_54_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_54_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '81' 'IN' port 'w2_rsc_54_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '81' 'IN' port 'w2_rsc_54_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_54_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '81' 'IN' port 'w2_rsc_54_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '81' 'IN' port 'w2_rsc_54_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_54_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '81' 'IN' port 'w2_rsc_54_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '82' 'IN' port 'w2_rsc_55_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_55_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '82' 'IN' port 'w2_rsc_55_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '82' 'IN' port 'w2_rsc_55_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_55_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '82' 'IN' port 'w2_rsc_55_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '82' 'IN' port 'w2_rsc_55_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_55_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '82' 'IN' port 'w2_rsc_55_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '82' 'IN' port 'w2_rsc_55_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_55_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '82' 'IN' port 'w2_rsc_55_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '82' 'IN' port 'w2_rsc_55_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_55_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '82' 'IN' port 'w2_rsc_55_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '82' 'IN' port 'w2_rsc_55_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_55_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '82' 'IN' port 'w2_rsc_55_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '82' 'IN' port 'w2_rsc_55_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_55_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '82' 'IN' port 'w2_rsc_55_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '83' 'IN' port 'w2_rsc_56_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_56_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '83' 'IN' port 'w2_rsc_56_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '83' 'IN' port 'w2_rsc_56_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_56_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '83' 'IN' port 'w2_rsc_56_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '83' 'IN' port 'w2_rsc_56_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_56_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '83' 'IN' port 'w2_rsc_56_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '83' 'IN' port 'w2_rsc_56_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_56_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '83' 'IN' port 'w2_rsc_56_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '83' 'IN' port 'w2_rsc_56_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_56_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '83' 'IN' port 'w2_rsc_56_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '83' 'IN' port 'w2_rsc_56_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_56_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '83' 'IN' port 'w2_rsc_56_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '83' 'IN' port 'w2_rsc_56_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_56_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '83' 'IN' port 'w2_rsc_56_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '84' 'IN' port 'w2_rsc_57_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_57_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '84' 'IN' port 'w2_rsc_57_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '84' 'IN' port 'w2_rsc_57_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_57_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '84' 'IN' port 'w2_rsc_57_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '84' 'IN' port 'w2_rsc_57_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_57_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '84' 'IN' port 'w2_rsc_57_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '84' 'IN' port 'w2_rsc_57_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_57_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '84' 'IN' port 'w2_rsc_57_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '84' 'IN' port 'w2_rsc_57_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_57_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '84' 'IN' port 'w2_rsc_57_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '84' 'IN' port 'w2_rsc_57_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_57_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '84' 'IN' port 'w2_rsc_57_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '84' 'IN' port 'w2_rsc_57_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_57_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '84' 'IN' port 'w2_rsc_57_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '85' 'IN' port 'w2_rsc_58_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_58_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '85' 'IN' port 'w2_rsc_58_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '85' 'IN' port 'w2_rsc_58_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_58_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '85' 'IN' port 'w2_rsc_58_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '85' 'IN' port 'w2_rsc_58_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_58_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '85' 'IN' port 'w2_rsc_58_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '85' 'IN' port 'w2_rsc_58_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_58_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '85' 'IN' port 'w2_rsc_58_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '85' 'IN' port 'w2_rsc_58_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_58_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '85' 'IN' port 'w2_rsc_58_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '85' 'IN' port 'w2_rsc_58_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_58_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '85' 'IN' port 'w2_rsc_58_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '85' 'IN' port 'w2_rsc_58_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_58_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '85' 'IN' port 'w2_rsc_58_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '86' 'IN' port 'w2_rsc_59_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_59_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '86' 'IN' port 'w2_rsc_59_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '86' 'IN' port 'w2_rsc_59_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_59_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '86' 'IN' port 'w2_rsc_59_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '86' 'IN' port 'w2_rsc_59_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_59_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '86' 'IN' port 'w2_rsc_59_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '86' 'IN' port 'w2_rsc_59_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_59_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '86' 'IN' port 'w2_rsc_59_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '86' 'IN' port 'w2_rsc_59_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_59_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '86' 'IN' port 'w2_rsc_59_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '86' 'IN' port 'w2_rsc_59_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_59_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '86' 'IN' port 'w2_rsc_59_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '86' 'IN' port 'w2_rsc_59_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_59_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '86' 'IN' port 'w2_rsc_59_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '87' 'IN' port 'w2_rsc_60_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_60_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '87' 'IN' port 'w2_rsc_60_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '87' 'IN' port 'w2_rsc_60_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_60_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '87' 'IN' port 'w2_rsc_60_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '87' 'IN' port 'w2_rsc_60_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_60_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '87' 'IN' port 'w2_rsc_60_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '87' 'IN' port 'w2_rsc_60_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_60_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '87' 'IN' port 'w2_rsc_60_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '87' 'IN' port 'w2_rsc_60_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_60_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '87' 'IN' port 'w2_rsc_60_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '87' 'IN' port 'w2_rsc_60_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_60_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '87' 'IN' port 'w2_rsc_60_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '87' 'IN' port 'w2_rsc_60_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_60_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '87' 'IN' port 'w2_rsc_60_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '88' 'IN' port 'w2_rsc_61_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_61_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '88' 'IN' port 'w2_rsc_61_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '88' 'IN' port 'w2_rsc_61_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_61_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '88' 'IN' port 'w2_rsc_61_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '88' 'IN' port 'w2_rsc_61_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_61_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '88' 'IN' port 'w2_rsc_61_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '88' 'IN' port 'w2_rsc_61_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_61_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '88' 'IN' port 'w2_rsc_61_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '88' 'IN' port 'w2_rsc_61_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_61_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '88' 'IN' port 'w2_rsc_61_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '88' 'IN' port 'w2_rsc_61_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_61_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '88' 'IN' port 'w2_rsc_61_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '88' 'IN' port 'w2_rsc_61_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_61_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '88' 'IN' port 'w2_rsc_61_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '89' 'IN' port 'w2_rsc_62_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_62_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '89' 'IN' port 'w2_rsc_62_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '89' 'IN' port 'w2_rsc_62_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_62_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '89' 'IN' port 'w2_rsc_62_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '89' 'IN' port 'w2_rsc_62_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_62_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '89' 'IN' port 'w2_rsc_62_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '89' 'IN' port 'w2_rsc_62_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_62_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '89' 'IN' port 'w2_rsc_62_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '89' 'IN' port 'w2_rsc_62_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_62_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '89' 'IN' port 'w2_rsc_62_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '89' 'IN' port 'w2_rsc_62_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_62_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '89' 'IN' port 'w2_rsc_62_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '89' 'IN' port 'w2_rsc_62_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_62_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '89' 'IN' port 'w2_rsc_62_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '90' 'IN' port 'w2_rsc_63_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_63_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '90' 'IN' port 'w2_rsc_63_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '90' 'IN' port 'w2_rsc_63_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_63_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '90' 'IN' port 'w2_rsc_63_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '90' 'IN' port 'w2_rsc_63_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_63_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '90' 'IN' port 'w2_rsc_63_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '90' 'IN' port 'w2_rsc_63_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_63_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '90' 'IN' port 'w2_rsc_63_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '90' 'IN' port 'w2_rsc_63_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_63_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '90' 'IN' port 'w2_rsc_63_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '90' 'IN' port 'w2_rsc_63_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_63_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '90' 'IN' port 'w2_rsc_63_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '90' 'IN' port 'w2_rsc_63_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w2_rsc_63_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '90' 'IN' port 'w2_rsc_63_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/b2_rsc_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '6' 'IN' port 'b2_rsc_dat' '2' 'OUT' port 'output1_rsc_dat'"

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

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '91' 'IN' port 'w4_rsc_0_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_0_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '91' 'IN' port 'w4_rsc_0_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '91' 'IN' port 'w4_rsc_0_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_0_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '91' 'IN' port 'w4_rsc_0_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '91' 'IN' port 'w4_rsc_0_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_0_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '91' 'IN' port 'w4_rsc_0_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '91' 'IN' port 'w4_rsc_0_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_0_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '91' 'IN' port 'w4_rsc_0_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '91' 'IN' port 'w4_rsc_0_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_0_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '91' 'IN' port 'w4_rsc_0_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '91' 'IN' port 'w4_rsc_0_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_0_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '91' 'IN' port 'w4_rsc_0_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '91' 'IN' port 'w4_rsc_0_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_0_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '91' 'IN' port 'w4_rsc_0_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '92' 'IN' port 'w4_rsc_1_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_1_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '92' 'IN' port 'w4_rsc_1_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '92' 'IN' port 'w4_rsc_1_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_1_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '92' 'IN' port 'w4_rsc_1_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '92' 'IN' port 'w4_rsc_1_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_1_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '92' 'IN' port 'w4_rsc_1_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '92' 'IN' port 'w4_rsc_1_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_1_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '92' 'IN' port 'w4_rsc_1_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '92' 'IN' port 'w4_rsc_1_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_1_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '92' 'IN' port 'w4_rsc_1_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '92' 'IN' port 'w4_rsc_1_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_1_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '92' 'IN' port 'w4_rsc_1_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '92' 'IN' port 'w4_rsc_1_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_1_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '92' 'IN' port 'w4_rsc_1_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '93' 'IN' port 'w4_rsc_2_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_2_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '93' 'IN' port 'w4_rsc_2_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '93' 'IN' port 'w4_rsc_2_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_2_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '93' 'IN' port 'w4_rsc_2_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '93' 'IN' port 'w4_rsc_2_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_2_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '93' 'IN' port 'w4_rsc_2_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '93' 'IN' port 'w4_rsc_2_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_2_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '93' 'IN' port 'w4_rsc_2_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '93' 'IN' port 'w4_rsc_2_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_2_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '93' 'IN' port 'w4_rsc_2_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '93' 'IN' port 'w4_rsc_2_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_2_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '93' 'IN' port 'w4_rsc_2_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '93' 'IN' port 'w4_rsc_2_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_2_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '93' 'IN' port 'w4_rsc_2_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '94' 'IN' port 'w4_rsc_3_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_3_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '94' 'IN' port 'w4_rsc_3_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '94' 'IN' port 'w4_rsc_3_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_3_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '94' 'IN' port 'w4_rsc_3_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '94' 'IN' port 'w4_rsc_3_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_3_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '94' 'IN' port 'w4_rsc_3_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '94' 'IN' port 'w4_rsc_3_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_3_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '94' 'IN' port 'w4_rsc_3_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '94' 'IN' port 'w4_rsc_3_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_3_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '94' 'IN' port 'w4_rsc_3_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '94' 'IN' port 'w4_rsc_3_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_3_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '94' 'IN' port 'w4_rsc_3_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '94' 'IN' port 'w4_rsc_3_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_3_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '94' 'IN' port 'w4_rsc_3_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '95' 'IN' port 'w4_rsc_4_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_4_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '95' 'IN' port 'w4_rsc_4_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '95' 'IN' port 'w4_rsc_4_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_4_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '95' 'IN' port 'w4_rsc_4_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '95' 'IN' port 'w4_rsc_4_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_4_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '95' 'IN' port 'w4_rsc_4_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '95' 'IN' port 'w4_rsc_4_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_4_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '95' 'IN' port 'w4_rsc_4_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '95' 'IN' port 'w4_rsc_4_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_4_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '95' 'IN' port 'w4_rsc_4_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '95' 'IN' port 'w4_rsc_4_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_4_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '95' 'IN' port 'w4_rsc_4_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '95' 'IN' port 'w4_rsc_4_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_4_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '95' 'IN' port 'w4_rsc_4_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '96' 'IN' port 'w4_rsc_5_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_5_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '96' 'IN' port 'w4_rsc_5_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '96' 'IN' port 'w4_rsc_5_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_5_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '96' 'IN' port 'w4_rsc_5_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '96' 'IN' port 'w4_rsc_5_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_5_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '96' 'IN' port 'w4_rsc_5_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '96' 'IN' port 'w4_rsc_5_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_5_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '96' 'IN' port 'w4_rsc_5_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '96' 'IN' port 'w4_rsc_5_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_5_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '96' 'IN' port 'w4_rsc_5_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '96' 'IN' port 'w4_rsc_5_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_5_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '96' 'IN' port 'w4_rsc_5_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '96' 'IN' port 'w4_rsc_5_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_5_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '96' 'IN' port 'w4_rsc_5_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '97' 'IN' port 'w4_rsc_6_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_6_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '97' 'IN' port 'w4_rsc_6_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '97' 'IN' port 'w4_rsc_6_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_6_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '97' 'IN' port 'w4_rsc_6_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '97' 'IN' port 'w4_rsc_6_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_6_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '97' 'IN' port 'w4_rsc_6_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '97' 'IN' port 'w4_rsc_6_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_6_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '97' 'IN' port 'w4_rsc_6_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '97' 'IN' port 'w4_rsc_6_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_6_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '97' 'IN' port 'w4_rsc_6_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '97' 'IN' port 'w4_rsc_6_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_6_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '97' 'IN' port 'w4_rsc_6_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '97' 'IN' port 'w4_rsc_6_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_6_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '97' 'IN' port 'w4_rsc_6_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '98' 'IN' port 'w4_rsc_7_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_7_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '98' 'IN' port 'w4_rsc_7_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '98' 'IN' port 'w4_rsc_7_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_7_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '98' 'IN' port 'w4_rsc_7_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '98' 'IN' port 'w4_rsc_7_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_7_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '98' 'IN' port 'w4_rsc_7_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '98' 'IN' port 'w4_rsc_7_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_7_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '98' 'IN' port 'w4_rsc_7_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '98' 'IN' port 'w4_rsc_7_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_7_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '98' 'IN' port 'w4_rsc_7_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '98' 'IN' port 'w4_rsc_7_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_7_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '98' 'IN' port 'w4_rsc_7_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '98' 'IN' port 'w4_rsc_7_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_7_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '98' 'IN' port 'w4_rsc_7_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '99' 'IN' port 'w4_rsc_8_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_8_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '99' 'IN' port 'w4_rsc_8_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '99' 'IN' port 'w4_rsc_8_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_8_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '99' 'IN' port 'w4_rsc_8_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '99' 'IN' port 'w4_rsc_8_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_8_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '99' 'IN' port 'w4_rsc_8_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '99' 'IN' port 'w4_rsc_8_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_8_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '99' 'IN' port 'w4_rsc_8_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '99' 'IN' port 'w4_rsc_8_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_8_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '99' 'IN' port 'w4_rsc_8_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '99' 'IN' port 'w4_rsc_8_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_8_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '99' 'IN' port 'w4_rsc_8_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '99' 'IN' port 'w4_rsc_8_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_8_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '99' 'IN' port 'w4_rsc_8_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '100' 'IN' port 'w4_rsc_9_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_9_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '100' 'IN' port 'w4_rsc_9_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '100' 'IN' port 'w4_rsc_9_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_9_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '100' 'IN' port 'w4_rsc_9_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '100' 'IN' port 'w4_rsc_9_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_9_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '100' 'IN' port 'w4_rsc_9_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '100' 'IN' port 'w4_rsc_9_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_9_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '100' 'IN' port 'w4_rsc_9_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '100' 'IN' port 'w4_rsc_9_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_9_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '100' 'IN' port 'w4_rsc_9_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '100' 'IN' port 'w4_rsc_9_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_9_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '100' 'IN' port 'w4_rsc_9_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '100' 'IN' port 'w4_rsc_9_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_9_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '100' 'IN' port 'w4_rsc_9_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '101' 'IN' port 'w4_rsc_10_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_10_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '101' 'IN' port 'w4_rsc_10_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '101' 'IN' port 'w4_rsc_10_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_10_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '101' 'IN' port 'w4_rsc_10_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '101' 'IN' port 'w4_rsc_10_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_10_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '101' 'IN' port 'w4_rsc_10_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '101' 'IN' port 'w4_rsc_10_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_10_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '101' 'IN' port 'w4_rsc_10_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '101' 'IN' port 'w4_rsc_10_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_10_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '101' 'IN' port 'w4_rsc_10_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '101' 'IN' port 'w4_rsc_10_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_10_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '101' 'IN' port 'w4_rsc_10_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '101' 'IN' port 'w4_rsc_10_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_10_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '101' 'IN' port 'w4_rsc_10_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '102' 'IN' port 'w4_rsc_11_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_11_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '102' 'IN' port 'w4_rsc_11_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '102' 'IN' port 'w4_rsc_11_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_11_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '102' 'IN' port 'w4_rsc_11_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '102' 'IN' port 'w4_rsc_11_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_11_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '102' 'IN' port 'w4_rsc_11_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '102' 'IN' port 'w4_rsc_11_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_11_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '102' 'IN' port 'w4_rsc_11_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '102' 'IN' port 'w4_rsc_11_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_11_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '102' 'IN' port 'w4_rsc_11_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '102' 'IN' port 'w4_rsc_11_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_11_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '102' 'IN' port 'w4_rsc_11_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '102' 'IN' port 'w4_rsc_11_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_11_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '102' 'IN' port 'w4_rsc_11_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '103' 'IN' port 'w4_rsc_12_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_12_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '103' 'IN' port 'w4_rsc_12_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '103' 'IN' port 'w4_rsc_12_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_12_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '103' 'IN' port 'w4_rsc_12_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '103' 'IN' port 'w4_rsc_12_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_12_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '103' 'IN' port 'w4_rsc_12_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '103' 'IN' port 'w4_rsc_12_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_12_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '103' 'IN' port 'w4_rsc_12_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '103' 'IN' port 'w4_rsc_12_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_12_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '103' 'IN' port 'w4_rsc_12_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '103' 'IN' port 'w4_rsc_12_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_12_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '103' 'IN' port 'w4_rsc_12_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '103' 'IN' port 'w4_rsc_12_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_12_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '103' 'IN' port 'w4_rsc_12_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '104' 'IN' port 'w4_rsc_13_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_13_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '104' 'IN' port 'w4_rsc_13_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '104' 'IN' port 'w4_rsc_13_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_13_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '104' 'IN' port 'w4_rsc_13_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '104' 'IN' port 'w4_rsc_13_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_13_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '104' 'IN' port 'w4_rsc_13_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '104' 'IN' port 'w4_rsc_13_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_13_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '104' 'IN' port 'w4_rsc_13_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '104' 'IN' port 'w4_rsc_13_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_13_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '104' 'IN' port 'w4_rsc_13_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '104' 'IN' port 'w4_rsc_13_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_13_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '104' 'IN' port 'w4_rsc_13_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '104' 'IN' port 'w4_rsc_13_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_13_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '104' 'IN' port 'w4_rsc_13_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '105' 'IN' port 'w4_rsc_14_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_14_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '105' 'IN' port 'w4_rsc_14_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '105' 'IN' port 'w4_rsc_14_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_14_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '105' 'IN' port 'w4_rsc_14_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '105' 'IN' port 'w4_rsc_14_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_14_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '105' 'IN' port 'w4_rsc_14_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '105' 'IN' port 'w4_rsc_14_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_14_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '105' 'IN' port 'w4_rsc_14_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '105' 'IN' port 'w4_rsc_14_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_14_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '105' 'IN' port 'w4_rsc_14_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '105' 'IN' port 'w4_rsc_14_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_14_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '105' 'IN' port 'w4_rsc_14_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '105' 'IN' port 'w4_rsc_14_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_14_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '105' 'IN' port 'w4_rsc_14_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '106' 'IN' port 'w4_rsc_15_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_15_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '106' 'IN' port 'w4_rsc_15_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '106' 'IN' port 'w4_rsc_15_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_15_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '106' 'IN' port 'w4_rsc_15_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '106' 'IN' port 'w4_rsc_15_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_15_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '106' 'IN' port 'w4_rsc_15_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '106' 'IN' port 'w4_rsc_15_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_15_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '106' 'IN' port 'w4_rsc_15_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '106' 'IN' port 'w4_rsc_15_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_15_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '106' 'IN' port 'w4_rsc_15_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '106' 'IN' port 'w4_rsc_15_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_15_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '106' 'IN' port 'w4_rsc_15_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '106' 'IN' port 'w4_rsc_15_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_15_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '106' 'IN' port 'w4_rsc_15_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '107' 'IN' port 'w4_rsc_16_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_16_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '107' 'IN' port 'w4_rsc_16_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '107' 'IN' port 'w4_rsc_16_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_16_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '107' 'IN' port 'w4_rsc_16_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '107' 'IN' port 'w4_rsc_16_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_16_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '107' 'IN' port 'w4_rsc_16_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '107' 'IN' port 'w4_rsc_16_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_16_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '107' 'IN' port 'w4_rsc_16_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '107' 'IN' port 'w4_rsc_16_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_16_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '107' 'IN' port 'w4_rsc_16_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '107' 'IN' port 'w4_rsc_16_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_16_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '107' 'IN' port 'w4_rsc_16_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '107' 'IN' port 'w4_rsc_16_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_16_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '107' 'IN' port 'w4_rsc_16_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '108' 'IN' port 'w4_rsc_17_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_17_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '108' 'IN' port 'w4_rsc_17_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '108' 'IN' port 'w4_rsc_17_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_17_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '108' 'IN' port 'w4_rsc_17_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '108' 'IN' port 'w4_rsc_17_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_17_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '108' 'IN' port 'w4_rsc_17_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '108' 'IN' port 'w4_rsc_17_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_17_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '108' 'IN' port 'w4_rsc_17_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '108' 'IN' port 'w4_rsc_17_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_17_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '108' 'IN' port 'w4_rsc_17_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '108' 'IN' port 'w4_rsc_17_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_17_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '108' 'IN' port 'w4_rsc_17_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '108' 'IN' port 'w4_rsc_17_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_17_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '108' 'IN' port 'w4_rsc_17_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '109' 'IN' port 'w4_rsc_18_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_18_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '109' 'IN' port 'w4_rsc_18_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '109' 'IN' port 'w4_rsc_18_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_18_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '109' 'IN' port 'w4_rsc_18_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '109' 'IN' port 'w4_rsc_18_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_18_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '109' 'IN' port 'w4_rsc_18_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '109' 'IN' port 'w4_rsc_18_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_18_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '109' 'IN' port 'w4_rsc_18_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '109' 'IN' port 'w4_rsc_18_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_18_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '109' 'IN' port 'w4_rsc_18_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '109' 'IN' port 'w4_rsc_18_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_18_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '109' 'IN' port 'w4_rsc_18_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '109' 'IN' port 'w4_rsc_18_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_18_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '109' 'IN' port 'w4_rsc_18_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '110' 'IN' port 'w4_rsc_19_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_19_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '110' 'IN' port 'w4_rsc_19_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '110' 'IN' port 'w4_rsc_19_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_19_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '110' 'IN' port 'w4_rsc_19_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '110' 'IN' port 'w4_rsc_19_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_19_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '110' 'IN' port 'w4_rsc_19_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '110' 'IN' port 'w4_rsc_19_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_19_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '110' 'IN' port 'w4_rsc_19_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '110' 'IN' port 'w4_rsc_19_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_19_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '110' 'IN' port 'w4_rsc_19_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '110' 'IN' port 'w4_rsc_19_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_19_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '110' 'IN' port 'w4_rsc_19_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '110' 'IN' port 'w4_rsc_19_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_19_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '110' 'IN' port 'w4_rsc_19_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '111' 'IN' port 'w4_rsc_20_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_20_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '111' 'IN' port 'w4_rsc_20_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '111' 'IN' port 'w4_rsc_20_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_20_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '111' 'IN' port 'w4_rsc_20_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '111' 'IN' port 'w4_rsc_20_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_20_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '111' 'IN' port 'w4_rsc_20_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '111' 'IN' port 'w4_rsc_20_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_20_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '111' 'IN' port 'w4_rsc_20_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '111' 'IN' port 'w4_rsc_20_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_20_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '111' 'IN' port 'w4_rsc_20_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '111' 'IN' port 'w4_rsc_20_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_20_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '111' 'IN' port 'w4_rsc_20_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '111' 'IN' port 'w4_rsc_20_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_20_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '111' 'IN' port 'w4_rsc_20_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '112' 'IN' port 'w4_rsc_21_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_21_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '112' 'IN' port 'w4_rsc_21_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '112' 'IN' port 'w4_rsc_21_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_21_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '112' 'IN' port 'w4_rsc_21_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '112' 'IN' port 'w4_rsc_21_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_21_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '112' 'IN' port 'w4_rsc_21_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '112' 'IN' port 'w4_rsc_21_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_21_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '112' 'IN' port 'w4_rsc_21_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '112' 'IN' port 'w4_rsc_21_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_21_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '112' 'IN' port 'w4_rsc_21_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '112' 'IN' port 'w4_rsc_21_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_21_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '112' 'IN' port 'w4_rsc_21_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '112' 'IN' port 'w4_rsc_21_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_21_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '112' 'IN' port 'w4_rsc_21_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '113' 'IN' port 'w4_rsc_22_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_22_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '113' 'IN' port 'w4_rsc_22_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '113' 'IN' port 'w4_rsc_22_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_22_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '113' 'IN' port 'w4_rsc_22_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '113' 'IN' port 'w4_rsc_22_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_22_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '113' 'IN' port 'w4_rsc_22_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '113' 'IN' port 'w4_rsc_22_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_22_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '113' 'IN' port 'w4_rsc_22_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '113' 'IN' port 'w4_rsc_22_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_22_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '113' 'IN' port 'w4_rsc_22_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '113' 'IN' port 'w4_rsc_22_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_22_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '113' 'IN' port 'w4_rsc_22_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '113' 'IN' port 'w4_rsc_22_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_22_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '113' 'IN' port 'w4_rsc_22_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '114' 'IN' port 'w4_rsc_23_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_23_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '114' 'IN' port 'w4_rsc_23_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '114' 'IN' port 'w4_rsc_23_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_23_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '114' 'IN' port 'w4_rsc_23_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '114' 'IN' port 'w4_rsc_23_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_23_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '114' 'IN' port 'w4_rsc_23_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '114' 'IN' port 'w4_rsc_23_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_23_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '114' 'IN' port 'w4_rsc_23_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '114' 'IN' port 'w4_rsc_23_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_23_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '114' 'IN' port 'w4_rsc_23_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '114' 'IN' port 'w4_rsc_23_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_23_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '114' 'IN' port 'w4_rsc_23_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '114' 'IN' port 'w4_rsc_23_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_23_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '114' 'IN' port 'w4_rsc_23_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '115' 'IN' port 'w4_rsc_24_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_24_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '115' 'IN' port 'w4_rsc_24_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '115' 'IN' port 'w4_rsc_24_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_24_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '115' 'IN' port 'w4_rsc_24_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '115' 'IN' port 'w4_rsc_24_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_24_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '115' 'IN' port 'w4_rsc_24_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '115' 'IN' port 'w4_rsc_24_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_24_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '115' 'IN' port 'w4_rsc_24_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '115' 'IN' port 'w4_rsc_24_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_24_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '115' 'IN' port 'w4_rsc_24_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '115' 'IN' port 'w4_rsc_24_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_24_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '115' 'IN' port 'w4_rsc_24_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '115' 'IN' port 'w4_rsc_24_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_24_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '115' 'IN' port 'w4_rsc_24_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '116' 'IN' port 'w4_rsc_25_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_25_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '116' 'IN' port 'w4_rsc_25_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '116' 'IN' port 'w4_rsc_25_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_25_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '116' 'IN' port 'w4_rsc_25_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '116' 'IN' port 'w4_rsc_25_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_25_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '116' 'IN' port 'w4_rsc_25_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '116' 'IN' port 'w4_rsc_25_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_25_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '116' 'IN' port 'w4_rsc_25_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '116' 'IN' port 'w4_rsc_25_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_25_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '116' 'IN' port 'w4_rsc_25_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '116' 'IN' port 'w4_rsc_25_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_25_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '116' 'IN' port 'w4_rsc_25_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '116' 'IN' port 'w4_rsc_25_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_25_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '116' 'IN' port 'w4_rsc_25_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '117' 'IN' port 'w4_rsc_26_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_26_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '117' 'IN' port 'w4_rsc_26_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '117' 'IN' port 'w4_rsc_26_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_26_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '117' 'IN' port 'w4_rsc_26_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '117' 'IN' port 'w4_rsc_26_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_26_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '117' 'IN' port 'w4_rsc_26_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '117' 'IN' port 'w4_rsc_26_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_26_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '117' 'IN' port 'w4_rsc_26_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '117' 'IN' port 'w4_rsc_26_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_26_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '117' 'IN' port 'w4_rsc_26_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '117' 'IN' port 'w4_rsc_26_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_26_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '117' 'IN' port 'w4_rsc_26_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '117' 'IN' port 'w4_rsc_26_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_26_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '117' 'IN' port 'w4_rsc_26_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '118' 'IN' port 'w4_rsc_27_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_27_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '118' 'IN' port 'w4_rsc_27_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '118' 'IN' port 'w4_rsc_27_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_27_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '118' 'IN' port 'w4_rsc_27_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '118' 'IN' port 'w4_rsc_27_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_27_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '118' 'IN' port 'w4_rsc_27_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '118' 'IN' port 'w4_rsc_27_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_27_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '118' 'IN' port 'w4_rsc_27_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '118' 'IN' port 'w4_rsc_27_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_27_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '118' 'IN' port 'w4_rsc_27_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '118' 'IN' port 'w4_rsc_27_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_27_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '118' 'IN' port 'w4_rsc_27_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '118' 'IN' port 'w4_rsc_27_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_27_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '118' 'IN' port 'w4_rsc_27_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '119' 'IN' port 'w4_rsc_28_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_28_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '119' 'IN' port 'w4_rsc_28_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '119' 'IN' port 'w4_rsc_28_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_28_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '119' 'IN' port 'w4_rsc_28_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '119' 'IN' port 'w4_rsc_28_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_28_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '119' 'IN' port 'w4_rsc_28_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '119' 'IN' port 'w4_rsc_28_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_28_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '119' 'IN' port 'w4_rsc_28_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '119' 'IN' port 'w4_rsc_28_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_28_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '119' 'IN' port 'w4_rsc_28_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '119' 'IN' port 'w4_rsc_28_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_28_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '119' 'IN' port 'w4_rsc_28_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '119' 'IN' port 'w4_rsc_28_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_28_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '119' 'IN' port 'w4_rsc_28_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '120' 'IN' port 'w4_rsc_29_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_29_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '120' 'IN' port 'w4_rsc_29_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '120' 'IN' port 'w4_rsc_29_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_29_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '120' 'IN' port 'w4_rsc_29_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '120' 'IN' port 'w4_rsc_29_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_29_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '120' 'IN' port 'w4_rsc_29_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '120' 'IN' port 'w4_rsc_29_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_29_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '120' 'IN' port 'w4_rsc_29_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '120' 'IN' port 'w4_rsc_29_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_29_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '120' 'IN' port 'w4_rsc_29_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '120' 'IN' port 'w4_rsc_29_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_29_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '120' 'IN' port 'w4_rsc_29_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '120' 'IN' port 'w4_rsc_29_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_29_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '120' 'IN' port 'w4_rsc_29_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '121' 'IN' port 'w4_rsc_30_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_30_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '121' 'IN' port 'w4_rsc_30_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '121' 'IN' port 'w4_rsc_30_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_30_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '121' 'IN' port 'w4_rsc_30_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '121' 'IN' port 'w4_rsc_30_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_30_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '121' 'IN' port 'w4_rsc_30_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '121' 'IN' port 'w4_rsc_30_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_30_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '121' 'IN' port 'w4_rsc_30_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '121' 'IN' port 'w4_rsc_30_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_30_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '121' 'IN' port 'w4_rsc_30_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '121' 'IN' port 'w4_rsc_30_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_30_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '121' 'IN' port 'w4_rsc_30_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '121' 'IN' port 'w4_rsc_30_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_30_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '121' 'IN' port 'w4_rsc_30_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '122' 'IN' port 'w4_rsc_31_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_31_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '122' 'IN' port 'w4_rsc_31_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '122' 'IN' port 'w4_rsc_31_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_31_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '122' 'IN' port 'w4_rsc_31_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '122' 'IN' port 'w4_rsc_31_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_31_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '122' 'IN' port 'w4_rsc_31_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '122' 'IN' port 'w4_rsc_31_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_31_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '122' 'IN' port 'w4_rsc_31_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '122' 'IN' port 'w4_rsc_31_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_31_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '122' 'IN' port 'w4_rsc_31_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '122' 'IN' port 'w4_rsc_31_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_31_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '122' 'IN' port 'w4_rsc_31_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '122' 'IN' port 'w4_rsc_31_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_31_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '122' 'IN' port 'w4_rsc_31_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '123' 'IN' port 'w4_rsc_32_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_32_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '123' 'IN' port 'w4_rsc_32_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '123' 'IN' port 'w4_rsc_32_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_32_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '123' 'IN' port 'w4_rsc_32_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '123' 'IN' port 'w4_rsc_32_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_32_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '123' 'IN' port 'w4_rsc_32_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '123' 'IN' port 'w4_rsc_32_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_32_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '123' 'IN' port 'w4_rsc_32_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '123' 'IN' port 'w4_rsc_32_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_32_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '123' 'IN' port 'w4_rsc_32_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '123' 'IN' port 'w4_rsc_32_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_32_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '123' 'IN' port 'w4_rsc_32_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '123' 'IN' port 'w4_rsc_32_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_32_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '123' 'IN' port 'w4_rsc_32_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '124' 'IN' port 'w4_rsc_33_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_33_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '124' 'IN' port 'w4_rsc_33_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '124' 'IN' port 'w4_rsc_33_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_33_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '124' 'IN' port 'w4_rsc_33_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '124' 'IN' port 'w4_rsc_33_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_33_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '124' 'IN' port 'w4_rsc_33_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '124' 'IN' port 'w4_rsc_33_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_33_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '124' 'IN' port 'w4_rsc_33_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '124' 'IN' port 'w4_rsc_33_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_33_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '124' 'IN' port 'w4_rsc_33_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '124' 'IN' port 'w4_rsc_33_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_33_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '124' 'IN' port 'w4_rsc_33_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '124' 'IN' port 'w4_rsc_33_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_33_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '124' 'IN' port 'w4_rsc_33_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '125' 'IN' port 'w4_rsc_34_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_34_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '125' 'IN' port 'w4_rsc_34_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '125' 'IN' port 'w4_rsc_34_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_34_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '125' 'IN' port 'w4_rsc_34_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '125' 'IN' port 'w4_rsc_34_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_34_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '125' 'IN' port 'w4_rsc_34_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '125' 'IN' port 'w4_rsc_34_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_34_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '125' 'IN' port 'w4_rsc_34_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '125' 'IN' port 'w4_rsc_34_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_34_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '125' 'IN' port 'w4_rsc_34_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '125' 'IN' port 'w4_rsc_34_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_34_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '125' 'IN' port 'w4_rsc_34_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '125' 'IN' port 'w4_rsc_34_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_34_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '125' 'IN' port 'w4_rsc_34_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '126' 'IN' port 'w4_rsc_35_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_35_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '126' 'IN' port 'w4_rsc_35_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '126' 'IN' port 'w4_rsc_35_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_35_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '126' 'IN' port 'w4_rsc_35_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '126' 'IN' port 'w4_rsc_35_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_35_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '126' 'IN' port 'w4_rsc_35_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '126' 'IN' port 'w4_rsc_35_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_35_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '126' 'IN' port 'w4_rsc_35_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '126' 'IN' port 'w4_rsc_35_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_35_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '126' 'IN' port 'w4_rsc_35_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '126' 'IN' port 'w4_rsc_35_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_35_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '126' 'IN' port 'w4_rsc_35_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '126' 'IN' port 'w4_rsc_35_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_35_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '126' 'IN' port 'w4_rsc_35_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '127' 'IN' port 'w4_rsc_36_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_36_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '127' 'IN' port 'w4_rsc_36_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '127' 'IN' port 'w4_rsc_36_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_36_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '127' 'IN' port 'w4_rsc_36_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '127' 'IN' port 'w4_rsc_36_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_36_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '127' 'IN' port 'w4_rsc_36_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '127' 'IN' port 'w4_rsc_36_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_36_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '127' 'IN' port 'w4_rsc_36_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '127' 'IN' port 'w4_rsc_36_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_36_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '127' 'IN' port 'w4_rsc_36_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '127' 'IN' port 'w4_rsc_36_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_36_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '127' 'IN' port 'w4_rsc_36_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '127' 'IN' port 'w4_rsc_36_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_36_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '127' 'IN' port 'w4_rsc_36_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '128' 'IN' port 'w4_rsc_37_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_37_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '128' 'IN' port 'w4_rsc_37_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '128' 'IN' port 'w4_rsc_37_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_37_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '128' 'IN' port 'w4_rsc_37_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '128' 'IN' port 'w4_rsc_37_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_37_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '128' 'IN' port 'w4_rsc_37_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '128' 'IN' port 'w4_rsc_37_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_37_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '128' 'IN' port 'w4_rsc_37_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '128' 'IN' port 'w4_rsc_37_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_37_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '128' 'IN' port 'w4_rsc_37_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '128' 'IN' port 'w4_rsc_37_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_37_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '128' 'IN' port 'w4_rsc_37_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '128' 'IN' port 'w4_rsc_37_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_37_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '128' 'IN' port 'w4_rsc_37_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '129' 'IN' port 'w4_rsc_38_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_38_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '129' 'IN' port 'w4_rsc_38_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '129' 'IN' port 'w4_rsc_38_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_38_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '129' 'IN' port 'w4_rsc_38_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '129' 'IN' port 'w4_rsc_38_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_38_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '129' 'IN' port 'w4_rsc_38_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '129' 'IN' port 'w4_rsc_38_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_38_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '129' 'IN' port 'w4_rsc_38_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '129' 'IN' port 'w4_rsc_38_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_38_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '129' 'IN' port 'w4_rsc_38_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '129' 'IN' port 'w4_rsc_38_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_38_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '129' 'IN' port 'w4_rsc_38_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '129' 'IN' port 'w4_rsc_38_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_38_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '129' 'IN' port 'w4_rsc_38_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '130' 'IN' port 'w4_rsc_39_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_39_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '130' 'IN' port 'w4_rsc_39_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '130' 'IN' port 'w4_rsc_39_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_39_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '130' 'IN' port 'w4_rsc_39_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '130' 'IN' port 'w4_rsc_39_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_39_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '130' 'IN' port 'w4_rsc_39_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '130' 'IN' port 'w4_rsc_39_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_39_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '130' 'IN' port 'w4_rsc_39_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '130' 'IN' port 'w4_rsc_39_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_39_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '130' 'IN' port 'w4_rsc_39_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '130' 'IN' port 'w4_rsc_39_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_39_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '130' 'IN' port 'w4_rsc_39_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '130' 'IN' port 'w4_rsc_39_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_39_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '130' 'IN' port 'w4_rsc_39_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '131' 'IN' port 'w4_rsc_40_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_40_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '131' 'IN' port 'w4_rsc_40_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '131' 'IN' port 'w4_rsc_40_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_40_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '131' 'IN' port 'w4_rsc_40_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '131' 'IN' port 'w4_rsc_40_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_40_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '131' 'IN' port 'w4_rsc_40_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '131' 'IN' port 'w4_rsc_40_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_40_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '131' 'IN' port 'w4_rsc_40_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '131' 'IN' port 'w4_rsc_40_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_40_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '131' 'IN' port 'w4_rsc_40_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '131' 'IN' port 'w4_rsc_40_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_40_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '131' 'IN' port 'w4_rsc_40_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '131' 'IN' port 'w4_rsc_40_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_40_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '131' 'IN' port 'w4_rsc_40_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '132' 'IN' port 'w4_rsc_41_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_41_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '132' 'IN' port 'w4_rsc_41_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '132' 'IN' port 'w4_rsc_41_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_41_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '132' 'IN' port 'w4_rsc_41_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '132' 'IN' port 'w4_rsc_41_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_41_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '132' 'IN' port 'w4_rsc_41_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '132' 'IN' port 'w4_rsc_41_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_41_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '132' 'IN' port 'w4_rsc_41_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '132' 'IN' port 'w4_rsc_41_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_41_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '132' 'IN' port 'w4_rsc_41_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '132' 'IN' port 'w4_rsc_41_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_41_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '132' 'IN' port 'w4_rsc_41_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '132' 'IN' port 'w4_rsc_41_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_41_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '132' 'IN' port 'w4_rsc_41_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '133' 'IN' port 'w4_rsc_42_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_42_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '133' 'IN' port 'w4_rsc_42_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '133' 'IN' port 'w4_rsc_42_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_42_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '133' 'IN' port 'w4_rsc_42_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '133' 'IN' port 'w4_rsc_42_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_42_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '133' 'IN' port 'w4_rsc_42_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '133' 'IN' port 'w4_rsc_42_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_42_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '133' 'IN' port 'w4_rsc_42_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '133' 'IN' port 'w4_rsc_42_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_42_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '133' 'IN' port 'w4_rsc_42_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '133' 'IN' port 'w4_rsc_42_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_42_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '133' 'IN' port 'w4_rsc_42_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '133' 'IN' port 'w4_rsc_42_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_42_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '133' 'IN' port 'w4_rsc_42_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '134' 'IN' port 'w4_rsc_43_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_43_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '134' 'IN' port 'w4_rsc_43_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '134' 'IN' port 'w4_rsc_43_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_43_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '134' 'IN' port 'w4_rsc_43_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '134' 'IN' port 'w4_rsc_43_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_43_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '134' 'IN' port 'w4_rsc_43_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '134' 'IN' port 'w4_rsc_43_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_43_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '134' 'IN' port 'w4_rsc_43_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '134' 'IN' port 'w4_rsc_43_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_43_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '134' 'IN' port 'w4_rsc_43_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '134' 'IN' port 'w4_rsc_43_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_43_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '134' 'IN' port 'w4_rsc_43_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '134' 'IN' port 'w4_rsc_43_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_43_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '134' 'IN' port 'w4_rsc_43_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '135' 'IN' port 'w4_rsc_44_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_44_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '135' 'IN' port 'w4_rsc_44_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '135' 'IN' port 'w4_rsc_44_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_44_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '135' 'IN' port 'w4_rsc_44_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '135' 'IN' port 'w4_rsc_44_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_44_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '135' 'IN' port 'w4_rsc_44_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '135' 'IN' port 'w4_rsc_44_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_44_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '135' 'IN' port 'w4_rsc_44_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '135' 'IN' port 'w4_rsc_44_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_44_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '135' 'IN' port 'w4_rsc_44_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '135' 'IN' port 'w4_rsc_44_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_44_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '135' 'IN' port 'w4_rsc_44_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '135' 'IN' port 'w4_rsc_44_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_44_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '135' 'IN' port 'w4_rsc_44_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '136' 'IN' port 'w4_rsc_45_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_45_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '136' 'IN' port 'w4_rsc_45_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '136' 'IN' port 'w4_rsc_45_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_45_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '136' 'IN' port 'w4_rsc_45_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '136' 'IN' port 'w4_rsc_45_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_45_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '136' 'IN' port 'w4_rsc_45_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '136' 'IN' port 'w4_rsc_45_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_45_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '136' 'IN' port 'w4_rsc_45_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '136' 'IN' port 'w4_rsc_45_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_45_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '136' 'IN' port 'w4_rsc_45_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '136' 'IN' port 'w4_rsc_45_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_45_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '136' 'IN' port 'w4_rsc_45_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '136' 'IN' port 'w4_rsc_45_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_45_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '136' 'IN' port 'w4_rsc_45_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '137' 'IN' port 'w4_rsc_46_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_46_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '137' 'IN' port 'w4_rsc_46_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '137' 'IN' port 'w4_rsc_46_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_46_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '137' 'IN' port 'w4_rsc_46_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '137' 'IN' port 'w4_rsc_46_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_46_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '137' 'IN' port 'w4_rsc_46_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '137' 'IN' port 'w4_rsc_46_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_46_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '137' 'IN' port 'w4_rsc_46_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '137' 'IN' port 'w4_rsc_46_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_46_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '137' 'IN' port 'w4_rsc_46_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '137' 'IN' port 'w4_rsc_46_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_46_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '137' 'IN' port 'w4_rsc_46_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '137' 'IN' port 'w4_rsc_46_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_46_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '137' 'IN' port 'w4_rsc_46_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '138' 'IN' port 'w4_rsc_47_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_47_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '138' 'IN' port 'w4_rsc_47_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '138' 'IN' port 'w4_rsc_47_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_47_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '138' 'IN' port 'w4_rsc_47_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '138' 'IN' port 'w4_rsc_47_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_47_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '138' 'IN' port 'w4_rsc_47_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '138' 'IN' port 'w4_rsc_47_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_47_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '138' 'IN' port 'w4_rsc_47_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '138' 'IN' port 'w4_rsc_47_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_47_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '138' 'IN' port 'w4_rsc_47_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '138' 'IN' port 'w4_rsc_47_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_47_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '138' 'IN' port 'w4_rsc_47_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '138' 'IN' port 'w4_rsc_47_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_47_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '138' 'IN' port 'w4_rsc_47_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '139' 'IN' port 'w4_rsc_48_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_48_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '139' 'IN' port 'w4_rsc_48_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '139' 'IN' port 'w4_rsc_48_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_48_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '139' 'IN' port 'w4_rsc_48_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '139' 'IN' port 'w4_rsc_48_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_48_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '139' 'IN' port 'w4_rsc_48_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '139' 'IN' port 'w4_rsc_48_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_48_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '139' 'IN' port 'w4_rsc_48_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '139' 'IN' port 'w4_rsc_48_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_48_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '139' 'IN' port 'w4_rsc_48_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '139' 'IN' port 'w4_rsc_48_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_48_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '139' 'IN' port 'w4_rsc_48_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '139' 'IN' port 'w4_rsc_48_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_48_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '139' 'IN' port 'w4_rsc_48_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '140' 'IN' port 'w4_rsc_49_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_49_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '140' 'IN' port 'w4_rsc_49_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '140' 'IN' port 'w4_rsc_49_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_49_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '140' 'IN' port 'w4_rsc_49_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '140' 'IN' port 'w4_rsc_49_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_49_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '140' 'IN' port 'w4_rsc_49_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '140' 'IN' port 'w4_rsc_49_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_49_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '140' 'IN' port 'w4_rsc_49_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '140' 'IN' port 'w4_rsc_49_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_49_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '140' 'IN' port 'w4_rsc_49_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '140' 'IN' port 'w4_rsc_49_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_49_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '140' 'IN' port 'w4_rsc_49_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '140' 'IN' port 'w4_rsc_49_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_49_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '140' 'IN' port 'w4_rsc_49_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '141' 'IN' port 'w4_rsc_50_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_50_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '141' 'IN' port 'w4_rsc_50_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '141' 'IN' port 'w4_rsc_50_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_50_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '141' 'IN' port 'w4_rsc_50_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '141' 'IN' port 'w4_rsc_50_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_50_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '141' 'IN' port 'w4_rsc_50_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '141' 'IN' port 'w4_rsc_50_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_50_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '141' 'IN' port 'w4_rsc_50_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '141' 'IN' port 'w4_rsc_50_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_50_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '141' 'IN' port 'w4_rsc_50_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '141' 'IN' port 'w4_rsc_50_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_50_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '141' 'IN' port 'w4_rsc_50_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '141' 'IN' port 'w4_rsc_50_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_50_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '141' 'IN' port 'w4_rsc_50_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '142' 'IN' port 'w4_rsc_51_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_51_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '142' 'IN' port 'w4_rsc_51_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '142' 'IN' port 'w4_rsc_51_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_51_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '142' 'IN' port 'w4_rsc_51_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '142' 'IN' port 'w4_rsc_51_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_51_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '142' 'IN' port 'w4_rsc_51_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '142' 'IN' port 'w4_rsc_51_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_51_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '142' 'IN' port 'w4_rsc_51_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '142' 'IN' port 'w4_rsc_51_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_51_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '142' 'IN' port 'w4_rsc_51_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '142' 'IN' port 'w4_rsc_51_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_51_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '142' 'IN' port 'w4_rsc_51_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '142' 'IN' port 'w4_rsc_51_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_51_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '142' 'IN' port 'w4_rsc_51_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '143' 'IN' port 'w4_rsc_52_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_52_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '143' 'IN' port 'w4_rsc_52_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '143' 'IN' port 'w4_rsc_52_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_52_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '143' 'IN' port 'w4_rsc_52_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '143' 'IN' port 'w4_rsc_52_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_52_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '143' 'IN' port 'w4_rsc_52_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '143' 'IN' port 'w4_rsc_52_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_52_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '143' 'IN' port 'w4_rsc_52_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '143' 'IN' port 'w4_rsc_52_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_52_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '143' 'IN' port 'w4_rsc_52_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '143' 'IN' port 'w4_rsc_52_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_52_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '143' 'IN' port 'w4_rsc_52_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '143' 'IN' port 'w4_rsc_52_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_52_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '143' 'IN' port 'w4_rsc_52_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '144' 'IN' port 'w4_rsc_53_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_53_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '144' 'IN' port 'w4_rsc_53_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '144' 'IN' port 'w4_rsc_53_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_53_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '144' 'IN' port 'w4_rsc_53_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '144' 'IN' port 'w4_rsc_53_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_53_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '144' 'IN' port 'w4_rsc_53_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '144' 'IN' port 'w4_rsc_53_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_53_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '144' 'IN' port 'w4_rsc_53_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '144' 'IN' port 'w4_rsc_53_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_53_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '144' 'IN' port 'w4_rsc_53_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '144' 'IN' port 'w4_rsc_53_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_53_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '144' 'IN' port 'w4_rsc_53_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '144' 'IN' port 'w4_rsc_53_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_53_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '144' 'IN' port 'w4_rsc_53_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '145' 'IN' port 'w4_rsc_54_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_54_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '145' 'IN' port 'w4_rsc_54_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '145' 'IN' port 'w4_rsc_54_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_54_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '145' 'IN' port 'w4_rsc_54_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '145' 'IN' port 'w4_rsc_54_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_54_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '145' 'IN' port 'w4_rsc_54_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '145' 'IN' port 'w4_rsc_54_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_54_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '145' 'IN' port 'w4_rsc_54_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '145' 'IN' port 'w4_rsc_54_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_54_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '145' 'IN' port 'w4_rsc_54_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '145' 'IN' port 'w4_rsc_54_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_54_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '145' 'IN' port 'w4_rsc_54_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '145' 'IN' port 'w4_rsc_54_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_54_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '145' 'IN' port 'w4_rsc_54_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '146' 'IN' port 'w4_rsc_55_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_55_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '146' 'IN' port 'w4_rsc_55_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '146' 'IN' port 'w4_rsc_55_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_55_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '146' 'IN' port 'w4_rsc_55_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '146' 'IN' port 'w4_rsc_55_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_55_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '146' 'IN' port 'w4_rsc_55_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '146' 'IN' port 'w4_rsc_55_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_55_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '146' 'IN' port 'w4_rsc_55_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '146' 'IN' port 'w4_rsc_55_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_55_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '146' 'IN' port 'w4_rsc_55_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '146' 'IN' port 'w4_rsc_55_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_55_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '146' 'IN' port 'w4_rsc_55_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '146' 'IN' port 'w4_rsc_55_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_55_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '146' 'IN' port 'w4_rsc_55_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '147' 'IN' port 'w4_rsc_56_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_56_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '147' 'IN' port 'w4_rsc_56_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '147' 'IN' port 'w4_rsc_56_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_56_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '147' 'IN' port 'w4_rsc_56_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '147' 'IN' port 'w4_rsc_56_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_56_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '147' 'IN' port 'w4_rsc_56_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '147' 'IN' port 'w4_rsc_56_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_56_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '147' 'IN' port 'w4_rsc_56_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '147' 'IN' port 'w4_rsc_56_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_56_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '147' 'IN' port 'w4_rsc_56_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '147' 'IN' port 'w4_rsc_56_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_56_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '147' 'IN' port 'w4_rsc_56_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '147' 'IN' port 'w4_rsc_56_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_56_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '147' 'IN' port 'w4_rsc_56_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '148' 'IN' port 'w4_rsc_57_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_57_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '148' 'IN' port 'w4_rsc_57_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '148' 'IN' port 'w4_rsc_57_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_57_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '148' 'IN' port 'w4_rsc_57_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '148' 'IN' port 'w4_rsc_57_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_57_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '148' 'IN' port 'w4_rsc_57_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '148' 'IN' port 'w4_rsc_57_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_57_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '148' 'IN' port 'w4_rsc_57_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '148' 'IN' port 'w4_rsc_57_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_57_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '148' 'IN' port 'w4_rsc_57_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '148' 'IN' port 'w4_rsc_57_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_57_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '148' 'IN' port 'w4_rsc_57_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '148' 'IN' port 'w4_rsc_57_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_57_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '148' 'IN' port 'w4_rsc_57_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '149' 'IN' port 'w4_rsc_58_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_58_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '149' 'IN' port 'w4_rsc_58_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '149' 'IN' port 'w4_rsc_58_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_58_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '149' 'IN' port 'w4_rsc_58_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '149' 'IN' port 'w4_rsc_58_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_58_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '149' 'IN' port 'w4_rsc_58_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '149' 'IN' port 'w4_rsc_58_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_58_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '149' 'IN' port 'w4_rsc_58_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '149' 'IN' port 'w4_rsc_58_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_58_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '149' 'IN' port 'w4_rsc_58_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '149' 'IN' port 'w4_rsc_58_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_58_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '149' 'IN' port 'w4_rsc_58_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '149' 'IN' port 'w4_rsc_58_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_58_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '149' 'IN' port 'w4_rsc_58_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '150' 'IN' port 'w4_rsc_59_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_59_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '150' 'IN' port 'w4_rsc_59_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '150' 'IN' port 'w4_rsc_59_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_59_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '150' 'IN' port 'w4_rsc_59_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '150' 'IN' port 'w4_rsc_59_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_59_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '150' 'IN' port 'w4_rsc_59_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '150' 'IN' port 'w4_rsc_59_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_59_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '150' 'IN' port 'w4_rsc_59_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '150' 'IN' port 'w4_rsc_59_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_59_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '150' 'IN' port 'w4_rsc_59_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '150' 'IN' port 'w4_rsc_59_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_59_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '150' 'IN' port 'w4_rsc_59_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '150' 'IN' port 'w4_rsc_59_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_59_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '150' 'IN' port 'w4_rsc_59_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '151' 'IN' port 'w4_rsc_60_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_60_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '151' 'IN' port 'w4_rsc_60_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '151' 'IN' port 'w4_rsc_60_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_60_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '151' 'IN' port 'w4_rsc_60_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '151' 'IN' port 'w4_rsc_60_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_60_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '151' 'IN' port 'w4_rsc_60_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '151' 'IN' port 'w4_rsc_60_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_60_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '151' 'IN' port 'w4_rsc_60_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '151' 'IN' port 'w4_rsc_60_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_60_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '151' 'IN' port 'w4_rsc_60_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '151' 'IN' port 'w4_rsc_60_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_60_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '151' 'IN' port 'w4_rsc_60_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '151' 'IN' port 'w4_rsc_60_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_60_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '151' 'IN' port 'w4_rsc_60_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '152' 'IN' port 'w4_rsc_61_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_61_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '152' 'IN' port 'w4_rsc_61_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '152' 'IN' port 'w4_rsc_61_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_61_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '152' 'IN' port 'w4_rsc_61_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '152' 'IN' port 'w4_rsc_61_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_61_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '152' 'IN' port 'w4_rsc_61_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '152' 'IN' port 'w4_rsc_61_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_61_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '152' 'IN' port 'w4_rsc_61_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '152' 'IN' port 'w4_rsc_61_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_61_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '152' 'IN' port 'w4_rsc_61_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '152' 'IN' port 'w4_rsc_61_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_61_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '152' 'IN' port 'w4_rsc_61_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '152' 'IN' port 'w4_rsc_61_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_61_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '152' 'IN' port 'w4_rsc_61_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '153' 'IN' port 'w4_rsc_62_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_62_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '153' 'IN' port 'w4_rsc_62_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '153' 'IN' port 'w4_rsc_62_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_62_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '153' 'IN' port 'w4_rsc_62_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '153' 'IN' port 'w4_rsc_62_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_62_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '153' 'IN' port 'w4_rsc_62_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '153' 'IN' port 'w4_rsc_62_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_62_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '153' 'IN' port 'w4_rsc_62_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '153' 'IN' port 'w4_rsc_62_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_62_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '153' 'IN' port 'w4_rsc_62_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '153' 'IN' port 'w4_rsc_62_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_62_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '153' 'IN' port 'w4_rsc_62_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '153' 'IN' port 'w4_rsc_62_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_62_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '153' 'IN' port 'w4_rsc_62_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '154' 'IN' port 'w4_rsc_63_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_63_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '154' 'IN' port 'w4_rsc_63_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '154' 'IN' port 'w4_rsc_63_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_63_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '154' 'IN' port 'w4_rsc_63_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '154' 'IN' port 'w4_rsc_63_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_63_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '154' 'IN' port 'w4_rsc_63_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '154' 'IN' port 'w4_rsc_63_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_63_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '154' 'IN' port 'w4_rsc_63_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '154' 'IN' port 'w4_rsc_63_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_63_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '154' 'IN' port 'w4_rsc_63_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '154' 'IN' port 'w4_rsc_63_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_63_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '154' 'IN' port 'w4_rsc_63_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '154' 'IN' port 'w4_rsc_63_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w4_rsc_63_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '154' 'IN' port 'w4_rsc_63_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/b4_rsc_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '8' 'IN' port 'b4_rsc_dat' '2' 'OUT' port 'output1_rsc_dat'"

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

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '155' 'IN' port 'w6_rsc_0_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_0_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '155' 'IN' port 'w6_rsc_0_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '155' 'IN' port 'w6_rsc_0_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_0_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '155' 'IN' port 'w6_rsc_0_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '155' 'IN' port 'w6_rsc_0_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_0_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '155' 'IN' port 'w6_rsc_0_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '155' 'IN' port 'w6_rsc_0_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_0_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '155' 'IN' port 'w6_rsc_0_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '155' 'IN' port 'w6_rsc_0_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_0_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '155' 'IN' port 'w6_rsc_0_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '155' 'IN' port 'w6_rsc_0_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_0_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '155' 'IN' port 'w6_rsc_0_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '155' 'IN' port 'w6_rsc_0_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_0_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '155' 'IN' port 'w6_rsc_0_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '156' 'IN' port 'w6_rsc_1_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_1_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '156' 'IN' port 'w6_rsc_1_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '156' 'IN' port 'w6_rsc_1_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_1_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '156' 'IN' port 'w6_rsc_1_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '156' 'IN' port 'w6_rsc_1_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_1_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '156' 'IN' port 'w6_rsc_1_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '156' 'IN' port 'w6_rsc_1_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_1_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '156' 'IN' port 'w6_rsc_1_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '156' 'IN' port 'w6_rsc_1_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_1_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '156' 'IN' port 'w6_rsc_1_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '156' 'IN' port 'w6_rsc_1_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_1_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '156' 'IN' port 'w6_rsc_1_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '156' 'IN' port 'w6_rsc_1_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_1_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '156' 'IN' port 'w6_rsc_1_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '157' 'IN' port 'w6_rsc_2_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_2_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '157' 'IN' port 'w6_rsc_2_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '157' 'IN' port 'w6_rsc_2_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_2_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '157' 'IN' port 'w6_rsc_2_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '157' 'IN' port 'w6_rsc_2_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_2_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '157' 'IN' port 'w6_rsc_2_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '157' 'IN' port 'w6_rsc_2_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_2_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '157' 'IN' port 'w6_rsc_2_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '157' 'IN' port 'w6_rsc_2_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_2_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '157' 'IN' port 'w6_rsc_2_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '157' 'IN' port 'w6_rsc_2_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_2_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '157' 'IN' port 'w6_rsc_2_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '157' 'IN' port 'w6_rsc_2_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_2_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '157' 'IN' port 'w6_rsc_2_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '158' 'IN' port 'w6_rsc_3_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_3_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '158' 'IN' port 'w6_rsc_3_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '158' 'IN' port 'w6_rsc_3_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_3_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '158' 'IN' port 'w6_rsc_3_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '158' 'IN' port 'w6_rsc_3_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_3_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '158' 'IN' port 'w6_rsc_3_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '158' 'IN' port 'w6_rsc_3_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_3_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '158' 'IN' port 'w6_rsc_3_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '158' 'IN' port 'w6_rsc_3_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_3_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '158' 'IN' port 'w6_rsc_3_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '158' 'IN' port 'w6_rsc_3_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_3_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '158' 'IN' port 'w6_rsc_3_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '158' 'IN' port 'w6_rsc_3_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_3_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '158' 'IN' port 'w6_rsc_3_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '159' 'IN' port 'w6_rsc_4_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_4_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '159' 'IN' port 'w6_rsc_4_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '159' 'IN' port 'w6_rsc_4_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_4_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '159' 'IN' port 'w6_rsc_4_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '159' 'IN' port 'w6_rsc_4_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_4_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '159' 'IN' port 'w6_rsc_4_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '159' 'IN' port 'w6_rsc_4_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_4_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '159' 'IN' port 'w6_rsc_4_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '159' 'IN' port 'w6_rsc_4_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_4_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '159' 'IN' port 'w6_rsc_4_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '159' 'IN' port 'w6_rsc_4_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_4_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '159' 'IN' port 'w6_rsc_4_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '159' 'IN' port 'w6_rsc_4_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_4_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '159' 'IN' port 'w6_rsc_4_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '160' 'IN' port 'w6_rsc_5_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_5_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '160' 'IN' port 'w6_rsc_5_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '160' 'IN' port 'w6_rsc_5_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_5_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '160' 'IN' port 'w6_rsc_5_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '160' 'IN' port 'w6_rsc_5_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_5_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '160' 'IN' port 'w6_rsc_5_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '160' 'IN' port 'w6_rsc_5_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_5_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '160' 'IN' port 'w6_rsc_5_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '160' 'IN' port 'w6_rsc_5_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_5_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '160' 'IN' port 'w6_rsc_5_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '160' 'IN' port 'w6_rsc_5_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_5_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '160' 'IN' port 'w6_rsc_5_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '160' 'IN' port 'w6_rsc_5_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_5_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '160' 'IN' port 'w6_rsc_5_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '161' 'IN' port 'w6_rsc_6_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_6_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '161' 'IN' port 'w6_rsc_6_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '161' 'IN' port 'w6_rsc_6_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_6_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '161' 'IN' port 'w6_rsc_6_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '161' 'IN' port 'w6_rsc_6_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_6_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '161' 'IN' port 'w6_rsc_6_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '161' 'IN' port 'w6_rsc_6_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_6_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '161' 'IN' port 'w6_rsc_6_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '161' 'IN' port 'w6_rsc_6_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_6_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '161' 'IN' port 'w6_rsc_6_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '161' 'IN' port 'w6_rsc_6_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_6_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '161' 'IN' port 'w6_rsc_6_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '161' 'IN' port 'w6_rsc_6_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_6_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '161' 'IN' port 'w6_rsc_6_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '162' 'IN' port 'w6_rsc_7_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_7_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '162' 'IN' port 'w6_rsc_7_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '162' 'IN' port 'w6_rsc_7_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_7_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '162' 'IN' port 'w6_rsc_7_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '162' 'IN' port 'w6_rsc_7_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_7_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '162' 'IN' port 'w6_rsc_7_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '162' 'IN' port 'w6_rsc_7_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_7_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '162' 'IN' port 'w6_rsc_7_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '162' 'IN' port 'w6_rsc_7_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_7_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '162' 'IN' port 'w6_rsc_7_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '162' 'IN' port 'w6_rsc_7_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_7_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '162' 'IN' port 'w6_rsc_7_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '162' 'IN' port 'w6_rsc_7_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_7_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '162' 'IN' port 'w6_rsc_7_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '163' 'IN' port 'w6_rsc_8_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_8_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '163' 'IN' port 'w6_rsc_8_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '163' 'IN' port 'w6_rsc_8_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_8_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '163' 'IN' port 'w6_rsc_8_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '163' 'IN' port 'w6_rsc_8_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_8_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '163' 'IN' port 'w6_rsc_8_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '163' 'IN' port 'w6_rsc_8_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_8_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '163' 'IN' port 'w6_rsc_8_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '163' 'IN' port 'w6_rsc_8_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_8_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '163' 'IN' port 'w6_rsc_8_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '163' 'IN' port 'w6_rsc_8_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_8_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '163' 'IN' port 'w6_rsc_8_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '163' 'IN' port 'w6_rsc_8_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_8_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '163' 'IN' port 'w6_rsc_8_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '164' 'IN' port 'w6_rsc_9_0_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_9_0_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '164' 'IN' port 'w6_rsc_9_0_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '164' 'IN' port 'w6_rsc_9_0_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_9_0_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '164' 'IN' port 'w6_rsc_9_0_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '164' 'IN' port 'w6_rsc_9_0_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_9_0_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '164' 'IN' port 'w6_rsc_9_0_dat' '2' 'OUT' port 'output1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '164' 'IN' port 'w6_rsc_9_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_9_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '164' 'IN' port 'w6_rsc_9_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '164' 'IN' port 'w6_rsc_9_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_9_0_dat[*]}] -to [find / -port {*/const_size_in_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '164' 'IN' port 'w6_rsc_9_0_dat' '3' 'OUT' port 'const_size_in_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '164' 'IN' port 'w6_rsc_9_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_9_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '164' 'IN' port 'w6_rsc_9_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '164' 'IN' port 'w6_rsc_9_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/w6_rsc_9_0_dat[*]}] -to [find / -port {*/const_size_out_1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '164' 'IN' port 'w6_rsc_9_0_dat' '4' 'OUT' port 'const_size_out_1_rsc_dat'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '1' 'IN' port 'input1_rsc_rdy'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/input1_rsc_rdy}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '1' 'IN' port 'input1_rsc_rdy'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '2' 'OUT' port 'output1_rsc_vld'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/output1_rsc_vld}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '2' 'OUT' port 'output1_rsc_vld'"

    puts "-- Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '2' 'OUT' port 'output1_rsc_dat'"
    report timing -full_pin_names -from [find / -port {*/b6_rsc_dat[*]}] -to [find / -port {*/output1_rsc_dat[*]}]
    puts "-- END Synthesis input_to_output:timing report for design 'mnist_mlp' '10' 'IN' port 'b6_rsc_dat' '2' 'OUT' port 'output1_rsc_dat'"

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

  }

if {$hls_status} {
  puts "Warning: Check transcript for errors hls_status=$hls_status"
}
puts "[clock format [clock seconds] -format {%a %b %d %H:%M:%S %Z %Y}]"
change_names -verilog
#write_hdl mnist_mlp > $RTL_TOOL_SCRIPT_DIR/gate.rc.v.v
#puts "-- Netlist for design 'mnist_mlp' written to $RTL_TOOL_SCRIPT_DIR/gate.rc.v.v"
#write_sdc > $RTL_TOOL_SCRIPT_DIR/gate.rc.v.sdc
#puts "-- SDC for design 'mnist_mlp' written to $RTL_TOOL_SCRIPT_DIR/gate.rc.v.sdc"
#write_sdf > $RTL_TOOL_SCRIPT_DIR/gate.rc.v.sdf
#puts "-- SDF for design 'mnist_mlp' written to $RTL_TOOL_SCRIPT_DIR/gate.rc.v.sdf"
write_hdl mnist_mlp > gate.rc.v.v
puts "-- Netlist for design 'mnist_mlp' written to gate.rc.v.v"
write_sdc > gate.rc.v.sdc
puts "-- SDC for design 'mnist_mlp' written to gate.rc.v.sdc"
write_sdf > gate.rc.v.sdf
puts "-- SDF for design 'mnist_mlp' written to gate.rc.v.sdf"

# Source potential custom script
source_custom_script final
puts "-- Synthesis finished for design 'mnist_mlp'"

