#################
#    HLS4ML
#################
array set opt {
  reset      0
  csim       0
  synth      1
  cosim      0
  validation 0
  vhdl       1
  verilog    1
  export     0
  vsynth     0
  bitfile    0
  fifo_opt   0
  ran_frame  2
  sw_opt     0
  power      0
  da         0
  bup        0
}

# Get pathname to this script to use as dereference path for relative file pathnames
set sfd [file dirname [info script]]

if { [info exists ::argv] } {
  foreach arg $::argv {
    foreach {optname optval} [split $arg '='] {}
    if { [info exists opt($optname)] } {
      if {[string is integer -strict $optval]} {
        set opt($optname) $optval
      } else {
        set opt($optname) [string is true -strict $optval]
      }
    }
  }
}

puts "***** INVOKE OPTIONS *****"
foreach x [lsort [array names opt]] {
  puts "[format {   %-20s %s} $x $opt($x)]"
}
puts ""

proc report_time { op_name time_start time_end } {
  set time_taken [expr $time_end - $time_start]
  set time_s [expr ($time_taken / 1000) % 60]
  set time_m [expr ($time_taken / (1000*60)) % 60]
  set time_h [expr ($time_taken / (1000*60*60)) % 24]
  puts "***** ${op_name} COMPLETED IN ${time_h}h${time_m}m${time_s}s *****"
}

proc setup_xilinx_part { part } {
  # Map Xilinx PART into Catapult library names
  set part_sav $part
  set libname [lindex [library get /CONFIG/PARAMETERS/Vivado/PARAMETERS/Xilinx/PARAMETERS/*/PARAMETERS/*/PARAMETERS/$part/LIBRARIES/*/NAME -match glob -ret v] 0]
  puts "Library Name: $libname"
  if { [llength $libname] == 1 } {
    set libpath [library get /CONFIG/PARAMETERS/Vivado/PARAMETERS/Xilinx/PARAMETERS/*/PARAMETERS/*/PARAMETERS/$part/LIBRARIES/*/NAME -match glob -ret p]
    puts "Library Path: $libpath"
    if { [regexp {/CONFIG/PARAMETERS/(\S+)/PARAMETERS/(\S+)/PARAMETERS/(\S+)/PARAMETERS/(\S+)/PARAMETERS/(\S+)/.*} $libpath dummy rtltool vendor family speed part] } {
      solution library add $libname -- -rtlsyntool $rtltool -vendor $vendor -family $family -speed $speed -part $part_sav
    } else {
      solution library add $libname -- -rtlsyntool Vivado
    }
  } else {
    logfile message "Could not find specific Xilinx base library for part '$part'. Using KINTEX-u\n" warning
    solution library add mgc_Xilinx-KINTEX-u-2_beh -- -rtlsyntool Vivado -manufacturer Xilinx -family KINTEX-u -speed -2 -part xcku115-flvb2104-2-i
  }
  solution library add Xilinx_RAMS
  solution library add Xilinx_ROMS
  solution library add Xilinx_FIFO
}


proc setup_asic_libs { args } {
  set do_saed 0
  foreach lib $args {
    solution library add $lib -- -rtlsyntool DesignCompiler
    if { [lsearch -exact {saed32hvt_tt0p78v125c_beh saed32lvt_tt0p78v125c_beh saed32rvt_tt0p78v125c_beh} $lib] != -1 } {
      set do_saed 1
    }
  }
  solution library add ccs_sample_mem
  solution library add ccs_sample_rom
  solution library add hls4ml_lib
  go libraries

  # special exception for SAED32 for use in power estimation
  if { $do_saed } {
    # SAED32 selected - enable DC settings to access Liberty data for power estimation
    source [application get /SYSTEM/ENV_MGC_HOME]/pkgs/siflibs/saed/setup_saedlib.tcl
  }
}

options set Input/CppStandard {c++17}
options set Input/CompilerFlags -DRANDOM_FRAMES=$opt(ran_frame)
options set Input/SearchPath {$MGC_HOME/shared/include/nnet_utils} -append
options set ComponentLibs/SearchPath {$MGC_HOME/shared/pkgs/ccs_hls4ml} -append

if {$opt(reset)} {
  project load CATAPULT_DIR.ccs
  go new
} else {
  project new -name CATAPULT_DIR
}

#--------------------------------------------------------
# Configure Catapult Options
# downgrade HIER-10
options set Message/ErrorOverride HIER-10 -remove
solution options set Message/ErrorOverride HIER-10 -remove

if {$opt(vhdl)}    {
  options set Output/OutputVHDL true
} else {
  options set Output/OutputVHDL false
}
if {$opt(verilog)} {
  options set Output/OutputVerilog true
} else {
  options set Output/OutputVerilog false
}

#--------------------------------------------------------
# Configure Catapult Flows
if { [info exists ::env(XILINX_PCL_CACHE)] } {
options set /Flows/Vivado/PCL_CACHE $::env(XILINX_PCL_CACHE)
solution options set /Flows/Vivado/PCL_CACHE $::env(XILINX_PCL_CACHE)
}

# Turn on HLS4ML flow (wrapped in a cache so that older Catapult installs still work)
catch {flow package require /HLS4ML}

# Turn on SCVerify flow
flow package require /SCVerify
#  flow package option set /SCVerify/INVOKE_ARGS {$sfd/firmware/weights $sfd/tb_data/tb_input_features.dat $sfd/tb_data/tb_output_predictions.dat}
#hls-fpga-machine-learning insert invoke_args

# Turn on VSCode flow
# flow package require /VSCode
# To launch VSCode on the C++ HLS design:
#   cd my-Catapult-test
#   code Catapult.code-workspace

#--------------------------------------------------------
#    Start of HLS script
set design_top myproject
solution file add $sfd/firmware/myproject.cpp
solution file add $sfd/myproject_test.cpp -exclude true

# Parse parameters.h to determine config info to control directives/pragmas
set IOType io_stream
if { ![file exists $sfd/firmware/parameters.h] } {
  logfile message "Could not locate firmware/parameters.h. Unable to determine network configuration.\n" warning
} else {
  set pf [open "$sfd/firmware/parameters.h" "r"]
  while {![eof $pf]} {
    gets $pf line
    if { [string match {*io_type = nnet::io_stream*} $line] } {
      set IOType io_stream
      break
    }
  }
  close $pf
}

if { $IOType == "io_stream" } {
solution options set Architectural/DefaultRegisterThreshold 2050
}
directive set -RESET_CLEARS_ALL_REGS no
# Constrain arrays to map to memory only over a certain size
directive set -MEM_MAP_THRESHOLD [expr 2048 * 16 + 1]
# The following line gets modified by the backend writer
set hls_clock_period 5

go analyze

# NORMAL TOP DOWN FLOW
if { ! $opt(bup) } {

go compile

if {$opt(csim)} {
  puts "***** C SIMULATION *****"
  set time_start [clock clicks -milliseconds]
  flow run /SCVerify/launch_make ./scverify/Verify_orig_cxx_osci.mk {} SIMTOOL=osci sim
  set time_end [clock clicks -milliseconds]
  report_time "C SIMULATION" $time_start $time_end
}

puts "***** SETTING TECHNOLOGY LIBRARIES *****"
#hls-fpga-machine-learning insert techlibs

directive set -CLOCKS [list clk [list -CLOCK_PERIOD $hls_clock_period -CLOCK_EDGE rising -CLOCK_OFFSET 0.000000 -CLOCK_UNCERTAINTY 0.0 -RESET_KIND sync -RESET_SYNC_NAME rst -RESET_SYNC_ACTIVE high -RESET_ASYNC_NAME arst_n -RESET_ASYNC_ACTIVE low -ENABLE_NAME {} -ENABLE_ACTIVE high]]

if {$opt(synth)} {
  puts "***** C/RTL SYNTHESIS *****"
  set time_start [clock clicks -milliseconds]

  go assembly

  go architect

  go allocate

  go schedule

  go extract
  set time_end [clock clicks -milliseconds]
  report_time "C/RTL SYNTHESIS" $time_start $time_end
}

# BOTTOM-UP FLOW
} else {
  # Start at 'go analyze'
  go analyze

  # Build the design bottom-up
  directive set -CLOCKS [list clk [list -CLOCK_PERIOD $hls_clock_period -CLOCK_EDGE rising -CLOCK_OFFSET 0.000000 -CLOCK_UNCERTAINTY 0.0 -RESET_KIND sync -RESET_SYNC_NAME rst -RESET_SYNC_ACTIVE high -RESET_ASYNC_NAME arst_n -RESET_ASYNC_ACTIVE low -ENABLE_NAME {} -ENABLE_ACTIVE high]]

  set blocks [solution get /HIERCONFIG/USER_HBS/*/RESOLVED_NAME -match glob -rec 1 -ret v -state analyze]
  set bu_mappings {}
  set top [lindex $blocks 0]
  foreach block [lreverse [lrange $blocks 1 end]] {
    # skip blocks that are net nnet:: functions
    if { [string match {nnet::*} $block] == 0 } { continue }
    go analyze
    solution design set $block -top
    go compile
    solution library remove *
    puts "***** SETTING TECHNOLOGY LIBRARIES *****"
#hls-fpga-machine-learning insert techlibs
    go extract
    set block_soln "[solution get /TOP/name -checkpath 0].[solution get /VERSION -checkpath 0]"
    lappend bu_mappings [solution get /CAT_DIR] /$top/$block "\[Block\] $block_soln"
  }

  # Move to top design
  go analyze
  solution design set $top -top
  go compile

  if {$opt(csim)} {
    puts "***** C SIMULATION *****"
    set time_start [clock clicks -milliseconds]
    flow run /SCVerify/launch_make ./scverify/Verify_orig_cxx_osci.mk {} SIMTOOL=osci sim
    set time_end [clock clicks -milliseconds]
    report_time "C SIMULATION" $time_start $time_end
  }
  foreach {d i l} $bu_mappings {
    logfile message "solution options set ComponentLibs/SearchPath $d -append\n" info
    solution options set ComponentLibs/SearchPath $d -append
  }

  # Add bottom-up blocks
  puts "***** SETTING TECHNOLOGY LIBRARIES *****"
  solution library remove *
#hls-fpga-machine-learning insert techlibs
  # need to revert back to go compile
  go compile
  foreach {d i l} $bu_mappings {
    logfile message "solution library add [list $l]\n" info
    eval solution library add [list $l]
  }
  go libraries

  # Map to bottom-up blocks
  foreach {d i l} $bu_mappings {
    # Make sure block exists
    set cnt [directive get $i/* -match glob -checkpath 0 -ret p]
    if { $cnt != {} } {
      logfile message "directive set $i -MAP_TO_MODULE [list $l]\n" info
      eval directive set $i -MAP_TO_MODULE [list $l]
    }
  }
  go assembly
  set design [solution get -name]
  logfile message "Adjusting FIFO_DEPTH for top-level interconnect channels\n" warning
  # FIFO interconnect between layers
  foreach ch_fifo_m2m [directive get -match glob -checkpath 0 -ret p $design/*_out:cns/MAP_TO_MODULE] {
    set ch_fifo [join [lrange [split $ch_fifo_m2m '/'] 0 end-1] /]/FIFO_DEPTH
    logfile message "directive set -match glob $ch_fifo 1\n" info
    directive set -match glob "$ch_fifo" 1
  }
  # For bypass paths - the depth will likely need to be larger than 1
  foreach ch_fifo_m2m [directive get -match glob -checkpath 0 -ret p $design/*_cpy*:cns/MAP_TO_MODULE] {
    set ch_fifo [join [lrange [split $ch_fifo_m2m '/'] 0 end-1] /]/FIFO_DEPTH
    logfile message "Bypass FIFO '$ch_fifo' depth set to 1 - larger value may be required to prevent deadlock\n" warning
    logfile message "directive set -match glob $ch_fifo 1\n" info
    directive set -match glob "$ch_fifo" 1
  }
  go architect
  go allocate
  go schedule
  go dpfsm
  go extract
}

project save

if {$opt(cosim) || $opt(validation)} {
  if {$opt(verilog)} {
    flow run /SCVerify/launch_make ./scverify/Verify_rtl_v_msim.mk {} SIMTOOL=msim sim
  }
  if {$opt(vhdl)} {
    flow run /SCVerify/launch_make ./scverify/Verify_rtl_vhdl_msim.mk {} SIMTOOL=msim sim
  }
}

if {$opt(export)} {
  puts "***** EXPORT IP *****"
  set time_start [clock clicks -milliseconds]
# Not yet implemented. Do we need to include value of $version ?
#  flow package option set /Vivado/BoardPart xilinx.com:zcu102:part0:3.1
#  flow package option set /Vivado/IP_Taxonomy {/Catapult}
#  flow run /Vivado/launch_package_ip -shell ./vivado_concat_v/concat_v_package_ip.tcl
  set time_end [clock clicks -milliseconds]
  report_time "EXPORT IP" $time_start $time_end
}
if {$opt(sw_opt)} {
  puts "***** Pre Power Optimization *****"
  go switching
  if {$opt(verilog)} {
    flow run /PowerAnalysis/report_pre_pwropt_Verilog
  }
  if {$opt(vhdl)} {
    flow run /PowerAnalysis/report_pre_pwropt_VHDL
  }
}

if {$opt(power)} {
  puts "***** Power Optimization *****"
  go power
}

if {$opt(vsynth)} {
  puts "***** VIVADO SYNTHESIS *****"
  set time_start [clock clicks -milliseconds]
  flow run /Vivado/synthesize -shell vivado_concat_v/concat_rtl.v.xv
  set time_end [clock clicks -milliseconds]
  report_time "VIVADO SYNTHESIS" $time_start $time_end
}

if {$opt(bitfile)} {
  puts "***** Option bitfile not supported yet *****"
}

if {$opt(da)} {
  puts "***** Launching DA *****"
  flow run /DesignAnalyzer/launch
}

if { [catch {flow package present /HLS4ML}] == 0 } {
  flow run /HLS4ML/collect_reports
}
