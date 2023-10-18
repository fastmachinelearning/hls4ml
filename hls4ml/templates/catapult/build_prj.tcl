#################
#    HLS4ML
#################
array set opt {
  reset      0
  csim       0
  synth      1
  cosim      0
  validation 0
  vhdl       0
  verilog    1
  export     0
  vsynth     0
  bitfile    0
  ran_frame  2
  sw_opt     0
  power      0
  da         0
}

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
  set libname [library get /CONFIG/PARAMETERS/Vivado/PARAMETERS/Xilinx/PARAMETERS/*/PARAMETERS/*/PARAMETERS/$part/LIBRARIES/*/NAME -match glob -ret v]
  if { [llength $libname] == 1 } {
    set libpath [library get /CONFIG/PARAMETERS/Vivado/PARAMETERS/Xilinx/PARAMETERS/*/PARAMETERS/*/PARAMETERS/$part/LIBRARIES/*/NAME -match glob -ret p]
    if { [regexp {/CONFIG/PARAMETERS/(\S+)/PARAMETERS/(\S+)/PARAMETERS/(\S+)/PARAMETERS/(\S+)/PARAMETERS/(\S+)/.*} $libpath dummy rtltool vendor family speed part] } {
      solution library add $libname -- -rtlsyntool $rtltool -vendor $vendor -family $family -speed $speed
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
  go libraries

  # special exception for SAED32 for use in power estimation
  if { $do_saed } {
    # SAED32 selected - enable DC settings to access Liberty data for power estimation
    source [application get /SYSTEM/ENV_MGC_HOME]/pkgs/siflibs/saed/setup_saedlib.tcl
  }
}

options set Input/CppStandard {c++17}
options set Input/CompilerFlags -DRANDOM_FRAMES=$opt(ran_frame)

if {$opt(reset)} {
  project load myproject_prj.ccs
  go new
} else {
  project new -name myproject_prj
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
# Ideally, the path to the weights/testbench data should be runtime configurable
# instead of compile-time WEIGHTS_DIR macro. If the nnet_helpers.h load_ functions
# are ever enhanced to take a path option then this setting can be used:
#   flow package option set /SCVerify/INVOKE_ARGS {firmware/weights tb_data}
# For now, copy weights and tb_data to the current directory
logfile message "Copying weights text file(s)\n" warning
file mkdir weights
foreach i [glob -nocomplain firmware/weights/*.txt] {
  file copy -force $i weights
}

# Turn on VSCode flow
flow package require /VSCode
# To launch VSCode on the C++ HLS design:
#   cd my-Catapult-test
#   code Catapult.code-workspace

#--------------------------------------------------------
#    Start of HLS script
solution file add firmware/myproject.cpp
solution file add myproject_test.cpp -exclude true

# Parse parameters.h to determine config info to control directives/pragmas
set IOType io_stream
if { ![file exists firmware/parameters.h] } {
  logfile message "Could not locate firmware/parameters.h. Unable to determine network configuration.\n" warning
} else {
  set pf [open "firmware/parameters.h" "r"]
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
# setup_xilinx_part xcku115-flvb2104-2-i

#  solution library add amba

# Constrain arrays to map to memory only over a certain size
directive set -MEM_MAP_THRESHOLD [expr 2048 * 16 + 1]

set hls_clock_period 5
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

project save

if {$opt(cosim) || $opt(validation)} {
  flow run /SCVerify/launch_make ./scverify/Verify_rtl_v_msim.mk {} SIMTOOL=msim sim
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
  flow run /PowerAnalysis/report_pre_pwropt_Verilog
  flow run /PowerAnalysis/report_pre_pwropt_VHDL
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

