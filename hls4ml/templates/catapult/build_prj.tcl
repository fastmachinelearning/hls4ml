#################
#    HLS4ML
#################
array set opt {
  reset      0
  csim       0
  synth      1
  cosim      0
  validation 0
  export     0
  vsynth     0
}

if { [info exists ::argv] } {
  foreach arg $::argv {
    foreach {optname optval} [split $arg '='] {}
    if { [info exists opt($optname)] } {
      set opt($optname) [string is true -strict $optval]
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

options set Input/CppStandard {c++17}

if {$opt(reset)} {
  project load myproject_prj.ccs
  go new
} else {
  project new -name myproject_prj
}

if { [info exists ::env(XILINX_PCL_CACHE)] } {
options set /Flows/Vivado/PCL_CACHE $::env(XILINX_PCL_CACHE)
solution options set /Flows/Vivado/PCL_CACHE $::env(XILINX_PCL_CACHE)
}

# downgrade HIER-10
options set Message/ErrorOverride HIER-10 -remove
solution options set Message/ErrorOverride HIER-10 -remove

solution file add firmware/myproject.cpp
solution file add myproject_test.cpp -exclude true

# Copy weights and tb_data
logfile message "Copying weights text file(s)\n" warning
file mkdir weights
foreach i [glob -nocomplain firmware/weights/*.txt] {
  file copy -force $i weights
}

# Parse parameters.h to determine config info to control directives/pragmas
set IOType io_stream
if { ![file exists firmware/parameters.h] } {
  logfile message "Could not locate firmware/parameters.h. Unable to determine network configuration.\n" warning
} else {
  set pf [open "firmware/parameters.h" "r"]
  while {![eof $pf]} {
    gets $pf line
    if { [string match {*io_type = nnet::io_stream*} $line] } {
      set IoType io_stream
      break
    }
  }
  close $pf
}

if { $IOType == "io_stream" } {
solution options set Architectural/DefaultRegisterThreshold 2050
}

flow package require /SCVerify
# Ideally, the path to the weights/testbench data should be runtime configurable
# instead of compile-time WEIGHTS_DIR macro. If the nnet_helpers.h load_ functions
# are ever enhanced to take a path option then this setting can be used:
#   flow package option set /SCVerify/INVOKE_ARGS {firmware/weights tb_data}


go compile

if {$opt(csim)} {
  puts "***** C SIMULATION *****"
  set time_start [clock clicks -milliseconds]
  flow run /SCVerify/launch_make ./scverify/Verify_orig_cxx_osci.mk {} SIMTOOL=osci sim
  set time_end [clock clicks -milliseconds]
  report_time "C SIMULATION" $time_start $time_end
}

if {$opt(synth)} {
  puts "***** C/RTL SYNTHESIS *****"
  set time_start [clock clicks -milliseconds]

  solution library add mgc_Xilinx-KINTEX-u-2_beh -- -rtlsyntool Vivado -manufacturer Xilinx -family KINTEX-u -speed -2 -part xcku115-flvb2104-2-i
  solution library add Xilinx_RAMS
  solution library add Xilinx_ROMS
  solution library add Xilinx_FIFO
#  solution library add amba

# Constrain arrays to map to memory only over a certain size
  directive set -MEM_MAP_THRESHOLD [expr 2048 * 16 + 1]

  directive set -CLOCKS {clk {-CLOCK_PERIOD 5 -CLOCK_EDGE rising -CLOCK_HIGH_TIME 2.5 -CLOCK_OFFSET 0.000000 -CLOCK_UNCERTAINTY 0.0 -RESET_KIND sync -RESET_SYNC_NAME rst -RESET_SYNC_ACTIVE high -RESET_ASYNC_NAME arst_n -RESET_ASYNC_ACTIVE low -ENABLE_NAME {} -ENABLE_ACTIVE high}}

  go assembly

  # Specifically for io_stream
  if { $IOType == "io_stream" } {
    # Bug in Catapult - hls_resource inserted by catapult_writer.py not applied to static var. workaround is placed in build_prj.tcl
    directive set /myproject/layer*_out:cns -match glob -MAP_TO_MODULE ccs_ioport.ccs_pipe
    #directive set /myproject/layer*_out:cns -match glob -MAP_TO_MODULE Xilinx_FIFO.FIFO_SYNC

#    # Pipeline init interval for dense_wrapper (should be a pragma in nnet_dense_stream.h)
#    directive set /myproject/nnet::dense*/core/nnet::dense* -match glob -PIPELINE_INIT_INTERVAL 1

    # Workaround for pipeline init interval for streaming dense() function - the conditional pragmas
    # at nnet_utils/nnet_dense_stream.h:43 and 60 do not seem to be honored.
    catch {directive set /myproject/nnet::dense*/core/main -match glob -PIPELINE_INIT_INTERVAL 1}

if { 0 } {
    # Control internal interfaces
    directive set /myproject/nnet::dense*/data_stream:rsc -match glob -MAP_TO_MODULE ccs_ioport.ccs_in_wait
    directive set /myproject/nnet::dense*/res_stream:rsc -match glob -MAP_TO_MODULE ccs_ioport.ccs_out_wait
    foreach {p v} [solution get /TOP/PARTITIONS/*/name -checkpath 0 -match glob -ret pv] {
      if { $v == "myproject:core" } {
        foreach {p1 v1} [solution get [file dirname $p]/INTERFACE/*/DATAMODE -checkpath 0 -match glob -ret pv] {
          # Skip clock/reset
          if { [solution get [file dirname $p1]/FF_CONTROL -checkpath 0 -match glob -ret v] != {} } { continue }
          set varname [solution get [file dirname $p1]/name -checkpath 0]
          # Skip non-hls4ml layer interconnect names
          if { ![string match {layer*_out} $varname] } { continue }
          set rscname [solution get [file dirname $p1]/PROPERTIES/RESOURCE/VALUE -checkpath 0]
          switch -- $v1 {
            "IN" { directive set /myproject/myproject:core/$rscname -match glob -MAP_TO_MODULE ccs_ioport.ccs_in_wait }
            "OUT" { directive set /myproject/myproject:core/$rscname -match glob -MAP_TO_MODULE ccs_ioport.ccs_out_wait }
          }
        }
        break
      }
    }

    # Control internal arrays
    directive set /myproject/nnet::dense*/core/data:rsc -match glob -MAP_TO_MODULE {[Register]}
    directive set /myproject/nnet::dense*/core/res:rsc -match glob -MAP_TO_MODULE {[Register]}
    directive set /myproject/nnet::dense*/core/DataPrepare*:rsc -match glob -MAP_TO_MODULE {[Register]}
    directive set /myproject/nnet::dense*/core/*:mult:rsc -match glob -MAP_TO_MODULE {[Register]}
    directive set /myproject/nnet::dense*/core/*:acc:rsc -match glob -MAP_TO_MODULE {[Register]}
    directive set /myproject/nnet::dense*/core/ResWrite*:rsc -match glob -MAP_TO_MODULE {[Register]}

    directive set /myproject/myproject:core/core/ReLUActLoop*:rsc -match glob -MAP_TO_MODULE {[Register]}
}

  }
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
# Not yet implemented
#  flow package option set /Vivado/BoardPart xilinx.com:zcu102:part0:3.1
#  flow package option set /Vivado/IP_Taxonomy {/Catapult}
#  flow run /Vivado/launch_package_ip -shell ./vivado_concat_v/concat_v_package_ip.tcl
  set time_end [clock clicks -milliseconds]
  report_time "EXPORT IP" $time_start $time_end
}

if {$opt(vsynth)} {
  puts "***** VIVADO SYNTHESIS *****"
  set time_start [clock clicks -milliseconds]
  flow run /Vivado/synthesize -shell vivado_concat_v/concat_rtl.v.xv
  set time_end [clock clicks -milliseconds]
  report_time "VIVADO SYNTHESIS" $time_start $time_end
}

