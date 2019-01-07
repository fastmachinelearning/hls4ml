#################
#    HLS4ML
#################
array set opt {
  csim   1
  synth  1
  cosim  1
  export 1
}

foreach arg $::argv {
  foreach o [lsort [array names opt]] {
    regexp "$o=+(\\w+)" $arg unused opt($o)
  }
}

proc report_time { op_name time_start time_end } {
  set time_taken [expr $time_end - $time_start]
  set time_s [expr ($time_taken / 1000) % 60]
  set time_m [expr ($time_taken / (1000*60)) % 60]
  set time_h [expr ($time_taken / (1000*60*60)) % 24]
  puts "***** ${op_name} COMPLETED IN ${time_h}h${time_m}m${time_s}s *****"
}

open_project -reset myproject_prj
set_top myproject
add_files firmware/myproject.cpp -cflags "-I[file normalize nnet_utils] -std=c++0x"
add_files -tb myproject_test.cpp -cflags "-I[file normalize nnet_utils] -std=c++0x"
add_files -tb firmware/weights
#add_files -tb tb_data
open_solution -reset "solution1"
catch {config_array_partition -maximum_size 4096}
set_part {xc7vx690tffg1927-2}
create_clock -period 5 -name default

if {$opt(csim)} {
  puts "***** C SIMULATION *****"
  set time_start [clock clicks -milliseconds]
  csim_design
  set time_end [clock clicks -milliseconds]
  report_time "C SIMULATION" $time_start $time_end  
}

if {$opt(synth)} {
  puts "***** C/RTL SYNTHESIS *****"
  set time_start [clock clicks -milliseconds]
  csynth_design
  set time_end [clock clicks -milliseconds]
  report_time "C/RTL SYNTHESIS" $time_start $time_end
  if {$opt(cosim)} {
    puts "***** C/RTL SIMULATION *****"
    set time_start [clock clicks -milliseconds]
    cosim_design -trace_level all
    set time_end [clock clicks -milliseconds]
    report_time "C/RTL SIMULATION" $time_start $time_end
  }
  if {$opt(export)} {
    puts "***** EXPORT IP *****"
    set time_start [clock clicks -milliseconds]
    export_design -format ip_catalog
    set time_end [clock clicks -milliseconds]
    report_time "EXPORT IP" $time_start $time_end
  }
}

exit
