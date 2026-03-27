#################
#    HLS4ML
#################

set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]
source [file join $tcldir build_opt.tcl]

proc remove_recursive_log_wave {} {
    set tcldir [file dirname [info script]]
    source [file join $tcldir project.tcl]

    set filename ${project_name}_prj/solution1/sim/verilog/${project_name}.tcl
    set timestamp [clock format [clock seconds] -format {%Y%m%d%H%M%S}]
    set temp     $filename.new.$timestamp
    # set backup   $filename.bak.$timestamp

    set in  [open $filename r]
    set out [open $temp     w]

    # line-by-line, read the original file
    while {[gets $in line] != -1} {
        if {[string equal "$line" "log_wave -r /"]} {
            set line { }
        }
        puts $out $line
    }

    close $in
    close $out

    # move the new data to the proper filename
    file delete -force $filename
    file rename -force $temp $filename
}

proc add_vcd_instructions_tcl {} {
    set tcldir [file dirname [info script]]
    source [file join $tcldir project.tcl]

    set filename ${project_name}_prj/solution1/sim/verilog/${project_name}.tcl
    set timestamp [clock format [clock seconds] -format {%Y%m%d%H%M%S}]
    set temp     $filename.new.$timestamp
    # set backup   $filename.bak.$timestamp

    set in  [open $filename r]
    set out [open $temp     w]

    # line-by-line, read the original file
    while {[gets $in line] != -1} {
        if {[string equal "$line" "log_wave -r /"]} {
            set line {source "../../../../project.tcl"
                if {[string equal "$backend" "vivadoaccelerator"]} {
                    current_scope [get_scopes -regex "/apatb_${project_name}_axi_top/AESL_inst_${project_name}_axi/${project_name}_U0.*"]
                    set scopes [get_scopes -regexp {layer(\d*)_.*data_0_V_U.*}]
                    append scopes { }
                    current_scope "/apatb_${project_name}_axi_top/AESL_inst_${project_name}_axi"
                    append scopes [get_scopes -regexp {(in_local_V_data.*_0_.*)}]
                    append scopes { }
                    append scopes [get_scopes -regexp {(out_local_V_data.*_0_.*)}]
                } else {
                    current_scope [get_scopes -regex "/apatb_${project_name}_top/AESL_inst_${project_name}"]
                    set scopes [get_scopes -regexp {layer(\d*)_.*data_0_V_U.*}]
                }
                open_vcd fifo_opt.vcd
                foreach scope $scopes {
                    current_scope $scope
                    if {[catch [get_objects usedw]] == 0} {
                        puts "$scope skipped"
                        continue
                    }
                    set usedw [get_objects usedw]
                    set depth [get_objects DEPTH]
                    add_wave $usedw
                    log_vcd $usedw
                    log_wave $usedw
                    add_wave $depth
                    log_vcd $depth
                    log_wave $depth
                }
            }
        }

        if {[string equal "$line" "quit"]} {
            set line {flush_vcd
                close_vcd
                quit
            }
        }
        # then write the transformed line
        puts $out $line
    }

    close $in
    close $out

    # move the new data to the proper filename
    file delete -force $filename
    file rename -force $temp $filename
}

# Generate RTL simulation JSON report from transaction file (latency and II in clock cycles)
proc generate_rtl_sim_report { project_name } {
    set transaction_file ${project_name}_prj/solution1/sim/verilog/${project_name}.performance.result.transaction.xml
    file mkdir vivado_reports
    set report_json vivado_reports/rtl_sim_${project_name}_report.json
    if {![file exists $transaction_file]} {
        puts "WARNING: Transaction file not found: $transaction_file (skipping RTL sim report)"
        return
    }
    set latency_min 0
    set latency_max 0
    set latency_sum 0
    set latency_count 0
    set ii_min 0
    set ii_max 0
    set ii_sum 0
    set ii_count 0
    set first_latency 1
    set first_ii 1
    set fh [open $transaction_file r]
    while {[gets $fh line] >= 0} {
        if {[regexp {transaction\s+\d+:\s+(\d+)\s+(\d+|x)} $line -> lat_val ii_val]} {
                if {[string is integer -strict $lat_val]} {
                    set lat [expr {int($lat_val)}]
                    if $first_latency {
                        set latency_min $lat
                        set latency_max $lat
                        set latency_sum $lat
                        set latency_count 1
                        set first_latency 0
                    } else {
                        if {$lat < $latency_min} { set latency_min $lat }
                        if {$lat > $latency_max} { set latency_max $lat }
                        set latency_sum [expr {$latency_sum + $lat}]
                        incr latency_count
                    }
                }
                if {$ii_val != "x" && [string is integer -strict $ii_val]} {
                    set ii [expr {int($ii_val)}]
                    if $first_ii {
                        set ii_min $ii
                        set ii_max $ii
                        set ii_sum $ii
                        set ii_count 1
                        set first_ii 0
                    } else {
                        if {$ii < $ii_min} { set ii_min $ii }
                        if {$ii > $ii_max} { set ii_max $ii }
                        set ii_sum [expr {$ii_sum + $ii}]
                        incr ii_count
                    }
                }
        }
    }
    close $fh
    set latency_avg 0
    if {$latency_count > 0} {
        set latency_avg [expr {double($latency_sum) / $latency_count}]
    }
    set ii_avg 0
    if {$ii_count > 0} {
        set ii_avg [expr {double($ii_sum) / $ii_count}]
    }
    set ofile [open $report_json w]
    puts $ofile "\{"
    puts $ofile "  \"transaction_count\": $latency_count,"
    puts $ofile "  \"latency\": \{"
    puts $ofile "    \"min\": $latency_min,"
    puts $ofile "    \"max\": $latency_max,"
    puts $ofile "    \"avg\": $latency_avg"
    puts $ofile "  \},"
    puts $ofile "  \"initiation_interval\": \{"
    puts $ofile "    \"min\": $ii_min,"
    puts $ofile "    \"max\": $ii_max,"
    puts $ofile "    \"avg\": $ii_avg"
    puts $ofile "  \}"
    puts $ofile "\}"
    flush $ofile
    close $ofile
    puts "INFO: RTL sim report written to $report_json"
    return $report_json
}

proc report_time { op_name time_start time_end } {
    set time_taken [expr $time_end - $time_start]
    set time_s [expr ($time_taken / 1000) % 60]
    set time_m [expr ($time_taken / (1000*60)) % 60]
    set time_h [expr ($time_taken / (1000*60*60)) % 24]
    puts "***** ${op_name} COMPLETED IN ${time_h}h${time_m}m${time_s}s *****"
}

# Compare file content: 1 = same, 0 = different
proc compare_files {file_1 file_2} {
    # Check if files exist, error otherwise
    if {! ([file exists $file_1] && [file exists $file_2])} {
        return 0
    }
    # Files with different sizes are obviously different
    if {[file size $file_1] != [file size $file_2]} {
        return 0
    }

    # String compare the content of the files
    set fh_1 [open $file_1 r]
    set fh_2 [open $file_2 r]
    set equal [string equal [read $fh_1] [read $fh_2]]
    close $fh_1
    close $fh_2
    return $equal
}

file mkdir tb_data
set CSIM_RESULTS "./tb_data/csim_results.log"
set RTL_COSIM_RESULTS "./tb_data/rtl_cosim_results.log"

if {$opt(reset)} {
    open_project -reset ${project_name}_prj
} else {
    open_project ${project_name}_prj
}
set_top ${project_name}
add_files firmware/${project_name}.cpp -cflags "-std=c++0x"
add_files -tb ${project_name}_test.cpp -cflags "-std=c++0x"
add_files -tb firmware/weights
add_files -tb tb_data
if {$opt(reset)} {
    open_solution -reset "solution1"
} else {
    open_solution "solution1"
}
catch {config_array_partition -maximum_size $maximum_size}
config_compile -name_max_length 80
set_part $part
config_schedule -enable_dsp_full_reg=false
create_clock -period $clock_period -name default
set_clock_uncertainty $clock_uncertainty default


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
}

if {$opt(cosim)} {
    puts "***** C/RTL SIMULATION *****"
    # TODO: This is a workaround (Xilinx defines __RTL_SIMULATION__ only for SystemC testbenches).
    add_files -tb ${project_name}_test.cpp -cflags "-std=c++0x -DRTL_SIM"
    set time_start [clock clicks -milliseconds]

    cosim_design -trace_level all -setup

    if {$opt(fifo_opt)} {
        puts "\[hls4ml\] - FIFO optimization started"

        if {[string equal "$backend" "vivado"] || [string equal $backend "vivadoaccelerator"]} {
            add_vcd_instructions_tcl
        }
    }

    remove_recursive_log_wave
    set old_pwd [pwd]
    cd ${project_name}_prj/solution1/sim/verilog/
    source run_sim.tcl
    cd $old_pwd

    set time_end [clock clicks -milliseconds]
    puts "INFO:"
    if {[string equal "$backend" "vivadoaccelerator"]} {
        set report_path [generate_rtl_sim_report ${project_name}_axi]
        puts [read [open $report_path r]]
    } else {
        set report_path [generate_rtl_sim_report ${project_name}]
        puts [read [open $report_path r]]
    }
    report_time "C/RTL SIMULATION" $time_start $time_end
}

if {$opt(validation)} {
    puts "***** C/RTL VALIDATION *****"
    if {[compare_files $CSIM_RESULTS $RTL_COSIM_RESULTS]} {
        puts "INFO: Test PASSED"
    } else {
        puts "ERROR: Test failed"
        puts "ERROR: - csim log:      $CSIM_RESULTS"
        puts "ERROR: - RTL-cosim log: $RTL_COSIM_RESULTS"
        exit 1
    }
}

if {$opt(export)} {
    puts "***** EXPORT IP *****"
    set time_start [clock clicks -milliseconds]
    export_design -format ip_catalog -version $version
    set time_end [clock clicks -milliseconds]
    report_time "EXPORT IP" $time_start $time_end
}

if {$opt(vsynth)} {
    puts "***** VIVADO SYNTHESIS *****"
    if {[file exist ${project_name}_prj/solution1/syn/verilog]} {
        set time_start [clock clicks -milliseconds]
        exec vivado -mode batch -source vivado_synth.tcl >@ stdout
        set time_end [clock clicks -milliseconds]
        report_time "VIVADO SYNTHESIS" $time_start $time_end
    } else {
        puts "ERROR: Cannot find generated Verilog files. Did you run C synthesis?"
        exit 1
    }
}

exit
