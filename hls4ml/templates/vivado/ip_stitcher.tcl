# ======================================================
# The script connects the output ports of each subgraph IP
# instance to the input ports of the next one in sequence,
# and makes important signals as external
#
# Run this script from the base directory containing the
# subgraph project folders (e.g., {proj_name}_graph1, etc.)
# ======================================================

puts "###########################################################"

array set opt {
    stitch_design           1
    sim_design              0
    export_design           0
    stitch_project_name     ""
    original_project_name   ""
    sim_verilog_file        ""
}

foreach arg $::argv {
    if {[regexp {([^=]+)=(.*)} $arg -> key value]} {
        if {[info exists opt($key)]} {
            set opt($key) $value
        } else {
            puts "Warning: Unknown option $key"
        }
    } else {
        puts "Warning: Ignoring argument $arg"
    }
}

set stitch_design [expr {$opt(stitch_design)}]
set sim_design [expr {$opt(sim_design)}]
set export_design [expr {$opt(export_design)}]
set sim_verilog_file $opt(sim_verilog_file)
set stitch_project_name $opt(stitch_project_name)
set original_project_name $opt(original_project_name)

# Project base dir
set base_dir [pwd]
set original_project_path "$base_dir/../../"
puts $base_dir
# Name of the block design
set bd_name "stitched_design"

# Find a directory that ends with "graph1", "graph2", etc. in the parent project folder
set project_dirs [glob -nocomplain -directory $original_project_path *graph[0-9]]

# Check if a matching directory is found
if {[llength $project_dirs] == 0} {
    puts "Error: No project directory ending with 'graph{id}' found in $original_project_path"
} else {
    # Get the first matching directory
    set project_dir [lindex $project_dirs 0]
    set project_tcl_file [file join $project_dir project.tcl]

    # Check if project.tcl exists and source it
    if {[file exists $project_tcl_file]} {
        puts "Sourcing $project_tcl_file from $project_dir"
        source $project_tcl_file
    } else {
        puts "Error: project.tcl not found in $project_dir"
        exit 1
    }
}

# Procedure for stitching the project
proc stitch_procedure {base_dir stitch_project_name original_project_name bd_name part} {

    puts "###########################################################"
    puts "#   Starting the IP connection process...                  "
    puts "###########################################################"


    # Create New Vivado Project
    create_project $stitch_project_name . -part $part -force

    # Add repositories
    # Initialize the repo count
    set repo_count 0
    # Loop through potential project directories
    for {set i 1} {[file exists "$base_dir/graph$i/${original_project_name}_graph${i}_prj"]} {incr i} {
        set repo_path "$base_dir/graph$i/${original_project_name}_graph${i}_prj/solution1/impl/ip"
        # Check if the repository path exists
        if {[file isdirectory $repo_path]} {
            # Add repository path to current project's IP repository paths
            set_property ip_repo_paths [concat [get_property ip_repo_paths [current_project]] $repo_path] [current_project]

            # Increment the repo count
            incr repo_count

            puts "Added IP repository path: $repo_path"
        } else {
            puts "Directory does not exist: $repo_path"
        }
    }

    if { $repo_count == 0 } {
        puts "No IP repositories were found in the specified directories."
    } else {
        puts "Total IP repositories added: $repo_count"
    }
    # Rescan repositories
    update_ip_catalog

    create_bd_design $bd_name

    # Add IPs to block design
    for {set i 1} {$i <= $repo_count} {incr i} {
        set vlnv "xilinx.com:hls:${original_project_name}_graph${i}:1.0"
        create_bd_cell -type ip -vlnv $vlnv "${original_project_name}_graph${i}_0"
    }

    # Collect all IP instance names in a list
    set ip_instances {}
    for {set i 1} {$i <= $repo_count} {incr i} {
        set ip_name "${original_project_name}_graph${i}_0"
        lappend ip_instances $ip_name
    }

    # Collect 'ap_clk' and 'ap_rst' signals from all IPs
    set ap_clk_ports {}
    set ap_rst_ports {}

    foreach ip $ip_instances {
        set ip_cell [get_bd_cells $ip]
        set ip_pins [get_bd_pins -of $ip_cell]
        foreach pin $ip_pins {
            set pin_name [get_property NAME $pin]
            if {[string match "ap_clk*" $pin_name]} {
                lappend ap_clk_ports $pin
            } elseif {[string match "ap_rst*" $pin_name]} {
                lappend ap_rst_ports $pin
            }
        }
    }

    # Create external ports for 'ap_clk' and 'ap_rst'
    # ap_clk
    if {[llength $ap_clk_ports] > 0} {
        set clk_freq [get_property CONFIG.FREQ_HZ [lindex $ap_clk_ports 0]]

        # Warn if modules are synthesized with different clk
        foreach clk_pin $ap_clk_ports {
            if {[get_property CONFIG.FREQ_HZ $clk_pin] ne $clk_freq} {
                puts "Warning: Inconsistent CONFIG.FREQ_HZ for ap_clk ports."
                break
            }
        }
        # NOTE: Probably we will need the lowest clock frequency among all IPs here
        create_bd_port -dir I -type clk -freq_hz 100000000 ap_clk
        set ap_clk_port [get_bd_ports ap_clk]
        # Connect all 'ap_clk' pins to the 'ap_clk' port
        foreach clk_pin $ap_clk_ports {
            connect_bd_net $ap_clk_port $clk_pin
        }
    }

    # ap_rst
    if {[llength $ap_rst_ports] > 0} {
        # Get the CONFIG.POLARITY property from one of the IP's 'ap_rst' pins
        set sample_rst_pin [lindex $ap_rst_ports 0]
        set rst_polarity [get_property CONFIG.POLARITY $sample_rst_pin]

        foreach ap_rst_port $ap_rst_ports {
            # All ports should have the same polarity
            if {[get_property CONFIG.POLARITY $ap_rst_port] ne $rst_polarity} {
                puts "Error: Inconsistent CONFIG.POLARITY for ap_rst ports. Aborting."
                exit 1
            }
        }

        # Only proceed if the polarity is defined
        if {$rst_polarity ne ""} {
            # Create the 'ap_rst' port
            set rst_port_name "ap_rst"
            create_bd_port -dir I -type rst $rst_port_name
            set ap_rst_port [get_bd_ports ap_rst]

            # Set the CONFIG.POLARITY property of the 'ap_rst' port based on the retrieved polarity
            set_property CONFIG.POLARITY $rst_polarity $ap_rst_port

            # Rename the port based on polarity
            if {$rst_polarity eq "ACTIVE_LOW"} {
                set rst_port_name "ap_rst_n"
                set_property NAME $rst_port_name $ap_rst_port
                puts "Setting reset port ap_rst_n (ACTIVE_LOW)."
            } else {
                puts "Setting reset port ap_rst (ACTIVE_HIGH)."
            }
            # Connect all 'ap_rst' pins to the 'ap_rst' port
            foreach rst_pin $ap_rst_ports {
                connect_bd_net $ap_rst_port $rst_pin
            }
        } else {
            # Fallback: Undefined polarity, no port created
            puts "Warning: CONFIG.POLARITY of ap_rst is undefined. No reset port created."
        }
    } else {
        puts "Error: No reset ports found."
    }

    # Determine interface type
    set first_ip [lindex $ip_instances 0]
    set first_ip_cell [get_bd_cells $first_ip]
    set first_ip_pins [get_bd_pins -of $first_ip_cell]

    set interface_type "unknown"
    foreach port $first_ip_pins {
        set port_name [get_property NAME $port]
        if {[string match "*_TDATA" $port_name]} {
            set interface_type "axi_stream"
            break
        } elseif {[regexp {^layer(?:\d+_)?out_(\d+)$} $port_name]} {
            set interface_type "partition"
            break
        }
    }

    if {$interface_type == "unknown"} {
        puts "Error: Could not determine interface type."
        exit 1
    } else {
        puts "Interface type detected: $interface_type"
    }

    # Collect 'ap_start' signals from all IPs
    set ap_start_ports {}
    foreach ip $ip_instances {
        set ip_cell [get_bd_cells $ip]
        set ip_pins [get_bd_pins -of $ip_cell]
        foreach pin $ip_pins {
            set pin_name [get_property NAME $pin]
            if {[string match "ap_start" $pin_name]} {
                lappend ap_start_ports $pin
            }
        }
    }

    # Loop over IP instances to connect outputs to inputs
    for {set i 0} {$i < [expr {[llength $ip_instances] - 1}]} {incr i} {
        # Get current IP and next IP
        set ip_i [lindex $ip_instances $i]
        set ip_i_plus1 [lindex $ip_instances [expr {$i + 1}]]

        # Get bd_cells for each IP
        set ip_i_cell [get_bd_cells $ip_i]
        set ip_i_plus1_cell [get_bd_cells $ip_i_plus1]

        if {$interface_type == "partition"} {
            # Existing partitioned interface connection logic
            # Get all output pins from ip_i
            set output_ports [get_bd_pins -of $ip_i_cell]

            # Initialize arrays for output ports
            array unset layer_out_ports_by_index
            array unset layer_out_vld_ports_by_index

            # Filter output ports and extract indices
            foreach port $output_ports {
                set port_name [get_property NAME $port]
                # Match 'layer_out_<index>' or 'layer<layerN>_out_<index>'
                if {[regexp {^layer(?:\d+_)?out_(\d+)$} $port_name all index]} {
                    set layer_out_ports_by_index($index) $port
                } elseif {[regexp {^layer(?:\d+_)?out_(\d+)_ap_vld$} $port_name all index]} {
                    set layer_out_vld_ports_by_index($index) $port
                } else {
                    # NOTE: We expect data ports to follow the previous naming pattern
                    # NOTE: This is not treated as an error because it might be a valid control port or non-standard signal
                }
            }

            # Get all input pins from ip_i_plus1
            set input_ports [get_bd_pins -of $ip_i_plus1_cell]

            # Initialize arrays for input ports
            array unset input_ports_by_index
            array unset input_vld_ports_by_index

            # Filter input ports and extract indices
            foreach port $input_ports {
                set port_name [get_property NAME $port]
                # Match '{name}_input_{index}'
                if {[regexp {^\w+_input_(\d+)$} $port_name all index]} {
                    set input_ports_by_index($index) $port
                } elseif {[regexp {^\w+_input_(\d+)_ap_vld$} $port_name all index]} {
                    set input_vld_ports_by_index($index) $port
                }
            }

            # Connect data signals
            foreach index [array names layer_out_ports_by_index] {
                set out_port $layer_out_ports_by_index($index)
                if {[info exists input_ports_by_index($index)]} {
                    set in_port $input_ports_by_index($index)
                    # Connect the ports
                    connect_bd_net $out_port $in_port
                } else {
                    puts "Warning: No matching input port found for output [get_property NAME $out_port]"
                }
            }

            # Connect ap_vld signals
            foreach index [array names layer_out_vld_ports_by_index] {
                set out_vld_port $layer_out_vld_ports_by_index($index)
                if {[info exists input_vld_ports_by_index($index)]} {
                    set in_vld_port $input_vld_ports_by_index($index)
                    # Connect the ports
                    connect_bd_net $out_vld_port $in_vld_port
                } else {
                    puts "Error: No matching input ap_vld port found for output [get_property NAME $out_vld_port]"
                    exit 1
                }
            }

            # Connect 'ap_done' of ip_i to 'ap_start' of ip_i_plus1
            # Get 'ap_done' pin of ip_i
            set ip_i_pins [get_bd_pins -of $ip_i_cell]
            set ap_done_pin ""
            foreach pin $ip_i_pins {
                set pin_name [get_property NAME $pin]
                if {[string match "ap_done" $pin_name]} {
                    set ap_done_pin $pin
                    break
                }
            }

            # Get 'ap_start' pin of ip_i_plus1
            set ip_i_plus1_pins [get_bd_pins -of $ip_i_plus1_cell]
            set ap_start_pin ""
            foreach pin $ip_i_plus1_pins {
                set pin_name [get_property NAME $pin]
                if {[string match "ap_start" $pin_name]} {
                    set ap_start_pin $pin
                    break
                }
            }

            # Connect 'ap_done' of ip_i to 'ap_start' of ip_i_plus1
            if {[string length $ap_done_pin] > 0 && [string length $ap_start_pin] > 0} {
                connect_bd_net $ap_done_pin $ap_start_pin
                puts "Connected 'ap_done' of $ip_i to 'ap_start' of $ip_i_plus1"
            } else {
                puts "Warning: Could not find 'ap_done' or 'ap_start' pin for IPs $ip_i and $ip_i_plus1"
            }
        } elseif {$interface_type == "axi_stream"} {
            # Get AXI Stream interface pins from ip_i and ip_i_plus1
            set ip_i_intf_pins [get_bd_intf_pins -of $ip_i_cell]
            set ip_i_plus1_intf_pins [get_bd_intf_pins -of $ip_i_plus1_cell]
            set ip_i_axis_master ""
            set ip_i_plus1_axis_slave ""

            # Identify the Master (output) AXI Stream interface of ip_i
            foreach intf_pin $ip_i_intf_pins {
                set pin_name [get_property NAME $intf_pin]
                # Assuming output interfaces have names ending with 'out'
                if {[string match "*out" $pin_name]} {
                    set ip_i_axis_master $intf_pin
                    break
                }
            }

            # Identify the Slave (input) AXI Stream interface of ip_i_plus1
            foreach intf_pin $ip_i_plus1_intf_pins {
                set pin_name [get_property NAME $intf_pin]
                # Assuming input interfaces have names ending with 'input'
                if {[string match "*input" $pin_name]} {
                    set ip_i_plus1_axis_slave $intf_pin
                    break
                }
            }

            # Check if both interfaces are found
            if {[string length $ip_i_axis_master] > 0 && [string length $ip_i_plus1_axis_slave] > 0} {
                # Connect the AXI Stream interfaces
                connect_bd_intf_net $ip_i_axis_master $ip_i_plus1_axis_slave
                puts "Connected AXI Stream interface between $ip_i and $ip_i_plus1"
            } else {
                puts "Warning: Could not find matching AXI Stream interfaces for $ip_i and $ip_i_plus1"
            }
        }
    }

    if {$interface_type == "axi_stream"} {
        # Create external port for 'ap_start' and connect all 'ap_start' pins
        # ap_start in streaming IPs needs to be constantly high
        if {[llength $ap_start_ports] > 0} {
            create_bd_port -dir I ap_start
            set ap_start_port [get_bd_ports ap_start]
            foreach start_pin $ap_start_ports {
                connect_bd_net $ap_start_port $start_pin
            }
        }

        # Make external all input interfaces of the first IP
        set first_ip_cell [get_bd_cells [lindex $ip_instances 0]]
        if {[string length $first_ip_cell] == 0} {
            puts "Error: Could not find the first IP cell."
            return
        }
        set first_ip_intf_pins [get_bd_intf_pins -of $first_ip_cell]
        set input_pin_names {}
        foreach intf_pin $first_ip_intf_pins {
            set intf_mode [get_property MODE $intf_pin]
            set vlnv [get_property VLNV $intf_pin]
            if {$intf_mode eq "Slave" && [string match "*:axis_rtl:*" $vlnv]} {
                # Make the interface pin external
                make_bd_intf_pins_external $intf_pin
                set pin_name [get_property NAME $intf_pin]
                # Retrieve the external interface port
                set external_intf_port [get_bd_intf_ports -filter "NAME =~ \"${pin_name}*\""]
                # Change name to base_name
                set_property NAME $pin_name $external_intf_port
                lappend input_pin_names $pin_name
            }
        }
        if {[llength $input_pin_names] == 0} {
            puts "Error: Could not find any input AXI Stream interfaces for first IP."
            return
        }

        # Make external all output interfaces of the last IP
        set last_ip_cell [get_bd_cells [lindex $ip_instances end]]
        if {[string length $last_ip_cell] == 0} {
            puts "Error: Could not find the last IP cell."
            return
        }
        set last_ip_intf_pins [get_bd_intf_pins -of $last_ip_cell]
        set output_pin_names {}
        foreach intf_pin $last_ip_intf_pins {
            set intf_mode [get_property MODE $intf_pin]
            set vlnv [get_property VLNV $intf_pin]
            if {$intf_mode eq "Master" && [string match "*:axis_rtl:*" $vlnv]} {
                # Make the interface pin external
                make_bd_intf_pins_external $intf_pin
                set pin_name [get_property NAME $intf_pin]
                # Retrieve the external interface port and change name to base name
                set external_intf_port [get_bd_intf_ports -filter "NAME =~ \"${pin_name}*\""]
                set_property NAME $pin_name $external_intf_port
                lappend output_pin_names $pin_name
            }
        }
        if {[llength $output_pin_names] == 0} {
            puts "Error: Could not find any output AXI Stream interfaces for last IP."
            return
        }

        # Associate input, output, and ap_rst to run at 'ap_clk'
        # Join interface names with colons to match the required format
        set associated_busif [join [concat $input_pin_names $output_pin_names] ":"]
        set_property CONFIG.ASSOCIATED_BUSIF {$associated_busif} [get_bd_ports /ap_clk]
        set_property CONFIG.ASSOCIATED_RESET $rst_port_name [get_bd_ports /ap_clk]

        # Make external the 'ap_done' signal of the last IP
        set last_ip_pins [get_bd_pins -of $last_ip_cell]
        set last_ap_done_pin ""
        foreach pin $last_ip_pins {
            set pin_name [get_property NAME $pin]
            if {[string match "ap_done" $pin_name]} {
                set last_ap_done_pin $pin
                break
            }
        }
        if {[string length $last_ap_done_pin] > 0} {
            create_bd_port -dir O ap_done
            set ap_done_port [get_bd_ports ap_done]
            connect_bd_net $ap_done_port $last_ap_done_pin
        } else {
            puts "Warning: Could not find 'ap_done' pin for last IP"
        }

    } elseif {$interface_type == "partition"} {
        # Make 'ap_start' of the first IP external
        set first_ip_cell [get_bd_cells [lindex $ip_instances 0]]
        if {[string length $first_ip_cell] == 0} {
            puts "Error: Could not find the first IP cell."
            return
        }
        set first_ip_pins [get_bd_pins -of $first_ip_cell]
        set first_ap_start_pin ""
        foreach pin $first_ip_pins {
            set pin_name [get_property NAME $pin]
            if {[string match "ap_start" $pin_name]} {
                set first_ap_start_pin $pin
                break
            }
        }
        if {[string length $first_ap_start_pin] > 0} {
            create_bd_port -dir I ap_start
            set ap_start_port [get_bd_ports ap_start]
            connect_bd_net $ap_start_port $first_ap_start_pin
        } else {
            puts "Warning: Could not find 'ap_start' pin for first IP"
        }

        # Make 'ap_done' of the last IP external
        set last_ip_cell [get_bd_cells [lindex $ip_instances end]]
        if {[string length $last_ip_cell] == 0} {
            puts "Error: Could not find the last IP cell."
            return
        }
        set last_ip_pins [get_bd_pins -of $last_ip_cell]
        set last_ap_done_pin ""
        foreach pin $last_ip_pins {
            set pin_name [get_property NAME $pin]
            if {[string match "ap_done" $pin_name]} {
                set last_ap_done_pin $pin
                break
            }
        }
        if {[string length $last_ap_done_pin] > 0} {
            create_bd_port -dir O ap_done
            set ap_done_port [get_bd_ports ap_done]
            connect_bd_net $ap_done_port $last_ap_done_pin
        } else {
            puts "Warning: Could not find 'ap_done' pin for last IP"
        }

        set control_pins {ap_clk ap_rst ap_start ap_done ap_idle ap_ready}

        # Make external all inputs of the first IP (including 'vld' signals)
        set input_pin_names {}
        foreach pin $first_ip_pins {
            set pin_name [get_property NAME $pin]
            set pin_dir [get_property DIR $pin]
            # Match patterns for inputs and input valid pins
            if {$pin_dir eq "I" && [lsearch -exact $control_pins $pin_name] == -1} {
                puts "Found NN model input pin: $pin_name"

                # Make the pin external
                make_bd_pins_external $pin
                # Retrieve the external port and change name to base name
                set external_port [get_bd_ports -filter "NAME =~ \"${pin_name}*\""]
                set_property NAME $pin_name $external_port
                lappend input_pin_names $pin_name
            }
        }
        if {[llength $input_pin_names] == 0} {
            puts "Error: Could not find any input pins for first IP."
            return
        }

        # Make external all outputs of the last IP (including 'vld' signals)
        set output_pin_names {}
        foreach pin $last_ip_pins {
            set pin_name [get_property NAME $pin]
            set pin_dir [get_property DIR $pin]
            # Match patterns for outputs and output valid pins
            if {$pin_dir eq "O" && [lsearch -exact $control_pins $pin_name] == -1} {
                puts "Found NN model output pin: $pin_name"
                # Make the pin external
                make_bd_pins_external $pin
                # Retrieve the external port and change name to base name
                set external_port [get_bd_ports -filter "NAME =~ \"${pin_name}*\""]
                set_property NAME $pin_name $external_port
                lappend output_pin_names $pin_name
            }
        }
        if {[llength $output_pin_names] == 0} {
            puts "Error: Could not find any output pins for last IP."
            return
        }
    }

    validate_bd_design

    regenerate_bd_layout

    save_bd_design

    puts "###########################################################"
    puts "#   Successfully connected the ports of each IP instance   "
    puts "#   A total of $repo_count IPs were connected.             "
    puts "###########################################################"

}

if {$stitch_design} {
    set start_time [clock seconds]
    stitch_procedure $original_project_path $stitch_project_name $original_project_name $bd_name $part
    set end_time [clock seconds]
    set elapsed_time [expr {$end_time - $start_time}]
    puts "====================================================="
    puts "\[Stitch\] Elapsed Time : $elapsed_time seconds"
    puts "====================================================="
} else {
    #set existing_stitch_project_name [file join $stitch_project_name "$stitch_project_name.xpr"]
    if {[file exists "$stitch_project_name.xpr"]} {
        puts "Opening existing project: $stitch_project_name.xpr"
        open_project "$stitch_project_name.xpr"
    } else {
        puts "Error: Project file "$stitch_project_name.xpr" does not exist."
        exit 1
    }
}

if {$export_design} {
    set start_time [clock seconds]
    puts "Exporting stitched IP..."
    set stitched_ip_dir "ip_repo"
    ipx::package_project -root_dir $stitched_ip_dir \
        -vendor user.org -library user -taxonomy /UserIP -module $bd_name \
        -import_files
    set_property description "This IP core integrates all NN subgraph IPs into one." [ipx::find_open_core user.org:user:stitched_design:1.0]
    set_property core_revision 2 [ipx::find_open_core user.org:user:stitched_design:1.0]
    ipx::create_xgui_files [ipx::find_open_core user.org:user:stitched_design:1.0]
    ipx::update_checksums [ipx::find_open_core user.org:user:stitched_design:1.0]
    ipx::check_integrity [ipx::find_open_core user.org:user:stitched_design:1.0]
    ipx::save_core [ipx::find_open_core user.org:user:stitched_design:1.0]
    puts "Stitched IP has been exported to '$stitched_ip_dir' folder"
    puts "====================================================="
    puts "\[Export\] Elapsed Time : $elapsed_time seconds"
    puts "====================================================="
}

if {$sim_design} {
    set start_time [clock seconds]
    if {$sim_verilog_file == ""} {
        puts "Error: sim_verilog_file not provided."
        exit 1
    }
    if {![file exists "$base_dir/$sim_verilog_file"]} {
        puts "Error: Simulation file not found: $base_dir/$sim_verilog_file"
        exit 1
    }
    if {[llength [get_filesets sim_1]] == 0} {
        create_fileset -simset sim_1
    }
    set_property SOURCE_SET sources_1 [get_filesets sim_1]
    add_files -fileset sim_1 -norecurse -scan_for_includes "$base_dir/$sim_verilog_file"
    update_compile_order -fileset sim_1
    puts "Simulation Verilog file added: $base_dir/$sim_verilog_file"
    set_property top tb_design_1_wrapper [get_filesets sim_1]
    set_property -name {xsim.simulate.runtime} -value {1000000ns} -objects [get_filesets sim_1]

   # Check if snapshot already exists
    set snapshot_name "tb_design_1_wrapper_behav"
    set xsim_folder_path "${base_dir}/vivado_stitched_design.sim/sim_1/behav/xsim"
    puts "##########################"
    puts "#  Running Simulation... #"
    puts "##########################"
    if {[file exists "${xsim_folder_path}/${snapshot_name}.wdb"]} {
        puts "Using existing snapshot..."
        cd $xsim_folder_path
        exec xsim $snapshot_name -R
    } else {
        launch_simulation
    }
    set end_time [clock seconds]
    set elapsed_time [expr {$end_time - $start_time}]
    puts "====================================================="
    puts "\[Simulation\] Elapsed Time : $elapsed_time seconds"
    puts "====================================================="
}


close_project
