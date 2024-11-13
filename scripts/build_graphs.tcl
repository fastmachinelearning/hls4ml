# ======================================================
# The script connects the output ports of each subgraph IP
# instance to the input ports of the next one in sequence,
# and makes important signals as external
#
# Run this script from the base directory containing the
# subgraph project folders (e.g., {proj_name}_graph1, etc.)
# ======================================================

puts "###########################################################"

# Project base dir
set base_dir [pwd]

# Find a directory that ends with "graph1", "graph2", etc.
set project_dirs [glob -nocomplain -directory $base_dir *graph[0-9]]

# Check if a matching directory is found
if {[llength $project_dirs] == 0} {
    puts "Error: No project directory ending with 'graph{id}' found in $base_dir"
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

puts "###########################################################"
puts "#   Starting the IP connection process...                  "
puts "###########################################################"


# Create New Vivado Project
set project_name "vivado_final_graph"
file mkdir $project_name
cd $project_name
create_project $project_name . -part $part

# Add repositories
# Initialize the repo count
set repo_count 0
# Loop through potential project directories
for {set i 1} {[file exists "$base_dir/hls4ml_prj_graph$i/myproject_graph${i}_prj"]} {incr i} {
    set repo_path "$base_dir/hls4ml_prj_graph$i/myproject_graph${i}_prj/solution1/impl/ip"
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

# Name of the block design 
set bd_name "stitched_design"
create_bd_design $bd_name

# Add IPs to block design
for {set i 1} {$i <= $repo_count} {incr i} {
    set vlnv "xilinx.com:hls:myproject_graph$i:1.0"
    create_bd_cell -type ip -vlnv $vlnv "myproject_graph${i}_0"
}

# Collect all IP instance names in a list
set ip_instances {}
for {set i 1} {$i <= $repo_count} {incr i} {
    set ip_name "myproject_graph${i}_0"
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
    # Create the 'ap_rst' port
    create_bd_port -dir I -type rst ap_rst
    set ap_rst_port [get_bd_ports ap_rst]
    
    # Set the CONFIG.POLARITY property of the 'ap_rst' port based on the retrieved polarity
    if {$rst_polarity ne ""} {
        set_property CONFIG.POLARITY $rst_polarity $ap_rst_port
    } else {
        # Fallback to ACTIVE_HIGH if the retrieved polarity is not defined
        set_property CONFIG.POLARITY ACTIVE_HIGH $ap_rst_port
    }
    # Connect all 'ap_rst' pins to the 'ap_rst' port
    foreach rst_pin $ap_rst_ports {
        connect_bd_net $ap_rst_port $rst_pin
    }
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
        set interface_type "unpacked"
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

    if {$interface_type == "unpacked"} {
        # Existing unpacked interface connection logic
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
                puts "Warning: No matching input ap_vld port found for output [get_property NAME $out_vld_port]"
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
        } else {
            puts "Warning: Could not find 'ap_done' or 'ap_start' pin for IPs $ip_i and $ip_i_plus1"
        }
    } elseif {$interface_type == "axi_stream"} {
        # Get AXI Stream interface pins from ip_i and ip_i_plus1
        set ip_i_intf_pins [get_bd_intf_pins -of $ip_i_cell]
        set ip_i_plus1_intf_pins [get_bd_intf_pins -of $ip_i_plus1_cell]

        # Initialize variables
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
        if {[string length $ip_i_axis_master] && [string length $ip_i_plus1_axis_slave]} {
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
    if {[llength $ap_start_ports] > 0} {
        create_bd_port -dir I ap_start
        set ap_start_port [get_bd_ports ap_start]
        foreach start_pin $ap_start_ports {
            connect_bd_net $ap_start_port $start_pin
        }
    }

    # Make external the input interface of the first IP
    set first_ip_cell [get_bd_cells [lindex $ip_instances 0]]
    if {[string length $first_ip_cell] == 0} {
        puts "Error: Could not find the first IP cell."
        return
    }
    set first_ip_intf_pins [get_bd_intf_pins -of $first_ip_cell]
    set first_ip_axis_slave ""
    foreach intf_pin $first_ip_intf_pins {
        set pin_name [get_property NAME $intf_pin]
        if {[string match "*s_axis*" $pin_name] || [string match "*input*" $pin_name]} {
            set first_ip_axis_slave $intf_pin
            break
        }
    }
    if {[string length $first_ip_axis_slave] > 0} {
        # Make the interface pin external
        make_bd_intf_pins_external $first_ip_axis_slave
        # Retrieve the external interface port
        set external_intf_port [get_bd_intf_ports -filter "NAME =~ \"${pin_name}*\""]
        # Change name to base_name and associate clock
        set_property NAME $pin_name $external_intf_port
        set input_pin_name $pin_name
    } else {
        puts "Error: Could not find input AXI Stream interface for first IP."
        return
    }


    # Make external the output interface of the last IP
    set last_ip_cell [get_bd_cells [lindex $ip_instances end]]
    if {[string length $last_ip_cell] == 0} {
        puts "Error: Could not find the last IP cell."
        return
    }
    set last_ip_intf_pins [get_bd_intf_pins -of $last_ip_cell]
    set last_ip_axis_master ""
    foreach intf_pin $last_ip_intf_pins {
        set pin_name [get_property NAME $intf_pin]
          if {[string match "*m_axis*" $pin_name] || [string match "*out*" $pin_name]} {
            set last_ip_axis_master $intf_pin
            break
        }
    }
    if {[string length $last_ip_axis_master] > 0} {
        # Make the interface pin external
        make_bd_intf_pins_external $last_ip_axis_master
        # Retrieve the external interface port
        set external_intf_port [get_bd_intf_ports -filter "NAME =~ \"${pin_name}*\""]
        # Change name to base_name and associate clock
        set_property NAME $pin_name $external_intf_port
        set output_pin_name $pin_name
    } else {
        puts "Error: Could not find output AXI Stream interface for last IP."
        return
    }

    # associate input and output bus interfaces to run at ap_clk
    set_property CONFIG.ASSOCIATED_BUSIF [list "${input_pin_name}:${output_pin_name}"] [get_bd_ports /ap_clk]
    
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
    
} elseif {$interface_type == "unpacked"} {
    # Make 'ap_start' of the first IP external
    set first_ip_cell [get_bd_cells [lindex $ip_instances 0]]
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

    # Make external the inputs of the first IP (including 'vld' signals)
    set first_ip_input_ports [get_bd_pins -of $first_ip_cell]
    foreach pin $first_ip_input_ports {
         set pin_name [get_property NAME $pin]
         # Match patterns for inputs and input valid pins
         if {[regexp {^\w+_input_(\d+)$} $pin_name] || [regexp {^\w+_input_(\d+)_ap_vld$} $pin_name]} {
             # Get pin properties
             set pin_dir [get_property DIR $pin]
             set pin_left [get_property LEFT $pin]
             set pin_right [get_property RIGHT $pin]
             set pin_type [get_property TYPE $pin]
             if {$pin_left ne "" && $pin_right ne ""} {
                 # Create an external port with the same name, bit range and type
                 set ext_port [create_bd_port -dir $pin_dir -from $pin_left -to $pin_right -type $pin_type $pin_name]
             } else {
                 # For single-bit signals where LEFT and RIGHT may not be defined
                 set ext_port [create_bd_port -dir $pin_dir -type $pin_type $pin_name]
             }
             connect_bd_net $ext_port $pin
         }
    }

    # Make external the outputs of the last IP (including 'vld' signals)
    set last_ip_output_ports [get_bd_pins -of $last_ip_cell]
    foreach pin $last_ip_output_ports {
        set pin_name [get_property NAME $pin]
         # Match patterns for ouputs and output valid pins
        if {[regexp {^layer(?:\d+_)?out_(\d+)$} $pin_name] || [regexp {^layer(?:\d+_)?out_(\d+)_ap_vld$} $pin_name]} {
             # Get pin properties
             set pin_dir [get_property DIR $pin]
             set pin_left [get_property LEFT $pin]
             set pin_right [get_property RIGHT $pin]
             set pin_type [get_property TYPE $pin]
             if {$pin_left ne "" && $pin_right ne ""} {
                 # Create an external port with the same name, bit range and type
                 set ext_port [create_bd_port -dir $pin_dir -from $pin_left -to $pin_right -type $pin_type $pin_name]
             } else {
                 # For single-bit signals where LEFT and RIGHT may not be defined
                 set ext_port [create_bd_port -dir $pin_dir -type $pin_type $pin_name]
             }
             connect_bd_net $ext_port $pin
        }
    }
}

save_bd_design

regenerate_bd_layout
close_project


puts "###########################################################"                                     
puts "#   Successfully connected the ports of each IP instance   "
puts "#   from '[lindex $ip_instances 0]' to '[lindex $ip_instances [expr {$repo_count - 1}]]'."
puts "#   A total of $repo_count IPs were connected.             "
puts "###########################################################"

