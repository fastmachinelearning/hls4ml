# ======================================================
# The script connects the output ports of each subgraph IP
# instance to the input ports of the next one in sequence.
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
set bd_name "design_1"
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

save_bd_design

regenerate_bd_layout
close_project

puts "###########################################################"                                     
puts "#   Successfully connected the ports of each IP instance   "
puts "#   from '[lindex $ip_instances 0]' to '[lindex $ip_instances [expr {$repo_count - 1}]]'."
puts "#   A total of $repo_count IPs were connected.             "
puts "###########################################################"
