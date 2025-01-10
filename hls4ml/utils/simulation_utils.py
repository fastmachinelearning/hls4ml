import os
from lxml import etree
import json
import numpy as np
import pandas as pd 

def parse_component_xml(component_xml_path):
    """
    Parse the given component.xml file and return structured information
    about the input and output ports.

    Returns:
        inputs (list): A list of dicts, each containing 'name', 'direction', and 'width' for input ports.
        outputs (list): A list of dicts, each containing 'name', 'direction', and 'width' for output ports.
    """
    if not os.path.exists(component_xml_path):
        raise FileNotFoundError(f"component.xml not found at {component_xml_path}")

    # Parse the XML file
    tree = etree.parse(component_xml_path)
    root = tree.getroot()

    # Define the namespaces
    ns = {
        'spirit': 'http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009',
        'xilinx': 'http://www.xilinx.com',
        'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
    }

    # Extract ports
    ports = root.findall('.//spirit:model/spirit:ports/spirit:port', namespaces=ns)
    inputs = []
    outputs = []

    for port in ports:
        name = port.find('spirit:name', namespaces=ns).text
        wire = port.find('spirit:wire', namespaces=ns)
        if wire is not None:
            direction = wire.find('spirit:direction', namespaces=ns).text
            vector = wire.find('spirit:vector', namespaces=ns)
            if vector is not None:
                left = vector.find('spirit:left', namespaces=ns).text
                right = vector.find('spirit:right', namespaces=ns).text
                width = abs(int(left) - int(right)) + 1
            else:
                width = 1

            port_info = {'name': name, 'direction': direction, 'width': width}
            if direction == 'in':
                inputs.append(port_info)
            elif direction == 'out':
                outputs.append(port_info)

    return inputs, outputs


def write_verilog_testbench(nn_config, testbench_output_path):
    """
    Generate a Verilog testbench for a given neural network configuration.
    The testbench includes:
      - Clock and reset logic
      - DUT instantiation and AXI4-Stream interfaces
      - Stimulus generation for inputs
      - Data capture and logging for outputs
      - Latency measurement
    """
    inputs = nn_config['inputs']
    outputs = nn_config['outputs']

    input_signals = []
    output_signals = []

    # Collect input signals (name and total bitwidth)
    for input_item in inputs:
        total_bits = input_item['integer_bits'] + input_item['fractional_bits']
        input_signals.append((input_item['name'], total_bits))

    # Collect output signals (name and total bitwidth)
    for output_item in outputs:
        total_bits = output_item['integer_bits'] + output_item['fractional_bits']
        output_signals.append((output_item['name'], total_bits))

    with open(testbench_output_path, 'w') as f:
        #----------------------------------------------------------------------
        # Header and Module Declaration
        #----------------------------------------------------------------------
        f.write('`timescale 1ns / 1ps\n\n')
        f.write('module tb_design_1_wrapper;\n\n')

        #----------------------------------------------------------------------
        # Clock and Reset Signals
        #----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // Clock and Reset Signals\n')
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    reg ap_clk;\n')
        f.write('    reg ap_rst_n;\n\n')

        #----------------------------------------------------------------------
        # Control and Handshaking Signals
        #----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // Control and Handshaking Signals\n')
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    reg  ap_start;\n')
        f.write('    wire ap_done;\n\n')

        #----------------------------------------------------------------------
        # AXI4-Stream Input Interfaces
        #----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // AXI4-Stream Input Interfaces\n')
        f.write('    //------------------------------------------------------------------------\n')

        for layer in nn_config['inputs']:
            total_bits = layer['integer_bits'] + layer['fractional_bits']
            batch_size = layer['batch_size']
            f.write(f'    reg  [{(total_bits * batch_size) - 1}:0] {layer["name"]}_tdata;\n')
            f.write(f'    reg  {layer["name"]}_tvalid;\n')
            f.write(f'    wire {layer["name"]}_tready;\n\n')

        #----------------------------------------------------------------------
        # AXI4-Stream Output Interfaces
        #----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // AXI4-Stream Output Interfaces\n')
        f.write('    //------------------------------------------------------------------------\n')

        for layer in nn_config['outputs']:
            total_bits = layer['integer_bits'] + layer['fractional_bits']
            batch_size = layer['batch_size']
            f.write(f'    wire [{(total_bits * batch_size) - 1}:0] {layer["name"]}_tdata;\n')
            f.write(f'    wire {layer["name"]}_tvalid;\n')
            f.write(f'    reg  {layer["name"]}_tready;\n\n')

        #----------------------------------------------------------------------
        # DUT Instantiation
        #----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // DUT Instantiation\n')
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    stitched_design dut (\n')
        f.write('        .ap_clk(ap_clk),\n')
        f.write('        .ap_done(ap_done),\n')
        f.write('        .ap_rst_n(ap_rst_n),\n')
        f.write('        .ap_start(ap_start),\n')

        # Connect input interfaces
        for layer in nn_config['inputs']:
            name = layer["name"]
            f.write(f'        .{name}_tdata({name}_tdata),\n')
            f.write(f'        .{name}_tready({name}_tready),\n')
            f.write(f'        .{name}_tvalid({name}_tvalid),\n')

        # Connect output interfaces (all but last have trailing comma)
        for layer in nn_config['outputs'][:-1]:
            name = layer["name"]
            f.write(f'        .{name}_tdata({name}_tdata),\n')
            f.write(f'        .{name}_tready({name}_tready),\n')
            f.write(f'        .{name}_tvalid({name}_tvalid),\n')

        # Last output interface (no trailing comma)
        last_output_layer = nn_config['outputs'][-1]
        name = last_output_layer["name"]
        f.write(f'        .{name}_tdata({name}_tdata),\n')
        f.write(f'        .{name}_tready({name}_tready),\n')
        f.write(f'        .{name}_tvalid({name}_tvalid)\n')
        f.write('    );\n\n')

        #----------------------------------------------------------------------
        # Clock Generation
        #----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // Clock Generation (100 MHz => 10 ns period)\n')
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    initial begin\n')
        f.write('        ap_clk = 0;\n')
        f.write('        forever #5 ap_clk = ~ap_clk;\n')
        f.write('    end\n\n')

        #----------------------------------------------------------------------
        # Reset Generation
        #----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // Reset Generation\n')
        f.write('    // Wait for a few cycles and then release reset.\n')
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    initial begin\n')
        f.write('        ap_rst_n = 0;\n')
        f.write('        repeat (5) @(posedge ap_clk);\n')
        f.write('        ap_rst_n = 1;\n')
        f.write('    end\n\n')

        #----------------------------------------------------------------------
        # Signal Initialization
        #----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // Signal Initialization\n')
        f.write('    // Initialize control signals, input valid, and output ready.\n')
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    initial begin\n')
        f.write('        ap_start = 0;\n')
        for name, _ in input_signals:
            f.write(f'        {name}_tvalid = 0;\n')
        for name, _ in output_signals:
            f.write(f'        {name}_tready = 1;\n')
        f.write('    end\n\n')

        #----------------------------------------------------------------------
        # Variables for Logging and Measurement
        #----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // Logging and Measurement Variables\n')
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    integer csv_file;\n')
        f.write('    integer file, r, value;\n')
        f.write('    integer j;\n')
        f.write('    integer total_bits;\n')
        f.write('    reg [63:0] cycle_count = 0;\n')
        f.write('    reg [63:0] start_cycle = 0;\n')
        f.write('    reg [63:0] end_cycle = 0;\n')
        f.write('    reg [1:0] done_counter = 0;\n')
        f.write('    reg       old_ap_done = 0;\n\n')

        #----------------------------------------------------------------------
        # Cycle Counting
        #----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // Cycle Counting\n')
        f.write('    // Count cycles to measure latency.\n')
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    always @(posedge ap_clk) begin\n')
        f.write('        if (!ap_rst_n)\n')
        f.write('            cycle_count <= 0;\n')
        f.write('        else\n')
        f.write('            cycle_count <= cycle_count + 1;\n')
        f.write('    end\n\n')

        #----------------------------------------------------------------------
        # Data Transmission (Stimulus Generation)
        #----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // Data Transmission (Stimulus)\n')
        f.write('    // Send input patterns to the DUT.\n')
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    initial begin\n')
        f.write('        // Wait until reset is de-asserted\n')
        f.write('        wait (ap_rst_n == 1);\n')
        f.write('        repeat (2) @(posedge ap_clk);\n\n')

        f.write('        // Open CSV log file\n')
        f.write('        csv_file = $fopen("../../../../testbench_log.csv", "w");\n')
        f.write('        if (csv_file == 0) begin\n')
        f.write('            $display("ERROR: Could not open CSV log file.");\n')
        f.write('            $finish;\n')
        f.write('        end\n')
        f.write('        $fwrite(csv_file, "output_name,index,value\\n");\n\n')

        f.write('        // Start the DUT\n')
        f.write('        ap_start = 1;\n\n')

        # Send first pattern of inputs (all zeroes)
        for layer in nn_config['inputs']:
            i_bits = layer["integer_bits"]
            f_bits = layer["fractional_bits"]
            total_bits = i_bits + f_bits
            batch_size = layer['batch_size']
            fifo_depth = layer["fifo_depth"]
            name = layer["name"]
            f.write(f'        // Sending 1st patern of inputs for {name}\n')
            f.write(f'        {name}_tvalid = 1;\n')
            f.write(f'        for (j = 0; j < {fifo_depth}; j = j + 1) begin\n')
            for k in range(batch_size):
                upper = (k + 1) * total_bits - 1
                lower = k * total_bits
                f.write(f'            {name}_tdata[{upper}:{lower}] = 0;\n')
            f.write(f'            while ({name}_tready == 0) @(posedge ap_clk);\n')
            f.write('            @(posedge ap_clk);\n')
            f.write('        end\n')
            f.write(f'        {name}_tvalid = 0;\n\n')

        # Send second pattern of inputs (read from file)
        for layer in nn_config['inputs']:
            i_bits = layer["integer_bits"]
            f_bits = layer["fractional_bits"]
            total_bits = i_bits + f_bits
            batch_size = layer['batch_size']
            fifo_depth = layer["fifo_depth"]
            name = layer["name"]
            input_file = f"{name}_input_data.txt"
            f.write(f'        // Sending 2nd pattern of inputs for {name}\n')
            f.write(f'        {name}_tvalid = 1;\n')
            f.write(f'        file = $fopen("../../../../{input_file}", "r");\n')
            f.write(f'        if (file == 0) begin\n')
            f.write(f'            $display("Error opening file {input_file}");\n')
            f.write(f'            $finish;\n')
            f.write(f'        end\n')
            f.write(f'        for (j = 0; j < {fifo_depth}; j = j + 1) begin\n')
            # For each line, read batch_size values:
            for k in range(batch_size):
                upper = (k + 1) * total_bits - 1
                lower = k * total_bits
                f.write(f'            r = $fscanf(file, "%d", value);\n')
                f.write(f'            {name}_tdata[{upper}:{lower}] = value;\n')
            f.write(f'            while ({name}_tready == 0) @(posedge ap_clk);\n')
            f.write('            @(posedge ap_clk);\n')
            f.write('        end\n')

        f.write('    end\n\n')

        #----------------------------------------------------------------------
        # Output Data Capture and Logging
        #----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // Output Data Capture and Logging\n')
        f.write('    // Capture output for 2nd input (done_counter == 1) and log them to CSV.\n')
        f.write('    //------------------------------------------------------------------------\n\n')

        for i, layer in enumerate(nn_config['outputs']):
            i_bits = layer['integer_bits']
            f_bits = layer['fractional_bits']
            total_bits = i_bits + f_bits
            layer_name = layer["name"]

            f.write(f'    //Output capture for {layer_name}\n')
            f.write(f'    integer idx_{i};\n')
            f.write(f'    reg signed [{total_bits-1}:0] fixed_val_{i};\n')
            f.write(f'    real real_val_{i};\n')
            f.write(f'    always @(posedge ap_clk) begin\n')
            f.write(f'        if (done_counter == 1 && {layer_name}_tvalid && {layer_name}_tready) begin\n')
            f.write(f'            for (idx_{i} = 0; idx_{i} < {layer["batch_size"]}; idx_{i} = idx_{i} + 1) begin\n')
            f.write(f'                fixed_val_{i} = {layer_name}_tdata[(idx_{i}+1)*{total_bits}-1 -: {total_bits}];\n')
            f.write(f'                real_val_{i}  = fixed_val_{i} / (1.0 * (1 << {f_bits}));\n')
            f.write(f'                $display("Output {layer_name}[%0d]: integer_bits=%0d fractional_bits=%0d value=%f", idx_{i}, {i_bits}, {f_bits}, real_val_{i});\n')
            f.write('                // Log result to CSV\n')
            f.write(f'                $fwrite(csv_file, "%s,%0d,%f\\n", "{layer_name}", idx_{i}, real_val_{i});\n')
            f.write('            end\n')
            f.write('        end\n')
            f.write('    end\n\n')

        #----------------------------------------------------------------------
        # Latency Measurement and Test End
        #----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // Latency Measurement\n')
        f.write('    // Measures the cycle count between start and subsequent ap_done signals.\n')
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    always @(posedge ap_clk) begin\n')
        f.write('        if (!ap_rst_n) begin\n')
        f.write('            old_ap_done <= 0;\n')
        f.write('        end else begin\n')
        f.write('            old_ap_done <= ap_done;\n')
        f.write('            // Detect rising edge of ap_done\n')
        f.write('            if (ap_done && !old_ap_done) begin\n')
        f.write('                done_counter <= done_counter + 1;\n')
        f.write('                if (done_counter == 0) begin\n')
        f.write('                    start_cycle = cycle_count;\n')
        f.write('                    $display("Worst latency (first input set): %0d cycles", cycle_count);\n')
        f.write('                    $fwrite(csv_file, "%s,%0d,%0d\\n", "WorstLatency", 0, cycle_count);\n')
        f.write('                end else if (done_counter == 1) begin\n')
        f.write('                    end_cycle = cycle_count;\n')
        f.write('                    $display("Best latency (second input set): %0d cycles", end_cycle - start_cycle);\n')
        f.write('                    $fwrite(csv_file, "%s,%0d,%0d\\n", "BestLatency", 0, end_cycle - start_cycle);\n')
        f.write('                    $fclose(csv_file);\n')
        f.write('                    $finish;\n')
        f.write('                end\n')
        f.write('            end\n')
        f.write('        end\n')
        f.write('    end\n\n')

        f.write('endmodule\n')

def float_to_fixed(float_value, integer_bits=6, fractional_bits=10):
    scaling_factor = 1 << fractional_bits
    total_bits = integer_bits + fractional_bits
    max_val = (1 << (total_bits - 1)) - 1
    min_val = -(1 << (total_bits - 1))

    float_value = float(float_value)  # Convert to Python float if it's a numpy type

    fixed_value = int(np.round(float_value * scaling_factor))
    fixed_value = max(min(fixed_value, max_val), min_val)

    if fixed_value < 0:
        fixed_value = fixed_value + (1 << total_bits)  # Two's complement

    return fixed_value

def write_testbench_input(float_inputs, file_name, integer_bits=6, fractional_bits=10):
    """
    Convert 1D or 2D arrays (or lists of floats) to fixed-point and write to file.

    If 'float_inputs' is 1D: writes a single line.
    If 'float_inputs' is 2D: flattens each row and writes one line per row.
    """
    with open(file_name, "w") as f:
        if len(float_inputs) > 0 and isinstance(float_inputs[0], (list, np.ndarray)):
            for row in float_inputs:
                row_array = np.array(row).ravel()  # flatten if necessary
                fixed_line = [float_to_fixed(val, integer_bits, fractional_bits) for val in row_array]
                f.write(" ".join(map(str, fixed_line)) + "\n")
        else:
            flattened = np.array(float_inputs).ravel()  # ensure it's a flat array of scalars
            fixed_line = [float_to_fixed(val, integer_bits, fractional_bits) for val in flattened]
            f.write(" ".join(map(str, fixed_line)) + "\n")


def prepare_zero_input(layer):
        batch_size = layer['batch_size']
        fifo_depth = layer['fifo_depth']       
        zero_input = np.zeros((fifo_depth, batch_size), dtype=np.int32)
        return zero_input

def prepare_testbench_input(data, fifo_depth, batch_size):
    data_arr = np.array(data)
    # Ensure that total elements = fifo_depth * batch_size
    total_elements = fifo_depth * batch_size
    if data_arr.size != total_elements:
        raise ValueError(
            f"Data size {data_arr.size} does not match fifo_depth * batch_size = {total_elements}"
        )
    data_reshaped = data_arr.reshape((fifo_depth, batch_size))
    return data_reshaped

def read_testbench_log(testbench_log_path):
    """
    Reads the testbench log file and returns a dictionary 
    """
    if not os.path.exists(testbench_log_path):
        print(f"Error: The file '{testbench_log_path}' does not exist.")
        return {}

    try:
        df = pd.read_csv(testbench_log_path)
        BestLatency = df[df['output_name'] == 'BestLatency']['value'].iloc[0]
        WorstLatency = df[df['output_name'] == 'WorstLatency']['value'].iloc[0]
        output_df = df[~df['output_name'].isin(['BestLatency', 'WorstLatency'])]
        
        sim_dict = {
            'BestLatency': int(BestLatency),
            'WorstLatency': int(WorstLatency),
            'BehavSimResults': []
        }

        grouped = output_df.groupby('output_name')
        for name, group in grouped:
            indices = group['index'].astype(int)
            values = group['value'].astype(float)
            array = np.zeros(max(indices) + 1, dtype=np.float64)
            array[indices] = values
            sim_dict['BehavSimResults'].append(array)

        if len(sim_dict['BehavSimResults']) == 1:
            sim_dict['BehavSimResults'] = sim_dict['BehavSimResults'][0]

        return sim_dict

    except (KeyError, IndexError) as e:
        print(f"Error: Missing expected columns or values in the file: {e}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}
