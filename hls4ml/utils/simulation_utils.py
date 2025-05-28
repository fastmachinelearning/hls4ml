import csv
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np


def parse_component_xml(component_xml_path):
    """
    Parse the component.xml from the generated Vivado IP and return a dict of the port metadata.
    """
    if not os.path.exists(component_xml_path):
        raise FileNotFoundError(f"component.xml not found at {component_xml_path}")

    tree = ET.parse(component_xml_path)
    root = tree.getroot()

    ns = {
        'spirit': 'http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009',
        'xilinx': 'http://www.xilinx.com',
        'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
    }

    ports = root.findall('.//spirit:model/spirit:ports/spirit:port', namespaces=ns)
    port_dict = {}

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

            port_dict[name] = {'direction': direction, 'width': width}

    return port_dict


def annotate_axis_stream_widths(nn_config, vivado_project_path):
    """
    Annotate nn_config with Vivado IP AXIS bus and per-element widths.
    """
    component_xml_path = os.path.join(vivado_project_path, 'solution1/impl/ip/component.xml')
    port_dict = parse_component_xml(component_xml_path)
    for layer in nn_config.get('outputs', []):
        batch = layer['batch_size']
        # find the TDATA port (case-insensitive)
        port, info = next(((p, i) for p, i in port_dict.items() if p.lower() == f"{layer['name']}_tdata"), (None, None))
        if port is None:
            raise KeyError(f"No TDATA port for '{layer['name']}' in {component_xml_path}")
        w = info['width']
        if w % batch:
            raise ValueError(f"Bus width ({w}) not divisible by No of output elements ({batch})")
        layer.update({'axis_bus_width': w, 'axis_element_width': w // batch})


def write_verilog_testbench(nn_config, testbench_output_path):
    """
    Generate a Verilog testbench for a given neural network configuration.
    The testbench includes:
      - Clock and reset logic
      - DUT instantiation and AXI4-Stream/Partition interfaces
      - Stimulus generation for inputs
      - Data capture and logging for outputs
      - Latency measurement
    """
    pragma = nn_config['inputs'][0]['pragma']
    # NOTE we usually have active-low in stream interfaces and active-high in partitioned interfaces.
    rst_name = 'ap_rst_n' if pragma == 'stream' else 'ap_rst'

    with open(testbench_output_path, 'w') as f:
        # ----------------------------------------------------------------------
        # Header and Module Declaration
        # ----------------------------------------------------------------------
        f.write('`timescale 1ns / 1ps\n\n')
        f.write('module tb_design_1_wrapper;\n\n')

        # ----------------------------------------------------------------------
        # Clock and Reset Signals
        # ----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // Clock and Reset Signals\n')
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    reg ap_clk;\n')
        f.write(f'    reg {rst_name};\n\n')

        # ----------------------------------------------------------------------
        # Control and Handshaking Signals
        # ----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // Control and Handshaking Signals\n')
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    reg  ap_start;\n')
        f.write('    wire ap_done;\n\n')

        if pragma == 'stream':
            # ----------------------------------------------------------------------
            # AXI4-Stream Input Interfaces
            # ----------------------------------------------------------------------
            f.write('    //------------------------------------------------------------------------\n')
            f.write('    // AXI4-Stream Input Interfaces\n')
            f.write('    //------------------------------------------------------------------------\n')

            for layer in nn_config['inputs']:
                name = layer["name"]
                total_bits = layer['integer_bits'] + layer['fractional_bits']
                batch_size = layer['batch_size']
                f.write(f'    reg  [{(total_bits * batch_size) - 1}:0] {name}_tdata;\n')
                f.write(f'    reg  {name}_tvalid;\n')
                f.write(f'    wire {name}_tready;\n\n')

            # ----------------------------------------------------------------------
            # AXI4-Stream Output Interfaces
            # ----------------------------------------------------------------------
            f.write('    //------------------------------------------------------------------------\n')
            f.write('    // AXI4-Stream Output Interfaces\n')
            f.write('    //------------------------------------------------------------------------\n')

            for layer in nn_config['outputs']:
                name = layer["name"]
                total_bits = layer['integer_bits'] + layer['fractional_bits']
                batch_size = layer['batch_size']
                axis_bus_width = layer['axis_bus_width']
                f.write(f'    wire [{axis_bus_width - 1}:0] {name}_tdata;\n')
                f.write(f'    wire {name}_tvalid;\n')
                f.write(f'    reg  {name}_tready;\n\n')
        else:
            # ----------------------------------------------------------------------
            # Partitioned Input Interfaces
            # ----------------------------------------------------------------------
            f.write('    //------------------------------------------------------------------------\n')
            f.write('    // Partitioned Input Interfaces\n')
            f.write('    //------------------------------------------------------------------------\n')

            for layer in nn_config['inputs']:
                name = layer["name"]
                total_bits = layer['integer_bits'] + layer['fractional_bits']
                batch_size = layer['batch_size']
                for idx in range(batch_size):
                    f.write(f'    reg  [{total_bits - 1}:0] {name}_{idx};\n')
                    f.write(f'    reg {name}_{idx}_ap_vld;\n')

            # ----------------------------------------------------------------------
            # Partitioned Output Interfaces
            # ----------------------------------------------------------------------
            f.write('    //------------------------------------------------------------------------\n')
            f.write('    // Partitioned Output Interfaces\n')
            f.write('    //------------------------------------------------------------------------\n')

            for layer in nn_config['outputs']:
                name = layer["name"]
                total_bits = layer['integer_bits'] + layer['fractional_bits']
                batch_size = layer['batch_size']
                for idx in range(batch_size):
                    f.write(f'    wire [{total_bits - 1}:0] {name}_{idx};\n')
                    f.write(f'    wire {name}_{idx}_ap_vld;\n')

        # ----------------------------------------------------------------------
        # DUT Instantiation
        # ----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // DUT Instantiation\n')
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    stitched_design dut (\n')
        f.write('        .ap_clk(ap_clk),\n')
        f.write('        .ap_done(ap_done),\n')
        f.write(f'        .{rst_name}({rst_name}),\n')
        f.write('        .ap_start(ap_start),\n')

        # Connect input interfaces
        for layer in nn_config['inputs']:
            name = layer["name"]
            batch_size = layer['batch_size']
            if pragma == 'stream':
                f.write(f'        .{name}_tdata({name}_tdata),\n')
                f.write(f'        .{name}_tready({name}_tready),\n')
                f.write(f'        .{name}_tvalid({name}_tvalid),\n')
            else:
                for idx in range(batch_size):
                    f.write(f'        .{name}_{idx}({name}_{idx}),\n')
                    f.write(f'        .{name}_{idx}_ap_vld({name}_{idx}_ap_vld),\n')

        # Connect output interfaces (all but last have trailing comma)
        for layer in nn_config['outputs'][:-1]:
            name = layer["name"]
            batch_size = layer['batch_size']
            if pragma == 'stream':
                f.write(f'        .{name}_tdata({name}_tdata),\n')
                f.write(f'        .{name}_tready({name}_tready),\n')
                f.write(f'        .{name}_tvalid({name}_tvalid),\n')
            else:
                for idx in range(batch_size):
                    f.write(f'        .{name}_{idx}({name}_{idx}),\n')
                    f.write(f'        .{name}_{idx}_ap_vld({name}_{idx}_ap_vld),\n')

        # Last output interface (no trailing comma)
        last_output_layer = nn_config['outputs'][-1]
        name = last_output_layer["name"]
        batch_size = last_output_layer['batch_size']
        if pragma == 'stream':
            f.write(f'        .{name}_tdata({name}_tdata),\n')
            f.write(f'        .{name}_tready({name}_tready),\n')
            f.write(f'        .{name}_tvalid({name}_tvalid)\n')
        else:
            for idx in range(batch_size):
                f.write(f'        .{name}_{idx}({name}_{idx}),\n')
                if idx < batch_size - 1:
                    f.write(f'        .{name}_{idx}_ap_vld({name}_{idx}_ap_vld),\n')
                else:
                    f.write(f'        .{name}_{idx}_ap_vld({name}_{idx}_ap_vld)\n')
        f.write('    );\n\n')

        # ----------------------------------------------------------------------
        # Clock Generation
        # ----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // Clock Generation (100 MHz => 10 ns period)\n')
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    initial begin\n')
        f.write('        ap_clk = 0;\n')
        f.write('        forever #5 ap_clk = ~ap_clk;\n')
        f.write('    end\n\n')

        # ----------------------------------------------------------------------
        # Reset Generation
        # ----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // Reset Generation\n')
        f.write('    // Wait for a cycle and then release reset.\n')
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    initial begin\n')
        if rst_name == 'ap_rst_n':
            f.write(f'        {rst_name} = 0;\n')
            f.write('        repeat (1) @(posedge ap_clk);\n')
            f.write(f'        {rst_name} = 1;\n')
        else:
            f.write(f'        {rst_name} = 1;\n')
            f.write('        repeat (1) @(posedge ap_clk);\n')
            f.write(f'        {rst_name} = 0;\n')
        f.write('    end\n\n')

        # ----------------------------------------------------------------------
        # Signal Initialization
        # ----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // Signal Initialization\n')
        f.write('    // Initialize control signals, input valid, and output ready.\n')
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    initial begin\n')
        f.write('        ap_start = 0;\n')

        for layer in nn_config['inputs']:
            name = layer['name']
            batch_size = layer['batch_size']
            if pragma == 'stream':
                f.write(f'        {name}_tvalid = 0;\n')
            else:
                for idx in range(batch_size):
                    f.write(f'        {name}_{idx}_ap_vld = 0;\n')
        if pragma == 'stream':
            for layer in nn_config['outputs']:
                name = layer['name']
                f.write(f'        {name}_tready = 1;\n')
        f.write('    end\n\n')

        # ----------------------------------------------------------------------
        # Variables for Logging and Measurement
        # ----------------------------------------------------------------------
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
        f.write('    reg old_ap_done = 0;\n\n')

        # ----------------------------------------------------------------------
        # Cycle Counting
        # ----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // Cycle Counting\n')
        f.write('    // Count cycles to measure latency.\n')
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    always @(posedge ap_clk) begin\n')
        if rst_name == 'ap_rst_n':
            f.write(f'        if (!{rst_name})\n')
            f.write('            cycle_count <= 0;\n')
        else:
            f.write(f'        if ({rst_name})\n')
            f.write('            cycle_count <= 0;\n')
        f.write('        else\n')
        f.write('            cycle_count <= cycle_count + 1;\n')
        f.write('    end\n\n')

        # ----------------------------------------------------------------------
        # Data Transmission (Stimulus Generation)
        # ----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // Data Transmission (Stimulus)\n')
        f.write('    // Send input patterns to the DUT.\n')
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    initial begin\n')
        f.write('        // Wait until reset is de-asserted\n')
        if rst_name == 'ap_rst_n':
            f.write(f'        wait ({rst_name} == 1);\n')
        else:
            f.write(f'        wait ({rst_name} == 0);\n')

        f.write('        // Open CSV log file\n')
        f.write('        csv_file = $fopen("../../../../testbench_log.csv", "w");\n')
        f.write('        if (csv_file == 0) begin\n')
        f.write('            $display("ERROR: Could not open CSV log file.");\n')
        f.write('            $finish;\n')
        f.write('        end\n')
        f.write('        $fwrite(csv_file, "output_name,index,value\\n");\n\n')

        if pragma == 'stream':
            f.write('        // Start the DUT\n')
            f.write('        ap_start = 1;\n\n')

        # ----------------------------------------------------------------------
        # Sending first pattern of inputs (all zeroes)
        # ----------------------------------------------------------------------
        for layer in nn_config['inputs']:
            i_bits = layer["integer_bits"]
            f_bits = layer["fractional_bits"]
            total_bits = i_bits + f_bits
            batch_size = layer['batch_size']
            fifo_depth = layer["fifo_depth"]
            name = layer["name"]
            f.write(f'        // Sending first patern of inputs for {name}\n')
            if pragma == 'stream':
                f.write(f'        {name}_tvalid = 1;\n')
            f.write(f'        for (j = 0; j < {fifo_depth}; j = j + 1) begin\n')
            for k in range(batch_size):
                upper = (k + 1) * total_bits - 1
                lower = k * total_bits
                if pragma == 'stream':
                    f.write(f'            {name}_tdata[{upper}:{lower}] = 0;\n')
                else:
                    f.write(f'            {name}_{k} = 0;\n')
            if pragma == 'stream':
                f.write(f'            while ({name}_tready == 0) @(posedge ap_clk);\n')
                f.write('            @(posedge ap_clk);\n')
            f.write('        end\n')
            if pragma == 'stream':
                f.write(f'        {name}_tvalid = 0;\n\n')
            else:
                f.write('        // Assert valid signals\n')
                for k in range(batch_size):
                    f.write(f'        {name}_{k}_ap_vld = 1;\n')
                f.write('        // Start the DUT\n')
                f.write('        ap_start = 1;\n')
                f.write('        @(posedge ap_clk);\n')
                f.write('        ap_start = 0;\n')
                f.write('        // Deassert valid signals\n')
                for k in range(batch_size):
                    f.write(f'        {name}_{k}_ap_vld = 0;\n')
                f.write('\n')
                f.write('        // Wait for ap_done to go high\n')
                f.write('        wait (ap_done);\n')
                f.write('        // Wait for ap_done to go low before sending next input\n')
                f.write('        wait (!ap_done);\n')
                f.write('        // Wait for ap_done to go high\n')

        # ----------------------------------------------------------------------
        # Sending second pattern of inputs (read from file)
        # ----------------------------------------------------------------------
        for layer in nn_config['inputs']:
            i_bits = layer["integer_bits"]
            f_bits = layer["fractional_bits"]
            total_bits = i_bits + f_bits
            batch_size = layer['batch_size']
            fifo_depth = layer["fifo_depth"]
            name = layer["name"]
            input_file = f"{name}_input_data.txt"
            f.write(f'        // Sending second pattern of inputs for {name}\n')
            if pragma == 'stream':
                f.write(f'        {name}_tvalid = 1;\n')
            f.write(f'        file = $fopen("../../../../{input_file}", "r");\n')
            f.write('        if (file == 0) begin\n')
            f.write(f'            $display("Error opening file {input_file}");\n')
            f.write('            $finish;\n')
            f.write('        end\n')
            f.write(f'        for (j = 0; j < {fifo_depth}; j = j + 1) begin\n')
            # For each line, read batch_size values:
            for k in range(batch_size):
                upper = (k + 1) * total_bits - 1
                lower = k * total_bits
                f.write('            r = $fscanf(file, "%d", value);\n')
                if pragma == 'stream':
                    f.write(f'            {name}_tdata[{upper}:{lower}] = value;\n')
                else:
                    f.write(f'            {name}_{k} = value;\n')
            if pragma == 'stream':
                f.write(f'            while ({name}_tready == 0) @(posedge ap_clk);\n')
                f.write('            @(posedge ap_clk);\n')
            f.write('        end\n')
            if pragma == 'partition':
                f.write('        // Assert valid signals\n')
                for k in range(batch_size):
                    f.write(f'        {name}_{k}_ap_vld = 1;\n')
                f.write('        // Start the DUT\n')
                f.write('        ap_start = 1;\n')
                f.write('        @(posedge ap_clk);\n')
                f.write('        ap_start = 0;\n')
                f.write('        // Deassert valid signals\n')
                for k in range(batch_size):
                    f.write(f'        {name}_{k}_ap_vld = 0;\n')
                f.write('\n')

        f.write('    end\n\n')

        # ----------------------------------------------------------------------
        # Output Data Capture and Logging
        # ----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // Output Data Capture and Logging\n')
        f.write('    // Capture output for 2nd input (done_counter == 1) and log them to CSV.\n')
        f.write('    //------------------------------------------------------------------------\n\n')

        for i, layer in enumerate(nn_config['outputs']):
            i_bits = layer['integer_bits']
            f_bits = layer['fractional_bits']
            total_bits = i_bits + f_bits
            layer_name = layer["name"]
            batch_size = layer["batch_size"]

            f.write(f'    //Output capture for {layer_name}\n')
            f.write(f'    integer idx_{i};\n')
            if pragma == 'stream':
                f.write(f'    reg signed [{total_bits-1}:0] fixed_val_{i};\n')
                f.write(f'    real real_val_{i};\n')
            else:
                f.write(f'    reg signed [{total_bits-1}:0] fixed_val_{i};\n')
                f.write(f'    real real_val_{i};\n')
            f.write('    always @(posedge ap_clk) begin\n')
            if pragma == 'stream':
                axis_element_width = layer['axis_element_width']
                f.write(f'        if (done_counter == 1 && {layer_name}_tvalid && {layer_name}_tready) begin\n')
                f.write(f'            for (idx_{i} = 0; idx_{i} < {batch_size}; idx_{i} = idx_{i} + 1) begin\n')
                f.write(
                    f'                fixed_val_{i} = {layer_name}_tdata[idx_{i}*{axis_element_width} +: {total_bits}];\n'
                )
                f.write(f'                real_val_{i}  = fixed_val_{i} / (1.0 * (1 << {f_bits}));\n')
                f.write(f'                $display("Output {layer_name}[%0d]: %.15f", idx_{i}, real_val_{i});\n')
                f.write('                // Log result to CSV\n')
                f.write(f'                $fwrite(csv_file, "%s,%0d,%.15f\\n", "{layer_name}", idx_{i}, real_val_{i});\n')
                f.write('            end\n')
            else:
                f.write(
                    '        // Note: The usual expected behavior is to have valid outputs (ap_vld=1) when ap_done = 1\n'
                )
                f.write('        if (done_counter == 1 && ap_done == 1) begin\n')
                for idx in range(batch_size):
                    f.write(f'            fixed_val_{i} = {layer_name}_{idx}[{total_bits - 1}:0];\n')
                    f.write(f'            real_val_{i} = fixed_val_{i} / (1.0 * (1 << {f_bits}));\n')
                    f.write(f'            $display("Output {layer_name}_{idx}: %.15f", real_val_{i});\n')
                    f.write('            // Log result to CSV\n')
                    f.write(f'            $fwrite(csv_file, "%s,%0d,%.15f\\n", "{layer_name}", {idx}, real_val_{i});\n')
            f.write('        end\n')
            f.write('    end\n\n')

        # ----------------------------------------------------------------------
        # Latency Measurement and Test End
        # ----------------------------------------------------------------------
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    // Latency Measurement\n')
        f.write('    // Measures the cycle count between start and subsequent ap_done signals.\n')
        f.write('    //------------------------------------------------------------------------\n')
        f.write('    always @(posedge ap_clk) begin\n')
        if rst_name == 'ap_rst_n':
            f.write(f'        if (!{rst_name}) begin\n')
        else:
            f.write(f'        if ({rst_name}) begin\n')
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


def prepare_zero_inputs(input_layers):
    zero_list = [np.zeros((layer['fifo_depth'], layer['batch_size']), dtype=np.float32) for layer in input_layers]
    return zero_list[0] if len(zero_list) == 1 else zero_list


def prepare_tb_inputs(simulation_input_data, input_layers):
    if simulation_input_data is None:
        return prepare_zero_inputs(input_layers)

    if isinstance(simulation_input_data, np.ndarray):
        data_list = [simulation_input_data]
    elif isinstance(simulation_input_data, (list, tuple)):
        data_list = list(simulation_input_data)
    else:
        raise TypeError(f"simulation_input_data must be None, ndarray or list/tuple, got {type(simulation_input_data)}")

    if len(data_list) != len(input_layers):
        raise ValueError(f"Expected {len(input_layers)} input arrays, got {len(data_list)}.")

    reshaped = []
    for data, layer in zip(data_list, input_layers):
        arr = np.asarray(data)
        total = layer['fifo_depth'] * layer['batch_size']
        if arr.size != total:
            raise ValueError(f"Layer '{layer['name']}' has {arr.size} elements; expected {total}.")
        reshaped.append(arr.reshape((layer['fifo_depth'], layer['batch_size'])))

    return reshaped[0] if len(reshaped) == 1 else reshaped


def read_testbench_log(testbench_log_path, outputs):
    """
    Reads the testbench log file and returns a dictionary
    """
    if not os.path.exists(testbench_log_path):
        print(f"Error: The file '{testbench_log_path}' does not exist.")
        return {}

    try:
        with open(testbench_log_path, encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)
            required_columns = {'output_name', 'value', 'index'}
            if not required_columns.issubset(set(header)):
                print("Error: Missing required columns in the CSV file.")
                return {}

            col_index = {col: idx for idx, col in enumerate(header)}
            best_latency = worst_latency = None
            output_data = defaultdict(list)

            for row in reader:
                output_name = row[col_index['output_name']]
                value = row[col_index['value']]
                if output_name == 'BestLatency':
                    best_latency = int(value)
                elif output_name == 'WorstLatency':
                    worst_latency = int(value)
                else:
                    index = int(row[col_index['index']])
                    output_data[output_name].append((index, float(value)))

            if best_latency is None or worst_latency is None:
                print("Error: BestLatency or WorstLatency not found.")
                return {}
            sim_dict = {'BestLatency': best_latency, 'WorstLatency': worst_latency, 'BehavSimResults': []}
            ordered_output_names = [entry['name'] for entry in outputs]

            for name in ordered_output_names:
                if name not in output_data:
                    print(f"Warning: Expected output '{name}' not found in testbench log.")
                    continue

                indices_values = output_data[name]
                max_index = max(index for index, _ in indices_values)
                array = np.zeros(max_index + 1, dtype=np.float64)
                for index, value in indices_values:
                    array[index] = value
                sim_dict['BehavSimResults'].append(array)

            # If only one set of results, return it as a single array instead of a list
            if len(sim_dict['BehavSimResults']) == 1:
                sim_dict['BehavSimResults'] = sim_dict['BehavSimResults'][0]

            return sim_dict

    except (KeyError, IndexError, ValueError) as e:
        print(f"Error: Issue with CSV file format or data: {e}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}
