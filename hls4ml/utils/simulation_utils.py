import os
from lxml import etree

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


def generate_verilog_testbench(nn_config, testbench_output_path):
    inputs = nn_config['inputs']
    outputs = nn_config['outputs']

    input_signals = []
    output_signals = []

    for input_item in inputs:
        total_bits = input_item['integer_bits'] + input_item['fractional_bits']
        input_signals.append((input_item['name'], total_bits))

    for output_item in outputs:
        total_bits = output_item['integer_bits'] + output_item['fractional_bits']
        output_signals.append((output_item['name'], total_bits))

    with open(testbench_output_path, 'w') as f:
        # Write the initial part of the testbench
        f.write('`timescale 1ns / 1ps\n\n')
        f.write('module tb_design_1_wrapper;\n\n')
        f.write('    // Clock and Reset Signals\n')
        f.write('    reg ap_clk;\n')
        f.write('    reg ap_rst_n;\n\n')
        f.write('    // Control Signals\n')
        f.write('    reg ap_start;\n')
        f.write('    wire ap_done;\n\n')

        # Generate AXI4-Stream interface signals for inputs
        for layer in nn_config['inputs']:
            total_bits = layer['integer_bits'] + layer['fractional_bits']
            f.write(f'    reg [{(total_bits * layer["batch_size"]) - 1}:0] {layer["name"]}_tdata;\n')
            f.write(f'    reg {layer["name"]}_tvalid;\n')
            f.write(f'    wire {layer["name"]}_tready;\n\n')

        # Generate AXI4-Stream interface signals for outputs
        for layer in nn_config['outputs']:
            total_bits = layer['integer_bits'] + layer['fractional_bits']
            f.write(f'    wire [{(total_bits * layer["batch_size"]) - 1}:0] {layer["name"]}_tdata;\n')
            f.write(f'    wire {layer["name"]}_tvalid;\n')
            f.write(f'    reg {layer["name"]}_tready;\n\n')

        # Instantiate the DUT
        f.write('    // Instantiate the Design Under Test (DUT)\n')
        f.write('    stitched_design dut (\n')
        f.write('        .ap_clk(ap_clk),\n')
        f.write('        .ap_done(ap_done),\n')
        f.write('        .ap_rst_n(ap_rst_n),\n')
        f.write('        .ap_start(ap_start),\n')
        # Connect input AXI4-Stream interfaces
        for layer in nn_config['inputs']:
            name = layer["name"]
            f.write(f'        .{name}_tdata({name}_tdata),\n')
            f.write(f'        .{name}_tready({name}_tready),\n')
            f.write(f'        .{name}_tvalid({name}_tvalid),\n')
        # Connect output AXI4-Stream interfaces
        for layer in nn_config['outputs'][:-1]:
            name = layer["name"]
            f.write(f'        .{name}_tdata({name}_tdata),\n')
            f.write(f'        .{name}_tready({name}_tready),\n')
            f.write(f'        .{name}_tvalid({name}_tvalid),\n')
        # Handle the last output layer without a trailing comma
        last_output_layer = nn_config['outputs'][-1]
        name = last_output_layer["name"]
        f.write(f'        .{name}_tdata({name}_tdata),\n')
        f.write(f'        .{name}_tready({name}_tready),\n')
        f.write(f'        .{name}_tvalid({name}_tvalid)\n')
        f.write('    );\n\n')

        # Add clock generation
        f.write('    // Clock Generation (100 MHz)\n')
        f.write('    initial begin\n')
        f.write('        ap_clk = 0;\n')
        f.write('        forever #5 ap_clk = ~ap_clk; // Clock period of 10 ns\n')
        f.write('    end\n\n')

        # Reset generation
        f.write('    // Reset Generation\n')
        f.write('    initial begin\n')
        f.write('        ap_rst_n  = 0;\n')
        f.write('        repeat (5) @(posedge ap_clk);\n')
        f.write('        ap_rst_n = 1;\n')
        f.write('    end\n\n')

        # Initialize Control Signals
        f.write('    // Control Signal Initialization\n')
        f.write('    initial begin\n')
        f.write('        ap_start = 0;\n')
        for name, _ in input_signals:
            f.write(f'        {name}_tvalid = 0;\n')
        for name, _ in output_signals:
            f.write(f'        {name}_tready = 1;\n')
        f.write('    end\n\n')

        # Cycle counter
        f.write('    // Cycle counter\n')
        f.write('    reg [63:0] cycle_count = 0;\n')
        f.write('    reg [63:0] start_cycle = 0;\n')
        f.write('    reg [63:0] end_cycle = 0;\n')
        f.write('    always @(posedge ap_clk) begin\n')
        f.write('        if (!ap_rst_n)\n')
        f.write('            cycle_count <= 0;\n')
        f.write('        else\n')
        f.write('            cycle_count <= cycle_count + 1;\n')
        f.write('    end\n\n')

        # Data Transmission
        f.write('    // Data Transmission\n')
        f.write('    integer i, j;\n')
        f.write('    integer total_bits;\n')
        f.write('    initial begin\n')
        f.write('        // Wait for reset deassertion\n')
        f.write('        wait (ap_rst_n == 1);\n')
        f.write('        repeat (2) @(posedge ap_clk);\n\n')

        f.write('        // Start the operation\n')
        f.write('        ap_start = 1;\n')

        # First Data Pattern: All Zeros
        for layer in nn_config['inputs']:
            f.write(f'        // Sending all zeros for {layer["name"]}\n')
            f.write(f'        total_bits = {layer["integer_bits"] + layer["fractional_bits"]};\n')
            f.write(f'        {layer["name"]}_tvalid = 1;\n\n')
            f.write(f'        for (j = 0; j < {layer["fifo_depth"]}; j = j + 1) begin\n')
            for k in range(layer['batch_size']):
                upper = (k + 1) * (layer["integer_bits"] + layer["fractional_bits"]) - 1
                lower = k * (layer["integer_bits"] + layer["fractional_bits"])
                f.write(f'            {layer["name"]}_tdata[{upper}:{lower}] = 0;\n')
            f.write(f'            while ({layer["name"]}_tready == 0) @(posedge ap_clk);\n')
            f.write(f'            @(posedge ap_clk);\n')
            f.write(f'        end\n')
            f.write(f'        {layer["name"]}_tvalid = 0;\n\n')

        # Second Data Pattern: Fixed Value of 1
        for layer in nn_config['inputs']:
            f.write(f'        // Sending fixed value 1 for {layer["name"]}\n')
            f.write(f'        total_bits = {layer["integer_bits"] + layer["fractional_bits"]};\n')
            f.write(f'        {layer["name"]}_tvalid = 1;\n\n')
            f.write(f'        for (j = 0; j < {layer["fifo_depth"]}; j = j + 1) begin\n')
            for k in range(layer['batch_size']):
                upper = (k + 1) * (layer["integer_bits"] + layer["fractional_bits"]) - 1
                lower = k * (layer["integer_bits"] + layer["fractional_bits"])
                f.write(f'            {layer["name"]}_tdata[{upper}:{lower}] = 1 << {layer["fractional_bits"]};\n')
            f.write(f'            while ({layer["name"]}_tready == 0) @(posedge ap_clk);\n')
            f.write(f'            @(posedge ap_clk);\n')
            f.write(f'        end\n')
            f.write(f'        {layer["name"]}_tvalid = 0;\n\n')

        f.write('        start_cycle = cycle_count;\n\n')
        # Third Data Pattern: All zeros (this is where we measure cycles)
        for layer in nn_config['inputs']:
            f.write(f'        // Sending all zeros for {layer["name"]} (this is where we measure cycles)\n')
            f.write(f'        total_bits = {layer["integer_bits"] + layer["fractional_bits"]};\n')
            f.write(f'        {layer["name"]}_tvalid = 1;\n\n')
            f.write(f'        for (j = 0; j < {layer["fifo_depth"]}; j = j + 1) begin\n')
            for k in range(layer['batch_size']):
                upper = (k + 1) * (layer["integer_bits"] + layer["fractional_bits"]) - 1
                lower = k * (layer["integer_bits"] + layer["fractional_bits"])
                f.write(f'            {layer["name"]}_tdata[{upper}:{lower}] = 0;\n')
            f.write(f'            while ({layer["name"]}_tready == 0) @(posedge ap_clk);\n')
            f.write(f'            @(posedge ap_clk);\n')
            f.write(f'        end\n')
            f.write(f'        {layer["name"]}_tvalid = 0;\n\n')

        f.write('        // Wait for operation to complete\n')
        f.write('        wait (ap_done == 1);\n')
        f.write('        end_cycle = cycle_count;\n')
        f.write('        $display("Total cycles from start to done: %0d", end_cycle - start_cycle);\n')
        f.write('        repeat (5) @(posedge ap_clk);\n')
        f.write('        $finish;\n')
        f.write('    end\n\n')

        # Output Handling
        f.write('    // Output Data Capture\n')
        f.write('    // Decode and display outputs in fixed-point format\n')
        for layer in nn_config['outputs']:
            signed_str = layer.get('signed', 1)
            i_bits = layer['integer_bits']
            f_bits = layer['fractional_bits']
            total_bits = i_bits + f_bits
            f.write(f'    integer idx;\n')
            f.write(f'    reg signed [{total_bits-1}:0] fixed_val;\n')
            f.write(f'    real real_val;\n')

            # We'll add an always block per output to print whenever valid & ready
            f.write(f'    always @(posedge ap_clk) begin\n')
            f.write(f'        if ({layer["name"]}_tvalid && {layer["name"]}_tready) begin\n')
            # For simplicity, assume batch_size = 1 here. If you have multiple batch elements, you'd need to loop.
            # If batch_size > 1, we would display each slice separately.
            f.write(f'            for (idx = 0; idx < {layer["batch_size"]}; idx = idx + 1) begin\n')
            f.write(f'                fixed_val = {layer["name"]}_tdata[(idx+1)*{total_bits}-1 -: {total_bits}];\n')
            # If signed, sign-extend was already done due to reg signed
            # Convert to real by dividing by 2^(fractional_bits)
            f.write(f'                real_val = fixed_val / (1.0 * (1 << {f_bits}));\n')
            f.write(f'                $display("Output {layer["name"]}[%0d]: integer_bits=%0d fractional_bits=%0d value=%f", idx, {i_bits}, {f_bits}, real_val);\n')
            f.write(f'            end\n')
            f.write('        end\n')
            f.write('    end\n\n')

        f.write('endmodule\n')