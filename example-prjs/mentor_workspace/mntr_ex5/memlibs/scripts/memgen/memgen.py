#!/usr/bin/python3

import sys
import os
import re
import math

### Helpers ###
def print_usage():
    print("Usage                    : ./memgen.py <tech_path> <infile> <out_path>")
    print("")
    print("")
    print("      <tech_path>        : Path to technology liraries for memories. A list of available SRAMS should")
    print("                           be available at the path <tech_path>/lib.txt")
    print("                         : In addition the folder <tech_path> should contain all wrappers and/or")
    print("                           necessary behavioral models of the SRAMs listed in lib.txt")
    print("                           While wrappers must exist, behavioral models may be compiled separately from")
    print("                           a third party library")
    print("")
    print("      <infile>           : Path to the List of required memories to generate. Each memory is specified")
    print("                           with one memory descriptor per line.")
    print("")
    print("      <out_path>         : Path where the generated verilog and log files will be created.")
    print("                           If the specified folder doesn't exist, it will be created.")
    print("")
    print("")
    print("")
    print("Memory descriptor syntax : <name> <words> <width> <parallel_op_list>")
    print("")
    print("")
    print("      <name>             : Memory name")
    print("")
    print("      <words>            : Number of logic words in memory")
    print("")
    print("      <width>            : Word bit-width")
    print("")
    print("      <parallel_op_list> : List of parallel accesses to memory. These may require one or more ports.")
    print("")
    print("")
    print("")
    print("Operation-list element   : <write_pattern>:<read_pattern>")
    print("")
    print("")
    print("      <write_pattern>    : 0w                 -> no write operation. The first operation in the list cannot have")
    print("                                                 zero write interfaces or the testbench will fail")
    print("                         : 1w                 -> 1 write operation")
    print("                         : <[2|4|8|16]>w      -> 2 parallel write operations with known (modulo) address pattern.")
    print("                                                 Data are distributed across banks and the number of banks must")
    print("                                                 be a power of two to have low-overhead bank-selection logic")
    print("                         : 2wu                -> 2 parallel write operations with unknown address pattern. This is")
    print("                                                 viable by using both physical ports of dual-port banks, but only")
    print("                                                 in combination with \"0r\" as read pattern.")
    print("")
    print("      <read_pattern>     : 0r                 -> no read operation")
    print("                         : 1r                 -> 1 read operation")
    print("                         : <[2|4|8|16]>r      -> parallel read operations with known (modulo) address pattern.")
    print("                                                 Data are distributed across banks and the number of banks must")
    print("                                                 be a power of two to have low-overhead bank-selection logic")
    print("                         : <N>ru              -> N parallel read operations with unknown address pattern. This")
    print("                                                 option incurs data and memory duplication. N can be any number")
    print("                                                 from 2 to 16.")
    print("")
    sys.exit(1)


def die_werr(message):
    print("  ERROR: " + message)
    sys.exit(2)


def warn(message):
    print("  WARNING: " + message)


def is_power2z(n):
    # True if zero or power of 2
    return ((n & (n - 1)) == 0)

ASSERT_ON = True

### Default memory delay (must be included int he mem/lib.txt file"
mem_delay = 0.2
mem_setup = 0.06

### Data structures ###
class sram():

    def __init__(self, name, words, width, area, ports):
        self.name = name
        self.words = words
        self.width = width
        self.area = area
        self.ports = ports

    def print(self):
        token1 = self.name
        token1 = format(token1, '>30')
        token2 = str(self.words)
        token2 = format(token2, '>7')
        token3 = str(self.width)
        token3 = format(token3, '>3')
        token4 = str(self.ports)
        token4 = format(token4, '>2')
        print("  INFO: Found SRAM definition " + \
              token1 + token2 + token3 + "-bit words " + \
              token4 + " read/write ports ")



class memory_operation():

    def __init__(self, rn, rp, wn, wp):
        self.rn = rn
        self.rp = rp
        self.wn = wn
        self.wp = wp

    def __str__(self):
        if self.rp == "modulo":
            rp = "r"
        else:
            rp = "ru"
        if self.wp == "modulo":
            wp = "w"
        else:
            wp = "wu"
        return str(self.wn) + wp + ":" + str(self.rn) + rp



class memory():

    def __init__(self, name, words, width, ops):
        self.name = name
        if words <= 0:
            die_werr("Memory "+name+" has illegal number of words")
        if width <= 0:
            die_werr("Memory "+name+" has illegal bit-width")
        if len(ops) == 0:
            die_werr("No operation specified for memory \"" + name + "\"")
        self.words = words
        self.width = width
        self.ops = ops
        self.read_interfaces = 1
        self.write_interfaces = 1
        self.need_dual_port = False
        self.need_parallel_rw = False
        self.duplication_factor = 1
        self.distribution_factor = 1
        # Horizontally duplicated banks when read pattern is unknown
        self.dbanks = 1
        # Horizontally composed banks to obtain desired parallelism
        self.hbanks = 1
        # Vertically composed banks to obtain desired word count
        self.vbanks = 1
        # Horizontally composed banks to obtain desired bit-width
        self.hhbanks = 1
        # Type of SRAM chosen to implement banks
        self.bank_type = None
        # Total area
        self.area = float('inf')
        # Port assignment array
        self.read_ports = [ ]
        self.write_ports = [ ]


    def print(self):
        operations = " ".join(map(str, self.ops))
        print("  INFO: Generating " + self.name + "...")
        print("        " + str(self.words) + " words, " + str(self.width) + " bits, " + operations)

    def __find_hbanks(self):
        for op in self.ops:
            self.read_interfaces = max(self.read_interfaces, op.rn)
            self.write_interfaces = max(self.write_interfaces, op.wn)
            # parallel_rw
            self.need_parallel_rw = (self.need_parallel_rw or
                                     (op.rn > 0 and op.wn > 0))
            # Note that we force dual-port memories to get half the number of hbanks when possible
            # dual_port
            self.need_dual_port = (self.need_dual_port or
                                   self.need_parallel_rw or
                                   (op.wn == 2 and op.wp == "unknown") or
                                   ((not self.need_parallel_rw) and (op.rn > 1 or op.wn > 1)))

        for op in self.ops:
            # Duplication
            op_duplication_factor = 1
            if (op.rp == "unknown" and op.rn > 1):
                if (op.wn != 0 or self.need_parallel_rw):
                    op_duplication_factor = op.rn
                else:
                    op_duplication_factor = int(math.ceil(op.rn / 2))
            if (op.wp == "unknown" and op.wn > 1):
                if (op.rn != 0 or self.need_parallel_rw):
                    op_duplication_factor = max(op_duplication_factor, op.wn)
                else:
                    op_duplication_factor = int(math.ceil(op.wn / 2))
            self.duplication_factor = max(self.duplication_factor, op_duplication_factor)

        for op in self.ops:
            # Distribution
            op_distribution_factor = 1
            if (op.rp == "modulo" and op.rn > 1):
                if (op.wn != 0 or self.need_parallel_rw):
                    op_distribution_factor = op.rn
                else:
                    op_distribution_factor = op.rn >> 1
            if (op.wp == "modulo" and op.wn > 1):
                if (op.rn != 0 or self.need_parallel_rw):
                    op_distribution_factor = max(op_distribution_factor, op.wn)
                else:
                    op_distribution_factor = op.wn >> 1
            self.distribution_factor = max(self.distribution_factor, op_distribution_factor)

        # Number of distributed banks and duplicated bank sets
        self.dbanks = self.duplication_factor
        self.hbanks = self.distribution_factor

    def __find_vbanks(self, lib):
        words_per_hbank = int(math.ceil(self.words / self.hbanks))
        d = self.dbanks
        h = self.hbanks
        for ram in lib:
            if self.need_dual_port and (ram.ports < 2):
                continue
            hh = int(math.ceil(self.width / ram.width))
            v = int(math.ceil(words_per_hbank / ram.words))
            new_area = d * h * hh * v * ram.area
            if self.area > (new_area):
                self.vbanks = v
                self.hhbanks = hh
                self.bank_type = ram
                self.area = new_area

    def __write_check_access_task(self, fd):
        fd.write("\n")
        fd.write("  task check_access;\n")
        fd.write("    input integer iface;\n")
        fd.write("    input integer d;\n")
        fd.write("    input integer h;\n")
        fd.write("    input integer v;\n")
        fd.write("    input integer hh;\n")
        fd.write("    input integer p;\n")
        fd.write("  begin\n")
        fd.write("    if ((check_bank_access[d][h][v][hh][p] != -1) &&\n")
        fd.write("        (check_bank_access[d][h][v][hh][p] != iface)) begin\n")
        fd.write("      $display(\"ASSERTION FAILED in %m: port conflict on bank\", h, \"h\", v, \"v\", hh, \"hh\", \" for port\", p, \" involving interfaces\", check_bank_access[d][h][v][hh][p], iface);\n")
        fd.write("      $finish;\n")
        fd.write("    end\n")
        fd.write("    else begin\n")
        fd.write("      check_bank_access[d][h][v][hh][p] = iface;\n")
        fd.write("    end\n")
        fd.write("  end\n")
        fd.write("  endtask\n")



    def __write_ctrl_assignment(self, fd, bank_addr_range_str, hh_range_str, last_hh_range_str, duplicated_bank_set, port, iface, is_write, parallelism):
        ce_str = [ ]
        a_str = [ ]
        d_str = [ ]
        we_str = [ ]
        wem_str = [ ]
        for d in range(0, self.dbanks):
            ce_str_i = [ ]
            a_str_i = [ ]
            d_str_i = [ ]
            we_str_i = [ ]
            wem_str_i = [ ]
            for p in range(0, self.bank_type.ports):
                ce_str_i.append ("                bank_CE["  + str(d) + "][h][v][hh]["  + str(p) + "]  = CE")
                a_str_i.append  ("                bank_A["   + str(d) + "][h][v][hh]["  + str(p) + "]   = A")
                d_str_i.append  ("                bank_D["   + str(d) + "][h][v][hh]["  + str(p) + "]   = D")
                we_str_i.append ("                bank_WE["  + str(d) + "][h][v][hh]["  + str(p) + "]  = WE")
                wem_str_i.append("                bank_WEM[" + str(d) + "][h][v][hh]["  + str(p) + "] = WEM")
            ce_str.append(ce_str_i)
            a_str.append(a_str_i)
            d_str.append(d_str_i)
            we_str.append(we_str_i)
            wem_str.append(wem_str_i)

        # # This handles cases in which there are multiple parallel ops with pattern "modulo", but different distribution factor.
        # if parallelism != 0:
        #     normalized_iface = iface
        #     # If access patter is modulo, we know it's a power of two. If it's not modulo parallelism is set to 0 when calling this method
        #     if not is_write:
        #         normalized_iface = iface - self.write_interfaces
        #     normalized_iface = normalized_iface % self.hbanks
        #     normalized_parallelism = min(parallelism, self.hbanks)
        #     fd.write("          if (h % " + str(normalized_parallelism) + " == " + str(normalized_iface) + ") begin\n")
        fd.write("            if (ctrlh[" + str(iface) + "] == h && ctrlv[" + str(iface) + "] == v && CE" + str(iface) + " == 1'b1) begin\n")
        # Check that no port is accessed by more than one interface
        if (ASSERT_ON):
            fd.write("// synthesis translate_off\n")
            fd.write("// synopsys translate_off\n")
            fd.write("              check_access(" + str(iface) + ", " + str(duplicated_bank_set) + ", h, v, hh, " + str(port) + ");\n")
            fd.write("// synopsys translate_on\n")
            fd.write("// synthesis translate_on\n")
        fd.write(ce_str[duplicated_bank_set][port]  + str(iface)                       + ";\n")
        fd.write(a_str[duplicated_bank_set][port]   + str(iface) + bank_addr_range_str + ";\n")
        if is_write:
            fd.write("              if (hh != " + str(self.hhbanks - 1) + ") begin\n")
            fd.write(d_str[duplicated_bank_set][port]   + str(iface) + hh_range_str        + ";\n")
            fd.write(wem_str[duplicated_bank_set][port] + str(iface) + hh_range_str        + ";\n")
            fd.write("              end\n")
            fd.write("              else begin\n")
            fd.write(d_str[duplicated_bank_set][port]   + str(iface) + last_hh_range_str   + ";\n")
            fd.write(wem_str[duplicated_bank_set][port] + str(iface) + last_hh_range_str   + ";\n")
            fd.write("              end\n")
            fd.write(we_str[duplicated_bank_set][port]  + str(iface)                       + ";\n")
        fd.write("            end\n")
        # if parallelism != 0:
        #     fd.write("          end\n")

    def write_verilog(self, out_path):
        try:
            fd = open(out_path + "/" + self.name + ".v", 'w')
        except IOError as e:
            die_werr(e)
        fd.write("/**\n")
        fd.write("* Created with the ESP Memory Generator\n")
        fd.write("*\n")
        fd.write("* Copyright (c) 2014-2017, Columbia University\n")
        fd.write("*\n")
        fd.write("* @author Paolo Mantovani <paolo@cs.columbia.edu>\n")
        fd.write("*/\n")
        fd.write("\n")
        fd.write("`timescale  1 ps / 1 ps\n")
        fd.write("\n")
        fd.write("module " + self.name + "(\n")
        fd.write("    CLK")
        # Module interface
        for i in range(0, self.write_interfaces):
            fd.write(",\n    CE" + str(i))
            fd.write(",\n    A" + str(i))
            fd.write(",\n    D" + str(i))
            fd.write(",\n    WE" + str(i))
            fd.write(",\n    WEM" + str(i))
        for i in range(self.write_interfaces, self.write_interfaces + self.read_interfaces):
            fd.write(",\n    CE" + str(i))
            fd.write(",\n    A" + str(i))
            fd.write(",\n    Q" + str(i))
        fd.write("\n  );\n")
        fd.write("  input CLK;\n")
        for i in range(0, self.write_interfaces):
            fd.write("  " + "input CE" + str(i) + ";\n")
            fd.write("  " + "input " + "[" + str(int(math.ceil(math.log(self.words, 2)))-1) + ":0] A" + str(i) + ";\n")
            fd.write("  " + "input " + "[" + str(self.width-1) + ":0] D" + str(i) + ";\n")
            fd.write("  " + "input WE" + str(i) + ";\n")
            fd.write("  " + "input " + "[" + str(self.width-1) + ":0] WEM" + str(i) + ";\n")
        for i in range(self.write_interfaces, self.write_interfaces + self.read_interfaces):
            fd.write("  " + "input CE" + str(i) + ";\n")
            fd.write("  " + "input " + "[" + str(int(math.ceil(math.log(self.words, 2)))-1) + ":0] A" + str(i) + ";\n")
            fd.write("  " + "output " + "[" + str(self.width-1) + ":0] Q" + str(i) + ";\n")
        fd.write("  genvar d, h, v, hh;\n")
        fd.write("\n")

        # Wire for banks
        bank_wire_addr_width     = int(math.ceil(math.log(self.bank_type.words, 2)))
        bank_wire_data_width     = self.bank_type.width
        sel_dbank_reg_width      = int(math.ceil(math.log(self.dbanks, 2)))
        sel_hbank_reg_width      = int(math.ceil(math.log(self.hbanks, 2)))
        sel_vbank_reg_width      = int(math.ceil(math.log(self.vbanks, 2)))
        signle_wire_width_str    = "            "
        bank_wire_dims_str       = "[" + str(self.dbanks - 1) + ":0]" + "[" + str(self.hbanks - 1) + ":0]" + "[" + str(self.vbanks - 1) + ":0]" + "[" + str(self.hhbanks - 1) + ":0]" + "[" + str(self.bank_type.ports - 1) + ":0]"
        bank_wire_addr_width_str = "[" + str(bank_wire_addr_width - 1) + ":0]"
        bank_wire_addr_width_str = format(bank_wire_addr_width_str, ">12")
        bank_wire_data_width_str = "[" + str(bank_wire_data_width - 1) + ":0]"
        bank_wire_data_width_str = format(bank_wire_data_width_str, ">12")
        ctrl_wire_dims_str        = "[" + str(self.write_interfaces + self.read_interfaces - 1) + ":0]"
        sel_reg_dims_str         = "[" + str(self.write_interfaces + self.read_interfaces - 1) + ":" + str(self.write_interfaces) + "]"
        if self.dbanks > 1:
            sel_dbank_reg_width_str  = "[" + str(sel_dbank_reg_width - 1) + ":0]"
        else:
            sel_dbank_reg_width_str  = "[0:0]"
        sel_dbank_reg_width_str  = format(sel_dbank_reg_width_str, ">12")
        if self.hbanks > 1:
            sel_hbank_reg_width_str  = "[" + str(sel_hbank_reg_width - 1) + ":0]"
        else:
            sel_hbank_reg_width_str  = "[0:0]"
        sel_hbank_reg_width_str  = format(sel_hbank_reg_width_str, ">12")
        if self.vbanks > 1:
            sel_vbank_reg_width_str  = "[" + str(sel_vbank_reg_width - 1) + ":0]"
        else:
            sel_vbank_reg_width_str  = "[0:0]"
        sel_vbank_reg_width_str  = format(sel_vbank_reg_width_str, ">12")
        fd.write("  " + "reg  " + signle_wire_width_str    + " bank_CE  " + bank_wire_dims_str + ";\n")
        fd.write("  " + "reg  " + bank_wire_addr_width_str + " bank_A   " + bank_wire_dims_str + ";\n")
        fd.write("  " + "reg  " + bank_wire_data_width_str + " bank_D   " + bank_wire_dims_str + ";\n")
        fd.write("  " + "reg  " + signle_wire_width_str    + " bank_WE  " + bank_wire_dims_str + ";\n")
        fd.write("  " + "reg  " + bank_wire_data_width_str + " bank_WEM " + bank_wire_dims_str + ";\n")
        fd.write("  " + "wire " + bank_wire_data_width_str + " bank_Q   " + bank_wire_dims_str + ";\n")
        fd.write("  " + "wire " + sel_dbank_reg_width_str  + " ctrld    " + sel_reg_dims_str   + ";\n")
        fd.write("  " + "wire " + sel_hbank_reg_width_str  + " ctrlh    " + ctrl_wire_dims_str + ";\n")
        fd.write("  " + "wire " + sel_vbank_reg_width_str  + " ctrlv    " + ctrl_wire_dims_str + ";\n")
        fd.write("  " + "reg  " + sel_dbank_reg_width_str  + " seld     " + sel_reg_dims_str   + ";\n")
        fd.write("  " + "reg  " + sel_hbank_reg_width_str  + " selh     " + sel_reg_dims_str   + ";\n")
        fd.write("  " + "reg  " + sel_vbank_reg_width_str  + " selv     " + sel_reg_dims_str   + ";\n")
        if (ASSERT_ON):
            fd.write("// synthesis translate_off\n")
            fd.write("// synopsys translate_off\n")
            fd.write("  " + "integer check_bank_access " + bank_wire_dims_str + ";\n")
            self.__write_check_access_task(fd)
            fd.write("// synopsys translate_on\n")
            fd.write("// synthesis translate_on\n")
        fd.write("\n")

        # Control selection
        for ri in range(self.write_interfaces, self.write_interfaces + self.read_interfaces):
            # For ru type of operations we guarantee to insist on different copies of the memory structure
            if self.dbanks > 1:
                fd.write("  assign ctrld[" + str(ri) + "] = " + str(ri % self.dbanks) + ";\n")
            else:
                fd.write("  assign ctrld[" + str(ri) + "] = 0;\n")
        for ri in range(0, self.write_interfaces + self.read_interfaces):
            if self.hbanks > 1:
                fd.write("  assign ctrlh[" + str(ri) + "] = A" + str(ri) + "[" + str(sel_hbank_reg_width - 1) + ":" + "0" + "];\n")
            else:
                fd.write("  assign ctrlh[" + str(ri) + "] = 0;\n")
        for ri in range(0, self.write_interfaces + self.read_interfaces):
            if self.vbanks > 1:
                fd.write("  assign ctrlv[" + str(ri) + "] = A" + str(ri) + "[" + str(bank_wire_addr_width + sel_hbank_reg_width + sel_vbank_reg_width - 1) + ":" +str(bank_wire_addr_width + sel_hbank_reg_width) + "];\n")
            else:
                fd.write("  assign ctrlv[" + str(ri) + "] = 0;\n")
        fd.write("\n")

        # Output bank selection
        fd.write("  always @(posedge CLK) begin\n")
        for ri in range(self.write_interfaces, self.write_interfaces + self.read_interfaces):
            fd.write("    seld[" + str(ri) + "] <= ctrld[" + str(ri) + "];\n")
            fd.write("    selh[" + str(ri) + "] <= ctrlh[" + str(ri) + "];\n")
            fd.write("    selv[" + str(ri) + "] <= ctrlv[" + str(ri) + "];\n")
        fd.write("  end\n")
        fd.write("\n")

        # Control ports CE, A, D, WE, WEM assignment
        hh_msb_str = str(bank_wire_data_width) + " * (hh + 1) - 1"
        hh_lsb_str = str(bank_wire_data_width) + " * hh"
        hh_range_str = "[" + hh_msb_str + ":" + hh_lsb_str + "]"
        last_hh_msb_str = str((self.width - 1) % bank_wire_data_width) + " + " + hh_lsb_str
        last_hh_lsb_str = hh_lsb_str
        last_hh_range_str = "[" + last_hh_msb_str + ":" + last_hh_lsb_str + "]"
        bank_addr_msb_str = str(min(int(math.ceil(math.log(self.words, 2))) - 1, bank_wire_addr_width + sel_hbank_reg_width - 1))
        bank_addr_lsb_str = str(sel_hbank_reg_width)
        bank_addr_range_str = "[" + bank_addr_msb_str + ":" + bank_addr_lsb_str + "]"
        fd.write("  generate\n")
        fd.write("  for (h = 0; h < " + str(self.hbanks) + "; h = h + 1) begin : gen_ctrl_hbanks\n")
        fd.write("    for (v = 0; v < " + str(self.vbanks) + "; v = v + 1) begin : gen_ctrl_vbanks\n")
        fd.write("      for (hh = 0; hh < " + str(self.hhbanks) + "; hh = hh + 1) begin : gen_ctrl_hhbanks\n")
        fd.write("\n")
        fd.write("        always @(*) begin : handle_ops\n")
        fd.write("\n")
        fd.write("// synthesis translate_off\n")
        fd.write("// synopsys translate_off\n")
        fd.write("          // Prevent assertions to trigger with false positive\n")
        fd.write("          # 1\n")
        fd.write("// synopsys translate_on\n")
        fd.write("// synthesis translate_on\n")
        fd.write("\n")
        fd.write("          /** Default **/\n")
        for d in range(0, self.dbanks):
            for p in range(0, self.bank_type.ports):
                if (ASSERT_ON):
                    # Initialize variable for conflicts check
                    fd.write("// synthesis translate_off\n")
                    fd.write("// synopsys translate_off\n")
                    fd.write("          check_bank_access["  + str(d) + "][h][v][hh]["  + str(p) + "] = -1;\n")
                    fd.write("// synopsys translate_on\n")
                    fd.write("// synthesis translate_on\n")
                # Dfault assignment
                fd.write("          bank_CE["  + str(d) + "][h][v][hh][" + str(p) + "]  = 0;\n")
                fd.write("          bank_A["   + str(d) + "][h][v][hh][" + str(p) + "]   = 0;\n")
                fd.write("          bank_D["   + str(d) + "][h][v][hh][" + str(p) + "]   = 0;\n")
                fd.write("          bank_WE["  + str(d) + "][h][v][hh][" + str(p) + "]  = 0;\n")
                fd.write("          bank_WEM[" + str(d) + "][h][v][hh][" + str(p) + "] = 0;\n")
            fd.write("\n")
        # Go through parallel accesses
        # In some cases we're building a full cross-bar, however most links will be trimmed away by
        # constant propagation if the accelerator is accessing data in a distributed fashion across
        # interfaces. This occurs, because ctrh[iface] becomes a constant.
        for op in self.ops:
            fd.write("          /** Handle " + str(op) + " **/\n")
            # Handle 2wu:0r
            if op.wp == "unknown" and op.wn == 2:
                for d in range(0, self.dbanks):
                    fd.write("          // Duplicated bank set " + str(d) + "\n")
                    for wi in range(0, op.wn):
                        p = self.write_ports[wi]
                        self.__write_ctrl_assignment(fd, bank_addr_range_str, hh_range_str, last_hh_range_str, d, p, wi, True, 0)
            # Handle <N>w:0r with N power of 2
            if op.rn == 0 and op.wp == "modulo":
                # Write to all duplicated sets
                for d in range(0, self.dbanks):
                    fd.write("          // Duplicated bank set " + str(d) + "\n")
                    for wi in range(0, op.wn):
                        p = self.write_ports[wi]
                        self.__write_ctrl_assignment(fd, bank_addr_range_str, hh_range_str, last_hh_range_str, d, p, wi, True, op.wn)
            # Handle 0w:<N>r with N power of 2
            if op.wn == 0 and op.rp == "modulo":
                # All duplicated sets would return the same data. 0 is correct even with no duplication
                d = 0
                fd.write("          // Always choose duplicated bank set " + str(d) + "\n")
                for ri in range(0, op.rn):
                    p = self.read_ports[ri]
                    self.__write_ctrl_assignment(fd, bank_addr_range_str, hh_range_str, last_hh_range_str, d, p, ri + self.write_interfaces, False, op.rn)

            # Handle <N>w:<M>r with N and M power of 2. In this case hbanks matches max(op.rn, op.wn) foreach op in the list of operations
            if op.wn > 0 and op.rn > 0 and op.wp == "modulo" and op.rp == "modulo":
                # Write to all duplicated sets
                for d in range(0, self.dbanks):
                    fd.write("          // Duplicated bank set " + str(d) + "\n")
                    for wi in range(0, op.wn):
                        p = self.write_ports[wi]
                        self.__write_ctrl_assignment(fd, bank_addr_range_str, hh_range_str, last_hh_range_str, d, p, wi, True, op.wn)
                # All duplicated sets would return the same data. 0 is correct even with no duplication
                d = 0
                fd.write("          // Always choose duplicated bank set " + str(d) + "\n")
                for ri in range(0, op.rn):
                    p = self.read_ports[ri]
                    self.__write_ctrl_assignment(fd, bank_addr_range_str, hh_range_str, last_hh_range_str, d, p, ri + self.write_interfaces, False, op.rn)
            # Handle <N>ru:0w with N > 1
            if op.rn > 1 and op.wn == 0 and op.rp == "unknown":
                # Duplicated set matches the read interface number
                for ri in range(0, op.rn):
                    p = self.read_ports[ri]
                    d = (ri + self.write_interfaces) % self.dbanks
                    self.__write_ctrl_assignment(fd, bank_addr_range_str, hh_range_str, last_hh_range_str, d, p, ri + self.write_interfaces, False, 0)
            # Handle <N>ru:<M>w with N > 1 and M power of 2
            if op.rn > 1 and op.wn > 0 and op.rp == "unknown" and op.wp == "modulo":
                # Write to all duplicated sets
                for d in range(0, self.dbanks):
                    fd.write("          // Duplicated bank set " + str(d) + "\n")
                    for wi in range(0, op.wn):
                        p = self.write_ports[wi]
                        self.__write_ctrl_assignment(fd, bank_addr_range_str, hh_range_str, last_hh_range_str, d, p, wi, True, op.wn)
                # Duplicated set matches the read interface number
                for ri in range(0, op.rn):
                    p = self.read_ports[ri]
                    d = (ri + self.write_interfaces) % self.dbanks
                    self.__write_ctrl_assignment(fd, bank_addr_range_str, hh_range_str, last_hh_range_str, d, p, ri + self.write_interfaces, False, 0)
            fd.write("\n")
        fd.write("        end\n")
        fd.write("\n")
        fd.write("      end\n")
        fd.write("    end\n")
        fd.write("  end\n")
        fd.write("  endgenerate\n")
        fd.write("\n")

        # Read port Q assignment
        # When parallel rw is required, port 0 is used for write and port 1 is used for read
        # Otherwise, modulo is applied to choose which port should be used.
        fd.write("  generate\n")
        fd.write("  for (hh = 0; hh < " + str(self.hhbanks) + "; hh = hh + 1) begin : gen_q_assign_hhbanks\n")
        q_last_hh_msb_str = str(int(min(self.width - 1, self.hhbanks * self.bank_type.width - 1)))
        q_last_hh_range_str = "[" + q_last_hh_msb_str + ":" + hh_lsb_str + "]"

        for ri in range(self.write_interfaces, self.write_interfaces + self.read_interfaces):
            p = self.read_ports[ri - self.write_interfaces]
            fd.write("    if (hh == " + str(self.hhbanks - 1) + " && (hh + 1) * " + str(self.bank_type.width) + " > " + str(self.width) + ") begin : gen_q_assign_hhbanks_last_" + str(ri) + " \n ")
            fd.write("      assign Q" + str(ri) + q_last_hh_range_str + " = bank_Q" + "[seld[" + str(ri) +"]]" + "[selh[" + str(ri) +"]]" + "[selv[" + str(ri) +"]]" + "[hh]" + "[" + str(p) + "][" + str((self.width - 1) % self.bank_type.width) + ":0];\n")
            fd.write("    end else begin : gen_q_assign_hhbanks_others_" + str(ri) + " \n")
            fd.write("      assign Q" + str(ri) + hh_range_str + " = bank_Q" + "[seld[" + str(ri) +"]]" + "[selh[" + str(ri) +"]]" + "[selv[" + str(ri) +"]]" + "[hh]" + "[" + str(p) + "];\n")
            fd.write("    end\n")
        fd.write("  end\n")
        fd.write("  endgenerate\n")
        fd.write("\n")

        # Bank instances
        fd.write("  generate\n")
        fd.write("  for (d = 0; d < " + str(self.dbanks) + "; d = d + 1) begin : gen_wires_dbanks\n")
        fd.write("    for (h = 0; h < " + str(self.hbanks) + "; h = h + 1) begin : gen_wires_hbanks\n")
        fd.write("      for (v = 0; v < " + str(self.vbanks) + "; v = v + 1) begin : gen_wires_vbanks\n")
        fd.write("        for (hh = 0; hh < " + str(self.hhbanks) + "; hh = hh + 1) begin : gen_wires_hhbanks\n")
        fd.write("\n")
        fd.write("          " + self.bank_type.name + " bank_i(\n")
        fd.write("              .CLK(CLK)")
        for p in range(0, self.bank_type.ports):
            fd.write(",\n              .CE"  + str(p) + "(bank_CE[d][h][v][hh]["  + str(p) + "])")
            fd.write(",\n              .A"   + str(p) + "(bank_A[d][h][v][hh]["   + str(p) + "])")
            fd.write(",\n              .D"   + str(p) + "(bank_D[d][h][v][hh]["   + str(p) + "])")
            fd.write(",\n              .WE"  + str(p) + "(bank_WE[d][h][v][hh]["  + str(p) + "])")
            fd.write(",\n              .WEM" + str(p) + "(bank_WEM[d][h][v][hh][" + str(p) + "])")
            fd.write(",\n              .Q"   + str(p) + "(bank_Q[d][h][v][hh]["   + str(p) + "])")
        fd.write("\n            );\n")
        fd.write("\n")
        if (ASSERT_ON):
            fd.write("// synthesis translate_off\n")
            fd.write("// synopsys translate_off\n")
            fd.write("            always @(posedge CLK) begin\n")
            for p0 in range(0, self.bank_type.ports):
                for p1 in range(p0 + 1, self.bank_type.ports):
                    fd.write("              if " + "((bank_CE[d][h][v][hh]["  + str(p0) + "] & " + "bank_CE[d][h][v][hh]["  + str(p1) + "]) &&\n")
                    fd.write("                 " + " (bank_WE[d][h][v][hh]["  + str(p0) + "] | " + "bank_WE[d][h][v][hh]["  + str(p1) + "]) &&\n")
                    fd.write("                 " + " (bank_A[d][h][v][hh]["  + str(p0) + "] == " + "bank_A[d][h][v][hh]["  + str(p1) + "])) begin\n")
                    fd.write("                $display(\"ASSERTION FAILED in %m: address conflict on bank\", h, \"h\", v, \"v\", hh, \"hh\");\n")
                    fd.write("                $finish;\n")
                    fd.write("              end\n")
                    fd.write("            end\n")
            fd.write("// synopsys translate_on\n")
            fd.write("// synthesis translate_on\n")
            fd.write("\n")
        fd.write("        end\n")
        fd.write("      end\n")
        fd.write("    end\n")
        fd.write("  end\n")
        fd.write("  endgenerate\n")
        fd.write("\n")
        fd.write("endmodule\n")
        fd.close()

    def __set_rwports(self):
        for wi in range(0, self.write_interfaces):
            if self.need_parallel_rw:
                self.write_ports.append(0)
            elif self.write_interfaces == 1:
                self.write_ports.append(0)
            elif self.write_interfaces == 2:
                self.write_ports.append(wi)
            else:
                self.write_ports.append((int(wi / self.hbanks) + (wi % self.bank_type.ports)) % self.bank_type.ports)
        for ri in range(0, self.read_interfaces):
            if self.need_parallel_rw:
                self.read_ports.append(1)
            elif self.read_interfaces == 1:
                self.read_ports.append(0)
            elif self.read_interfaces == 2:
                self.read_ports.append(ri)
            else:
                self.read_ports.append((int(ri / self.hbanks) + (ri % self.bank_type.ports)) % self.bank_type.ports)



    def gen(self, lib):
        # Determine memory requirements (first pass over ops list)
        self.__find_hbanks()
        self.__find_vbanks(lib)
        self.__set_rwports()
        print("        " + "read_interfaces " + str(self.read_interfaces))
        print("        " + "write_interfaces " + str(self.write_interfaces))
        print("        " + "duplication_factor " + str(self.duplication_factor))
        print("        " + "distribution_factor " + str(self.distribution_factor))
        print("        " + "need_dual_port " + str(self.need_dual_port))
        print("        " + "need_parallel_rw " + str(self.need_parallel_rw))
        print("        " + "d-banks " + str(self.dbanks))
        print("        " + "h-banks " + str(self.hbanks))
        print("        " + "v-banks " + str(self.vbanks))
        print("        " + "hh-banks " + str(self.hhbanks))
        print("        " + "bank type " + str(self.bank_type.name))
        print("        " + "Write interfaces are assigned to ports " + str(self.write_ports))
        print("        " + "Read interfaces are assigned to ports " + str(self.read_ports))
        print("        " + "Total area " + str(self.area))


    def write_tb(self, tb_path):
        try:
            fd = open(tb_path + "/" + self.name + "_tb.v", 'w')
        except IOError as e:
            die_werr(e)
        fd.write("/**\n")
        fd.write("* Created with the ESP Memory Generator\n")
        fd.write("*\n")
        fd.write("* Copyright (c) 2014-2017, Columbia University\n")
        fd.write("*\n")
        fd.write("* @author Paolo Mantovani <paolo@cs.columbia.edu>\n")
        fd.write("*/\n")
        fd.write("\n")
        fd.write("`timescale  1 ps / 1 ps\n")
        fd.write("\n")
        fd.write("module " + self.name + "_tb(\n")
        fd.write("\n  );\n")
        fd.write("  reg CLK;\n")
        for i in range(0, self.write_interfaces):
            fd.write("  " + "reg  CE" + str(i) + ";\n")
            fd.write("  " + "reg  " + "[" + str(int(math.ceil(math.log(self.words, 2)))-1) + ":0] A" + str(i) + ";\n")
            fd.write("  " + "reg  " + "[" + str(self.width-1) + ":0] D" + str(i) + ";\n")
            fd.write("  " + "reg  WE" + str(i) + ";\n")
            fd.write("  " + "reg  " + "[" + str(self.width-1) + ":0] WEM" + str(i) + ";\n")
        for i in range(self.write_interfaces, self.write_interfaces + self.read_interfaces):
            fd.write("  " + "reg  CE" + str(i) + ";\n")
            fd.write("  " + "reg  " + "[" + str(int(math.ceil(math.log(self.words, 2)))-1) + ":0] A" + str(i) + ";\n")
            fd.write("  " + "wire " + "[" + str(self.width-1) + ":0] Q" + str(i) + ";\n")
        fd.write("\n")
        fd.write("  initial begin\n")
        fd.write("    CLK = 0;\n")
        fd.write("    forever begin\n")
        fd.write("      #5000 CLK = !CLK;\n")
        fd.write("    end\n")
        fd.write("  end\n")
        fd.write("\n")
        fd.write("  initial begin\n")
        for iface in range(0, self.write_interfaces):
            fd.write("  CE" + str(iface) + " = 0;\n")
            fd.write("  A" + str(iface) + " = 0;\n")
            fd.write("  D" + str(iface) + " = 0;\n")
            fd.write("  WE" + str(iface) + " = 0;\n")
            fd.write("  WEM" + str(iface) + " = 0;\n")
        for iface in range(self.write_interfaces, self.write_interfaces + self.read_interfaces):
            fd.write("  CE" + str(iface) + " = 0;\n")
            fd.write("  A" + str(iface) + " = 0;\n")
        # Go through operations and merge them for testing purposes
        tb_ops = [ ]
        wmax = 0
        wumax = 0
        rmax = 0
        rumax = 0
        for op in self.ops:
            if op.wp == "modulo" and op.wn > wmax:
                wmax = op.wn
            if op.wp == "unknown" and op.wn == 2:
                wumax = 2
            if op.rp == "modulo" and op.rn > rmax:
                rmax = op.rn
            if op.rp == "unknown" and op.rn > rumax:
                rumax = op.rn
        # Do 2wu:0r first, if required
        if wumax != 0:
            tb_ops.append(memory_operation(0, "modulo", wumax, "unknown"))
            # Check what's written with any read-only op (even if not in the original list of ops)
            if rumax != 0:
                tb_ops.append(memory_operation(rumax, "unknown", 0, "modulo"))
            else:
                tb_ops.append(memory_operation(rmax, "modulo", 0, "modulo"))
        # Merge other modulo operations
        if rmax != 0 and wmax != 0 and self.need_parallel_rw:
            tb_ops.append(memory_operation(rmax, "modulo", wmax, "modulo"))
        else:
            if wmax != 0:
                tb_ops.append(memory_operation(0, "modulo", wmax, "modulo"))
            if rmax != 0:
                tb_ops.append(memory_operation(rmax, "modulo", 0, "modulo"))
        # Check parallel read/write with unknown pattern for read if required
        if rumax != 0 and wmax != 0 and self.need_parallel_rw:
            tb_ops.append(memory_operation(rumax, "unknown", wmax, "modulo"))
        elif rumax != 0:
            tb_ops.append(memory_operation(rumax, "unknown", 0, "modulo"))
        tb_operations = " ".join(map(str, tb_ops))
        print("  INFO: Generating testbench for " + self.name + " with merged operations " + tb_operations)
        # Write testbench
        for op in tb_ops:
            fd.write("  $display(\"\");\n")
            fd.write("  #500000 $display(\"* Testing parallel access " + str(op) + " *\");\n")
            # Reset memory content
            if op.wn != 0:
                fd.write("  $display(\"\");\n")
                fd.write("  $display(\"--- Set all memory cells to 0 for writing test ---\");\n")
                fd.write("  $display(\"\");\n")
                for addr in range(0, self.words):
                    wi = addr % op.wn
                    fd.write("  @ (posedge CLK) $display(\"Reset addr " + str(addr) + "\");\n")
                    fd.write("  CE" + str(wi) + " = 1'b1;\n")
                    fd.write("  A" + str(wi) + " = " + str(addr) + ";\n")
                    data = 0
                    data_str = str(data)
                    fd.write("  D" + str(wi) + " = " + data_str + ";\n")
                    fd.write("  WE" + str(wi) + " = 1'b1;\n")
                    fd.write("  WEM" + str(wi) + " = {" + str(self.width) + "{1'b1}};\n")
                    # Disable write interfaces before testing
                    fd.write("  @ (posedge CLK) CE" + str(wi) + " = 1'b0;\n")
            # By default we assume modulo access pattern
            wi_start = 0
            wi_end = op.wn
            wi_step = 1
            ri_start = self.write_interfaces
            ri_end = self.write_interfaces + op.rn
            ri_step = 1
            # Reversing the interface order makes them access different banks than the ones assigned for pattern modulo
            if op.wp == "unknown":
                wi_start = op.wn - 1
                wi_end = -1
                wi_step = -1
            if op.rp == "unknown":
                ri_start = self.write_interfaces + op.rn - 1
                ri_end =  self.write_interfaces - 1
                ri_step = -1
            fd.write("  $display(\"\");\n")
            fd.write("  $display(\"--- Begin test for " + str(op) + " ---\");\n")
            fd.write("  $display(\"\");\n")
            waddr = 0
            raddr = 0
            caddr = 0
            wcycle = 1
            rcycle = 1
            ccycle = 1
            format_str = "0" + str(int(math.ceil(self.width / 4))) + "x"
            while True:
                fd.write("  @ (posedge CLK) $display(\"Current waddr and raddr are " + str(waddr) + ", " + str(raddr) + "\");\n")
                if waddr >= self.words:
                    if op.wn != 0:
                        for wi in range(0, self.write_interfaces):
                            fd.write("  CE" + str(wi) + " = 1'b0;\n")
                    if op.rn == 0:
                        fd.write("  $display(\"\");\n")
                        fd.write("  $display(\"--- End of Test " + str(op) + " PASSED ---\");\n")
                        fd.write("  $display(\"\");\n")
                        break
                if raddr >= self.words:
                    if op.rn != 0:
                        for ri in range(self.write_interfaces, self.write_interfaces + self.read_interfaces):
                            fd.write("  CE" + str(ri) + " = 1'b0;\n")
                for wi in range(wi_start, wi_end, wi_step):
                    if waddr < self.words:
                        if op.wn != 0:
                            fd.write("  CE" + str(wi) + " = 1'b1;\n")
                            fd.write("  A" + str(wi) + " = " + str(waddr) + ";\n")
                            addr_width = int(math.ceil(math.log(self.words, 2)))
                            data = waddr
                            if ( addr_width > self.width):
                                data = waddr % (int(math.pow(2, self.width)) - 1)
                            if waddr % 2 != 0:
                                data = (data << (self.width - int(math.log(max(data, 1), 2)) - 1)) | data
                            data_str = str(self.width) + "'h" + format(data, format_str)
                            fd.write("  D" + str(wi) + " = " + data_str + ";\n")
                            fd.write("  WE" + str(wi) + " = 1'b1;\n")
                            fd.write("  WEM" + str(wi) + " = {" + str(self.width) + "{1'b1}};\n")
                        waddr = waddr + 1
                    else:
                        if op.wn != 0:
                            fd.write("  CE" + str(wi) + " = 1'b0;\n")
                wcycle = wcycle + 1
                if op.wn != 0:
                    if waddr < (min(raddr + op.rn - 1, self.words - 1)) or (wcycle - 1) / rcycle <= math.ceil(op.rn / op.wn):
                        continue
                for ri in range(ri_start, ri_end, ri_step):
                    if raddr < self.words:
                        if op.rn != 0:
                            fd.write("  CE" + str(ri) + " = 1'b1;\n")
                            fd.write("  A" + str(ri) + " = " + str(raddr) + ";\n")
                        raddr = raddr + 1
                    else:
                        if op.rn != 0:
                            fd.write("  CE" + str(ri) + " = 1'b0;\n")
                rcycle = rcycle + 1
                if raddr < (min(caddr + op.rn - 1, self.words - 1)) or rcycle - ccycle <= 1:
                    continue
                for ri in range(ri_start, ri_end, ri_step):
                    if caddr < self.words:
                        if op.rn != 0:
                            data = caddr
                            if ( addr_width > self.width):
                                data = caddr % (int(math.pow(2, self.width)) - 1)
                            if caddr % 2 != 0:
                                data = (data << (self.width - int(math.log(max(data, 1), 2)) - 1)) | data
                            data_str = str(self.width) + "'h" + format(data, format_str)
                            fd.write("  #200 ;\n")
                            fd.write("  if (Q" + str(ri) + " != " + str(data_str) + ") begin\n")
                            fd.write("    $display(\"Memory failure on interface " + str(ri) + " at address " + str(caddr) + ": reading %h, but expecting %h\", Q" + str(ri) + ", " + data_str + ");\n")
                            fd.write("    $finish;\n")
                            fd.write("  end\n")
                            fd.write("  else begin\n")
                            fd.write("    $display(\"Memory read on interface " + str(ri) + " at address " + str(caddr) + ": reading %h\", Q" + str(ri) + ");\n")
                            fd.write("  end\n")
                        caddr = caddr + 1
                ccycle = ccycle + 1
                if caddr >= self.words:
                    fd.write("  $display(\"\");\n")
                    fd.write("  $display(\"--- End of Test " + str(op) + " PASSED ---\");\n")
                    fd.write("  $display(\"\");\n")
                    break

        fd.write("  $display(\"\");\n")
        fd.write("  $display(\"*** Test completed successfully ***\");\n")
        fd.write("  $display(\"\");\n")
        fd.write("  $finish;\n")
        fd.write("  end\n")
        fd.write("\n")
        fd.write("  // Memory instance\n")
        fd.write("  " + self.name + " dut (\n")
        fd.write("    .CLK(CLK)")
        for i in range(0, self.write_interfaces):
            fd.write(",\n    .CE" + str(i) + "(CE" + str(i) + ")")
            fd.write(",\n    .A" + str(i) + "(A" + str(i) + ")")
            fd.write(",\n    .D" + str(i) + "(D" + str(i) + ")")
            fd.write(",\n    .WE" + str(i) + "(WE" + str(i) + ")")
            fd.write(",\n    .WEM" + str(i) + "(WEM" + str(i) + ")")
        for i in range(self.write_interfaces, self.write_interfaces + self.read_interfaces):
            fd.write(",\n    .CE" + str(i) + "(CE" + str(i) + ")")
            fd.write(",\n    .A" + str(i) + "(A" + str(i) + ")")
            fd.write(",\n    .Q" + str(i) + "(Q" + str(i) + ")")
        fd.write("\n  );\n")
        fd.write("\n")
        fd.write("endmodule\n")
        fd.close()


    def write_hpp(self):
        try:
            fd = open("./memlib/" + self.name + ".hpp", 'w')
        except IOError as e:
            die_werr(e)
        print("  INFO: Generating SystemC explicit memory definition for " + self.name)
        fd.write("/**\n")
        fd.write("* Created with the ESP Memory Generator\n")
        fd.write("*\n")
        fd.write("* Copyright (c) 2014-2017, Columbia University\n")
        fd.write("*\n")
        fd.write("* @author Paolo Mantovani <paolo@cs.columbia.edu>\n")
        fd.write("*/\n")
        fd.write("\n")
        fd.write("#ifndef __" + self.name.upper() + "_HPP__\n")
        fd.write("#define __" + self.name.upper() + "_HPP__\n")
        fd.write("#include \"" + self.name + ".h\"\n")
        fd.write("template<class T, unsigned S, typename ioConfig=CYN::PIN>\n")
        fd.write("class " + self.name +"_t : public sc_module\n")
        fd.write("{\n")
        fd.write("\n")
        fd.write("  HLS_INLINE_MODULE;\n")
        fd.write("public:\n")
        fd.write("  " + self.name + "_t(const sc_module_name& name = sc_gen_unique_name(\"" + self.name + "\"))\n")
        fd.write("  : sc_module(name)\n")
        fd.write("  , clk(\"clk\")\n")
        for iface in range(1, self.read_interfaces + self.write_interfaces + 1):
            fd.write("  , port" + str(iface) + "(\"port" + str(iface) + "\")\n")
        fd.write("  {\n")
        fd.write("    m_m0.clk_rst(clk);\n")
        for iface in range(1, self.read_interfaces + self.write_interfaces + 1):
            fd.write("    port" + str(iface) + "(m_m0.if" + str(iface) + ");\n")
        fd.write("  }\n")
        fd.write("\n")
        fd.write("  sc_in<bool> clk;\n")
        fd.write("\n")
        fd.write("  " + self.name + "::wrapper<ioConfig> m_m0;\n")
        fd.write("\n")
        for iface in range(1, self.read_interfaces + self.write_interfaces + 1):
            # TODO: there is a bug in Stratus that prevents a port interface to have a single dimension (so use [1] for now)
            fd.write("  typedef " + self.name + "::port_" + str(iface) + "<ioConfig, T[1][S]> Port" + str(iface) + "_t;\n")
        fd.write("\n")
        for iface in range(1, self.read_interfaces + self.write_interfaces + 1):
            fd.write("  Port" + str(iface) + "_t port" + str(iface) + ";\n")
        fd.write("};\n")
        fd.write("#endif\n")
        fd.close()

    def write_bdm(self, out_path, tech_path):
        try:
            fd = open("./memlib/" + self.name + ".bdm", 'w')
        except IOError as e:
            die_werr(e)
        print("  INFO: Generating Stratus bdw_memgen input for " + self.name)
        fd.write("setName " + self.name + "\n")
        fd.write("catch {setToolVersion \"16.20-p100\"}\n")
        fd.write("setModelStyle 1\n")
        fd.write("setWordSize " + str(self.width) + "\n")
        fd.write("setNumWords " + str(self.words) + "\n")
        fd.write("setArea " + str(self.area) + "\n")
        fd.write("setLatency 1\n")
        fd.write("setDelay " + str(mem_delay) + "\n")
        fd.write("setSetup " + str(mem_setup) + "\n")
        fd.write("setInputDataFormat 1\n")
        fd.write("setFileSuffix 0\n")
        fd.write("setHasReset 0\n")
        fd.write("setNoSpecReads 1\n")
        fd.write("setSharedAccess 1\n")
        fd.write("setMaxSharedPorts 32\n")
        fd.write("setIntRegMemIn 0\n")
        fd.write("setIntRegMemOut 0\n")
        fd.write("setExtRegMemIn 0\n")
        fd.write("setExtRegMemOut 0\n")
        fd.write("setIntRegsAtMemIn 0\n")
        fd.write("setIntRegsAtMemOut 0\n")
        fd.write("setExtRegsAtMemIn 0\n")
        fd.write("setExtRegsAtMemOut 0\n")
        fd.write("setClockMult 0\n")
        fd.write("setNumAccessPorts " + str(self.read_interfaces + self.write_interfaces) + "\n")
        fd.write("setNumExtraPorts 0\n")
        fd.write("setPipelined 0\n")
        fd.write("setVendorModel \"" + tech_path + "/" + self.bank_type.name + ".v\"\n")
        fd.write("setModelWrapper \"" + out_path + "/" + self.name + ".v\"\n")
        fd.write("setExtraPortsInWrapper 0\n")
        port_count = 0;
        for wi in range(self.write_interfaces):
            fd.write("setPortData " + str(port_count) + " setType 1\n")
            fd.write("setPortData " + str(port_count) + " setAddr \"A" + str(wi) + "\"\n")
            fd.write("setPortData " + str(port_count) + " setClk \"CLK\"\n")
            fd.write("setPortData " + str(port_count) + " setReqName \"REQ" + str(wi) + "\"\n")
            fd.write("setPortData " + str(port_count) + " setDin \"D" + str(wi) + "\"\n")
            fd.write("setPortData " + str(port_count) + " setHasRW 1\n")
            fd.write("setPortData " + str(port_count) + " setRwName \"WE" + str(wi) + "\"\n")
            fd.write("setPortData " + str(port_count) + " setRwActive 1\n")
            fd.write("setPortData " + str(port_count) + " setWMWord 1\n")
            fd.write("setPortData " + str(port_count) + " setWMName \"WEM" + str(wi) + "\"\n")
            fd.write("setPortData " + str(port_count) + " setWMActive 1\n")
            fd.write("setPortData " + str(port_count) + " setHasCE 1\n")
            fd.write("setPortData " + str(port_count) + " setCEName \"CE" + str(wi) + "\"\n")
            fd.write("setPortData " + str(port_count) + " setCEActive 1\n")
            port_count = port_count + 1;
        for ri in range(self.write_interfaces, self.read_interfaces + self.write_interfaces):
            fd.write("setPortData " + str(port_count) + " setType 0\n")
            fd.write("setPortData " + str(port_count) + " setAddr \"A" + str(ri) + "\"\n")
            fd.write("setPortData " + str(port_count) + " setClk \"CLK\"\n")
            fd.write("setPortData " + str(port_count) + " setReqName \"REQ" + str(ri) + "\"\n")
            fd.write("setPortData " + str(port_count) + " setDout \"Q" + str(ri) + "\"\n")
            fd.write("setPortData " + str(port_count) + " setHasOE 0\n")
            fd.write("setPortData " + str(port_count) + " setHasRE 0\n")
            fd.write("setPortData " + str(port_count) + " setHasCE 1\n")
            fd.write("setPortData " + str(port_count) + " setCEName \"CE" + str(ri) + "\"\n")
            fd.write("setPortData " + str(port_count) + " setCEActive 1\n")
            port_count = port_count + 1;
        fd.close()

        # for wi in range(self.write_interfaces - 1, -1, -1):
        #     fd.write("setPortData " + str(port_count) + " setType 1\n")
        #     fd.write("setPortData " + str(port_count) + " setAddr \"A" + str(wi) + "\"\n")
        #     fd.write("setPortData " + str(port_count) + " setClk \"CLK\"\n")
        #     fd.write("setPortData " + str(port_count) + " setReqName \"REQ" + str(wi) + "\"\n")
        #     fd.write("setPortData " + str(port_count) + " setDin \"D" + str(wi) + "\"\n")
        #     fd.write("setPortData " + str(port_count) + " setHasRW 1\n")
        #     fd.write("setPortData " + str(port_count) + " setRwName \"WE" + str(wi) + "\"\n")
        #     fd.write("setPortData " + str(port_count) + " setRwActive 1\n")
        #     fd.write("setPortData " + str(port_count) + " setWMWord 1\n")
        #     fd.write("setPortData " + str(port_count) + " setWMName \"WEM" + str(wi) + "\"\n")
        #     fd.write("setPortData " + str(port_count) + " setWMActive 1\n")
        #     fd.write("setPortData " + str(port_count) + " setHasCE 1\n")
        #     fd.write("setPortData " + str(port_count) + " setCEName \"CE" + str(wi) + "\"\n")
        #     fd.write("setPortData " + str(port_count) + " setCEActive 1\n")
        #     port_count = port_count + 1;
        # for ri in range(self.read_interfaces + self.write_interfaces - 1, self.write_interfaces - 1, -1):
        #     fd.write("setPortData " + str(port_count) + " setType 0\n")
        #     fd.write("setPortData " + str(port_count) + " setAddr \"A" + str(ri) + "\"\n")
        #     fd.write("setPortData " + str(port_count) + " setClk \"CLK\"\n")
        #     fd.write("setPortData " + str(port_count) + " setReqName \"REQ" + str(ri) + "\"\n")
        #     fd.write("setPortData " + str(port_count) + " setDout \"Q" + str(ri) + "\"\n")
        #     fd.write("setPortData " + str(port_count) + " setHasOE 0\n")
        #     fd.write("setPortData " + str(port_count) + " setHasRE 0\n")
        #     fd.write("setPortData " + str(port_count) + " setHasCE 1\n")
        #     fd.write("setPortData " + str(port_count) + " setCEName \"CE" + str(ri) + "\"\n")
        #     fd.write("setPortData " + str(port_count) + " setCEActive 1\n")
        #     port_count = port_count + 1;
        # fd.close()


### Input parsing ###
def parse_sram(s):
    item = s.split()
    words = int(item[0])
    width = int(item[1])
    name = item[2]
    area = float(item[3])
    # We assume rw ports supporting one read or one write per cycle
    ports = int(item[4])
    if ports < 1 or ports > 2:
        warn("Skipping SRAM type " + name + " with unsopported number of ports")
        return None
    return sram(name, words, width, area, ports)

def parse_op(op, mem_words):
    item = op.split(":")
    write_number = int(re.split('[a-z]+', item[0], re.M|re.I)[0])
    write_pattern_abbrv = str(re.split('[0-9]+', item[0], re.M|re.I)[1])
    read_number = int(re.split('[a-z]+', item[1], re.M|re.I)[0])
    read_pattern_abbrv = str(re.split('[0-9]+', item[1], re.M|re.I)[1])

    if read_number > mem_words or write_number > mem_words:
        die_werr("Too many ports for the specified number of words for "+op)

    if read_number > 16 or read_number < 0:
        die_werr("Too many paralle accesses specified for "+op);

    if re.match(r'ru', read_pattern_abbrv, re.M|re.I):
        read_pattern = "unknown"
    elif re.match(r'r', read_pattern_abbrv, re.M|re.I):
        read_pattern = "modulo"
        if not is_power2z(read_number):
            die_werr("Operation "+op+" implies known access patter (modulo), but the number of accesses is not a power of 2")
    else:
        die_werr("Parallel read access "+op+" not recognized")

    if write_number > 16 or write_number < 0:
        die_werr("Too many paralle accesses specified for "+op);

    if re.match(r'wu', write_pattern_abbrv, re.M|re.I):
        write_pattern = "unknown"
        if write_number > 2:
            die_werr("Too many parallel write accesses with unknown pattern for "+op)
        if write_number == 2 and read_number != 0:
            die_werr("2 parallel write accesses with unknown pattern for "+op+" have non-zero parallel read access")
    elif re.match(r'w', write_pattern_abbrv, re.M|re.I):
        write_pattern = "modulo"
        if not is_power2z(write_number):
            die_werr("Operation "+op+" implies known access patter (modulo), but the number of accesses is not a power of 2")
    else:
        die_werr("Parallel write access "+op+" not recognized")

    return memory_operation(read_number, read_pattern, write_number, write_pattern)



def read_techfile(tech_path, sram_list):
    global mem_delay
    global mem_setup
    try:
        fd = open(tech_path + "/lib.txt", 'r')
    except IOError as e:
        die_werr(e)
    for line in fd:
        line.strip()
        item = line.split()
        # Check for commented line
        if re.match(r'# delay*', line, re.M|re.I):
            mem_delay = float(item[2])
        if re.match(r'# setup*', line, re.M|re.I):
            mem_setup = float(item[2])
        if re.match(r'#\.*', line, re.M|re.I):
            continue
        ram = parse_sram(line)
        if ram == None:
            continue
        sram_list.append(ram)
    fd.close()


def read_infile(name, mem_list):
    try:
        fd = open(name, 'r')
    except IOError as e:
        die_werr(e)
    for line in fd:
        line.strip()
        item = line.split()
        # Check for commented line
        if re.match(r'#\.*', line, re.M|re.I):
            continue
        mem_name = item[0]
        mem_words = int(item[1])
        mem_width = int(item[2])
        mem_ops = []
        for i in range(3, len(item)):
            op = parse_op(item[i], mem_words)
            mem_ops.append(op)
        mem = memory(mem_name, mem_words, mem_width, mem_ops)
        mem_list.append(mem)
    fd.close()


### Start script ###
if len(sys.argv) != 4:
    print_usage()
tech_path = sys.argv[1]
infile = sys.argv[2]
out_path = sys.argv[3]
tb_path = out_path + "/tb"
mem_list = []
sram_list = []

print("  INFO: Target technology path: " + tech_path)
read_techfile(tech_path, sram_list)
for ram in sram_list:
    ram.print()

print("  INFO: Output path: " + out_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)

if not os.path.exists(tb_path):
    os.makedirs(tb_path)

print("  INFO: Stratus memory library path: ./memlib")
if not os.path.exists("memlib"):
    os.makedirs("memlib")


print("  INFO: Memory list file: " + infile)
read_infile(infile, mem_list)
if len(mem_list) == 0:
    print("  INFO: Memory list is empty")
    sys.exit(0)

for mem in mem_list:
    mem.print()
    mem.gen(sram_list)
    mem.write_verilog(out_path)
    # mem.write_tb(tb_path)
    mem.write_hpp()
    mem.write_bdm(out_path, tech_path)

