"""Helpers to elaborate and simulate a generated parameter-banks wrapper.

The interesting failures in this feature are structural (a top that does not
elaborate) or temporal (a bank selected one cycle late). Neither shows up in a
pack/unpack round trip, so these helpers drive the real RTL through xsim.
"""

import os
import re
import shutil
import subprocess
from fractions import Fraction

# ap_fixed<16,6>, AP_TRN / AP_WRAP -- the default hls4ml precision used by the tests
WIDTH, INTEGER = 16, 6
FRAC = WIDTH - INTEGER


def fx(value):
    """Quantize onto the ap_fixed grid, returning a Fraction."""
    scaled = Fraction(value).limit_denominator(1 << 30) * (1 << FRAC)
    code = (scaled.numerator // scaled.denominator) % (1 << WIDTH)
    if code & (1 << (WIDTH - 1)):
        code -= 1 << WIDTH
    return Fraction(code, 1 << FRAC)


def code_of(value):
    """Quantize onto the ap_fixed grid, returning the raw two's-complement code."""
    scaled = Fraction(value).limit_denominator(1 << 30) * (1 << FRAC)
    return (scaled.numerator // scaled.denominator) % (1 << WIDTH)


def dense_reference(x, weights, bias):
    """Fixed-point reference for nnet::dense_resource with accum_t == result_t.

    acc = (accum_t) bias; acc += (accum_t)(data * weight); res = (res_t) acc.
    """
    n_in, n_out = len(x), len(bias)
    xq = [fx(v) for v in x]
    wq = [[fx(weights[i][o]) for o in range(n_out)] for i in range(n_in)]
    out = []
    for o in range(n_out):
        acc = fx(bias[o])
        for i in range(n_in):
            acc = fx(acc + fx(xq[i] * wq[i][o]))
        out.append(acc)
    return out


def have_xsim():
    return all(shutil.which(tool) for tool in ('xvlog', 'xelab', 'xsim'))


def have_vivado():
    return shutil.which('vivado') is not None


def run_vivado_batch(tcl_name, cwd, marker, timeout=1800):
    """Run a Tcl script through batch-mode Vivado; returns (marker_seen, log).

    hls4ml has no build API for this - it stops at IP export - so the invocation
    lives here with the other tool calls rather than inline in a test.
    """
    result = subprocess.run(
        ['vivado', '-mode', 'batch', '-nojournal', '-nolog', '-source', tcl_name],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    log = result.stdout + result.stderr
    return marker in result.stdout, log


def write_testbench(path, project_name, input_codes, outputs, summary, banks, expected):
    """Emit a self-checking testbench for any number of banks and parameters.

    ``summary`` is what package() returned: the loader geometry and each
    parameter's id come from there, not from the test's own port order.
    ``banks`` is a list, one entry per bank, of {port_name: words or scalar codes};
    ``expected`` is a list, one per bank, of {output_name: [codes]}. The same input
    vector is applied to every bank, so an output difference can only come from
    bank selection.
    """
    # instantiate through the Verilog shim IP Integrator users get, so the
    # simulation also proves the shim forwards every port of the .sv top
    top = f'{project_name}_parameter_banks_bd'
    n_banks = len(banks)
    bank_bits = max((n_banks - 1).bit_length(), 1)
    n_in = len(input_codes)
    g = summary['loader']
    params = summary['banked_ports']
    first = params[0]
    pid_w, addr_w, data_w = g['param_id_width'], g['addr_width'], g['data_width']

    lines = [
        '`timescale 1 ns / 1 ps',
        '',
        'module tb;',
        '  reg ap_clk = 0, ap_rst = 1;',
        '  always #5 ap_clk = ~ap_clk;',
        '',
        '  reg ext_ap_start = 0;',
        f'  reg [{bank_bits - 1}:0] ext_bank_id = 0;',
        '  wire ext_ap_ready, ext_ap_done, ext_bank_id_bad;',
        f'  reg [{n_in * WIDTH - 1}:0] input_1 = 0;',
        '  reg input_1_ap_vld = 0;',
    ]
    for name, count in outputs:
        for o in range(count):
            lines.append(f'  wire [{WIDTH - 1}:0] {name}_{o};')
            lines.append(f'  wire {name}_{o}_ap_vld;')
            # hls4ml drives each output with its own ap_vld; capture on that pulse
            # rather than sampling at ap_done, which holds only for a single layer
            lines.append(f'  reg [{WIDTH - 1}:0] cap_{name}_{o};')
            lines.append(f'  reg seen_{name}_{o};')
    lines += [
        '  reg ld_req = 0;',
        f'  reg [{pid_w - 1}:0] ld_param_id = 0;',
        f'  reg [{bank_bits - 1}:0] ld_bank = 0;',
        f'  reg [{addr_w - 1}:0] ld_addr = 0;',
        f'  reg [{data_w - 1}:0] ld_data = 0;',
        '  wire ld_accept, ld_reject;',
    ]
    lines += [
        f'  wire [{bank_bits - 1}:0] cur_bank_id;',
        '  wire busy, quiescent;',
        '  integer errors = 0, n_done = 0, want_done;',
        '  always @(posedge ap_clk) if (ext_ap_done) n_done <= n_done + 1;',
        '',
        f'  {top} dut (',  # bank count is fixed by the packager
        '    .ap_clk(ap_clk), .ap_rst(ap_rst),',
        '    .ext_ap_start(ext_ap_start), .ext_bank_id(ext_bank_id),',
        '    .ext_ap_ready(ext_ap_ready), .ext_ap_done(ext_ap_done),',
        '    .ext_bank_id_bad(ext_bank_id_bad),',
        '    .input_1(input_1), .input_1_ap_vld(input_1_ap_vld),',
    ]
    for name, count in outputs:
        for o in range(count):
            lines.append(f'    .{name}_{o}({name}_{o}), .{name}_{o}_ap_vld({name}_{o}_ap_vld),')
    lines += [
        '    .ld_req(ld_req), .ld_param_id(ld_param_id), .ld_bank(ld_bank),',
        '    .ld_addr(ld_addr), .ld_data(ld_data), .ld_accept(ld_accept), .ld_reject(ld_reject),',
    ]
    lines += [
        '    .cur_bank_id(cur_bank_id), .busy(busy), .quiescent(quiescent));',
        '',
    ]
    for name, count in outputs:
        for o in range(count):
            lines.append(
                f'  always @(posedge ap_clk) if ({name}_{o}_ap_vld) begin '
                f"cap_{name}_{o} <= {name}_{o}; seen_{name}_{o} <= 1'b1; end"
            )
    lines += [
        '',
        '  // one loader write; want tells whether it must be accepted',
        '  task ld_write(input integer pid, input integer bank, input integer addr,',
        f'                input [{data_w - 1}:0] data, input want);',
        '    begin',
        '      @(negedge ap_clk);',
        f'      ld_param_id = pid[{pid_w - 1}:0]; ld_bank = bank[{bank_bits - 1}:0]; ld_addr = addr[{addr_w - 1}:0];',
        '      ld_data = data; ld_req = 1;',
        '      @(negedge ap_clk);',
        '      if (ld_accept !== want || ld_reject !== !want) begin',
        '        $display("FAIL: param %0d bank %0d addr %0d: accept=%b expected %b", pid, bank, addr, ld_accept, want);',
        '        errors = errors + 1;',
        '      end',
        '      ld_req = 0;',
        '    end',
        '  endtask',
        '',
        '  // ap_ctrl_hs: the request is taken at a rising edge where ext_ap_ready is high',
        '  task wait_accept;',
        '    begin',
        '      @(posedge ap_clk);',
        "      while (ext_ap_ready !== 1'b1) @(posedge ap_clk);",
        '      @(negedge ap_clk);',
        '    end',
        '  endtask',
        '',
        '  task run_bank(input integer bank);',
        '    begin',
        '      @(negedge ap_clk);',
        '      want_done = n_done + 1;',
        '      ext_bank_id = bank[%d:0]; ext_ap_start = 1; input_1_ap_vld = 1;' % (bank_bits - 1),
        '      wait_accept;',
        '      ext_ap_start = 0;',
        '      // a dataflow IP can raise ap_done no later than ap_ready',
        '      wait (n_done >= want_done);',
        '      @(negedge ap_clk);',
        '      input_1_ap_vld = 0;',
        '    end',
        '  endtask',
        '',
        '  initial begin',
        f"    input_1 = {n_in * WIDTH}'h{''.join(f'{c:04x}' for c in reversed(input_codes))};",
        '    repeat (5) @(negedge ap_clk);',
        '    ap_rst = 0;',
        '    repeat (2) @(negedge ap_clk);',
        '',
        '    // load every parameter of every bank while idle',
    ]
    for b, payload in enumerate(banks):
        for q in params:
            w = q['ld_data_width']
            for a, value in enumerate(payload[q['name']]):
                lines.append(f"    ld_write({q['param_id']}, {b}, {a}, {data_w}'h{value:0{(w + 3) // 4}x}, 1'b1);")
    # Every loader write costs two cycles, one per word and per scalar of every
    # bank, so the bound scales with the payload rather than being a constant that
    # quietly becomes too small for a larger model.
    writes = sum(len(v) for payload in banks for v in payload.values())
    timeout = 40 * writes + 20000 * n_banks + 20000

    lines.append('')
    lines.append('    // same input to every bank: any output difference is bank selection')
    seen_all = ' & '.join(f'seen_{name}_{o}' for name, count in outputs for o in range(count))
    for b in range(n_banks):
        lines.append('    @(negedge ap_clk);')
        for name, count in outputs:
            for o in range(count):
                lines.append(f"    cap_{name}_{o} = {WIDTH}'hx; seen_{name}_{o} = 1'b0;")
        lines.append(f'    run_bank({b});')
        # ap_done can precede an output's own ap_vld: with two heads the slower one
        # asserts valid a cycle after done. Wait for the data, not the transaction.
        lines.append(f'    wait ({seen_all});')
        lines.append('    @(negedge ap_clk);')
        for name, count in outputs:
            for o in range(count):
                exp = expected[b][name][o]
                lines += [
                    f"    if (cap_{name}_{o} !== {WIDTH}'h{exp:04x}) begin",
                    f'      $display("FAIL: bank {b} {name}_{o} = %04x expected {exp:04x}", cap_{name}_{o});',
                    '      errors = errors + 1;',
                    '    end',
                ]
    lines += [
        '',
        '    repeat (20) @(negedge ap_clk);',
        '    if (busy !== 1\'b0) begin $display("FAIL: busy stuck high"); errors = errors + 1; end',
        '',
        '    // an in-range write while idle must still be accepted',
        f"    ld_write({first['param_id']}, {n_banks - 1}, 0, 0, 1'b1);",
        '',
        '    // out-of-range values are rejected, not narrowed onto a valid location',
    ]
    # a parameter whose depth leaves an encodable address above it
    short = next((q for q in params if q['ld_depth'] < (1 << addr_w)), None)
    if short:
        lines.append(f"    ld_write({short['param_id']}, {n_banks - 1}, {short['ld_depth']}, 0, 1'b0);")
    if g['n_params'] < (1 << pid_w):
        lines.append(f"    ld_write({g['n_params']}, 0, 0, 0, 1'b0);")
    if n_banks < (1 << bank_bits):
        lines.append(f"    ld_write({first['param_id']}, {n_banks}, 0, 0, 1'b0);")
    lines += [
        '',
        '    // a write while a request is presented must be refused',
        '    @(negedge ap_clk);',
        '    want_done = n_done + 1;',
        '    ext_bank_id = 0; ext_ap_start = 1; input_1_ap_vld = 1;',
        f"    ld_write({first['param_id']}, 0, 0, 0, 1'b0);",
        '    wait_accept;',
        '    ext_ap_start = 0;',
        '    wait (n_done >= want_done);',
        '    @(negedge ap_clk); input_1_ap_vld = 0;',
        '',
        '    if (errors == 0) $display("PARAMETER_BANKS_PASS");',
        '    else $display("PARAMETER_BANKS_FAIL errors=%0d", errors);',
        '    $finish;',
        '  end',
        '',
        '  initial begin',
        f'    #{timeout};',
        '    $display("PARAMETER_BANKS_FAIL timeout");',
        '    $finish;',
        '  end',
        'endmodule',
    ]
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return path


def run_xsim(work_dir, rtl_dirs, tb_file, top='tb'):
    """Compile and run; returns (passed, log)."""
    os.makedirs(work_dir, exist_ok=True)
    sv, v = [], []
    for d in rtl_dirs:
        for name in sorted(os.listdir(d)):
            path = os.path.join(d, name)
            if name.endswith('.sv'):
                sv.append(path)
            elif name.endswith('.v'):
                v.append(path)
            elif name.endswith('.dat'):
                # Vitis initializes generated ROMs with $readmemh("./<name>.dat"),
                # a path relative to the simulator's cwd. Without this the ROM
                # stays X and the failure appears as an X-valued output, not as a
                # load error. dense_resource_rf_gt_nin_rem0 emits one.
                shutil.copy(path, work_dir)

    commands = []
    if sv:
        commands.append(['xvlog', '-sv', *sv, tb_file])
    if v:
        commands.append(['xvlog', *v])
    commands += [['xelab', top, '-s', 'pb_sim'], ['xsim', 'pb_sim', '-runall']]

    log = ''
    for cmd in commands:
        result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True, timeout=600)
        log += f'$ {" ".join(cmd)}\n{result.stdout}\n{result.stderr}\n'
        if result.returncode != 0 or re.search(r'^ERROR', result.stdout, re.M):
            return False, log
    return 'PARAMETER_BANKS_PASS' in log, log


LATCH_TB = r"""
`timescale 1 ns / 1 ps

// bank_select_latch against a pipelined IP model: one start per cycle, done L
// cycles later, ap_idle low while anything is in flight or ap_start is high.
module tb;
  localparam int NB = 3, W = 2, L = 4;

  reg clk = 0, rst = 1;
  always #5 clk = ~clk;

  reg ext_ap_start = 0;
  reg [W-1:0] ext_bank_id = 0;
  wire ext_ap_ready, ext_bank_id_bad, hls_ap_start, busy, quiescent;
  wire [W-1:0] cur_bank_id;

  reg [L-1:0] pipe = 0;
  wire hls_ap_ready = hls_ap_start;
  wire hls_ap_idle = (pipe == 0) & ~hls_ap_start;
  always @(posedge clk) pipe <= rst ? '0 : {pipe[L-2:0], hls_ap_start};

  bank_select_latch #(.BANK_ID_WIDTH(W), .N_BANKS(NB)) dut (
    .ap_clk(clk), .ap_rst(rst),
    .ext_ap_start(ext_ap_start), .ext_bank_id(ext_bank_id),
    .ext_ap_ready(ext_ap_ready), .ext_bank_id_bad(ext_bank_id_bad),
    .hls_ap_start(hls_ap_start), .hls_ap_ready(hls_ap_ready), .hls_ap_idle(hls_ap_idle),
    .cur_bank_id(cur_bank_id), .busy(busy), .quiescent(quiescent));

  integer errors = 0, accepts = 0, cyc = 0, t_req = 0;
  reg [W-1:0] prev_bank = 0;
  always @(posedge clk) begin
    cyc <= cyc + 1;
    if (!rst) begin
      if (hls_ap_start & hls_ap_ready) accepts = accepts + 1;
      if (pipe != 0 && cur_bank_id !== prev_bank) begin
        $display("FAIL: bank changed with a transaction in flight"); errors = errors + 1;
      end
    end
    prev_bank <= cur_bank_id;
  end

  initial begin
    repeat (4) @(negedge clk);
    rst = 0;
    @(negedge clk);

    ext_bank_id = 0; ext_ap_start = 1;
    repeat (6) @(negedge clk);
    if (accepts != 6) begin
      $display("FAIL: %0d of 6 requests for the selected bank accepted", accepts); errors = errors + 1;
    end
    if (quiescent !== 1'b0) begin $display("FAIL: quiescent while busy"); errors = errors + 1; end

    ext_bank_id = 2; accepts = 0; t_req = cyc;
    #1;
    if (ext_ap_ready !== 1'b0 || hls_ap_start !== 1'b0) begin
      $display("FAIL: bank change not held while busy"); errors = errors + 1;
    end
    wait (accepts == 1);
    if (cur_bank_id !== 2'd2) begin $display("FAIL: bank %0d after change", cur_bank_id); errors = errors + 1; end
    if (cyc - t_req < L) begin $display("FAIL: bank change before the IP drained"); errors = errors + 1; end

    @(negedge clk);
    ext_bank_id = 3;
    #1;
    if (ext_bank_id_bad !== 1'b1 || ext_ap_ready !== 1'b1 || hls_ap_start !== 1'b0) begin
      $display("FAIL: out-of-range bank not dropped"); errors = errors + 1;
    end
    @(negedge clk);
    if (cur_bank_id !== 2'd2) begin $display("FAIL: out-of-range bank selected"); errors = errors + 1; end

    ext_ap_start = 0;
    repeat (L + 2) @(negedge clk);
    if (busy !== 1'b0 || quiescent !== 1'b1) begin
      $display("FAIL: not quiescent after the last transaction"); errors = errors + 1;
    end

    if (errors == 0) $display("PARAMETER_BANKS_PASS");
    else $display("PARAMETER_BANKS_FAIL errors=%0d", errors);
    $finish;
  end

  initial begin
    #50000; $display("PARAMETER_BANKS_FAIL timeout"); $finish;
  end
endmodule
"""


def write_latch_testbench(path):
    """Standalone bench for bank_select_latch; needs no synthesized IP."""
    with open(path, 'w') as fh:
        fh.write(LATCH_TB)
    return path
