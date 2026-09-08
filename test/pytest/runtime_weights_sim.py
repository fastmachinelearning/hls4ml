"""Helpers to elaborate and simulate a generated runtime-weights wrapper.

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


def write_testbench(path, project_name, input_codes, outputs, bram_ports, scalar_ports, banks, expected):
    """Emit a self-checking testbench for any number of banks and parameters.

    ``banks`` is a list, one entry per bank, of {port_name: words or scalar codes};
    ``expected`` is a list, one per bank, of {output_name: [codes]}. The same input
    vector is applied to every bank, so an output difference can only come from
    bank selection.
    """
    top = f'{project_name}_runtime_weights'
    n_banks = len(banks)
    bank_bits = max((n_banks - 1).bit_length(), 1)
    n_in = len(input_codes)
    first_bram = bram_ports[0]['name']

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
    for p in bram_ports:
        n = p['name']
        word_bits = max((p['expected_depth'] - 1).bit_length(), 1)
        lines += [
            f'  reg ld_{n}_req = 0;',
            f'  reg [{bank_bits - 1}:0] ld_{n}_bank = 0;',
            f'  reg [{word_bits - 1}:0] ld_{n}_word = 0;',
            f'  reg [{p["actual_data_width"] - 1}:0] ld_{n}_wdata = 0;',
            f'  wire ld_{n}_accept, ld_{n}_reject;',
        ]
    for p in scalar_ports:
        n = p['name']
        idx_bits = max((p['n_scalars'] - 1).bit_length(), 1)
        lines += [
            f'  reg ld_{n}_we = 0;',
            f'  reg [{bank_bits - 1}:0] ld_{n}_bank = 0;',
            f'  reg [{idx_bits - 1}:0] ld_{n}_idx = 0;',
            f'  reg [{p["actual_width"] - 1}:0] ld_{n}_data = 0;',
            f'  wire ld_{n}_accept, ld_{n}_reject;',
        ]
    lines += [
        f'  wire [{bank_bits - 1}:0] cur_bank_id;',
        '  wire busy, quiescent;',
        '  integer errors = 0;',
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
    for p in bram_ports:
        n = p['name']
        lines += [
            f'    .ld_{n}_req(ld_{n}_req), .ld_{n}_bank(ld_{n}_bank),',
            f'    .ld_{n}_word(ld_{n}_word), .ld_{n}_wdata(ld_{n}_wdata),',
            f'    .ld_{n}_accept(ld_{n}_accept), .ld_{n}_reject(ld_{n}_reject),',
        ]
    for p in scalar_ports:
        n = p['name']
        lines += [
            f'    .ld_{n}_we(ld_{n}_we), .ld_{n}_bank(ld_{n}_bank),',
            f'    .ld_{n}_idx(ld_{n}_idx), .ld_{n}_data(ld_{n}_data),',
            f'    .ld_{n}_accept(ld_{n}_accept), .ld_{n}_reject(ld_{n}_reject),',
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
        '  task run_bank(input integer bank);',
        '    begin',
        '      @(negedge ap_clk);',
        '      ext_bank_id = bank[%d:0]; ext_ap_start = 1; input_1_ap_vld = 1;' % (bank_bits - 1),
        "      wait (ext_ap_ready === 1'b1);",
        '      @(negedge ap_clk);',
        '      ext_ap_start = 0;',
        "      wait (ext_ap_done === 1'b1);",
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
        for p in bram_ports:
            n, dw = p['name'], p['actual_data_width']
            for wi, word in enumerate(payload[n]):
                lines += [
                    '    @(negedge ap_clk);',
                    f'    ld_{n}_bank = {b}; ld_{n}_word = {wi};',
                    f"    ld_{n}_wdata = {dw}'h{word:0{dw // 4}x}; ld_{n}_req = 1;",
                    '    @(negedge ap_clk);',
                    f"    if (ld_{n}_accept !== 1'b1) begin",
                    f'      $display("FAIL: {n} load rejected (bank %0d word %0d)", {b}, {wi}); errors = errors + 1;',
                    '    end',
                    f'    ld_{n}_req = 0;',
                ]
        for p in scalar_ports:
            n, w = p['name'], p['actual_width']
            for bi, code in enumerate(payload[n]):
                lines += [
                    '    @(negedge ap_clk);',
                    f'    ld_{n}_bank = {b}; ld_{n}_idx = {bi};',
                    f"    ld_{n}_data = {w}'h{code:0{w // 4}x}; ld_{n}_we = 1;",
                    '    @(negedge ap_clk);',
                    f'    ld_{n}_we = 0;',
                ]
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
        '    @(negedge ap_clk);',
        '    if (busy !== 1\'b0) begin $display("FAIL: busy stuck high"); errors = errors + 1; end',
        '',
        '    // an in-range write while idle must still be accepted',
        '    @(negedge ap_clk);',
        f'    ld_{first_bram}_bank = {n_banks - 1}; ld_{first_bram}_word = 0; ld_{first_bram}_req = 1;',
        '    @(negedge ap_clk);',
        f"    if (ld_{first_bram}_accept !== 1'b1) begin",
        '      $display("FAIL: in-range bank write refused"); errors = errors + 1;',
        '    end',
        f'    ld_{first_bram}_req = 0;',
        '',
        '    // a write while an inference is active must be refused',
        '    @(negedge ap_clk);',
        '    ext_bank_id = 0; ext_ap_start = 1; input_1_ap_vld = 1;',
        "    wait (ext_ap_ready === 1'b1);",
        '    @(negedge ap_clk); ext_ap_start = 0;',
        f'    ld_{first_bram}_bank = 0; ld_{first_bram}_word = 0; ld_{first_bram}_req = 1;',
        '    @(negedge ap_clk);',
        f"    if (ld_{first_bram}_accept !== 1'b0 || ld_{first_bram}_reject !== 1'b1) begin",
        '      $display("FAIL: write accepted while inference active"); errors = errors + 1;',
        '    end',
        f'    ld_{first_bram}_req = 0;',
        "    wait (ext_ap_done === 1'b1);",
        '    @(negedge ap_clk); input_1_ap_vld = 0;',
        '',
        '    if (errors == 0) $display("RUNTIME_WEIGHTS_PASS");',
        '    else $display("RUNTIME_WEIGHTS_FAIL errors=%0d", errors);',
        '    $finish;',
        '  end',
        '',
        '  initial begin',
        f'    #{timeout};',
        '    $display("RUNTIME_WEIGHTS_FAIL timeout");',
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
    commands += [['xelab', top, '-s', 'rw_sim'], ['xsim', 'rw_sim', '-runall']]

    log = ''
    for cmd in commands:
        result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True, timeout=600)
        log += f'$ {" ".join(cmd)}\n{result.stdout}\n{result.stderr}\n'
        if result.returncode != 0 or re.search(r'^ERROR', result.stdout, re.M):
            return False, log
    return 'RUNTIME_WEIGHTS_PASS' in log, log


LATCH_TB = r"""
`timescale 1 ns / 1 ps

// Out-of-range bank rejection, checked on bank_select_latch directly.
//
// This cannot be exercised through a two-bank wrapper: with N_BANKS=2 the bank id
// is one bit wide, so every encodable value is valid. N_BANKS=3 makes id 3
// representable and invalid.
module tb;
  localparam int NB = 3;
  localparam int W  = $clog2(NB);

  reg clk = 0, rst = 1;
  always #5 clk = ~clk;

  reg ext_ap_start = 0;
  reg [W-1:0] ext_bank_id = 0;
  wire ext_ap_ready, ext_bank_id_bad, hls_ap_start;
  reg hls_ap_ready = 1, hls_ap_done = 0;

  // ap_ctrl_hs: the IP drops ap_idle when it accepts a start and raises it one
  // cycle AFTER ap_done, so there is a window where the transaction is over but
  // the IP is not yet idle. The wrapper must not accept a new bank in it.
  reg hls_ap_idle = 1, done_d = 0;
  always @(posedge clk) begin
    if (rst) begin
      hls_ap_idle <= 1'b1;
      done_d      <= 1'b0;
    end else begin
      done_d <= hls_ap_done;
      if (hls_ap_start & hls_ap_ready) hls_ap_idle <= 1'b0;
      else if (done_d)                 hls_ap_idle <= 1'b1;
    end
  end
  wire [W-1:0] cur_bank_id;
  wire busy, quiescent;
  integer errors = 0;

  bank_select_latch #(.BANK_ID_WIDTH(W), .N_BANKS(NB)) dut (
    .ap_clk(clk), .ap_rst(rst),
    .ext_ap_start(ext_ap_start), .ext_bank_id(ext_bank_id),
    .ext_ap_ready(ext_ap_ready), .ext_bank_id_bad(ext_bank_id_bad),
    .hls_ap_start(hls_ap_start), .hls_ap_ready(hls_ap_ready),
    .hls_ap_idle(hls_ap_idle), .hls_ap_done(hls_ap_done),
    .cur_bank_id(cur_bank_id), .busy(busy), .quiescent(quiescent));

  initial begin
    repeat (4) @(negedge clk);
    rst = 0;
    @(negedge clk);

    // id 3 is out of range for NB=3: flagged, and nothing starts
    ext_bank_id = 2'd3; ext_ap_start = 1;
    @(negedge clk);
    if (ext_bank_id_bad !== 1'b1) begin
      $display("FAIL: out-of-range bank not flagged"); errors = errors + 1;
    end
    if (hls_ap_start !== 1'b0) begin
      $display("FAIL: invalid bank started the IP"); errors = errors + 1;
    end
    @(negedge clk);
    if (busy !== 1'b0) begin
      $display("FAIL: invalid bank made the wrapper busy"); errors = errors + 1;
    end
    ext_ap_start = 0;
    @(negedge clk);

    // a valid id is accepted, and is stable before ap_start rises
    ext_bank_id = 2'd2; ext_ap_start = 1;
    @(negedge clk);
    if (ext_bank_id_bad !== 1'b0) begin
      $display("FAIL: valid bank flagged as bad"); errors = errors + 1;
    end
    if (cur_bank_id !== 2'd2) begin
      $display("FAIL: bank not committed before ap_start (got %0d)", cur_bank_id);
      errors = errors + 1;
    end
    if (hls_ap_start !== 1'b1) begin
      $display("FAIL: valid bank did not start the IP"); errors = errors + 1;
    end
    ext_ap_start = 0;
    @(negedge clk);
    if (busy !== 1'b1) begin $display("FAIL: not busy after accept"); errors = errors + 1; end

    // loads are refused while a transaction is in flight
    if (quiescent !== 1'b0) begin
      $display("FAIL: quiescent asserted while busy"); errors = errors + 1;
    end
    hls_ap_done = 1; @(negedge clk); hls_ap_done = 0;
    if (busy !== 1'b0) begin $display("FAIL: still busy after done"); errors = errors + 1; end

    // done has cleared busy but the IP has not raised ap_idle yet: the wrapper
    // must refuse both a new transaction and a load until it does
    if (hls_ap_idle !== 1'b0) begin
      $display("FAIL: bench did not model the done-to-idle window"); errors = errors + 1;
    end
    if (ext_ap_ready !== 1'b0) begin
      $display("FAIL: accepted a request before the IP went idle"); errors = errors + 1;
    end
    if (quiescent !== 1'b0) begin
      $display("FAIL: quiescent before the IP went idle"); errors = errors + 1;
    end

    @(negedge clk);
    if (hls_ap_idle !== 1'b1) begin
      $display("FAIL: IP model never returned to idle"); errors = errors + 1;
    end
    if (ext_ap_ready !== 1'b1) begin
      $display("FAIL: not ready once idle"); errors = errors + 1;
    end
    if (quiescent !== 1'b1) begin
      $display("FAIL: not quiescent once idle"); errors = errors + 1;
    end

    // a transaction whose ap_ready and ap_done coincide is already over: busy must
    // not latch high with no later done to clear it
    @(negedge clk);
    ext_bank_id = 2'd1; ext_ap_start = 1;
    @(negedge clk);
    ext_ap_start = 0;
    hls_ap_done = 1;          // same cycle the latch sees ap_ready
    @(negedge clk);
    hls_ap_done = 0;
    if (busy !== 1'b0) begin
      $display("FAIL: busy latched by a same-cycle ready/done transaction"); errors = errors + 1;
    end
    @(negedge clk); @(negedge clk);
    if (ext_ap_ready !== 1'b1) begin
      $display("FAIL: wrapper never became ready again"); errors = errors + 1;
    end

    // a transaction whose ap_ready and ap_done coincide is already over: busy must
    // not latch high with no later done left to clear it
    ext_bank_id = 2'd1; ext_ap_start = 1;
    @(negedge clk);
    ext_ap_start = 0; hls_ap_done = 1;      // same cycle the latch sees ap_ready
    @(negedge clk);
    hls_ap_done = 0;
    if (busy !== 1'b0) begin
      $display("FAIL: busy latched by a same-cycle ready/done transaction"); errors = errors + 1;
    end

    if (errors == 0) $display("RUNTIME_WEIGHTS_PASS");
    else $display("RUNTIME_WEIGHTS_FAIL errors=%0d", errors);
    $finish;
  end

  initial begin
    #50000; $display("RUNTIME_WEIGHTS_FAIL timeout"); $finish;
  end
endmodule
"""


def write_latch_testbench(path):
    """Standalone bench for bank_select_latch; needs no synthesized IP."""
    with open(path, 'w') as fh:
        fh.write(LATCH_TB)
    return path
