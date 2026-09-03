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


def write_testbench(path, project_name, n_in, n_out, input_codes, banks, expected, w_port, b_port):
    """Emit a self-checking two-bank testbench.

    ``banks`` is [(weight_words, bias_codes), ...]; ``expected`` is a list of
    per-bank lists of output codes. The same input vector is applied to every
    bank, so any difference in the outputs can only come from bank selection.
    """
    top = f'{project_name}_runtime_weights'
    w_name, b_name = w_port['name'], b_port['name']
    dw = w_port['actual_data_width']
    depth = w_port['expected_depth']
    word_bits = max((depth - 1).bit_length(), 1)
    idx_bits = max((b_port['n_scalars'] - 1).bit_length(), 1)
    n_banks = len(banks)
    bank_bits = max((n_banks - 1).bit_length(), 1)

    lines = [
        '`timescale 1 ns / 1 ps',
        '',
        'module tb;',
        '  localparam int NB = %d;' % n_banks,
        '  reg ap_clk = 0, ap_rst = 1;',
        '  always #5 ap_clk = ~ap_clk;',
        '',
        '  reg ext_ap_start = 0;',
        f'  reg [{bank_bits - 1}:0] ext_bank_id = 0;',
        '  wire ext_ap_ready, ext_ap_done, ext_bank_id_bad;',
        f'  reg [{n_in * WIDTH - 1}:0] input_1 = 0;',
        '  reg input_1_ap_vld = 0;',
    ]
    for o in range(n_out):
        lines.append(f'  wire [{WIDTH - 1}:0] layer2_out_{o};')
        lines.append(f'  wire layer2_out_{o}_ap_vld;')
    lines += [
        f'  reg ld_{w_name}_req = 0;',
        f'  reg [{bank_bits - 1}:0] ld_{w_name}_bank = 0;',
        f'  reg [{word_bits - 1}:0] ld_{w_name}_word = 0;',
        f'  reg [{dw - 1}:0] ld_{w_name}_wdata = 0;',
        f'  wire ld_{w_name}_accept, ld_{w_name}_reject;',
        f'  reg ld_{b_name}_we = 0;',
        f'  reg [{bank_bits - 1}:0] ld_{b_name}_bank = 0;',
        f'  reg [{idx_bits - 1}:0] ld_{b_name}_idx = 0;',
        f'  reg [{WIDTH - 1}:0] ld_{b_name}_data = 0;',
        f'  wire ld_{b_name}_accept;',
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
    for o in range(n_out):
        lines.append(f'    .layer2_out_{o}(layer2_out_{o}), .layer2_out_{o}_ap_vld(layer2_out_{o}_ap_vld),')
    lines += [
        f'    .ld_{w_name}_req(ld_{w_name}_req), .ld_{w_name}_bank(ld_{w_name}_bank),',
        f'    .ld_{w_name}_word(ld_{w_name}_word), .ld_{w_name}_wdata(ld_{w_name}_wdata),',
        f'    .ld_{w_name}_accept(ld_{w_name}_accept), .ld_{w_name}_reject(ld_{w_name}_reject),',
        f'    .ld_{b_name}_we(ld_{b_name}_we), .ld_{b_name}_bank(ld_{b_name}_bank),',
        f'    .ld_{b_name}_idx(ld_{b_name}_idx), .ld_{b_name}_data(ld_{b_name}_data),',
        f'    .ld_{b_name}_accept(ld_{b_name}_accept),',
        '    .cur_bank_id(cur_bank_id), .busy(busy), .quiescent(quiescent));',
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
        '    // load every bank while idle',
    ]
    for b, (words, bias_codes) in enumerate(banks):
        for wi, word in enumerate(words):
            lines += [
                '    @(negedge ap_clk);',
                f'    ld_{w_name}_bank = {b}; ld_{w_name}_word = {wi};',
                f"    ld_{w_name}_wdata = {dw}'h{word:0{dw // 4}x}; ld_{w_name}_req = 1;",
                '    @(negedge ap_clk);',
                f"    if (ld_{w_name}_accept !== 1'b1) begin",
                f'      $display("FAIL: weight load rejected (bank %0d word %0d)", {b}, {wi}); errors = errors + 1;',
                '    end',
                f'    ld_{w_name}_req = 0;',
            ]
        for bi, code in enumerate(bias_codes):
            lines += [
                '    @(negedge ap_clk);',
                f'    ld_{b_name}_bank = {b}; ld_{b_name}_idx = {bi};',
                f"    ld_{b_name}_data = {WIDTH}'h{code:04x}; ld_{b_name}_we = 1;",
                '    @(negedge ap_clk);',
                f'    ld_{b_name}_we = 0;',
            ]
    lines.append('')
    lines.append('    // same input to every bank: any output difference is bank selection')
    for b in range(n_banks):
        lines.append(f'    run_bank({b});')
        for o in range(n_out):
            exp = expected[b][o]
            lines += [
                f"    if (layer2_out_{o} !== {WIDTH}'h{exp:04x}) begin",
                f'      $display("FAIL: bank {b} out{o} = %04x expected {exp:04x}", layer2_out_{o});',
                '      errors = errors + 1;',
                '    end',
            ]
    lines += [
        '',
        '    @(negedge ap_clk);',
        '    if (busy !== 1\'b0) begin $display("FAIL: busy stuck high"); errors = errors + 1; end',
        '',
        '    // a write to an out-of-range bank must be refused, not acknowledged',
        '    @(negedge ap_clk);',
        f'    ld_{w_name}_bank = {n_banks - 1}; ld_{w_name}_word = 0; ld_{w_name}_req = 1;',
        '    @(negedge ap_clk);',
        f"    if (ld_{w_name}_accept !== 1'b1) begin",
        '      $display("FAIL: in-range bank write refused"); errors = errors + 1;',
        '    end',
        f'    ld_{w_name}_req = 0;',
        '',
        '    // a write while an inference is active must be refused',
        '    @(negedge ap_clk);',
        '    ext_bank_id = 0; ext_ap_start = 1; input_1_ap_vld = 1;',
        "    wait (ext_ap_ready === 1'b1);",
        '    @(negedge ap_clk); ext_ap_start = 0;',
        f'    ld_{w_name}_bank = 0; ld_{w_name}_word = 0; ld_{w_name}_req = 1;',
        '    @(negedge ap_clk);',
        f"    if (ld_{w_name}_accept !== 1'b0 || ld_{w_name}_reject !== 1'b1) begin",
        '      $display("FAIL: write accepted while inference active"); errors = errors + 1;',
        '    end',
        f'    ld_{w_name}_req = 0;',
        "    wait (ext_ap_done === 1'b1);",
        '    @(negedge ap_clk); input_1_ap_vld = 0;',
        '',
        '    if (errors == 0) $display("RUNTIME_WEIGHTS_PASS");',
        '    else $display("RUNTIME_WEIGHTS_FAIL errors=%0d", errors);',
        '    $finish;',
        '  end',
        '',
        '  initial begin',
        '    #200000;',
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
            (sv if name.endswith('.sv') else v if name.endswith('.v') else []).append(os.path.join(d, name))

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
  reg hls_ap_ready = 1, hls_ap_idle = 1, hls_ap_done = 0;
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
    @(negedge clk);
    if (busy !== 1'b0) begin $display("FAIL: still busy after done"); errors = errors + 1; end

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
