"""Post-export packaging: wrap an already-synthesized hls4ml IP with banked
parameter memories.

This runs AFTER ``build(synth=True)``. It only reads the generated project -- the
HLS compute IP is never modified, and its RTL stays byte-identical.

Not provided, by design: AXI or board support, any driver, a Vivado project or
block design, and any support for overlapping transactions. Bank selection is
idle-time only.
"""

import hashlib
import json
import os
import shutil

from hls4ml.contrib.parameter_banks import interface
from hls4ml.contrib.parameter_banks.interface import BramInterface, ScalarBundleInterface
from hls4ml.model.external_parameters import ExternalParameterManifest

# Only the Vitis flow has been verified end to end. This is about which hls4ml
# backend produced the IP; Vivado is still used to implement the result.
SUPPORTED_BACKENDS = frozenset({'Vitis'})

# The ownership latch drives ap_start/ap_ready directly, so anything else
# (ap_ctrl_chain, ap_ctrl_none) would be mis-driven rather than merely unsupported.
REQUIRED_CONTROL_PROTOCOL = 'ap_ctrl_hs'

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
STATIC_MODULES = ['bank_addr_mapper.sv', 'bank_select_latch.sv', 'parameter_bank.sv', 'scalar_bank_mux.sv']
SUMMARY_FILENAME = 'parameter_banks.json'

CONTROL_PORTS = {'ap_clk', 'ap_rst', 'ap_start', 'ap_done', 'ap_ready', 'ap_idle'}


class InterfaceUnsupported(Exception):
    """The model has an external parameter this version cannot bank."""


def fingerprint_ip(project_dir, project_name):
    """SHA-256 over the exported compute artifacts: the RTL and the data files that
    initialize its generated ROMs. Comparing a fingerprint taken before packaging
    with one taken after is what shows the IP was left alone."""
    verilog = interface.solution_verilog_dir(project_dir, project_name)
    entries = []
    for name in sorted(os.listdir(verilog)):
        # A .dat is part of the compute artifact: $readmemh loads it into a
        # generated ROM at elaboration, so it changes what the IP computes.
        if name.endswith(('.v', '.dat')):
            digest = hashlib.sha256(open(os.path.join(verilog, name), 'rb').read()).hexdigest()
            entries.append(f'{digest}  {name}')
    combined = hashlib.sha256('\n'.join(entries).encode()).hexdigest()
    return {'combined': combined, 'files': entries}


# --- loader geometry -----------------------------------------------------------


class LoaderSlot:
    """One parameter's place on the shared loader: its id, and the valid range of
    ``ld_addr`` and ``ld_data`` for it."""

    def __init__(self, param_id, iface):
        self.param_id = param_id
        self.interface = iface
        self.depth = iface.loader_depth  # valid ld_addr: 0 .. depth-1
        self.data_width = iface.loader_data_width  # valid ld_data: the low data_width bits
        self.addr_width = max((self.depth - 1).bit_length(), 1)  # the port's own narrow address

    @property
    def name(self):
        return self.interface.name

    def describe(self):
        return {'param_id': self.param_id, 'ld_depth': self.depth, 'ld_data_width': self.data_width}


class LoaderGeometry:
    """The one generic loader port set: as wide as the widest parameter needs.

    Every parameter takes one write per request, a BRAM addressed by word and a
    scalar bundle by element, so they can share one interface.
    """

    def __init__(self, interfaces):
        self.slots = [LoaderSlot(i, iface) for i, iface in enumerate(interfaces)]
        self.param_id_width = max((len(self.slots) - 1).bit_length(), 1)
        self.addr_width = max(s.addr_width for s in self.slots)
        self.data_width = max(s.data_width for s in self.slots)

    def __iter__(self):
        return iter(self.slots)

    def __len__(self):
        return len(self.slots)

    def slot(self, name):
        return next(s for s in self.slots if s.name == name)

    def describe(self):
        return {
            'n_params': len(self.slots),
            'param_id_width': self.param_id_width,
            'addr_width': self.addr_width,
            'data_width': self.data_width,
        }


# --- wrapper generation ----------------------------------------------------------


class Wrapper:
    """The generated top: the unmodified IP, its banked memories, and one port list
    shared by the SystemVerilog module and its Verilog shim."""

    def __init__(self, project_name, interfaces, rtl_ports, n_banks, control):
        self.project_name = project_name
        self.top_name = f'{project_name}_parameter_banks'
        self.bd_module = f'{self.top_name}_bd'
        self.interfaces = list(interfaces)
        self.bram = [i for i in self.interfaces if isinstance(i, BramInterface)]
        self.scalar = [i for i in self.interfaces if isinstance(i, ScalarBundleInterface)]
        self.rtl_ports = rtl_ports
        self.n_banks = n_banks
        self.bank_id_width = max(1, (n_banks - 1).bit_length())
        self.control = control
        self.loader = LoaderGeometry(self.interfaces)

    @property
    def passthrough(self):
        """The IP's ports the wrapper neither owns nor banks: the network data."""
        owned = set(CONTROL_PORTS)
        for b in self.bram:
            owned.update(b.signals.values())
        for s in self.scalar:
            owned.update(s.ports)
        return [p for p in self.rtl_ports if p['name'] not in owned]

    @property
    def init_params(self):
        return [f'{b.name.upper()}_INIT_HEX' for b in self.bram]

    # -- port list, shared by top and shim --

    def port_lines(self):
        """Declarations in order, with section comments and blank separators."""
        w = self.bank_id_width
        lines = [
            '    input  wire ap_clk',
            '    input  wire ap_rst',
            '',
            '    // transaction boundary',
            '    input  wire ext_ap_start',
            f'    input  wire [{w - 1}:0] ext_bank_id',
            '    output wire ext_ap_ready',
            '    output wire ext_ap_done',
            '    output wire ext_bank_id_bad',
            '',
        ]
        if self.passthrough:
            lines.append('    // network data ports, passed through to the IP')
            lines.extend(_decl(p) for p in self.passthrough)
            lines.append('')
        g = self.loader
        lines += [
            '    // loader: one write per request, routed by ld_param_id (accepted only while idle)',
            '    input  wire ld_req',
            f'    input  wire [{g.param_id_width - 1}:0] ld_param_id',
            f'    input  wire [{w - 1}:0] ld_bank',
            f'    input  wire [{g.addr_width - 1}:0] ld_addr',
            f'    input  wire [{g.data_width - 1}:0] ld_data',
            '    output wire ld_accept',
            '    output wire ld_reject',
            '',
            '    // observability',
            f'    output wire [{w - 1}:0] cur_bank_id',
            '    output wire busy',
            '    output wire quiescent',
        ]
        return lines

    def port_names(self):
        return [ln.split()[-1] for ln in self.port_lines() if _is_port_decl(ln)]

    def _write_header(self, f, module):
        f.write('`timescale 1 ns / 1 ps\n`default_nettype none\n\n')
        params = [f'    parameter {p} = ""' for p in self.init_params]
        f.write(f'module {module} #(\n' + ',\n'.join(params) + '\n) (\n' if params else f'module {module} (\n')
        lines = self.port_lines()
        remaining = sum(1 for ln in lines if _is_port_decl(ln))
        for ln in lines:
            if not _is_port_decl(ln):
                f.write(ln + '\n' if ln.strip() else '\n')
            else:
                remaining -= 1
                f.write(ln + ('\n' if remaining == 0 else ',\n'))
        f.write(');\n\n')

    # -- SystemVerilog top --

    def write_top(self, f):
        """Every port of the wrapped IP is declared and connected explicitly: control
        to the ownership latch, banked parameters to their memories, everything else
        passed through. No .* connection: the IP's data ports are not nets here."""
        f.write('// GENERATED by hls4ml.contrib.parameter_banks -- do not edit.\n')
        f.write(f'// Wraps the unmodified {self.project_name} IP with runtime-selected parameter banks.\n')
        f.write(f'// Control protocol: {self.control}. Bank selection is idle-time only.\n\n')
        self._write_header(f, self.top_name)

        f.write(f'  localparam int N_BANKS       = {self.n_banks};\n')
        f.write(f'  localparam int BANK_ID_WIDTH = {self.bank_id_width};\n\n')
        f.write('  wire hls_ap_start, hls_ap_done, hls_ap_idle, hls_ap_ready;\n\n')
        f.write('  bank_select_latch #(.BANK_ID_WIDTH(BANK_ID_WIDTH), .N_BANKS(N_BANKS)) u_latch (\n')
        f.write('      .ap_clk(ap_clk), .ap_rst(ap_rst),\n')
        f.write('      .ext_ap_start(ext_ap_start), .ext_bank_id(ext_bank_id),\n')
        f.write('      .ext_ap_ready(ext_ap_ready), .ext_bank_id_bad(ext_bank_id_bad),\n')
        f.write('      .hls_ap_start(hls_ap_start), .hls_ap_ready(hls_ap_ready),\n')
        f.write('      .hls_ap_idle(hls_ap_idle), .hls_ap_done(hls_ap_done),\n')
        f.write('      .cur_bank_id(cur_bank_id), .busy(busy), .quiescent(quiescent));\n\n')
        f.write('  assign ext_ap_done = hls_ap_done;\n\n')

        # Route each request to one parameter. The id and the full-width address
        # are checked here, before the address is narrowed to the port's own width,
        # so an out-of-range value is rejected rather than aliased onto a valid one.
        g = self.loader
        f.write(f'  wire [{len(g) - 1}:0] ld_hit, ld_acc;\n')
        for s in g:
            f.write(
                f"  assign ld_hit[{s.param_id}] = ld_req & (ld_param_id == {g.param_id_width}'d{s.param_id})"
                f" & (ld_addr < {g.addr_width + 1}'d{s.depth});\n"  # one bit wider: depth may be 2**addr_width
            )
        f.write('  assign ld_accept = |ld_acc;\n')
        f.write('  assign ld_reject = ld_req & ~ld_accept;\n\n')

        for b in self.bram:
            s, sig = g.slot(b.name), b.signals
            f.write(f'  wire [{b.addr_width - 1}:0] {sig["addr_a"]}, {sig["addr_b"]};\n')
            f.write(f'  wire {sig["en_a"]}, {sig["en_b"]}, {sig["rst_a"]};\n')
            # declared at the PHYSICAL width the IP drives; parameter_bank pads
            # between logical and physical explicitly
            f.write(f'  wire [{b.port_width - 1}:0] {sig["dout_a"]}, {sig["dout_b"]};\n')
            f.write(f'  parameter_bank #(.DATA_WIDTH({b.data_width}), .PORT_WIDTH({b.port_width}),\n')
            f.write(f'      .HLS_ADDR_WIDTH({b.addr_width}),\n')
            f.write(f'      .WORD_BYTES({b.addr_stride}), .LOCAL_WORDS({b.depth}),\n')
            f.write('      .N_BANKS(N_BANKS), .BANK_ID_WIDTH(BANK_ID_WIDTH),\n')
            f.write(f'      .BANK_STRIDE_WORDS({b.bank_stride_words}), .INIT_HEX({b.name.upper()}_INIT_HEX)) u_{b.name} (\n')
            f.write('      .ap_clk(ap_clk), .ap_rst(ap_rst), .cur_bank_id(cur_bank_id), .quiescent(quiescent),\n')
            f.write(f'      .hls_Addr_A({sig["addr_a"]}), .hls_EN_A({sig["en_a"]}),\n')
            f.write(f'      .hls_Dout_A({sig["dout_a"]}), .hls_Rst_A({sig["rst_a"]}),\n')
            f.write(f'      .hls_Addr_B({sig["addr_b"]}), .hls_EN_B({sig["en_b"]}), .hls_Dout_B({sig["dout_b"]}),\n')
            f.write(f'      .ld_req(ld_hit[{s.param_id}]), .ld_bank(ld_bank), .ld_word(ld_addr[{s.addr_width - 1}:0]),\n')
            f.write(f'      .ld_wdata(ld_data[{b.data_width - 1}:0]),\n')
            f.write(f'      .ld_accept(ld_acc[{s.param_id}]), .ld_reject(),\n')
            f.write('      .addr_padding_violation());\n\n')

        for sb in self.scalar:
            s, w, n = g.slot(sb.name), sb.width, sb.n_scalars
            f.write(f'  wire [{w * n - 1}:0] {sb.name}_flat;\n')
            f.write(f'  scalar_bank_mux #(.SCALAR_WIDTH({w}), .N_SCALARS({n}),\n')
            f.write(f'      .N_BANKS(N_BANKS), .BANK_ID_WIDTH(BANK_ID_WIDTH)) u_{sb.name} (\n')
            f.write('      .ap_clk(ap_clk), .ap_rst(ap_rst), .cur_bank_id(cur_bank_id), .quiescent(quiescent),\n')
            f.write(f'      .ld_we(ld_hit[{s.param_id}]), .ld_bank(ld_bank), .ld_idx(ld_addr[{s.addr_width - 1}:0]),\n')
            f.write(f'      .ld_data(ld_data[{w - 1}:0]), .ld_accept(ld_acc[{s.param_id}]),\n')
            f.write(f'      .ld_reject(), .q_flat({sb.name}_flat));\n\n')

        conns = [
            '.ap_clk(ap_clk)',
            '.ap_rst(ap_rst)',
            '.ap_start(hls_ap_start)',
            '.ap_done(hls_ap_done)',
            '.ap_ready(hls_ap_ready)',
            '.ap_idle(hls_ap_idle)',
        ]
        conns += [f'.{p["name"]}({p["name"]})' for p in self.passthrough]
        for b in self.bram:
            sig = b.signals
            # Both ports go to the IP as read ports; the bank lends port B to the
            # loader while quiescent. A layer that ignores port B leaves EN_B low.
            for role in ('addr_a', 'en_a', 'dout_a', 'rst_a', 'addr_b', 'en_b', 'dout_b'):
                conns.append(f'.{sig[role]}({sig[role]})')
            # Din/WEN dangling: the IP never writes (proven in verify); both clocks are ap_clk.
            for role in ('din_a', 'wen_a', 'clk_a', 'din_b', 'wen_b', 'clk_b', 'rst_b'):
                conns.append(f'.{sig[role]}()')
        for sb in self.scalar:
            w = sb.width
            for i, port in enumerate(sb.ports):
                conns.append(f'.{port}({sb.name}_flat[{w * i + w - 1}:{w * i}])')

        f.write(f'  // unmodified hls4ml IP\n  {self.project_name} u_{self.project_name} (\n')
        f.write(',\n'.join(f'      {c}' for c in conns))
        f.write('\n  );\n\nendmodule\n\n`default_nettype wire\n')

    # -- Verilog shim --

    def write_bd_shim(self, f):
        """A Verilog-2001 module forwarding every port and parameter 1:1. Vivado IP
        Integrator (checked on 2025.2) refuses a SystemVerilog top as a Module
        Reference; this is what goes into a block design."""
        f.write('// GENERATED by hls4ml.contrib.parameter_banks -- do not edit.\n')
        f.write(f'// Verilog shim over {self.top_name} for Vivado IP Integrator, which does not\n')
        f.write('// accept a SystemVerilog top as a Module Reference. Use this module in a block\n')
        f.write(f'// design; instantiate {self.top_name} directly from RTL.\n\n')
        self._write_header(f, self.bd_module)
        if self.init_params:
            f.write(f'  {self.top_name} #(\n')
            f.write(',\n'.join(f'      .{p}({p})' for p in self.init_params))
            f.write('\n  ) u_parameter_banks (\n')
        else:
            f.write(f'  {self.top_name} u_parameter_banks (\n')
        f.write(',\n'.join(f'      .{n}({n})' for n in self.port_names()))
        f.write('\n  );\n\nendmodule\n\n`default_nettype wire\n')

    def describe(self, rom_data, fingerprint):
        return {
            'project_name': self.project_name,
            'top': self.top_name,
            'bd_module': self.bd_module,
            'n_banks': self.n_banks,
            'bank_id_width': self.bank_id_width,
            'control_protocol': self.control,
            'bank_selection': 'idle-time (bank committed before ap_start, held to ap_done)',
            'passthrough_ports': [p['name'] for p in self.passthrough],
            'loader': self.loader.describe(),
            'banked_ports': [{**s.interface.describe(), **s.describe()} for s in self.loader],
            'rom_data_files': rom_data,
            'exported_ip_sha256': fingerprint,
        }


def _decl(port):
    direction = 'input  wire' if port['dir'] == 'input' else 'output wire'
    width = '' if port['width'] == 1 else f'[{port["width"] - 1}:0] '
    return f'    {direction} {width}{port["name"]}'


def _is_port_decl(line):
    return bool(line.strip()) and not line.strip().startswith('//')


# --- entry point -------------------------------------------------------------


def package(project, n_banks=2, output_dir=None):
    """Generate the banked-parameter wrapper for a synthesized hls4ml project.

    Args:
        project: a ModelGraph, or the path to its output directory.
        n_banks: how many banks to provision. Fixed in the generated RTL; changing
            it means re-running this packager, not re-running HLS.
        output_dir: where to write the wrapper (default ``<project>/parameter_banks``).

    Returns a dict describing what was written, including a fingerprint of the
    exported IP; the same dict is saved as ``parameter_banks.json``.
    """
    project_dir = project.config.get_output_dir() if hasattr(project, 'config') else str(project)
    manifest = ExternalParameterManifest.load(project_dir)
    if manifest.backend not in SUPPORTED_BACKENDS:
        raise ValueError(f'manifest backend is {manifest.backend!r}, expected one of {sorted(SUPPORTED_BACKENDS)}')
    project_name = manifest.project_name

    if manifest.undescribed:
        reasons = '\n'.join(f'  {p.name}: {p.note or "no reason recorded"}' for p in manifest.undescribed)
        raise InterfaceUnsupported(
            f'no verified adapter for external parameter(s) {[p.name for p in manifest.undescribed]}; banking only '
            "some of a model's parameters would leave the rest as unconnected top-level ports.\n" + reasons
        )
    if not manifest.described:
        raise ValueError('manifest describes no external parameters; nothing to bank')
    if isinstance(n_banks, bool) or not isinstance(n_banks, int) or n_banks < 2:
        raise ValueError(f'n_banks must be an integer of at least 2, got {n_banks!r}')

    interfaces, summary = interface.verify(manifest, project_dir)
    if summary.control != REQUIRED_CONTROL_PROTOCOL:
        raise interface.InterfaceMismatch(
            f'control protocol is {summary.control!r}; the idle-time wrapper requires {REQUIRED_CONTROL_PROTOCOL!r}'
        )
    rtl_ports = interface.parse_rtl_ports(project_dir, project_name)
    wrapper = Wrapper(project_name, interfaces, rtl_ports, n_banks, summary.control)

    if output_dir is None:
        output_dir = os.path.join(project_dir, 'parameter_banks')
    rtl_dir = os.path.join(output_dir, 'rtl')
    os.makedirs(rtl_dir, exist_ok=True)
    for module in STATIC_MODULES:
        shutil.copyfile(os.path.join(TEMPLATE_DIR, module), os.path.join(rtl_dir, module))

    # Vitis initializes generated ROMs with $readmemh("./<name>.dat"), resolved
    # against the tool's working directory. The generated Tcl runs from output_dir,
    # so the data has to be there too.
    hls_rtl_dir = interface.solution_verilog_dir(project_dir, project_name)
    rom_data = [n for n in sorted(os.listdir(hls_rtl_dir)) if n.endswith('.dat')]
    for name in rom_data:
        shutil.copyfile(os.path.join(hls_rtl_dir, name), os.path.join(output_dir, name))

    with open(os.path.join(rtl_dir, f'{wrapper.top_name}.sv'), 'w') as fh:
        wrapper.write_top(fh)
    with open(os.path.join(rtl_dir, f'{wrapper.bd_module}.v'), 'w') as fh:
        wrapper.write_bd_shim(fh)

    with open(os.path.join(TEMPLATE_DIR, 'create_parameter_banks.tcl')) as fh:
        tcl = fh.read()
    tcl = (
        tcl.replace('@PROJECT_NAME@', project_name)
        .replace('@TOP_NAME@', wrapper.top_name)
        .replace('@BD_MODULE@', wrapper.bd_module)
        .replace('@HLS_RTL_DIR@', os.path.abspath(hls_rtl_dir))
        .replace('@PART@', str(manifest.part))
        .replace('@N_BANKS@', str(n_banks))
        .replace('@CLOCK_PERIOD@', str(manifest.clock_period))
    )
    with open(os.path.join(output_dir, 'create_parameter_banks.tcl'), 'w') as fh:
        fh.write(tcl)

    description = wrapper.describe(rom_data, fingerprint_ip(project_dir, project_name))
    with open(os.path.join(output_dir, SUMMARY_FILENAME), 'w') as fh:
        json.dump(description, fh, indent=2)
    return description
