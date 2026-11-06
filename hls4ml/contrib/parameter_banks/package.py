"""Post-export packaging: wrap an already-synthesized hls4ml IP with banked
parameter memories.

This runs AFTER ``build(synth=True)``. It only reads the generated project -- the
HLS compute IP is never modified, and its RTL stays byte-identical.

Not provided, by design: AXI or board support, any driver, a Vivado project or
block design.
"""

import hashlib
import json
import os
import shutil

from hls4ml.contrib.parameter_banks import interface
from hls4ml.contrib.parameter_banks.interface import BramInterface, ScalarBundleInterface
from hls4ml.contrib.parameter_banks.pack import pack_banks
from hls4ml.model.external_parameters import ExternalParameterManifest

# the hls4ml backends whose IP has been verified end to end
SUPPORTED_BACKENDS = frozenset({'Vitis'})

# the latch drives ap_start/ap_ready as ap_ctrl_hs defines them
REQUIRED_CONTROL_PROTOCOL = 'ap_ctrl_hs'

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
STATIC_MODULES = ['bank_addr_mapper.sv', 'bank_select_latch.sv', 'parameter_bank.sv', 'scalar_bank_mux.sv']
SUMMARY_FILENAME = 'parameter_banks.json'

CONTROL_PORTS = {'ap_clk', 'ap_rst', 'ap_start', 'ap_done', 'ap_ready', 'ap_idle'}

MEMORY_STYLES = ('auto', 'block', 'distributed')
# Physical depth up to which 'auto' picks LUTRAM: a block RAM costs the same at any depth up to its own
AUTO_DISTRIBUTED_DEPTH = 64


class InterfaceUnsupported(Exception):
    """The model has an external parameter this version cannot bank."""


def fingerprint_ip(project_dir, project_name):
    """SHA-256 over the exported compute artifacts: the RTL and the data files that
    initialize its generated ROMs. Comparing a fingerprint taken before packaging
    with one taken after is what shows the IP was left alone."""
    verilog = interface.solution_verilog_dir(project_dir, project_name)
    entries = []
    for name in sorted(os.listdir(verilog)):
        # the ROM data is part of what the IP computes
        if name.endswith(('.v', '.dat')):
            digest = hashlib.sha256(open(os.path.join(verilog, name), 'rb').read()).hexdigest()
            entries.append(f'{digest}  {name}')
    combined = hashlib.sha256('\n'.join(entries).encode()).hexdigest()
    return {'combined': combined, 'files': entries}


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


class Wrapper:
    """The generated top: the unmodified IP, its banked memories, and one port list
    shared by the SystemVerilog module and its Verilog shim."""

    def __init__(self, project_name, interfaces, rtl_ports, n_banks, control, images=None, memory='auto'):
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
        self.images = images or {}  # parameter name -> preload file, bound as its INIT_HEX default
        self.memory = memory
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
        return [f'{i.name.upper()}_INIT_HEX' for i in self.bram + self.scalar]

    def init_default(self, interface):
        return self.images.get(interface.name, '')

    def ram_style(self, bram):
        if self.memory != 'auto':
            return self.memory
        return 'distributed' if self.n_banks * bram.bank_stride_words <= AUTO_DISTRIBUTED_DEPTH else 'block'

    def port_lines(self):
        """Declarations in order, with section comments and blank separators."""
        w = self.bank_id_width
        lines = [
            '    input  wire ap_clk',
            '    input  wire ap_rst',
            '',
            "    // the IP's ap_ctrl_hs control, plus the bank each inference uses",
            '    input  wire ap_start',
            '    output wire ap_done',
            '    output wire ap_idle',
            '    output wire ap_ready',
        ]
        if self.n_banks > 1:
            lines.append(f'    input  wire [{w - 1}:0] bank_id')
        lines += [
            '',
        ]
        if self.passthrough:
            lines.append('    // network data ports, passed through to the IP')
            lines.extend(_decl(p) for p in self.passthrough)
            lines.append('')
        g = self.loader
        lines += [
            '    // loader: rewrites banks at run time, only while the IP is idle; tie ld_req low if unused',
            '    input  wire ld_req',
            f'    input  wire [{g.param_id_width - 1}:0] ld_param_id',
        ]
        if self.n_banks > 1:
            lines.append(f'    input  wire [{w - 1}:0] ld_bank')
        lines += [
            f'    input  wire [{g.addr_width - 1}:0] ld_addr',
            f'    input  wire [{g.data_width - 1}:0] ld_data',
            '    output wire ld_accept',
            '    output wire ld_reject',
        ]
        return lines

    def port_names(self):
        return [ln.split()[-1] for ln in self.port_lines() if _is_port_decl(ln)]

    def _write_header(self, f, module):
        f.write('`timescale 1 ns / 1 ps\n`default_nettype none\n\n')
        params = [f'    parameter {p} = "{self.init_default(i)}"' for p, i in zip(self.init_params, self.bram + self.scalar)]
        f.write(f'module {module} #(\n' + ',\n'.join(params) + '\n) (\n')
        lines = self.port_lines()
        remaining = sum(1 for ln in lines if _is_port_decl(ln))
        for ln in lines:
            if not _is_port_decl(ln):
                f.write(ln + '\n' if ln.strip() else '\n')
            else:
                remaining -= 1
                f.write(ln + ('\n' if remaining == 0 else ',\n'))
        f.write(');\n\n')

    def write_top(self, f):
        """Every port of the wrapped IP is declared and connected explicitly: control
        to the ownership latch, banked parameters to their memories, everything else
        passed through. No .* connection: the IP's data ports are not nets here."""
        f.write('// GENERATED by hls4ml.contrib.parameter_banks -- do not edit.\n')
        f.write(f'// Wraps the unmodified {self.project_name} IP with runtime-selected parameter banks.\n')
        f.write(f'// Control protocol: {self.control}. A bank change waits for the IP to go idle.\n\n')
        self._write_header(f, self.top_name)

        f.write(f'  localparam int N_BANKS       = {self.n_banks};\n')
        f.write(f'  localparam int BANK_ID_WIDTH = {self.bank_id_width};\n\n')
        if self.n_banks == 1:
            f.write("  wire [0:0] bank_id = 1'b0, ld_bank = 1'b0;\n\n")
        f.write('  wire hls_ap_start, hls_ap_done, hls_ap_idle, hls_ap_ready;\n')
        f.write('  wire [BANK_ID_WIDTH-1:0] cur_bank_id;\n')
        f.write('  wire quiescent;\n\n')
        f.write('  bank_select_latch #(.BANK_ID_WIDTH(BANK_ID_WIDTH)) u_latch (\n')
        f.write('      .ap_clk(ap_clk), .ap_rst(ap_rst),\n')
        f.write('      .ext_ap_start(ap_start), .ext_bank_id(bank_id), .ext_ap_ready(ap_ready),\n')
        f.write('      .hls_ap_start(hls_ap_start), .hls_ap_ready(hls_ap_ready), .hls_ap_idle(hls_ap_idle),\n')
        f.write('      .cur_bank_id(cur_bank_id), .quiescent(quiescent));\n\n')
        f.write('  assign ap_done = hls_ap_done;\n')
        f.write('  assign ap_idle = hls_ap_idle;\n\n')

        # the full-width address is checked before it is narrowed, so it cannot alias a valid one
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
            f.write(f'  wire {sig["en_a"]}, {sig["en_b"]};\n')
            f.write(f'  wire [{b.port_width - 1}:0] {sig["dout_a"]}, {sig["dout_b"]};\n')
            f.write(f'  parameter_bank #(.DATA_WIDTH({b.data_width}), .PORT_WIDTH({b.port_width}),\n')
            f.write(f'      .HLS_ADDR_WIDTH({b.addr_width}),\n')
            f.write(f'      .WORD_BYTES({b.addr_stride}), .LOCAL_WORDS({b.depth}),\n')
            f.write('      .N_BANKS(N_BANKS), .BANK_ID_WIDTH(BANK_ID_WIDTH),\n')
            f.write(f'      .BANK_STRIDE_WORDS({b.bank_stride_words}), .INIT_HEX({b.name.upper()}_INIT_HEX),\n')
            f.write(f'      .RAM_STYLE("{self.ram_style(b)}"), .PORT_B_READ({int(b.reads_port_b)})) u_{b.name} (\n')
            f.write('      .ap_clk(ap_clk), .ap_rst(ap_rst), .cur_bank_id(cur_bank_id), .quiescent(quiescent),\n')
            f.write(f'      .hls_Addr_A({sig["addr_a"]}), .hls_EN_A({sig["en_a"]}),\n')
            f.write(f'      .hls_Dout_A({sig["dout_a"]}),\n')
            f.write(f'      .hls_Addr_B({sig["addr_b"]}), .hls_EN_B({sig["en_b"]}), .hls_Dout_B({sig["dout_b"]}),\n')
            f.write(f'      .ld_req(ld_hit[{s.param_id}]), .ld_bank(ld_bank), .ld_word(ld_addr[{s.addr_width - 1}:0]),\n')
            f.write(f'      .ld_wdata(ld_data[{b.data_width - 1}:0]),\n')
            f.write(f'      .ld_accept(ld_acc[{s.param_id}]), .ld_reject(),\n')
            f.write('      .addr_padding_violation());\n\n')

        for sb in self.scalar:
            s, w, n = g.slot(sb.name), sb.width, sb.n_scalars
            f.write(f'  wire [{w * n - 1}:0] {sb.name}_flat;\n')
            f.write(f'  scalar_bank_mux #(.SCALAR_WIDTH({w}), .N_SCALARS({n}),\n')
            f.write('      .N_BANKS(N_BANKS), .BANK_ID_WIDTH(BANK_ID_WIDTH),\n')
            f.write(f'      .INIT_HEX({sb.name.upper()}_INIT_HEX)) u_{sb.name} (\n')
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
            for role in ('addr_a', 'en_a', 'dout_a', 'addr_b', 'en_b', 'dout_b'):
                conns.append(f'.{sig[role]}({sig[role]})')
            # The IP never writes (proven in verify), the banks need no reset, both clocks are ap_clk.
            for role in ('din_a', 'wen_a', 'clk_a', 'rst_a', 'din_b', 'wen_b', 'clk_b', 'rst_b'):
                conns.append(f'.{sig[role]}()')
        for sb in self.scalar:
            w = sb.width
            for i, port in enumerate(sb.ports):
                conns.append(f'.{port}({sb.name}_flat[{w * i + w - 1}:{w * i}])')

        f.write(f'  // unmodified hls4ml IP\n  {self.project_name} u_{self.project_name} (\n')
        f.write(',\n'.join(f'      {c}' for c in conns))
        f.write('\n  );\n\nendmodule\n\n`default_nettype wire\n')

    def write_bd_shim(self, f):
        """A Verilog-2001 module forwarding every port and parameter 1:1. Vivado IP
        Integrator (checked on 2025.2) refuses a SystemVerilog top as a Module
        Reference; this is what goes into a block design."""
        f.write('// GENERATED by hls4ml.contrib.parameter_banks -- do not edit.\n')
        f.write(f'// Verilog shim over {self.top_name} for Vivado IP Integrator, which does not\n')
        f.write('// accept a SystemVerilog top as a Module Reference. Use this module in a block\n')
        f.write(f'// design; instantiate {self.top_name} directly from RTL.\n\n')
        self._write_header(f, self.bd_module)
        f.write(f'  {self.top_name} #(\n')
        f.write(',\n'.join(f'      .{p}({p})' for p in self.init_params))
        f.write('\n  ) u_parameter_banks (\n')
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
            'bank_selection': 'requests for the selected bank overlap; a bank change waits for the IP to go idle',
            'passthrough_ports': [p['name'] for p in self.passthrough],
            'loader': self.loader.describe(),
            'banked_ports': [{**s.interface.describe(), **s.describe()} for s in self.loader],
            'memory': {b.name: self.ram_style(b) for b in self.bram},
            'rom_data_files': rom_data,
            'preload_files': self.images,
            'exported_ip_sha256': fingerprint,
        }


def _decl(port):
    direction = 'input  wire' if port['dir'] == 'input' else 'output wire'
    width = '' if port['width'] == 1 else f'[{port["width"] - 1}:0] '
    return f'    {direction} {width}{port["name"]}'


def _is_port_decl(line):
    return bool(line.strip()) and not line.strip().startswith('//')


def package(project, n_banks=None, banks=None, memory='auto', output_dir=None):
    """Generate the banked-parameter wrapper for a synthesized hls4ml project.

    Args:
        project: a ModelGraph, or the path to its output directory.
        n_banks: how many banks to provision, a power of two; one bank gives
            reloadable parameters with no bank selection. Fixed in the generated
            RTL; changing it means re-running this packager, not re-running HLS.
            Defaults to the smallest that holds ``banks``, or 2.
        banks: parameter sets to preload, as for ``pack_banks()``; needs the
            ModelGraph. Their images are written next to the RTL and bound as the
            wrapper's defaults. Banks beyond them start as zero.
        memory: ``'block'`` or ``'distributed'`` RAM for the banks; ``'auto'`` picks
            LUTRAM for shallow banks, where a block RAM would be mostly empty.
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
            f'no verified layout for external parameter(s) {[p.name for p in manifest.undescribed]}; banking only '
            "some of a model's parameters would leave the rest as unconnected top-level ports.\n" + reasons
        )
    if not manifest.described:
        raise ValueError('manifest describes no external parameters; nothing to bank')
    if banks is not None and not hasattr(project, 'config'):
        raise ValueError('preloading banks needs the ModelGraph the IP was built from, not a path')
    if n_banks is None:
        n_banks = 1 << (len(banks) - 1).bit_length() if banks else 2
    if isinstance(n_banks, bool) or not isinstance(n_banks, int) or n_banks < 1 or n_banks & (n_banks - 1):
        raise ValueError(f'n_banks must be a power of two, got {n_banks!r}')
    if memory not in MEMORY_STYLES:
        raise ValueError(f'memory must be one of {MEMORY_STYLES}, got {memory!r}')
    if banks is not None and len(banks) > n_banks:
        raise ValueError(f'{len(banks)} parameter sets do not fit in {n_banks} banks')

    interfaces, control = interface.verify(manifest, project_dir)
    if control != REQUIRED_CONTROL_PROTOCOL:
        raise interface.InterfaceMismatch(
            f'control protocol is {control!r}; the wrapper requires {REQUIRED_CONTROL_PROTOCOL!r}'
        )
    rtl_ports = interface.parse_rtl_ports(project_dir, project_name)

    if output_dir is None:
        output_dir = os.path.join(project_dir, 'parameter_banks')
    rtl_dir = os.path.join(output_dir, 'rtl')
    os.makedirs(rtl_dir, exist_ok=True)

    images = {}
    if banks is not None:
        for name, image in pack_banks(project, banks).items():
            images[name] = f'{name}.mem'
            image.write_mem(os.path.join(rtl_dir, images[name]))
    wrapper = Wrapper(project_name, interfaces, rtl_ports, n_banks, control, images, memory)
    for module in STATIC_MODULES:
        shutil.copyfile(os.path.join(TEMPLATE_DIR, module), os.path.join(rtl_dir, module))

    # the HLS ROMs read ./<name>.dat, so the data goes next to the Tcl that runs from here
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
