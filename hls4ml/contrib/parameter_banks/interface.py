"""What HLS actually built for each external parameter, cross-checked against
what the writer asked for.

The manifest records the request. The C synthesis report (read through
``hls4ml.report``) and the generated Verilog record the result. Nothing here is
inferred from pragmas: a fact that is not in the generated output is not claimed.
"""

import os
import re

from hls4ml.report import parse_interface_summary

# A fully partitioned parameter lowers to plain input ports with no handshake.
EXPECTED_SCALAR_MODE = 'ap_none'

# How Vitis spells the signals of one external BRAM parameter port, and their
# direction from the IP's point of view. This is the only place that spelling
# appears: verify() resolves it against the real RTL and the wrapper generator
# takes the resulting names, never rebuilding one.
BRAM_SIGNALS = {
    'addr_a': ('Addr_A', 'output'),
    'en_a': ('EN_A', 'output'),
    'din_a': ('Din_A', 'output'),
    'dout_a': ('Dout_A', 'input'),
    'wen_a': ('WEN_A', 'output'),
    'clk_a': ('Clk_A', 'output'),
    'rst_a': ('Rst_A', 'output'),
    'addr_b': ('Addr_B', 'output'),
    'en_b': ('EN_B', 'output'),
    'din_b': ('Din_B', 'output'),
    'dout_b': ('Dout_B', 'input'),
    'wen_b': ('WEN_B', 'output'),
    'clk_b': ('Clk_B', 'output'),
    'rst_b': ('Rst_B', 'output'),
}


class InterfaceMismatch(Exception):
    """The synthesized interface disagrees with the manifest."""


# The one HLS solution everything is read from: the report, the exported RTL and
# the ROM data files must all describe the same synthesis run.
SOLUTION = 'solution1'


def _solution_dir(project_dir, project_name):
    return os.path.join(project_dir, f'{project_name}_prj', SOLUTION)


def solution_verilog_dir(project_dir, project_name):
    """Where Vitis writes the exported RTL. Callers should not spell this out."""
    return os.path.join(_solution_dir(project_dir, project_name), 'syn', 'verilog')


# --- verified interfaces -------------------------------------------------------


class ParameterInterface:
    """An external parameter together with the interface HLS built for it."""

    def __init__(self, parameter):
        self.parameter = parameter

    @property
    def name(self):
        return self.parameter.name

    @property
    def kind(self):
        return self.parameter.interface_kind

    # the loader addresses every parameter the same way: one word per request
    loader_depth = None  # valid ld_addr values are 0 .. loader_depth-1
    loader_data_width = None  # the low bits of ld_data this parameter uses

    def describe(self):
        """JSON-ready summary for parameter_banks.json."""
        return {'name': self.name, 'kind': self.kind, 'n_scalars': self.parameter.n_scalars}


class BramInterface(ParameterInterface):
    """A memory the IP reads through the HLS ``bram`` port protocol (address,
    enable, data). This names the interface, not the storage: what the wrapper
    implements behind it -- block RAM today -- is the bank module's choice.

    Three widths are in play and only the first is a design choice: the logical
    word the packing builds (``data_width``), the physical port that carries it
    (``port_width``, rounded up to a power-of-two byte count), and the byte stride
    between consecutive words (``addr_stride``).
    """

    def __init__(self, parameter, signals, data_width, addr_width):
        super().__init__(parameter)
        self.signals = signals  # role -> RTL signal name
        self.data_width = data_width
        self.addr_width = addr_width
        self.word_bytes = -(-data_width // 8)  # ceiling: a fixed-point word need not be byte-aligned
        self.addr_stride = 1 << (self.word_bytes - 1).bit_length()
        self.port_width = self.addr_stride * 8
        scalar_width = parameter.precision.width
        if data_width % scalar_width:
            raise InterfaceMismatch(
                f'{parameter.name}: {data_width}-bit word is not a whole number of {scalar_width}-bit scalars'
            )
        self.scalars_per_word = data_width // scalar_width
        if parameter.n_scalars % self.scalars_per_word:
            raise InterfaceMismatch(
                f'{parameter.name}: {parameter.n_scalars} scalars do not fill {self.scalars_per_word}-lane words'
            )
        self.depth = parameter.n_scalars // self.scalars_per_word

    @property
    def bank_stride_words(self):
        return 1 << (self.depth - 1).bit_length() if self.depth > 1 else 1

    @property
    def loader_depth(self):
        return self.depth

    @property
    def loader_data_width(self):
        return self.data_width

    def describe(self):
        d = super().describe()
        d.update(data_width=self.data_width, depth=self.depth, bank_stride_words=self.bank_stride_words)
        return d


class ScalarBundleInterface(ParameterInterface):
    """A fully partitioned parameter: one ``ap_none`` input port per scalar."""

    def __init__(self, parameter, ports, width):
        super().__init__(parameter)
        self.ports = list(ports)  # RTL port names, in element order
        self.width = width

    @property
    def n_scalars(self):
        return len(self.ports)

    @property
    def loader_depth(self):
        return self.n_scalars

    @property
    def loader_data_width(self):
        return self.width

    def describe(self):
        d = super().describe()
        d.update(data_width=self.width)
        return d


# --- cross-check ---------------------------------------------------------------


def verify(manifest, project_dir):
    """Cross-check every described parameter against synthesis.

    Returns ``(interfaces, summary)``: one ``BramInterface``/``ScalarBundleInterface``
    per described parameter, and the report's ``InterfaceSummary``. Raises
    ``InterfaceMismatch`` on any disagreement. Undescribed parameters are skipped,
    not guessed at.
    """
    project_name = manifest.project_name
    summary = parse_interface_summary(project_dir, solution=SOLUTION)
    rtl_ports = parse_rtl_ports(project_dir, project_name)
    by_name = {p['name']: p for p in rtl_ports}
    interfaces, problems = [], []

    for parameter in manifest.described:
        try:
            if parameter.interface_kind == 'bram':
                interfaces.append(_verify_bram(parameter, summary, by_name, project_dir, project_name))
            elif parameter.interface_kind == 'scalar_bundle':
                interfaces.append(_verify_scalar_bundle(parameter, summary, by_name))
            else:
                raise InterfaceMismatch(f'{parameter.name}: unknown interface kind {parameter.interface_kind!r}')
        except InterfaceMismatch as exc:
            problems.append(str(exc))

    if problems:
        raise InterfaceMismatch('; '.join(problems))
    return interfaces, summary


def _verify_bram(parameter, summary, rtl_by_name, project_dir, project_name):
    name = parameter.name
    key = f'{name}_PORTA'
    if key not in summary.bram:
        raise InterfaceMismatch(f'{name}: expected a BRAM interface {key}, not present in the synthesis report')
    geometry = summary.bram[key]
    data_width = geometry.data_width
    if data_width != parameter.data_width:
        raise InterfaceMismatch(f'{name}: data width {data_width} but manifest expected {parameter.data_width}')

    # every signal must exist in the RTL with the expected direction
    signals, missing, wrong_dir = {}, [], []
    for role, (suffix, want_dir) in BRAM_SIGNALS.items():
        port = rtl_by_name.get(f'{name}_{suffix}')
        if port is None:
            missing.append(f'{name}_{suffix}')
        elif port['dir'] != want_dir:
            wrong_dir.append(f'{port["name"]} is {port["dir"]}, expected {want_dir}')
        else:
            signals[role] = port
    if missing:
        raise InterfaceMismatch(f'{name}: generated RTL has no {", ".join(missing)}; the BRAM port naming has changed')
    if wrong_dir:
        raise InterfaceMismatch(f'{name}: {"; ".join(wrong_dir)}')

    addr_width = signals['addr_a']['width']
    if addr_width != geometry.addr_width:
        raise InterfaceMismatch(f'{name}: address is {addr_width} bits in RTL, {geometry.addr_width} in the report')
    interface = BramInterface(parameter, {r: s['name'] for r, s in signals.items()}, data_width, addr_width)
    if interface.depth != parameter.depth:
        raise InterfaceMismatch(f'{name}: implied depth {interface.depth} but manifest expected {parameter.depth}')

    # the address shift in the RTL is what fixes the byte stride between words
    shift, evidence = parse_addr_shift(project_dir, project_name, name)
    if shift is None:
        raise InterfaceMismatch(f'{name}: could not determine the byte-address shift from the generated RTL ({evidence})')
    if (1 << shift) != interface.addr_stride:
        raise InterfaceMismatch(
            f'{name}: RTL shifts address by {shift} but a {interface.word_bytes}-byte word '
            f'({data_width} bits) strides by {interface.addr_stride} -- {evidence}'
        )

    # The wrapper drives Addr/EN and reads Dout on both ports; Din/WEN are left
    # dangling once proven inactive, so the read interface is what must agree.
    bad = [
        f'{signals[b]["name"]} is {signals[b]["width"]} bits, {signals[a]["name"]} is {signals[a]["width"]}'
        for a, b in (('addr_a', 'addr_b'), ('dout_a', 'dout_b'), ('din_a', 'din_b'), ('wen_a', 'wen_b'))
        if signals[a]['width'] != signals[b]['width']
    ]
    bad += [
        f'{signals[role]["name"]} is {signals[role]["width"]} bits, expected 1'
        for role in ('en_a', 'en_b', 'clk_a', 'clk_b', 'rst_a', 'rst_b')
        if signals[role]['width'] != 1
    ]
    dout_name, dout_width = signals['dout_a']['name'], signals['dout_a']['width']
    report_port = summary.port(dout_name)
    if report_port is not None and report_port.bits != dout_width:
        bad.append(f'{dout_name} is {dout_width} bits in RTL, {report_port.bits} in the report')
    if dout_width != interface.port_width:
        bad.append(f'read port is {dout_width} bits; expected {interface.port_width} (rounded from {data_width})')
    if bad:
        raise InterfaceMismatch(f'{name}: BRAM interface disagrees ({"; ".join(bad)})')

    # The loader shares port B with the IP, which is sound only if the IP never
    # writes the memory and both ports run on ap_clk; prove both from the RTL.
    read_only, evidence = bram_is_read_only(project_dir, project_name, interface.signals)
    if not read_only:
        raise InterfaceMismatch(f'{name}: the IP is not provably read-only on this memory ({evidence})')
    same_clk, evidence = bram_ports_use_ap_clk(project_dir, project_name, interface.signals)
    if not same_clk:
        raise InterfaceMismatch(f'{name}: both port clocks must be ap_clk ({evidence})')
    return interface


def _verify_scalar_bundle(parameter, summary, rtl_by_name):
    name = parameter.name
    # Require exactly <name>_0 .. <name>_N-1: a matching count is not enough,
    # since a gap plus an extra index would also count correctly.
    expected = [f'{name}_{i}' for i in range(parameter.n_scalars)]
    members = sorted(
        (p.name for p in summary.ports if re.fullmatch(rf'{re.escape(name)}_\d+', p.name)),
        key=lambda p: int(p.rsplit('_', 1)[1]),
    )
    if members != expected:
        raise InterfaceMismatch(f'{name}: scalar ports are {members}, expected {expected}')
    modes = {summary.port(m).protocol for m in members}
    widths = {summary.port(m).bits for m in members}
    if len(modes) != 1 or len(widths) != 1:
        raise InterfaceMismatch(f'{name}: scalar ports disagree (modes={sorted(modes)}, widths={sorted(widths)})')
    mode, bits = modes.pop(), widths.pop()
    if mode != EXPECTED_SCALAR_MODE:
        raise InterfaceMismatch(f'{name}: scalar interface mode is {mode!r}, expected {EXPECTED_SCALAR_MODE!r}')
    if bits != parameter.data_width:
        raise InterfaceMismatch(f'{name}: scalar width {bits} but manifest expected {parameter.data_width}')

    problems = []
    for member in members:
        port = rtl_by_name.get(member)
        if port is None:
            problems.append(f'{member} is not a port of the generated RTL')
        elif port['dir'] != 'input':
            problems.append(f'{member} is {port["dir"]}, expected input')
        elif port['width'] != bits:
            problems.append(f'{member} is {port["width"]} bits, the report says {bits}')
    if problems:
        raise InterfaceMismatch(f'{name}: {"; ".join(problems)}')
    return ScalarBundleInterface(parameter, members, bits)


# --- generated Verilog ---------------------------------------------------------

# Ways HLS spells a constant left shift of an address. The scale of the shift is
# the fact; the spelling is not, so all of them are read the same way.
_ADDR_SHIFT_FORMS = (
    re.compile(r"<<\s*(?:\d+'[dh])?(\d+)$"),  # x << 32'd4, x << 4
    re.compile(r",\s*(\d+)'[bdh]0+\s*\}$"),  # {x, 4'd0}
)


def parse_addr_shift(project_dir, project_name, port):
    """Return (shift, evidence): how far HLS shifts this port's word index left.

    A one-byte word needs no shift and HLS emits none, so an unshifted driver means
    shift 0. ``(None, ...)`` means nothing drives the port, or drivers disagree.
    """
    verilog_dir = solution_verilog_dir(project_dir, project_name)
    if not os.path.isdir(verilog_dir):
        return None, f'no synthesized RTL at {verilog_dir}'

    # What sits between _Addr_A and the shift varies with pipeline style --
    # _local, _orig, a gep temporary, or nothing -- so match any driver.
    driver = re.compile(rf'assign\s+{re.escape(port)}_Addr_A(?:_\w+)?\s*=\s*([^;]+);')

    assigns, shifts = [], {}  # shift -> evidence
    for name in sorted(os.listdir(verilog_dir)):
        if not name.endswith('.v'):
            continue
        with open(os.path.join(verilog_dir, name)) as fh:
            for match in driver.finditer(fh.read()):
                rhs = ' '.join(match.group(1).split())
                assigns.append(f'{name}: {port}_Addr_A... = {rhs}')
                for form in _ADDR_SHIFT_FORMS:
                    shift = form.search(rhs)
                    if shift:
                        shifts.setdefault(int(shift.group(1)), assigns[-1])

    if not assigns:
        return None, f'nothing in {verilog_dir} drives {port}_Addr_A'
    if len(shifts) > 1:
        return None, 'conflicting shifts: ' + '; '.join(shifts.values())  # file order is not a tie-breaker
    if shifts:
        return next(iter(shifts.items()))
    return 0, 'no shift in ' + '; '.join(assigns)


def parse_rtl_ports(project_dir, project_name):
    """The generated top module's real port list, in header order, as
    ``{'name', 'dir', 'width'}``. The RTL is authoritative for what the wrapper
    must connect; deriving it from interface modes would mean guessing which
    companion signals HLS emitted."""
    path = os.path.join(solution_verilog_dir(project_dir, project_name), f'{project_name}.v')
    if not os.path.exists(path):
        raise FileNotFoundError(f'no exported RTL at {path}')
    text = open(path).read()

    header = re.search(rf'\bmodule\s+{re.escape(project_name)}\s*\((.*?)\)\s*;', text, re.S)
    if not header:
        raise ValueError(f'could not find module header for {project_name} in {path}')
    order = [p.strip() for p in header.group(1).split(',') if p.strip()]

    decls = {}
    for match in re.finditer(
        r'^\s*(input|output|inout)\s+(?:wire\s+|reg\s+)?(?:\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*)?(\w+)\s*;', text, re.M
    ):
        direction, msb, lsb, name = match.groups()
        width = (int(msb) - int(lsb) + 1) if msb is not None else 1
        decls[name] = {'name': name, 'dir': direction, 'width': width}

    ports, missing = [], []
    for name in order:
        if name in decls:
            ports.append(decls[name])
        else:
            missing.append(name)
    if missing:
        raise ValueError(f'{project_name}: no declaration found for header port(s) {missing}; refusing to guess')
    inout = [p['name'] for p in ports if p['dir'] == 'inout']
    if inout:
        raise ValueError(f'{project_name}: inout ports are not supported ({inout})')
    return ports


def bram_ports_use_ap_clk(project_dir, project_name, signals):
    """Prove from the RTL that both BRAM port clocks are ap_clk: the wrapper clocks
    the banked memory with ap_clk and leaves Clk_A/Clk_B dangling."""
    path = os.path.join(solution_verilog_dir(project_dir, project_name), f'{project_name}.v')
    if not os.path.exists(path):
        return False, 'exported RTL not found'
    text = open(path).read()
    evidence = {}
    for role in ('clk_a', 'clk_b'):
        name = signals[role]
        evidence[name] = bool(re.search(rf'assign\s+{re.escape(name)}\s*=\s*ap_clk\s*;', text))
    return all(evidence.values()), evidence


def bram_is_read_only(project_dir, project_name, signals):
    """Prove from the RTL that the IP never writes this parameter memory, which is
    what makes lending port B to the loader sound. Whether the IP *reads* port B is
    deliberately not checked: Dense leaves it idle, pointwise uses it."""
    path = os.path.join(solution_verilog_dir(project_dir, project_name), f'{project_name}.v')
    if not os.path.exists(path):
        return False, 'exported RTL not found'
    text = open(path).read()
    # A zero write enable is what makes it read-only; Din may be driven without
    # meaning anything, so it is not required to be tied off.
    evidence = {}
    for role in ('wen_a', 'wen_b'):
        name = signals[role]
        evidence[name] = bool(re.search(rf"assign\s+{re.escape(name)}\s*=\s*\d+'[bdh]0\s*;", text))
    return all(evidence.values()), evidence
