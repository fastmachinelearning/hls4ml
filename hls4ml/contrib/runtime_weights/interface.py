"""Classify the interface an hls4ml project actually synthesized, and cross-check
it against the manifest.

The manifest records what hls4ml *asked* HLS for. This module records what HLS
*built*, by reading the C/RTL synthesis report and the generated Verilog. Nothing
here is inferred from pragmas -- if a fact is not in the generated output, it is
not reported.
"""

import os
import re

# A fully partitioned parameter lowers to plain input ports with no handshake.
# Anything else would need its own wrapper treatment.
EXPECTED_SCALAR_MODE = 'ap_none'

# How Vitis spells the signals of one external BRAM parameter port. This map is the
# only place that spelling appears: verify() resolves it against the real RTL and
# hands the resulting names to the wrapper generator, which never rebuilds them.
BRAM_SIGNAL_SUFFIXES = {
    'addr_a': 'Addr_A',
    'en_a': 'EN_A',
    'din_a': 'Din_A',
    'dout_a': 'Dout_A',
    'wen_a': 'WEN_A',
    'clk_a': 'Clk_A',
    'rst_a': 'Rst_A',
    'addr_b': 'Addr_B',
    'en_b': 'EN_B',
    'din_b': 'Din_B',
    'dout_b': 'Dout_B',
    'wen_b': 'WEN_B',
    'clk_b': 'Clk_B',
    'rst_b': 'Rst_B',
}


# Direction of each BRAM signal from the IP's point of view. The wrapper drives the
# memory side, so anything mismatched here would be connected backwards.
BRAM_SIGNAL_DIRS = {
    'addr_a': 'output',
    'en_a': 'output',
    'din_a': 'output',
    'dout_a': 'input',
    'wen_a': 'output',
    'clk_a': 'output',
    'rst_a': 'output',
}


def solution_verilog_dir(project_dir, project_name):
    """Where Vitis writes the exported RTL. Callers should not spell this out."""
    return os.path.join(_solution_dir(project_dir, project_name), 'syn', 'verilog')


def bram_signals(rtl_ports, name):
    """Resolve one BRAM parameter's signals against the real RTL port list.

    Returns {role: {'name', 'width', 'dir'}}. Raises if the generated RTL does not
    spell the port the way this module expects, so a renamed or missing signal is
    reported here rather than becoming a wrapper that will not elaborate.
    """
    by_name = {p['name']: p for p in rtl_ports}
    signals, missing = {}, []
    for role, suffix in BRAM_SIGNAL_SUFFIXES.items():
        port = by_name.get(f'{name}_{suffix}')
        if port is None:
            missing.append(f'{name}_{suffix}')
        else:
            signals[role] = port
    if missing:
        raise InterfaceMismatch(f'{name}: generated RTL has no {", ".join(missing)}; the BRAM port naming has changed')
    return signals


class InterfaceMismatch(Exception):
    """The synthesized interface disagrees with the manifest."""


def _solution_dir(project_dir, project_name):
    return os.path.join(project_dir, f'{project_name}_prj', 'solution1')


def parse_csynth(project_dir, project_name):
    """Return {'bram': {iface: (data_width, addr_width)}, 'scalar': {port: (mode, bits)},
    'control': str}. Raises FileNotFoundError if synthesis has not been run."""
    report = os.path.join(_solution_dir(project_dir, project_name), 'syn', 'report', 'csynth.rpt')
    if not os.path.exists(report):
        raise FileNotFoundError(f'no C/RTL synthesis report at {report}; run build(synth=True) first')

    text = open(report).read()
    out = {'bram': {}, 'scalar': {}, 'control': None}

    section = re.search(r'\* BRAM\n(.*?)\n\n', text, re.S)
    if section:
        for line in section.group(1).splitlines():
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if len(cells) == 3 and cells[0] not in ('Interface', '') and not cells[0].startswith('-'):
                try:
                    out['bram'][cells[0]] = (int(cells[1]), int(cells[2]))
                except ValueError:
                    pass

    section = re.search(r'\* Other Ports\n(.*?)\n\n', text, re.S)
    if section:
        for line in section.group(1).splitlines():
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if len(cells) == 4 and cells[0] not in ('Port', '') and not cells[0].startswith('-'):
                try:
                    out['scalar'][cells[0]] = (cells[1], int(cells[3]))
                except ValueError:
                    pass

    control = re.search(r'\|\s*ap_ctrl\s*\|\s*(\w+)\s*\|', text)
    if control:
        out['control'] = control.group(1)
    return out


def parse_addr_shift(project_dir, project_name, port):
    """Recover N from `<port>_Addr_A_local = <port>_Addr_A_orig << 32'dN` in the RTL.

    This is how many bytes one memory word occupies; hls4ml BRAM ports are
    byte-addressed.
    """
    verilog_dir = os.path.join(_solution_dir(project_dir, project_name), 'syn', 'verilog')
    if not os.path.isdir(verilog_dir):
        return None
    # The shifted source has different names in different pipeline styles
    # (<port>_Addr_A_orig with dataflow, a gep temporary with pipeline), so match
    # any right-hand side. Being too specific here reports a parse failure instead
    # of whatever the real incompatibility is.
    pattern = re.compile(rf"assign\s+{re.escape(port)}_Addr_A_local\s*=\s*.+?<<\s*\d+'d(\d+)\s*;")
    for name in sorted(os.listdir(verilog_dir)):
        if not name.endswith('.v'):
            continue
        match = pattern.search(open(os.path.join(verilog_dir, name)).read())
        if match:
            return int(match.group(1))
    return None


def verify(manifest, project_dir, project_name):
    """Cross-check the manifest against synthesis. Returns a list of verified ports.

    Raises InterfaceMismatch on any disagreement. Ports the manifest declined to
    describe are skipped, not guessed at.
    """
    hardware = parse_csynth(project_dir, project_name)
    rtl_ports = parse_rtl_ports(project_dir, project_name)
    verified = []
    problems = []

    for port in manifest['ports']:
        kind = port['expected_interface_kind']
        name = port['name']

        if kind is None:
            continue  # outside schema scope; manifest claims nothing

        if kind == 'bram':
            key = f'{name}_PORTA'
            if key not in hardware['bram']:
                problems.append(f'{name}: expected a BRAM interface {key}, not present in csynth report')
                continue
            data_width, addr_width = hardware['bram'][key]
            if data_width != port['expected_data_width']:
                problems.append(f'{name}: data width {data_width} but manifest expected {port["expected_data_width"]}')
                continue

            # Three widths are in play and only the first comes from the report:
            #   logical width  the packed word this schema builds        (e.g. 96)
            #   physical width the RTL port that carries it              (e.g. 128)
            #   byte stride    how far consecutive words are apart       (e.g. 16)
            # Ceiling division, because a fixed-point word need not be byte-aligned.
            word_bytes = -(-data_width // 8)
            addr_stride = 1 << (word_bytes - 1).bit_length()
            shift = parse_addr_shift(project_dir, project_name, name)
            if shift is None:
                problems.append(f'{name}: could not determine the byte-address shift from the generated RTL')
                continue
            if (1 << shift) != addr_stride:
                problems.append(
                    f'{name}: RTL shifts address by {shift} but a {word_bytes}-byte word strides by {addr_stride}'
                )
                continue

            scalars_per_word = data_width // port['precision']['width']
            depth = port['n_scalars'] // scalars_per_word
            if depth != port['expected_depth']:
                problems.append(f'{name}: implied depth {depth} but manifest expected {port["expected_depth"]}')
                continue

            try:
                signals = bram_signals(rtl_ports, name)
            except InterfaceMismatch as exc:
                problems.append(str(exc))
                continue
            # csynth reports the logical width; the RTL port is rounded up to a
            # power of two (a 96-bit word is carried on a 128-bit port). Accept the
            # rounding, refuse anything else.
            wrong_dir = [
                f'{signals[role]["name"]} is {signals[role]["dir"]}, expected {want}'
                for role, want in BRAM_SIGNAL_DIRS.items()
                if signals[role]['dir'] != want
            ]
            if wrong_dir:
                problems.append(f'{name}: {"; ".join(wrong_dir)}')
                continue

            port_width = signals['din_a']['width']
            if port_width != (addr_stride * 8):
                problems.append(
                    f'{name}: csynth reports {data_width} bits but the RTL port is {port_width}; '
                    f'expected the {addr_stride * 8}-bit rounding'
                )
                continue

            verified.append(
                {
                    **port,
                    'actual_signals': {role: sig['name'] for role, sig in signals.items()},
                    'actual_port_width': port_width,
                    'actual_data_width': data_width,
                    'actual_addr_width': addr_width,
                    'actual_word_bytes': word_bytes,
                    'actual_addr_stride': addr_stride,
                    'actual_depth': depth,
                    'actual_scalars_per_word': scalars_per_word,
                }
            )

        elif kind == 'scalar_bundle':
            # Require exactly <name>_0 .. <name>_N-1: a matching count is not
            # enough, since a gap plus an extra index would also count correctly.
            expected_members = [f'{name}_{i}' for i in range(port['n_scalars'])]
            members = sorted(
                (p for p in hardware['scalar'] if re.fullmatch(rf'{re.escape(name)}_\d+', p)),
                key=lambda p: int(p.rsplit('_', 1)[1]),
            )
            if members != expected_members:
                problems.append(f'{name}: scalar ports are {members}, expected {expected_members}')
                continue
            modes = {hardware['scalar'][m][0] for m in members}
            widths = {hardware['scalar'][m][1] for m in members}
            if len(modes) != 1 or len(widths) != 1:
                problems.append(f'{name}: scalar ports disagree (modes={sorted(modes)}, widths={sorted(widths)})')
                continue
            mode, bits = modes.pop(), widths.pop()
            if mode != EXPECTED_SCALAR_MODE:
                problems.append(f'{name}: scalar interface mode is {mode!r}, expected {EXPECTED_SCALAR_MODE!r}')
                continue
            if bits != port['expected_data_width']:
                problems.append(f'{name}: scalar width {bits} but manifest expected {port["expected_data_width"]}')
                continue
            # csynth names the ports; the wrapper connects them, so confirm they are
            # really there, are inputs, and are the width csynth claimed.
            by_name = {p['name']: p for p in rtl_ports}
            rtl_problems = []
            for member in members:
                rtl_port = by_name.get(member)
                if rtl_port is None:
                    rtl_problems.append(f'{member} is not a port of the generated RTL')
                elif rtl_port['dir'] != 'input':
                    rtl_problems.append(f'{member} is {rtl_port["dir"]}, expected input')
                elif rtl_port['width'] != bits:
                    rtl_problems.append(f'{member} is {rtl_port["width"]} bits, csynth says {bits}')
            if rtl_problems:
                problems.append(f'{name}: {"; ".join(rtl_problems)}')
                continue

            verified.append({**port, 'actual_ports': members, 'actual_mode': mode, 'actual_width': bits})

        else:
            problems.append(f'{name}: unknown interface kind {kind!r}')

    if problems:
        raise InterfaceMismatch('; '.join(problems))

    return verified, hardware


def parse_rtl_ports(project_dir, project_name):
    """Read the generated top module's real port list.

    Returns an ordered list of {'name', 'dir', 'width'}. The RTL is authoritative:
    deriving the port list from interface *modes* would require guessing which
    companion signals (``_ap_vld`` and friends) HLS emitted.
    """
    path = os.path.join(_solution_dir(project_dir, project_name), 'syn', 'verilog', f'{project_name}.v')
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


def bram_port_clk_is_ap_clk(project_dir, project_name, signals):
    """Prove from the RTL that the IP ties this BRAM port's clock to ap_clk.

    The wrapper clocks the banked memory with ap_clk and leaves Clk_A dangling, so
    check rather than assume the two are the same clock. ``signals`` is the mapping
    verify() resolved, so no signal name is reconstructed here.
    """
    path = os.path.join(solution_verilog_dir(project_dir, project_name), f'{project_name}.v')
    if not os.path.exists(path):
        return False, 'exported RTL not found'
    text = open(path).read()
    clk = signals['clk_a']
    tied = bool(re.search(rf'assign\s+{re.escape(clk)}\s*=\s*ap_clk\s*;', text))
    return tied, {f'{clk}_tied_to_ap_clk': tied}


def bram_is_read_only(project_dir, project_name, signals):
    """Prove from the RTL that the IP never writes this parameter memory.

    That is what makes lending port B to the loader sound. Whether the IP *reads*
    port B is deliberately not checked: Dense leaves it idle, pointwise uses it, and
    the wrapper serves both. Returns (is_read_only, evidence).
    """
    path = os.path.join(solution_verilog_dir(project_dir, project_name), f'{project_name}.v')
    if not os.path.exists(path):
        return False, 'exported RTL not found'
    text = open(path).read()

    evidence = {}
    for role in ('wen_a', 'wen_b', 'din_a', 'din_b'):
        name = signals[role]
        evidence[name] = bool(re.search(rf"assign\s+{re.escape(name)}\s*=\s*\d+'[bdh]0\s*;", text))

    return all(evidence.values()), evidence
