"""Classify the interface an hls4ml project actually synthesized, and cross-check
it against the manifest.

The manifest records what hls4ml *asked* HLS for. This module records what HLS
*built*, by reading the C/RTL synthesis report and the generated Verilog. Nothing
here is inferred from pragmas -- if a fact is not in the generated output, it is
not reported.
"""

import os
import re


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
    pattern = re.compile(rf"{re.escape(port)}_Addr_A_local\s*=\s*{re.escape(port)}_Addr_A_orig\s*<<\s*\d+'d(\d+)")
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

            word_bytes = data_width // 8
            shift = parse_addr_shift(project_dir, project_name, name)
            if shift is not None and (1 << shift) != word_bytes:
                problems.append(f'{name}: RTL shifts address by {shift} but word is {word_bytes} bytes')
                continue

            scalars_per_word = data_width // port['precision']['width']
            depth = port['n_scalars'] // scalars_per_word
            if depth != port['expected_depth']:
                problems.append(f'{name}: implied depth {depth} but manifest expected {port["expected_depth"]}')
                continue

            verified.append(
                {
                    **port,
                    'actual_data_width': data_width,
                    'actual_addr_width': addr_width,
                    'actual_word_bytes': word_bytes,
                    'actual_depth': depth,
                    'actual_scalars_per_word': scalars_per_word,
                }
            )

        elif kind == 'scalar_bundle':
            members = sorted(p for p in hardware['scalar'] if re.fullmatch(rf'{re.escape(name)}_\d+', p))
            if len(members) != port['n_scalars']:
                problems.append(f'{name}: found {len(members)} scalar ports, expected {port["n_scalars"]}')
                continue
            mode, bits = hardware['scalar'][members[0]]
            if bits != port['expected_data_width']:
                problems.append(f'{name}: scalar width {bits} but manifest expected {port["expected_data_width"]}')
                continue
            verified.append({**port, 'actual_ports': members, 'actual_mode': mode, 'actual_width': bits})

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

    ports = []
    for name in order:
        if name in decls:
            ports.append(decls[name])
    return ports


def bram_port_b_is_unused(project_dir, project_name, port):
    """Prove from the RTL that HLS leaves port B of a BRAM interface idle.

    csynth declares both PORTA and PORTB for a ``bram`` interface, but hls4ml's
    generated top drives only port A and ties port B off. The wrapper may only
    borrow port B for the loader if that is actually true, so check rather than
    assume. Returns (is_unused, evidence).
    """
    path = os.path.join(_solution_dir(project_dir, project_name), 'syn', 'verilog', f'{project_name}.v')
    if not os.path.exists(path):
        return False, 'exported RTL not found'
    text = open(path).read()

    evidence = {}
    for signal, pattern in (
        ('EN_B', rf"assign\s+{re.escape(port)}_EN_B\s*=\s*1'b0\s*;"),
        ('WEN_B', rf"assign\s+{re.escape(port)}_WEN_B\s*=\s*\d+'[bd]0\s*;"),
        ('Addr_B', rf"assign\s+{re.escape(port)}_Addr_B\s*=\s*\d+'[bd]0\s*;"),
    ):
        evidence[signal] = bool(re.search(pattern, text))

    return evidence.get('EN_B', False), evidence
