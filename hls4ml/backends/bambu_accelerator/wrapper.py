"""
Verilog wrapper generation for Bambu-generated HLS modules.

Pure-Python library: no hls4ml imports, no CLI entry point.

Typical usage::

    content = Path('myproject_float.v').read_text()
    module_name, port_names, port_decls = parse_module(content)
    flow = detect_flow(port_names)
    wrapper_verilog = generate_wrapper_verilog(module_name, port_names, port_decls, flow)
    in_dw, out_dw = extract_data_widths(port_names, port_decls, flow)
    slots = extract_bram_depths(port_names, port_decls, flow)  # None for 'stream'
    localparams = generate_top_localparams(in_dw, n_in, out_dw, n_out, flow)
"""

import re

# PortDecls maps port name -> (direction, width_str)
# direction : 'input' | 'output'
# width_str : e.g. '[15:0]' or '' for a 1-bit port
PortDecls = dict[str, tuple[str, str]]

# BRAM interface suffixes used to detect and group memory ports (io_parallel)
_BRAM_SUFFIXES: list[str] = [
    '_address0',
    '_address1',
    '_ce0',
    '_ce1',
    '_we0',
    '_we1',
    '_d0',
    '_d1',
    '_q0',
    '_q1',
]

# AXI-Stream interface suffixes used to detect and group streaming ports (io_stream).
# Bambu emits TDATA/TVALID/TREADY on ports named like `<stream_name>_TDATA`.
_AXIS_SUFFIXES: list[str] = ['_TDATA', '_TVALID', '_TREADY']

# Standard HLS control ports — never renamed in the wrapper
_STANDARD_PORTS: frozenset[str] = frozenset({'clock', 'reset', 'start_port', 'done_port'})


def _parse_width(width_str: str) -> int:
    """Return the bit-width encoded in a Verilog range string.

    '[N:M]' -> N - M + 1.  Empty string -> 1 (scalar wire).
    """
    if not width_str:
        return 1
    m = re.match(r'\[(\d+):(\d+)\]', width_str.strip())
    return int(m.group(1)) - int(m.group(2)) + 1 if m else 1


def parse_module(content: str) -> tuple[str, list[str], PortDecls]:
    """Parse the last non-'myproject' module in a Verilog string.

    Returns:
        module_name : name of the HLS top-level module
        port_names  : port names in declaration order
        port_decls  : mapping from port name to (direction, width_str)
    """
    all_mods = list(re.finditer(r'\bmodule\s+(\w+)\s*\(', content))
    if not all_mods:
        raise ValueError('No module declarations found')

    target = next((m for m in reversed(all_mods) if m.group(1) != 'myproject'), None)
    if target is None:
        raise ValueError("Only a 'myproject' module was found — nothing to wrap")

    module_name = target.group(1)

    # Find the matching closing parenthesis of the port list
    paren_start = target.end() - 1  # position of '('
    depth, i = 0, paren_start
    while i < len(content):
        if content[i] == '(':
            depth += 1
        elif content[i] == ')':
            depth -= 1
            if depth == 0:
                break
        i += 1
    close_paren = i

    # Extract ordered port names from the header, stripping comments
    raw = content[paren_start + 1 : close_paren]
    raw = re.sub(r'//[^\n]*', '', raw)
    raw = re.sub(r'/\*.*?\*/', '', raw, flags=re.DOTALL)
    port_names = [p.strip() for p in raw.split(',') if p.strip()]

    # Extract direction/width from the module body (old-style port declarations)
    body_start = close_paren + 1
    end_match = re.search(r'\bendmodule\b', content[body_start:])
    body = content[body_start : body_start + end_match.start()] if end_match else content[body_start:]

    decl_pat = re.compile(
        r'^\s*(input|output)\s*(?:wire\s*)?(?:reg\s*)?(?:signed\s*)?\s*(\[[^\]]*\])?\s*(\w+)\s*;',
        re.MULTILINE,
    )
    port_decls: PortDecls = {m.group(3): (m.group(1), (m.group(2) or '').strip()) for m in decl_pat.finditer(body)}

    return module_name, port_names, port_decls


def detect_flow(port_names: list[str]) -> str:
    """Classify the HLS IP by inspecting its port suffixes.

    Returns 'stream' if any port ends in an AXI-Stream suffix
    (_TDATA/_TVALID/_TREADY), else 'parallel'.
    """
    for port in port_names:
        for suffix in _AXIS_SUFFIXES:
            if port.endswith(suffix):
                return 'stream'
    return 'parallel'


def _build_rename_map_bram(port_names: list[str]) -> dict[str, str]:
    """Return a rename map {original_port: generic_port} for BRAM memory ports.

    BRAM groups are identified by their suffix pattern and classified as:
      - input  group: carries _q0/_q1 (data read from BRAM into HLS)
      - output group: carries _d0/_d1/_we* (data written from HLS to BRAM)

    Multiple groups of the same class are indexed: input0_*, input1_*, ...
    Standard control ports are not renamed.
    """
    groups: dict[str, dict[str, str]] = {}
    for port in port_names:
        if port in _STANDARD_PORTS:
            continue
        for suffix in _BRAM_SUFFIXES:
            if port.endswith(suffix):
                prefix = port[: -len(suffix)]
                groups.setdefault(prefix, {})[suffix] = port
                break

    input_groups = sorted(p for p, m in groups.items() if {'_q0', '_q1'} & m.keys())
    output_groups = sorted(p for p, m in groups.items() if {'_d0', '_d1', '_we0', '_we1'} & m.keys())

    rename_map: dict[str, str] = {}
    for class_groups, base in ((input_groups, 'input'), (output_groups, 'output')):
        for idx, prefix in enumerate(class_groups):
            new_prefix = base if len(class_groups) == 1 else f'{base}{idx}'
            for suffix, old_name in groups[prefix].items():
                rename_map[old_name] = f'{new_prefix}{suffix}'

    return rename_map


def _build_rename_map_axis(port_names: list[str], port_decls: PortDecls) -> dict[str, str]:
    """Return a rename map {original_port: generic_port} for AXI-Stream ports.

    Groups ports by prefix (e.g. 'input_stream' from 'input_stream_TDATA').
    A group is an *input stream* (HLS IP consumes) when its _TDATA port is
    an input of the HLS module; an *output stream* (HLS IP produces) when
    its _TDATA port is an output.  Prefixes become `hls_in` / `hls_out`
    with lowercase `_t{data,valid,ready}` suffixes.

    Multiple input or output streams are indexed: hls_in0_t*, hls_in1_t*, ...
    """
    groups: dict[str, dict[str, str]] = {}
    for port in port_names:
        if port in _STANDARD_PORTS:
            continue
        for suffix in _AXIS_SUFFIXES:
            if port.endswith(suffix):
                prefix = port[: -len(suffix)]
                groups.setdefault(prefix, {})[suffix] = port
                break

    def _is_input_stream(prefix: str, members: dict[str, str]) -> bool:
        # Classify by the direction of the _TDATA port in the HLS module.
        tdata = members.get('_TDATA')
        if tdata is None:
            # Fallback: _TREADY is an output of the wrapper iff this is an
            # input stream (the IP asserts ready for incoming data).
            tready = members.get('_TREADY')
            if tready and port_decls.get(tready, ('', ''))[0] == 'output':
                return True
            return False
        return port_decls.get(tdata, ('', ''))[0] == 'input'

    input_prefixes = sorted(p for p, m in groups.items() if _is_input_stream(p, m))
    output_prefixes = sorted(p for p, m in groups.items() if not _is_input_stream(p, m))

    rename_map: dict[str, str] = {}
    for class_prefixes, base in ((input_prefixes, 'hls_in'), (output_prefixes, 'hls_out')):
        for idx, prefix in enumerate(class_prefixes):
            new_prefix = base if len(class_prefixes) == 1 else f'{base}{idx}'
            for suffix, old_name in groups[prefix].items():
                rename_map[old_name] = f'{new_prefix}{suffix.lower()}'

    return rename_map


def build_rename_map(port_names: list[str], port_decls: PortDecls, flow: str) -> dict[str, str]:
    """Return {original_hls_port: wrapper_port} for non-standard ports.

    Args:
        port_names: port names in declaration order (from parse_module)
        port_decls: mapping from port name to (direction, width_str)
        flow: 'stream' or 'parallel'

    Returns:
        dict mapping original HLS port names to renamed wrapper port names.
        Standard control ports (clock, reset, start_port, done_port) are absent.
    """
    if flow == 'stream':
        return _build_rename_map_axis(port_names, port_decls)
    return _build_rename_map_bram(port_names)


def generate_wrapper_verilog(module_name: str, port_names: list[str], port_decls: PortDecls, flow: str) -> str:
    """Return the 'myproject' wrapper Verilog instantiating module_name as u0.

    BRAM (parallel) or AXI-Stream (stream) ports are renamed to generic names
    appropriate to the flow; the instantiation connects each renamed wrapper
    port back to the original port name.
    """
    rename_map = build_rename_map(port_names, port_decls, flow)

    # Column-align the width field across all port declarations
    max_w = max((len(port_decls.get(p, ('', ''))[1]) for p in port_names), default=0)

    def width_col(width: str) -> str:
        """Padded text between 'wire' and the port name."""
        if max_w == 0:
            return ' '
        return f' {width:<{max_w}} ' if width else ' ' * (max_w + 2)

    lines = [
        '// Wrapper with a valid identifier for mixed-language (VHDL top) instantiation',
        'module myproject (',
    ]
    last = len(port_names) - 1
    for idx, port in enumerate(port_names):
        direction, width = port_decls.get(port, ('input', ''))
        display = rename_map.get(port, port)
        comma = '' if idx == last else ','
        # 'input ' / 'output' are both 6 chars, keeping 'wire' aligned
        lines.append(f'  {direction:<6} wire{width_col(width)}{display}{comma}')

    lines += [');', f'  {module_name} u0 (']
    for idx, port in enumerate(port_names):
        comma = '' if idx == last else ','
        lines.append(f'    .{port}({rename_map.get(port, port)}){comma}')
    lines += ['  );', 'endmodule', '']

    return '\n'.join(lines)


def extract_data_widths(port_names: list[str], port_decls: PortDecls, flow: str) -> tuple[int, int]:
    """Return (in_data_width_bits, out_data_width_bits) from the HLS port declarations.

    Parallel mode: input width from first _q0/_q1 port, output from first _d0/_d1.
    Stream mode: input width from first input-stream _TDATA port, output from
    first output-stream _TDATA port (direction determined by port direction in the IP).
    """
    if flow == 'stream':
        in_tdatas = [p for p in port_names if p.endswith('_TDATA') and port_decls.get(p, ('', ''))[0] == 'input']
        out_tdatas = [p for p in port_names if p.endswith('_TDATA') and port_decls.get(p, ('', ''))[0] == 'output']
        in_dw = _parse_width(port_decls[sorted(in_tdatas)[0]][1]) if in_tdatas else 0
        out_dw = _parse_width(port_decls[sorted(out_tdatas)[0]][1]) if out_tdatas else 0
        return in_dw, out_dw

    in_ports = sorted(p for p in port_names if p.endswith(('_q0', '_q1')))
    out_ports = sorted(p for p in port_names if p.endswith(('_d0', '_d1')))

    in_dw = _parse_width(port_decls[in_ports[0]][1]) if in_ports else 0
    out_dw = _parse_width(port_decls[out_ports[0]][1]) if out_ports else 0
    return in_dw, out_dw


def extract_bram_depths(port_names: list[str], port_decls: PortDecls, flow: str) -> dict[str, int] | None:
    """Return {'in': depth, 'out': depth} = 2 ** (width of each `*_address0` port).

    This is Bambu's BRAM depth, which it rounds up to a whole number of address
    bits: a 5-element output array gets a 3-bit address = 8 slots.  It is NOT
    the element count from the firmware headers.

    HLS_*_N_WORDS must carry the DEPTH, established empirically on NG-ULTRA
    (jet-tagger, 5 outputs, 3-bit address port):

        N_WORDS=4 (stale template)  ->  2958 LUT4, 7794 carry, works, but the
                                        AXI window is short and the board's
                                        read burst hangs (bsp_rc=4)
        N_WORDS=5 (element count)   ->   144 LUT4,    0 carry -- NxMap deletes
                                        the entire datapath.  The AXI slave's
                                        output_flat_padded path nominally
                                        supports a non-power-of-two word count;
                                        in practice synthesis collapses it.
        N_WORDS=8 (BRAM depth)      ->  3329 LUT4, 7794 carry, datapath intact

    Returns None for the stream flow: AXI-Stream IPs have no address ports, and
    AXISlaveStream's N_BEATS_*/LAST_BEAT_*_VALID arithmetic genuinely wants the
    element count.  Do not route this value there.
    """
    if flow == 'stream':
        return None

    inverse = {new: old for old, new in build_rename_map(port_names, port_decls, flow).items()}
    depths = {}
    for key, base in (('in', 'input'), ('out', 'output')):
        port = inverse.get(f'{base}_address0') or inverse.get(f'{base}_address1')
        if port is None:
            raise ValueError(f'No {base} address port found -- cannot size the {key} BRAM')
        depths[key] = 2 ** _parse_width(port_decls[port][1])
    return depths


def generate_top_localparams(in_dw: int, in_n: int, out_dw: int, out_n: int, flow: str) -> str:
    """Return the localparam block string for top_parallel.v / top_stream.v.

    `in_n`/`out_n` are BRAM DEPTHS for the parallel flow (see
    extract_bram_depths -- a non-power-of-two value makes NxMap delete the
    datapath) and ELEMENT COUNTS for the stream flow (AXISlaveStream's
    LAST_BEAT_*_VALID arithmetic needs the true count).  The caller routes it;
    this function only formats.

    Parallel flow emits the four HLS_*_DATA_W / HLS_*_N_WORDS parameters plus
    the two derived HLS_*_ADDR_W widths that AXISlaveParallel needs.
    Stream flow skips the ADDR_W lines (AXISlaveStream doesn't use them).
    """
    lines = [
        f'localparam HLS_IN_DATA_W   = {in_dw};',
        f'localparam HLS_OUT_DATA_W  = {out_dw};',
        f'localparam HLS_IN_N_WORDS  = {in_n};',
        f'localparam HLS_OUT_N_WORDS = {out_n};',
    ]
    if flow != 'stream':
        lines += [
            'localparam HLS_IN_ADDR_W   = (HLS_IN_N_WORDS  > 1) ? $clog2(HLS_IN_N_WORDS)  : 1;',
            'localparam HLS_OUT_ADDR_W  = (HLS_OUT_N_WORDS > 1) ? $clog2(HLS_OUT_N_WORDS) : 1;',
        ]
    return '\n'.join(lines)
