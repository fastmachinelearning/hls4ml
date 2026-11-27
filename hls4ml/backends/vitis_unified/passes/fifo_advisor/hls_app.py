import os
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import quoteattr


def _resolve(cfg_path, path):
    return os.path.abspath(os.path.join(cfg_path.parent, path))


def parse_kernel_cfg(cfg_path):
    """Return ``(top, syn_files, syn_cflags, tb_files, tb_cflags)`` from a kernel ``.cfg``, with absolute file paths.

    ``top`` is ``None`` if the ``.cfg`` has no ``syn.top``.
    """
    cfg_path = Path(cfg_path)
    syn_files, syn_cflags, tb_files, tb_cflags, top = [], {}, [], {}, None
    for line in cfg_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('[') or line.startswith('part=') or '=' not in line:
            continue
        key, _, val = line.partition('=')
        key, val = key.strip(), val.strip()
        if key == 'syn.top':
            top = val
        elif key == 'syn.file':
            syn_files.append(_resolve(cfg_path, val))
        elif key == 'tb.file':
            tb_files.append(_resolve(cfg_path, val))
        elif key == 'syn.file_cflags':
            path, _, flag = val.partition(',')
            syn_cflags.setdefault(_resolve(cfg_path, path), []).append(flag)
        elif key == 'tb.file_cflags':
            path, _, flag = val.partition(',')
            tb_cflags.setdefault(_resolve(cfg_path, path), []).append(flag)
    return top, syn_files, syn_cflags, tb_files, tb_cflags


def generate(cfg_path, hls_app_path, project_name):
    """Write an ``hls.app`` equivalent to the kernel ``.cfg``; raises ``RuntimeError`` if it has no ``syn.top``."""
    hls_app_path = Path(hls_app_path)
    top, syn_files, syn_cflags, tb_files, tb_cflags = parse_kernel_cfg(cfg_path)
    if top is None:
        raise RuntimeError(f'no syn.top in {cfg_path}')
    entries = []
    for f in syn_files:
        entries.append(
            f'        <file name={quoteattr(f)} sc="0" tb="false" '
            f'cflags={quoteattr(" ".join(syn_cflags.get(f, [])))} csimflags="" blackbox="false"/>'
        )
    for f in tb_files:
        entries.append(
            f'        <file name={quoteattr(f)} sc="0" tb="1" '
            f'cflags={quoteattr(" ".join(tb_cflags.get(f, [])))} csimflags="" blackbox="false"/>'
        )
    header = (
        '<AutoPilot:project xmlns:AutoPilot="com.autoesl.autopilot.project" '
        f'projectType="C/C++" name={quoteattr(project_name)} top={quoteattr(top)}>'
    )
    simflow = (
        '    <Simulation argv=""><SimFlow name="csim" setup="false" '
        'optimizeCompile="false" clean="false" ldflags="" mflags=""/></Simulation>'
    )
    body = '\n'.join(entries)
    hls_app_path.write_text(
        f'{header}\n'
        '    <files>\n'
        f'{body}\n'
        '    </files>\n'
        '    <solutions><solution name="solution1" status=""/></solutions>\n'
        f'{simflow}\n'
        '</AutoPilot:project>\n'
    )


def verify(cfg_path, hls_app_path):
    """Return ``(top_matches, files_match, n_files)`` for a generated ``hls.app`` and its ``.cfg``."""
    cfg_path, hls_app_path = Path(cfg_path), Path(hls_app_path)
    root = ET.parse(hls_app_path).getroot()
    app_top = root.attrib['top']
    app_files = sorted(f.attrib['name'] for f in root.iter('file'))
    cfg_top, cfg_files = None, []
    for line in cfg_path.read_text().splitlines():
        line = line.strip()
        if line.startswith('syn.top='):
            cfg_top = line.split('=', 1)[1]
        elif line.startswith(('syn.file=', 'tb.file=')):
            cfg_files.append(_resolve(cfg_path, line.split('=', 1)[1]))
    return app_top == cfg_top, app_files == sorted(cfg_files), len(app_files)
