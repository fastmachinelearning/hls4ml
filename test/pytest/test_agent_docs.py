"""Checks that the agent documentation in .agents/ still matches the tree.

These documents cite files, classes and functions. The cheapest way for them to become misleading is for one
of those to be renamed, so this test extracts every repository path they mention and asserts it exists.

Paths are written in several styles: relative to the package directory (``backends/vivado/passes``), relative
to the repository root (``test/pytest``), or as a bare filename (``nnet_dense.h``). A token with a separator
has to match the tail of a real path; a bare filename has to match some file's name. Files belonging to a
*generated* project rather than to the repository are listed in GENERATED and skipped.
"""

import re
from pathlib import Path

import pytest

repo_root = Path(__file__).parent.parent.parent
docs_dir = repo_root / '.agents'

CHECKED_SUFFIXES = ('py', 'h', 'md', 'rst', 'tcl', 'yml', 'yaml', 'toml', 'in', 'cfg')

# Names that appear in the documents but belong to a generated hls4ml project, not to this repository.
GENERATED = {
    'myproject.cpp',
    'myproject.h',
    'myproject_bridge.cpp',
    'myproject_test.cpp',
    'parameters.h',
    'defines.h',
    'hls4ml_config.yml',
    'build_prj.tcl',
    'build_opt.tcl',
    'project.tcl',
    'vitis_hls.log',
}

path_like = re.compile(r'`([A-Za-z0-9_./*<>-]+\.(?:' + '|'.join(CHECKED_SUFFIXES) + r'))`')


def _repo_index():
    """All tracked-looking files in the repository, as a set of paths and a set of names."""
    paths, names = set(), set()
    for path in repo_root.rglob('*'):
        if not path.is_file() or '.git' in path.parts:
            continue
        relative = path.relative_to(repo_root)
        paths.add(relative.as_posix())
        names.add(path.name)
    return paths, names


REPO_PATHS, REPO_NAMES = _repo_index()
doc_files = sorted(docs_dir.glob('*.md'))


def _is_known(token: str) -> bool:
    if token in GENERATED or Path(token).name in GENERATED:
        return True
    if token.startswith('firmware/') or '<' in token or '*' in token:
        return True  # generated output, or a placeholder such as <backend>_backend.py
    if '/' in token:
        return any(path == token or path.endswith('/' + token) for path in REPO_PATHS)
    return token in REPO_NAMES


@pytest.mark.parametrize('doc', doc_files, ids=lambda p: p.name)
def test_referenced_paths_exist(doc):
    """Every repository file named in a document should still exist."""
    missing = sorted({token for token in path_like.findall(doc.read_text()) if not _is_known(token)})
    assert not missing, f'{doc.name} refers to paths that no longer exist: {missing}'


@pytest.mark.parametrize('doc', doc_files, ids=lambda p: p.name)
def test_front_matter_is_well_formed(doc):
    """Each document carries the metadata the adapter script needs."""
    if doc.name == 'README.md':
        pytest.skip('the index carries no front matter')
    match = re.match(r'^---\n(.*?)\n---\n', doc.read_text(), re.S)
    assert match, f'{doc.name} has no front matter block'
    front = match.group(1)
    assert re.search(r'^name:\s*\S+', front, re.M), f'{doc.name} front matter has no name'
    assert re.search(r'^description:\s*>-', front, re.M), f'{doc.name} front matter has no description'
    assert re.search(r'^globs:', front, re.M), f'{doc.name} front matter has no globs'


@pytest.mark.parametrize('doc', doc_files, ids=lambda p: p.name)
def test_relative_links_resolve(doc):
    """Links between the documents should point at files that exist."""
    broken = [
        target
        for target in re.findall(r'\]\(([^)#][^)]*)\)', doc.read_text())
        if not target.startswith(('http://', 'https://', 'mailto:')) and not (doc.parent / target).resolve().exists()
    ]
    assert not broken, f'{doc.name} has broken relative links: {sorted(set(broken))}'
