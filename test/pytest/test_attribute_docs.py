"""Checks that docs/ir/attributes.rst matches what docs/attr_doc_gen.py generates from the layer attributes."""

import importlib.util
import os
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from hls4ml.backends.plugin_loader import ENTRY_POINT_GROUP, ENV_PLUGIN_MODULES

repo_root = Path(__file__).parent.parent.parent
docs_dir = repo_root / 'docs'


def test_attribute_docs_are_up_to_date(tmp_path, monkeypatch):
    if not docs_dir.is_dir():
        pytest.skip('the documentation is not part of the source distribution')
    if entry_points().select(group=ENTRY_POINT_GROUP) or os.environ.get(ENV_PLUGIN_MODULES):
        pytest.skip('backend plugins are installed, and the generated file would list them')

    spec = importlib.util.spec_from_file_location('attr_doc_gen', docs_dir / 'attr_doc_gen.py')
    attr_doc_gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attr_doc_gen)

    monkeypatch.chdir(tmp_path)  # the script writes attributes.rst into the current directory
    attr_doc_gen.write_all_attributes(attr_doc_gen.convert_to_attr_list())

    generated = (tmp_path / 'attributes.rst').read_text()
    committed = (docs_dir / 'ir' / 'attributes.rst').read_text()
    assert generated == committed, 'docs/ir/attributes.rst is out of date; regenerate it as CONTRIBUTING.md describes.'
