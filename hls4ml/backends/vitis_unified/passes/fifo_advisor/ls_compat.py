import importlib.machinery
import importlib.util
import sys

MODULE = 'lightningsim.trace_file'

_STOCK_IMPORT = 'from .model import BasicBlock, CDFGRegion, Instruction, Function, Solution'

_STOCK_PAYLOAD = """                            payload = event_instruction.operands[-1]
                            assert payload is not None
                            source_instruction = payload.source
                            assert isinstance(source_instruction, Instruction)
                            fifo_widths[entry.metadata.fifo.id] = (
                                source_instruction.bitwidth
                            )"""

_FIXED_PAYLOAD = """                            _operands = event_instruction.operands
                            payload = None
                            for _index in range(len(_operands) - 1, -1, -1):
                                _operand = _operands[_index]
                                if (
                                    _operand is not None
                                    and _operand.type == CDFGEdge.INPUT
                                    and isinstance(_operand.source, Instruction)
                                ):
                                    payload = _operand
                                    if _index == len(_operands) - 1:
                                        LS_COMPAT_STATS["last_operand_ok"] += 1
                                    else:
                                        LS_COMPAT_STATS["diverged"] += 1
                                    break
                            LS_COMPAT_STATS["resolutions"] += 1
                            if payload is not None:
                                source_instruction = payload.source
                                assert isinstance(source_instruction, Instruction)
                                fifo_widths[entry.metadata.fifo.id] = (
                                    source_instruction.bitwidth
                                )
                            else:
                                # Constant write: leave the width to a later write.
                                LS_COMPAT_STATS["constant_payload"] += 1"""

_STATS_DECL = """

# Inserted by hls4ml's fifo_advisor.ls_compat.
LS_COMPAT_STATS = {"resolutions": 0, "last_operand_ok": 0, "diverged": 0, "constant_payload": 0}
"""


def patch_source(source):
    """Return the patched source of ``trace_file.py``; raises ``RuntimeError`` if it is not the stock source.

    A ``fifo_write`` takes its width from the last data operand produced by an instruction, not ``operands[-1]``, and a
    constant write leaves the width to a later write to the same FIFO.
    """
    if source.count(_STOCK_PAYLOAD) != 1:
        raise RuntimeError(
            f'Unsupported LightningSim version: the code patched in {MODULE} was found '
            f'{source.count(_STOCK_PAYLOAD)} times instead of once.'
        )
    if source.count(_STOCK_IMPORT) != 1:
        raise RuntimeError(f'Unsupported LightningSim version: the expected import line was not found once in {MODULE}.')

    # The fixed code needs CDFGEdge, which trace_file does not import.
    source = source.replace(
        _STOCK_IMPORT,
        _STOCK_IMPORT + '\nfrom .model.cdfg_edge import CDFGEdge' + _STATS_DECL,
        1,
    )
    return source.replace(_STOCK_PAYLOAD, _FIXED_PAYLOAD, 1)


class _PatchingLoader(importlib.machinery.SourceFileLoader):
    def get_data(self, path):
        data = super().get_data(path)
        if path.endswith('trace_file.py'):
            data = patch_source(data.decode()).encode()
        return data

    def get_code(self, fullname):
        # Bypass the .pyc cache, which would skip get_data().
        path = self.get_filename(fullname)
        return self.source_to_code(self.get_data(path), path)


class _PatchingFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != MODULE:
            return None
        finders = [f for f in sys.meta_path if f is not self]
        for finder in finders:
            find_spec = getattr(finder, 'find_spec', None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is None or spec.origin is None:
                continue
            return importlib.util.spec_from_file_location(
                fullname, spec.origin, loader=_PatchingLoader(fullname, spec.origin)
            )
        return None


def install():
    """Install the import hook; raises ``RuntimeError`` if ``lightningsim.trace_file`` was imported unpatched."""
    if MODULE in sys.modules:
        if verify_installed():
            return
        raise RuntimeError(f'ls_compat.install() must be called before {MODULE} is imported.')
    if not any(isinstance(f, _PatchingFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchingFinder())


def stats():
    """Return the patch's counters, or an empty dict if the module is not loaded."""
    module = sys.modules.get(MODULE)
    if module is None:
        return {}
    return dict(getattr(module, 'LS_COMPAT_STATS', {}))


def verify_installed():
    """Return whether the loaded ``trace_file`` is the patched one."""
    module = sys.modules.get(MODULE)
    return module is not None and hasattr(module, 'LS_COMPAT_STATS')
