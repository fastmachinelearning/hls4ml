import json
import os
import re
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model

import hls4ml
from hls4ml.backends.vitis_unified.passes.fifo_advisor import dedup, hls_app, ls_compat, sweep

test_root_path = Path(__file__).parent

ADVISOR_FLOW = 'vitisunified:fifo_depth_optimization_advisor'

backend_options = ['VitisUnified']


def make_cfg(tmp_path):
    cfg = tmp_path / 'hls_kernel_config.cfg'
    cfg.write_text(
        '\n'.join(
            [
                'part=xczu9eg-ffvb1156-2-e',
                '[hls]',
                'syn.top=myproject_axi_master',
                f'syn.file={tmp_path}/firmware/myproject.cpp',
                f'syn.file_cflags={tmp_path}/firmware/myproject.cpp,-std=c++14',
                f'syn.file={tmp_path}/firmware/myproject_axi_master.cpp',
                f'tb.file={tmp_path}/myproject_test.cpp',
                f'tb.file_cflags={tmp_path}/myproject_test.cpp,-std=c++14',
                f'tb.file={tmp_path}/firmware/weights',
                'syn.cflags=-I.',
            ]
        )
    )
    return cfg


def test_hls_app_roundtrip(tmp_path):
    cfg = make_cfg(tmp_path)
    app = tmp_path / 'hls.app'
    hls_app.generate(cfg, app, 'myproject')
    ok_top, ok_files, n_files = hls_app.verify(cfg, app)
    assert ok_top and ok_files and n_files == 4


def test_hls_app_accepts_str_paths(tmp_path):
    cfg = make_cfg(tmp_path)
    app = tmp_path / 'hls.app'
    hls_app.generate(str(cfg), str(app), 'myproject')
    assert hls_app.verify(str(cfg), str(app))[:2] == (True, True)


def test_hls_app_resolves_relative_paths(tmp_path):
    cfg = tmp_path / 'vitis_workspace' / 'myproject' / 'hls_kernel_config_csim.cfg'
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        '\n'.join(
            [
                '[hls]',
                'syn.top=myproject_axi_master',
                'syn.file=../../firmware/myproject.cpp',
                'syn.file_cflags=../../firmware/myproject.cpp,-std=c++0x',
                'tb.file=../../tb_data',
            ]
        )
    )
    app = tmp_path / 'hls.app'
    hls_app.generate(cfg, app, 'myproject')
    assert hls_app.verify(cfg, app)[:2] == (True, True)
    files = {f.attrib['name']: f.attrib['cflags'] for f in ET.parse(app).getroot().iter('file')}
    assert files == {str(tmp_path / 'firmware' / 'myproject.cpp'): '-std=c++0x', str(tmp_path / 'tb_data'): ''}


def test_hls_app_requires_top(tmp_path):
    cfg = tmp_path / 'empty.cfg'
    cfg.write_text('[hls]\n')
    with pytest.raises(RuntimeError, match='no syn.top'):
        hls_app.generate(cfg, tmp_path / 'hls.app', 'myproject')


class FakeFifo:
    def __init__(self, fifo_id, name):
        self.id = fifo_id
        self.name = name
        self.width = 16


class FakeEnv:
    def __init__(self, names, invocations):
        self.fifos = [FakeFifo(i, name) for i, name in enumerate(names * invocations)]
        self.num_fifos = len(self.fifos)


def test_dedup_keeps_first_instance_of_each_name():
    env = FakeEnv(['a', 'b', 'c'], invocations=5)
    env, stats = dedup.dedup_env(env, verify_compiled=False)
    assert stats == {'total': 15, 'distinct': 3, 'invocations': 5}
    assert [f.id for f in env.fifos] == [0, 1, 2]
    assert env.num_fifos == 3


def test_dedup_refuses_nonuniform_multiplicity():
    env = FakeEnv(['a', 'b'], invocations=1)
    env.fifos.append(FakeFifo(99, 'a'))
    env.num_fifos = 3
    with pytest.raises(dedup.DedupError, match='do not repeat uniformly'):
        dedup.dedup_env(env, verify_compiled=False)


def test_dedup_refuses_empty():
    with pytest.raises(dedup.DedupError, match='no FIFOs'):
        dedup.dedup_env(FakeEnv([], invocations=1), verify_compiled=False)


def test_restrict_env_limits_search_space():
    env = FakeEnv(['layer2_out', 'layer3_out', 'batch_size_c'], invocations=1)
    env, n_kept = dedup.restrict_env(env, {'layer2_out', 'layer3_out'})
    assert n_kept == 2
    assert {f.name for f in env.fifos} == {'layer2_out', 'layer3_out'}


def test_restrict_env_refuses_no_overlap():
    env = FakeEnv(['a'], invocations=1)
    with pytest.raises(dedup.DedupError, match='matches a channel'):
        dedup.restrict_env(env, {'zzz'})


class FakeResult:
    def __init__(self, fifo_sizes):
        self.fifo_sizes = fifo_sizes


class CostingEnv:
    def __init__(self):
        self.costed = []

    def eval_solution_parallel(self, points):
        self.costed.append(points)
        return [FakeResult(p) for p in points]


def test_unique_points_drops_repeats():
    env = CostingEnv()
    evals = [FakeResult({0: 2, 1: 4}), FakeResult({0: 2, 1: 4}), FakeResult({0: 4, 1: 4})]
    kept = sweep.unique_points(env, evals)
    assert [r.fifo_sizes for r in kept] == [{0: 2, 1: 4}, {0: 4, 1: 4}]
    assert env.costed == []


def test_unique_points_costs_only_searched_fifos():
    env = CostingEnv()
    evals = [FakeResult({0: 2, 1: 2, 7: 2}), FakeResult({0: 2, 1: 2, 7: 128}), FakeResult({0: 4, 1: 2, 7: 2})]
    kept = sweep.unique_points(env, evals, searched={0, 1})
    assert env.costed == [[{0: 2, 1: 2}, {0: 4, 1: 2}]]
    assert [r.fifo_sizes for r in kept] == [{0: 2, 1: 2}, {0: 4, 1: 2}]


STOCK_SHAPED_SOURCE = (
    'from .model import BasicBlock, CDFGRegion, Instruction, Function, Solution\n'
    'def resolve_trace():\n'
    '    def do_sync_work_batch():\n'
    '        if True:\n'
    '            if True:\n'
    '                if True:\n'
    '                    if True:\n'
    '                        if True:\n'
    '                            payload = event_instruction.operands[-1]\n'
    '                            assert payload is not None\n'
    '                            source_instruction = payload.source\n'
    '                            assert isinstance(source_instruction, Instruction)\n'
    '                            fifo_widths[entry.metadata.fifo.id] = (\n'
    '                                source_instruction.bitwidth\n'
    '                            )\n'
)


def test_ls_compat_patch_applies_to_stock_shaped_source():
    patched = ls_compat.patch_source(STOCK_SHAPED_SOURCE)
    assert 'from .model.cdfg_edge import CDFGEdge' in patched
    assert 'LS_COMPAT_STATS' in patched
    assert 'payload = event_instruction.operands[-1]\n' not in patched


def test_ls_compat_refuses_changed_source():
    with pytest.raises(RuntimeError, match='Unsupported LightningSim version'):
        ls_compat.patch_source(STOCK_SHAPED_SOURCE.replace('operands[-1]', 'operands[0]'))
    with pytest.raises(RuntimeError, match='Unsupported LightningSim version'):
        ls_compat.patch_source(STOCK_SHAPED_SOURCE + STOCK_SHAPED_SOURCE)


def test_ls_compat_install_accepts_patched_module(monkeypatch):
    patched = types.ModuleType(ls_compat.MODULE)
    patched.LS_COMPAT_STATS = {}
    monkeypatch.setitem(sys.modules, ls_compat.MODULE, patched)
    ls_compat.install()


def test_ls_compat_install_refuses_unpatched_module(monkeypatch):
    monkeypatch.setitem(sys.modules, ls_compat.MODULE, types.ModuleType(ls_compat.MODULE))
    with pytest.raises(RuntimeError, match='must be called before'):
        ls_compat.install()


def test_advisor_pass_registered_alongside_cosim_pass():
    hls4ml.backends.get_backend('VitisUnified')
    from hls4ml.model.flow import get_flow
    from hls4ml.model.optimizer import get_optimizer

    opt = get_optimizer(ADVISOR_FLOW)
    assert type(opt).__name__ == 'FifoDepthOptimizationAdvisor'
    assert opt.search_scope == 'compute'

    flow = get_flow(ADVISOR_FLOW)
    assert flow.optimizers[0] == ADVISOR_FLOW
    assert get_flow('vitisunified:fifo_depth_optimization').name


def test_advisor_pass_rejects_io_parallel():
    hls4ml.backends.get_backend('VitisUnified')
    from hls4ml.model.optimizer import get_optimizer

    class StubConfig:
        def get_config_value(self, key):
            return 'io_parallel'

    class StubModel:
        config = StubConfig()

    with pytest.raises(RuntimeError, match=re.escape('`io_stream`')):
        get_optimizer(ADVISOR_FLOW).transform(StubModel())


def test_advisor_pass_rejects_axi_stream():
    hls4ml.backends.get_backend('VitisUnified')
    from hls4ml.model.optimizer import get_optimizer

    class StubConfig:
        def get_config_value(self, key):
            return {'IOType': 'io_stream', 'VitisUnifiedConfig': {'axi_mode': 'axi_stream'}}[key]

    class StubModel:
        config = StubConfig()

    with pytest.raises(RuntimeError, match="axi_mode='axi_master'"):
        get_optimizer(ADVISOR_FLOW).transform(StubModel())


def test_advisor_pass_rejects_unknown_scope():
    hls4ml.backends.get_backend('VitisUnified')
    from hls4ml.model.optimizer import get_optimizer

    class StubConfig:
        def get_config_value(self, key):
            return {'IOType': 'io_stream', 'VitisUnifiedConfig': {'axi_mode': 'axi_master'}}[key]

    class StubModel:
        config = StubConfig()

    opt = get_optimizer(ADVISOR_FLOW)
    opt.configure(search_scope='everything')
    try:
        with pytest.raises(ValueError, match='search_scope'):
            opt.transform(StubModel())
    finally:
        opt.configure(search_scope='compute')


def test_advisor_pass_validates_before_synthesis():
    hls4ml.backends.get_backend('VitisUnified')
    from hls4ml.model.optimizer import get_optimizer

    class StubConfig:
        def get_config_value(self, key):
            return {'IOType': 'io_stream', 'VitisUnifiedConfig': {'axi_mode': 'axi_master'}}[key]

    class StubModel:
        config = StubConfig()

    opt = get_optimizer(ADVISOR_FLOW)
    try:
        opt.configure(selection='fastest')
        with pytest.raises(ValueError, match='selection'):
            opt.transform(StubModel())
        opt.configure(selection='min_bram', solvers=('heuristic', 'genetic'))
        with pytest.raises(ValueError, match='unknown solver'):
            opt.transform(StubModel())
    finally:
        opt.configure(selection='min_bram', solvers=('heuristic', 'sa', 'group-sa', 'random', 'group-random'))


def test_advisor_pass_reports_missing_analysis_packages(monkeypatch):
    import importlib.util

    import hls4ml.backends.vitis_unified.passes.fifo_depth_optimization_advisor as advisor_mod

    hls4ml.backends.get_backend('VitisUnified')
    from hls4ml.model.optimizer import get_optimizer

    class StubConfig:
        def get_config_value(self, key):
            return {'IOType': 'io_stream', 'VitisUnifiedConfig': {'axi_mode': 'axi_master'}}[key]

    class StubModel:
        config = StubConfig()

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in ('lightningsim', 'fifo_advisor'):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(advisor_mod.importlib.util, 'find_spec', fake_find_spec)

    opt = get_optimizer(ADVISOR_FLOW)
    with pytest.raises(RuntimeError, match='lightningsim and fifo_advisor'):
        opt.transform(StubModel())


def test_advisor_pass_refuses_skip_synthesis_without_solution(tmp_path, monkeypatch):
    import importlib.util

    import hls4ml.backends.vitis_unified.passes.fifo_depth_optimization_advisor as advisor_mod

    hls4ml.backends.get_backend('VitisUnified')
    from hls4ml.model.optimizer import get_optimizer

    class StubWriter:
        def get_vitis_hls_exec_dir(self, model):
            return str(tmp_path / 'vitis_unified_project')

    class StubBackend:
        writer = StubWriter()

    class StubConfig:
        backend = StubBackend()

        def get_config_value(self, key):
            return {'IOType': 'io_stream', 'VitisUnifiedConfig': {'axi_mode': 'axi_master'}}[key]

        def get_output_dir(self):
            return str(tmp_path)

        def get_project_name(self):
            return 'myproject'

    class StubModel:
        config = StubConfig()

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in ('lightningsim', 'fifo_advisor'):
            return object()
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(advisor_mod.importlib.util, 'find_spec', fake_find_spec)

    opt = get_optimizer(ADVISOR_FLOW)
    opt.configure(skip_synthesis=True)
    try:
        with pytest.raises(RuntimeError, match='has not been synthesised'):
            opt.transform(StubModel())
    finally:
        opt.configure(skip_synthesis=False)


def build_tiny_cnn():
    x_in = Input(shape=(8, 8, 1), name='in')
    x = Conv2D(2, (3, 3), name='conv')(x_in)
    x = Activation('relu', name='act')(x)
    x = MaxPooling2D((2, 2), name='pool')(x)
    x = Flatten(name='flat')(x)
    x = Dense(3, name='dense')(x)
    return Model(inputs=x_in, outputs=x, name='tiny_cnn')


def build_skip_cnn():
    from tensorflow.keras.layers import Add

    x_in = Input(shape=(8, 8, 1), name='in')
    x = Conv2D(4, (3, 3), padding='same', name='conv1')(x_in)
    x = Activation('relu', name='act1')(x)
    skip = x
    x = Conv2D(4, (3, 3), padding='same', name='conv2')(x)
    x = Add(name='skip_add')([skip, x])
    x = Activation('relu', name='act2')(x)
    x = Flatten(name='flat')(x)
    x = Dense(3, name='dense')(x)
    return Model(inputs=x_in, outputs=x, name='skip_cnn')


def run_fifo_advisor_optimization_keras(
    model, output_dir, backend='VitisUnified', selection='min_bram', search_scope='compute'
):
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Flows'] = [ADVISOR_FLOW]
    hls4ml.model.optimizer.get_optimizer(ADVISOR_FLOW).configure(selection=selection, search_scope=search_scope)

    return hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        project_name='myproject',
        backend=backend,
        board='zcu102',
        part='xczu9eg-ffvb1156-2-e',
        clock_period=5,
        io_type='io_stream',
        axi_mode='axi_master',
    )


def fifo_advisor_optimization_checks(hls_model):
    output_dir = hls_model.config.get_output_dir()

    with open(output_dir + '/fifo_depths.json') as fifo_depths_file:
        fifo_depths = json.load(fifo_depths_file)
    assert fifo_depths, 'the pass applied no depths'
    assert all(fifo['optimized'] <= fifo['initial'] for fifo in fifo_depths.values())
    assert any(fifo['optimized'] < fifo['initial'] for fifo in fifo_depths.values())

    with open(output_dir + '/fifo_depths_advisor.json') as report_file:
        report = json.load(report_file)
    assert report['pareto_front'], 'no non-deadlocking assignment was found'
    assert report['winner']['latency'] is not None

    hls_model.build(reset=False, csim=False, synth=True, cosim=True, bitfile=False)

    exec_dir = hls_model.config.backend.writer.get_vitis_hls_exec_dir(hls_model)
    cosim_report = Path(exec_dir) / 'reports' / 'hls_cosim.rpt'
    assert os.path.isfile(cosim_report), 'co-simulation report not found'
    assert any('Pass' in line for line in cosim_report.read_text().splitlines())


@pytest.mark.skip(reason='Skipping synthesis tests for now')
@pytest.mark.parametrize('backend', backend_options)
def test_successful_execution_of_tiny_cnn(test_case_id, backend):
    hls_model = run_fifo_advisor_optimization_keras(build_tiny_cnn(), str(test_root_path / test_case_id), backend=backend)
    fifo_advisor_optimization_checks(hls_model)


@pytest.mark.skip(reason='Skipping synthesis tests for now')
@pytest.mark.parametrize('backend', backend_options)
def test_successful_execution_of_skip_cnn(test_case_id, backend):
    hls_model = run_fifo_advisor_optimization_keras(build_skip_cnn(), str(test_root_path / test_case_id), backend=backend)
    fifo_advisor_optimization_checks(hls_model)


@pytest.mark.skip(reason='Skipping synthesis tests for now')
@pytest.mark.parametrize('backend', backend_options)
def test_runtime_error_on_io_parallel(test_case_id, backend):
    message = 'To use this optimization you have to set `IOType` field to `io_stream` in the HLS config.'
    model = build_tiny_cnn()
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Flows'] = [ADVISOR_FLOW]
    with pytest.raises(RuntimeError, match=re.escape(message)):
        hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=config,
            output_dir=str(test_root_path / test_case_id),
            project_name='myproject',
            backend=backend,
            part='xczu9eg-ffvb1156-2-e',
            clock_period=5,
            io_type='io_parallel',
            axi_mode='axi_master',
        )


@pytest.mark.skip(reason='Skipping synthesis tests for now')
@pytest.mark.parametrize('backend', backend_options)
def test_predictions_match_after_optimization(test_case_id, backend):
    model = build_tiny_cnn()
    x = np.random.rand(5, 8, 8, 1).astype(np.float32)
    keras_prediction = model.predict(x)

    hls_model = run_fifo_advisor_optimization_keras(model, str(test_root_path / test_case_id), backend=backend)
    hls_model.compile()
    hls_prediction = hls_model.predict(np.ascontiguousarray(x)).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.05)
