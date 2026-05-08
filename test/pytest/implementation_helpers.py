import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest

DATASET_SCHEMA_VERSION = 'implementation-dataset/v1'
DATASET_DIR_ENV = 'IMPLEMENTATION_DATASET_DIR'
DEFAULT_DATASET_DIR = Path(__file__).parent / 'implementation'

EXPECTED_REPORT_KEYS = {
    'VivadoAccelerator': {'CSynthesisReport'},
}

REQUIRED_METADATA_FIELDS = {
    'VivadoAccelerator': {'board', 'part'},
}

BITFILE_REQUIRED_BACKENDS = {'VivadoAccelerator'}


def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _project_root():
    return Path(os.getenv('CI_PROJECT_DIR', Path(__file__).parents[2]))


def _portable_path(path):
    return os.path.relpath(path, _project_root())


def _collect_files(output_dir, suffixes=None):
    output_path = Path(output_dir)
    files = []
    if not output_path.exists():
        return files

    for path in sorted(output_path.rglob('*')):
        if not path.is_file() or (suffixes is not None and path.suffix not in suffixes):
            continue
        entry = {
            'path': str(path.relative_to(output_path)),
            'size_bytes': path.stat().st_size,
        }
        files.append(entry)
    return files


def _dataset_dir():
    return Path(os.getenv(DATASET_DIR_ENV, str(DEFAULT_DATASET_DIR)))


def _artifact_path(filename):
    dataset_dir = _dataset_dir()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir / filename


def _write_json(data, filename):
    out_path = _artifact_path(filename)
    with open(out_path, 'w') as fp:
        json.dump(data, fp, indent=4, sort_keys=True)
    return out_path


@contextmanager
def _capture_terminal_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    with open(path, 'w') as fp:
        os.dup2(fp.fileno(), 1)
        os.dup2(fp.fileno(), 2)
        try:
            yield path
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def _validate_metadata(backend, metadata):
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise TypeError('Implementation metadata must be a dictionary.')

    missing = REQUIRED_METADATA_FIELDS.get(backend, set()) - metadata.keys()
    if missing:
        raise AssertionError(f'Missing implementation metadata for {backend}: {sorted(missing)}')
    return metadata


def _build_dataset(
    config,
    hls_model,
    test_case_id,
    backend,
    metadata,
    report,
    build_args,
    build_started_at,
    build_finished_at,
    build_duration_seconds,
    hls4ml_report_path,
    build_output_path,
):
    output_dir = hls_model.config.get_output_dir()

    return {
        'schema_version': DATASET_SCHEMA_VERSION,
        'test_id': test_case_id,
        'hls_config': {
            'backend': backend,
            'project_name': hls_model.config.get_project_name(),
            'build_args': build_args,
        },
        'metadata': metadata,
        'toolchain': {
            'version': config.get('tools_version', {}).get(backend, 'unknown'),
        },
        'ci': {
            'commit_sha': os.getenv('CI_COMMIT_SHA'),
            'commit_ref': os.getenv('CI_COMMIT_REF_NAME'),
            'commit_tag': os.getenv('CI_COMMIT_TAG'),
            'pipeline_id': os.getenv('CI_PIPELINE_ID'),
            'job_id': os.getenv('CI_JOB_ID'),
            'project_url': os.getenv('CI_PROJECT_URL'),
            'job_image': os.getenv('CI_JOB_IMAGE'),
            'runner_description': os.getenv('CI_RUNNER_DESCRIPTION'),
            'runner_tags': os.getenv('CI_RUNNER_TAGS'),
        },
        'build': {
            'status': 'success',
            'started_at_utc': build_started_at,
            'finished_at_utc': build_finished_at,
            'duration_seconds': build_duration_seconds,
        },
        'hls4ml_report': report,
        'artifacts': {
            'output_dir': _portable_path(output_dir),
            'hls4ml_report_json': _portable_path(hls4ml_report_path),
            'build_output_log': _portable_path(build_output_path),
            'bitstreams': _collect_files(output_dir, {'.bit'}),
            'reports': _collect_files(output_dir, {'.rpt', '.xml'}),
            'logs': _collect_files(output_dir, {'.log'}),
            'raw_output_files': _collect_files(output_dir),
        },
    }


def run_implementation_collection_test(config, hls_model, test_case_id, backend, metadata=None):
    """
    Build an implementation target and emit a dataset record plus raw backend reports.
    """
    metadata = _validate_metadata(backend, metadata)
    artifact_id = metadata.get('artifact_id', test_case_id)
    build_args = config.get('implementation_build_args', {}).get(backend, config.get('build_args', {}).get(backend, {}))
    build_output_path = _artifact_path(f'{artifact_id}_build.log')

    started_at = _utc_now()
    started = time.monotonic()
    try:
        with _capture_terminal_output(build_output_path):
            report = hls_model.build(**build_args)
    except Exception as e:
        pytest.fail(f'hls_model.build failed: {e}')
    finished_at = _utc_now()
    duration = round(time.monotonic() - started, 3)

    expected_keys = EXPECTED_REPORT_KEYS.get(backend, set())
    assert report and expected_keys.issubset(report.keys()), (
        f'Implementation failed: missing expected report keys: expected {expected_keys}, got {set(report.keys())}'
    )

    output_dir = hls_model.config.get_output_dir()
    bitfiles = _collect_files(output_dir, {'.bit'})
    if backend in BITFILE_REQUIRED_BACKENDS:
        assert bitfiles, f'Implementation failed: no bitstream was generated in {output_dir}'

    hls4ml_report_path = _write_json(report, f'{artifact_id}_hls4ml_report.json')
    dataset = _build_dataset(
        config=config,
        hls_model=hls_model,
        test_case_id=test_case_id,
        backend=backend,
        metadata=metadata,
        report=report,
        build_args=build_args,
        build_started_at=started_at,
        build_finished_at=finished_at,
        build_duration_seconds=duration,
        hls4ml_report_path=hls4ml_report_path,
        build_output_path=build_output_path,
    )
    _write_json(dataset, f'{artifact_id}_dataset.json')
