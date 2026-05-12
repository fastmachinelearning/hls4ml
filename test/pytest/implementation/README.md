# Implementation CI Suite

This directory contains manually listed implementation tests. These jobs run full backend implementation flows and collect dataset artifacts with actual tool reports and generated project files.

The normal pytest CI should stay separate from this suite. Implementation tests are intended for manually triggered dataset collection, not for the regular fast test matrix.

## Pipeline Layout

Implementation CI uses:

- `test/pytest/implementation/pytests.yml`: static job list
- `test/pytest/implementation/ci-template.yml`: implementation-specific GitLab templates
- `test/pytest/implementation/implementation_helpers.py`: dataset artifact helpers used by implementation tests

Jobs in `pytests.yml` extend backend-specific runtime templates from `ci-template.yml`, for example:

- `.pytest-implementation-vivadoaccelerator`
- `.pytest-implementation-vivado-runtime`

Each concrete job sets `PYTESTFILE` to one specific test function, so each model/backend run becomes a separate CI job and artifact set.

## Dataset Artifacts

Implementation tests should use `run_implementation_collection_test()` from `implementation_helpers.py`. The helper:

- runs `hls_model.build(...)` with backend-specific implementation build args
- captures full terminal output to `*_build.log`
- writes the hls4ml report returned by the backend to `*_hls4ml_report.json`
- writes one compact dataset record to `*_dataset.json`
- compresses the generated project directory to `*_project.zip`
- records the project archive path, size, and SHA256 in the dataset record

The dataset record includes:

- hls4ml source commit and repository URL
- example-models source commit and model file names
- backend, project name, board/part metadata, and build args
- backend-specific toolchain versions
- CI metadata for finding the run later
- build timing and the hls4ml report
- pointers to the log, hls4ml report JSON, bitstream files, and project archive

`IMPLEMENTATION_DATASET_DIR` controls where artifacts are written. In CI it is set to:

```bash
test/pytest/implementation
```

## Test Policy

Implementation tests should load models from the `example-models` submodule instead of defining models inline. This keeps dataset records tied to a model name, model file, and exact `example-models` commit.

Keep test code focused on:

- loading the example model
- converting it with the backend under test
- passing clear metadata to `run_implementation_collection_test()`

Avoid duplicating dataset/report parsing in individual tests. Add backend-specific dataset behavior in `implementation_helpers.py` when needed.

## Adding A Test

1. Add or update a `test_*.py` file in this directory.
2. Load a model from `example-models`.
3. Add model metadata including name, source, source commit, and model file paths.
4. Add a static job in `pytests.yml` with `PYTESTFILE` pointing to the specific test function.
5. Set tool versions and backend runtime template through the job/template variables.
