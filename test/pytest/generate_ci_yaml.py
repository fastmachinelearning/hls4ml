import ast
import itertools
import os
from pathlib import Path

import yaml

"""
Create a Gitlab CI yml file with a separate entry for each test_* file
in the pytests directory to parallelise the CI jobs.
"""

template = """
pytest.{}:
  extends: {}
  variables:
    PYTESTFILE: {}
    EXAMPLEMODEL: {}
    VIVADO_VERSION: "2020.1"
    VITIS_VERSION: "2024.1"
    RUN_SYNTHESIS: "true"
"""

template_keras3_backend = (
    template
    + """
    KERAS_BACKEND: {}
"""
)

n_test_files_per_yml = int(os.environ.get('N_TESTS_PER_YAML', 4))

# Blacklisted tests will be skipped
BLACKLIST = {'test_reduction'}

# Long-running tests will not be bundled with other tests
LONGLIST = {'test_hgq_layers', 'test_hgq_players', 'test_qkeras', 'test_pytorch_api'}
KERAS3_LIST = {
    'test_keras_v3_api',
    'test_hgq2_mha',
    'test_einsum_dense',
    'test_qeinsum',
    'test_multiout_onnx',
    'test_keras_v3_profiling',
    'test_sparsepixels',
}
QKERAS3_LIST = {'test_qkerasV3'}
KERAS3_BACKEND_SPECIFIC_LIST = {
    'test_pquant_keras': 'tensorflow',
    'test_pquant_pytorch': 'torch',
}

# Test files to split by individual test cases
# Value = chunk size per CI job
SPLIT_BY_TEST_CASE = {
    'test_keras_api': 1,
}

# Files that parametrize both Bambu and non-Bambu backends. They run twice:
# once in the standard image with the Bambu cases excluded, and once in the
# Bambu image with only the Bambu cases selected.
BAMBU_SHARED_TESTS = {
    'test_activations',
    'test_auto_precision',
    'test_build_bambu',
    'test_dense_unrolled',
    'test_keras_api',
    'test_multi_dense',
    'test_pooling',
    'test_softmax',
}

# Bambu tests not expressed through backend parametrization, so they have to be
# named individually.
BAMBU_ONLY_TESTS = {'test_report.py::test_bambu_report'}

# Whole files that only make sense in the Bambu image. They need Bambu installed
# but do not parametrize a ``backend``, so the filter args cannot select them.
# They are kept out of the standard matrix and get their own unfiltered job in
# the Bambu image.
BAMBU_ONLY_FILES = {'test_build_bambu_accelerator'}

BAMBU_FILTER_ARG = '--backend-filter=Bambu'
BAMBU_EXCLUDE_ARG = '--backend-exclude=Bambu'


def collect_test_functions_from_ast(test_file):
    """Collect all test function names using AST parsing (no imports)."""
    with open(test_file, encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=str(test_file))

    test_funcs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test'):
            test_funcs.append(f'{test_file}::{node.name}')
    return test_funcs


def batched(iterable, batch_size):
    it = iter(iterable)
    while batch := list(itertools.islice(it, batch_size)):
        yield batch


def path_to_name(test_path):
    path = Path(test_path)
    name = path.stem.replace('test_', '')
    return name


def uses_example_model(test_filename):
    with open(test_filename) as f:
        content = f.read()
        return 'example-models' in content


def standard_extra_args(batch_paths):
    """Pytest args that keep the Bambu cases out of the standard image."""
    args = []
    if any(path.stem in BAMBU_SHARED_TESTS for path in batch_paths):
        args.append(BAMBU_EXCLUDE_ARG)
    args += [
        f'--ci-exclude-nodeid={nodeid}'
        for nodeid in sorted(BAMBU_ONLY_TESTS)
        if any(nodeid.startswith(f'{path.name}::') for path in batch_paths)
    ]
    return args


def generate_test_yaml(test_root='.'):
    test_root = Path(test_root)
    test_paths = [
        path
        for path in test_root.glob('**/test_*.py')
        if path.stem
        not in (
            BLACKLIST
            | LONGLIST
            | BAMBU_ONLY_FILES
            | set(SPLIT_BY_TEST_CASE.keys())
            | KERAS3_LIST
            | set(KERAS3_BACKEND_SPECIFIC_LIST.keys() | QKERAS3_LIST)
        )
    ]
    need_example_models = [uses_example_model(path) for path in test_paths]

    idxs = list(range(len(need_example_models)))
    idxs = sorted(idxs, key=lambda i: f'{need_example_models[i]}_{path_to_name(test_paths[i])}')

    yml = None
    for batch_idxs in batched(idxs, n_test_files_per_yml):
        batch_paths: list[Path] = [test_paths[i] for i in batch_idxs]
        names = [path_to_name(path) for path in batch_paths]
        name = '+'.join(names)
        test_files = ' '.join([str(path.relative_to(test_root)) for path in batch_paths] + standard_extra_args(batch_paths))
        batch_need_example_model = int(any([need_example_models[i] for i in batch_idxs]))
        diff_yml = yaml.safe_load(template.format(name, '.pytest', test_files, batch_need_example_model))
        if yml is None:
            yml = diff_yml
        else:
            yml.update(diff_yml)

    test_paths = [path for path in test_root.glob('**/test_*.py') if path.stem in LONGLIST]
    for path in test_paths:
        name = path.stem.replace('test_', '')
        test_file = ' '.join([str(path.relative_to(test_root))] + standard_extra_args([path]))
        needs_examples = uses_example_model(path)
        diff_yml = yaml.safe_load(template.format(name, '.pytest', test_file, int(needs_examples)))
        yml.update(diff_yml)

    test_paths = [path for path in test_root.glob('**/test_*.py') if path.stem in SPLIT_BY_TEST_CASE]
    for path in test_paths:
        stem = path.stem
        name_base = stem.replace('test_', '')
        test_file = str(path.relative_to(test_root))
        test_ids = collect_test_functions_from_ast(test_file)
        chunk_size = SPLIT_BY_TEST_CASE[stem]
        needs_examples = uses_example_model(path)

        for i, batch in enumerate(batched(test_ids, chunk_size)):
            job_name = f'{name_base}_part{i}'
            test_file_args = ' '.join(list(batch) + standard_extra_args([path])).strip().replace('\n', ' ')
            diff_yml = yaml.safe_load(template.format(job_name, '.pytest', test_file_args, int(needs_examples)))
            if yml is None:
                yml = diff_yml
            else:
                yml.update(diff_yml)

    keras3_paths = [path for path in test_root.glob('**/test_*.py') if path.stem in KERAS3_LIST]
    keras3_need_examples = [uses_example_model(path) for path in keras3_paths]

    k3_idxs = list(range(len(keras3_need_examples)))
    k3_idxs = sorted(k3_idxs, key=lambda i: f'{keras3_need_examples[i]}_{path_to_name(keras3_paths[i])}')

    for batch_idxs in batched(k3_idxs, n_test_files_per_yml):
        batch_paths: list[Path] = [keras3_paths[i] for i in batch_idxs]
        names = [path_to_name(path) for path in batch_paths]
        name = 'keras3-' + '+'.join(names)
        test_files = ' '.join([str(path.relative_to(test_root)) for path in batch_paths])
        batch_need_example_model = int(any([keras3_need_examples[i] for i in batch_idxs]))
        diff_yml = yaml.safe_load(template.format(name, '.pytest-keras3-only', test_files, batch_need_example_model))
        yml.update(diff_yml)

    qkeras3_paths = [path for path in test_root.glob('**/test_*.py') if path.stem in QKERAS3_LIST]
    qkeras3_need_examples = [uses_example_model(path) for path in qkeras3_paths]

    qk3_idxs = list(range(len(qkeras3_need_examples)))
    qk3_idxs = sorted(qk3_idxs, key=lambda i: f'{qkeras3_need_examples[i]}_{path_to_name(qkeras3_paths[i])}')

    for batch_idxs in batched(qk3_idxs, n_test_files_per_yml):
        batch_paths: list[Path] = [qkeras3_paths[i] for i in batch_idxs]
        names = [path_to_name(path) for path in batch_paths]
        name = 'qkerasV3'
        test_files = ' '.join([str(path.relative_to(test_root)) for path in batch_paths])
        batch_need_example_model = int(any([qkeras3_need_examples[i] for i in batch_idxs]))
        diff_yml = yaml.safe_load(template.format(name, '.pytest-qkeras-v3-only', test_files, batch_need_example_model))
        yml.update(diff_yml)

    backend_specific_paths = [path for path in test_root.glob('**/test_*.py') if path.stem in KERAS3_BACKEND_SPECIFIC_LIST]
    for path in backend_specific_paths:
        name = path.stem.replace('test_', '')
        test_file = str(path.relative_to(test_root))
        needs_examples = uses_example_model(path)
        backend = KERAS3_BACKEND_SPECIFIC_LIST[path.stem]
        diff_yml = yaml.safe_load(
            template_keras3_backend.format(name, '.pytest-keras3-only', test_file, int(needs_examples), backend)
        )
        yml.update(diff_yml)

    # Bambu jobs: the same test files, Bambu cases only, in the Bambu image.
    bambu_paths = sorted(
        (path for path in test_root.glob('**/test_*.py') if path.stem in BAMBU_SHARED_TESTS), key=path_to_name
    )
    for batch_paths in batched(bambu_paths, n_test_files_per_yml):
        name = 'bambu-' + '+'.join(path_to_name(path) for path in batch_paths)
        test_files = ' '.join([str(path.relative_to(test_root)) for path in batch_paths] + [BAMBU_FILTER_ARG])
        batch_need_example_model = int(any(uses_example_model(path) for path in batch_paths))
        yml.update(yaml.safe_load(template.format(name, '.pytest-bambu', test_files, batch_need_example_model)))

    for nodeid in sorted(BAMBU_ONLY_TESTS):
        name = 'bambu-' + nodeid.split('::', 1)[1].replace('test_', '')
        yml.update(yaml.safe_load(template.format(name, '.pytest-bambu', nodeid, 0)))

    # Bambu-only files run whole and unfiltered: they have no backend parameter
    # for BAMBU_FILTER_ARG to match.
    bambu_only_paths = sorted(
        (path for path in test_root.glob('**/test_*.py') if path.stem in BAMBU_ONLY_FILES), key=path_to_name
    )
    for path in bambu_only_paths:
        name = 'bambu-' + path_to_name(path)
        test_file = str(path.relative_to(test_root))
        yml.update(yaml.safe_load(template.format(name, '.pytest-bambu', test_file, int(uses_example_model(path)))))

    return yml


if __name__ == '__main__':
    yml = generate_test_yaml(Path(__file__).parent)
    with open('pytests.yml', 'w') as yamlfile:
        yaml.safe_dump(yml, yamlfile)
