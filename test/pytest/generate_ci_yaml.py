import itertools
import os
import subprocess
from pathlib import Path

import yaml

'''
Create a Gitlab CI yml file with a separate entry for each test_* file
in the pytests directory to parallelise the CI jobs.
'''


template = """
pytest.{}:
  extends: .pytest
  variables:
    PYTESTFILE: {}
    EXAMPLEMODEL: {}
"""


n_test_files_per_yml = int(os.environ.get('N_TESTS_PER_YAML', 4))

# Blacklisted tests will be skipped
BLACKLIST = {'test_reduction'}

# Long-running tests will not be bundled with other tests
LONGLIST = {'test_hgq_layers', 'test_hgq_players', 'test_qkeras', 'test_pytorch_api'}

# Test files to split by individual test cases (stem only, no .py)
# Value = chunk size per CI job
SPLIT_BY_TEST_CASE = {
    'test_keras_api': 20,
}


def collect_test_cases(test_file):
    result = subprocess.run(['pytest', '--collect-only', '-q', str(test_file)], capture_output=True, text=True)

    lines = result.stdout.splitlines()
    test_ids = [line.strip().split('/')[-1] for line in lines if "::" in line]  # get only filename + nodeid
    return test_ids


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


def generate_test_yaml(test_root='.'):
    test_root = Path(test_root)
    test_paths = [
        path
        for path in test_root.glob('**/test_*.py')
        if path.stem not in (BLACKLIST | LONGLIST | set(SPLIT_BY_TEST_CASE.keys()))
    ]
    need_example_models = [uses_example_model(path) for path in test_paths]

    idxs = list(range(len(need_example_models)))
    idxs = sorted(idxs, key=lambda i: f'{need_example_models[i]}_{path_to_name(test_paths[i])}')

    yml = None
    for batch_idxs in batched(idxs, n_test_files_per_yml):
        batch_paths: list[Path] = [test_paths[i] for i in batch_idxs]
        names = [path_to_name(path) for path in batch_paths]
        name = '+'.join(names)
        test_files = ' '.join([str(path.relative_to(test_root)) for path in batch_paths])
        batch_need_example_model = int(any([need_example_models[i] for i in batch_idxs]))
        diff_yml = yaml.safe_load(template.format(name, test_files, batch_need_example_model))
        if yml is None:
            yml = diff_yml
        else:
            yml.update(diff_yml)

    test_paths = [path for path in test_root.glob('**/test_*.py') if path.stem in LONGLIST]
    for path in test_paths:
        name = path.stem.replace('test_', '')
        test_file = str(path.relative_to(test_root))
        needs_examples = uses_example_model(path)
        diff_yml = yaml.safe_load(template.format(name, test_file, int(needs_examples)))
        yml.update(diff_yml)

    # Handle split-by-test-case files
    test_paths = [path for path in test_root.glob('**/test_*.py') if path.stem in SPLIT_BY_TEST_CASE]
    for path in test_paths:
        stem = path.stem
        name_base = stem.replace('test_', '')
        test_file = str(path.relative_to(test_root))
        test_ids = collect_test_cases(path)
        chunk_size = SPLIT_BY_TEST_CASE[stem]
        needs_examples = uses_example_model(path)

        for i, batch in enumerate(batched(test_ids, chunk_size)):
            job_name = f"{name_base}_part{i}"
            batch_ids = " ".join(batch).strip().replace("\n", " ")  # flat single-line string
            job_key = f"pytest.{job_name}"
            job_entry = {
                job_key: {"extends": ".pytest", "variables": {"PYTESTFILE": batch_ids, "EXAMPLEMODEL": int(needs_examples)}}
            }
            if yml is None:
                yml = job_entry
            else:
                yml.update(job_entry)

    return yml


if __name__ == '__main__':
    yml = generate_test_yaml(Path(__file__).parent)
    with open('pytests.yml', 'w') as yamlfile:
        yaml.safe_dump(yml, yamlfile)
