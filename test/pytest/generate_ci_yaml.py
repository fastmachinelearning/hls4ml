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
"""


n_test_files_per_yml = int(os.environ.get('N_TESTS_PER_YAML', 4))

# Blacklisted tests will be skipped
BLACKLIST = {'test_reduction'}

# Long-running tests will not be bundled with other tests
LONGLIST = {'test_hgq_layers', 'test_hgq_players', 'test_qkeras', 'test_pytorch_api'}
KERAS3_LIST = {'test_keras_v3_api', 'test_hgq2_mha', 'test_einsum_dense', 'test_qeinsum'}


def path_to_name(test_path):
    path = Path(test_path)
    name = path.stem.replace('test_', '')
    return name


def batched(iterable, chunk_size):
    iterator = iter(iterable)
    while chunk := tuple(itertools.islice(iterator, chunk_size)):
        yield chunk


def uses_example_model(test_filename):
    with open(test_filename) as f:
        content = f.read()
        return 'example-models' in content


def generate_test_yaml(test_root='.'):
    test_root = Path(test_root)
    test_paths = [path for path in test_root.glob('**/test_*.py') if path.stem not in (BLACKLIST | LONGLIST | KERAS3_LIST)]
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
        diff_yml = yaml.safe_load(template.format(name, '.pytest', test_files, batch_need_example_model))
        if yml is None:
            yml = diff_yml
        else:
            yml.update(diff_yml)

    test_paths = [path for path in test_root.glob('**/test_*.py') if path.stem in LONGLIST]
    for path in test_paths:
        name = path.stem.replace('test_', '')
        test_file = str(path.relative_to(test_root))
        needs_examples = uses_example_model(path)
        diff_yml = yaml.safe_load(template.format(name, '.pytest', test_file, int(needs_examples)))
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
        diff_yml = yaml.safe_load(template.format(name, 'pytest-keras3-only', test_files, batch_need_example_model))
        yml.update(diff_yml)

    return yml


if __name__ == '__main__':
    yml = generate_test_yaml(Path(__file__).parent)
    with open('pytests.yml', 'w') as yamlfile:
        yaml.safe_dump(yml, yamlfile)
