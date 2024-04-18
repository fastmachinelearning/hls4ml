import glob
import itertools
import os

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


def batched(iterable, chunk_size):
    iterator = iter(iterable)
    while chunk := tuple(itertools.islice(iterator, chunk_size)):
        yield chunk


def uses_example_model(test_filename):
    with open(test_filename) as f:
        content = f.read()
        return 'example-models' in content


yml = None
tests = glob.glob('test_*.py')
for test_batch in batched(tests, n_test_files_per_yml):
    name = '+'.join([test.replace('test_', '').replace('.py', '') for test in test_batch])
    test_files = ' '.join(list(test_batch))
    uses_example_models = int(any([uses_example_model(test) for test in test_batch]))

    new_yml = yaml.safe_load(template.format(name, test_files, uses_example_models))
    if yml is None:
        yml = new_yml
    else:
        yml.update(new_yml)

# hls4ml Optimization API
tests = glob.glob('test_optimization/test_*.py')
for test in tests:
    name = test.replace('test_optimization/', '').replace('test_', '').replace('.py', '')
    new_yml = yaml.safe_load(template.format(name, f'test_optimization/test_{name}.py', int(uses_example_model(test))))
    if yml is None:
        yml = new_yml
    else:
        yml.update(new_yml)

tests = glob.glob('test_optimization/test_keras/test_*.py')
for test in tests:
    # For now, skip Keras Surgeon [conflicting versions]
    if 'test_reduction' not in test:
        name = test.replace('test_optimization/test_keras/', '').replace('test_', '').replace('.py', '')
        new_yml = yaml.safe_load(
            template.format(name, f'test_optimization/test_keras/test_{name}.py', int(uses_example_model(test)))
        )
        if yml is None:
            yml = new_yml
        else:
            yml.update(new_yml)

yamlfile = open('pytests.yml', 'w')
yaml.safe_dump(yml, yamlfile)
