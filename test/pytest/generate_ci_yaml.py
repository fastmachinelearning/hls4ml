import glob

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


def uses_example_model(test_filename):
    with open(test_filename) as f:
        content = f.read()
        return 'example-models' in content


yml = None
tests = glob.glob('test_*.py')
for test in tests:
    name = test.replace('test_', '').replace('.py', '')
    new_yml = yaml.safe_load(template.format(name, f'test_{name}.py', int(uses_example_model(test))))
    if yml is None:
        yml = new_yml
    else:
        yml.update(new_yml)

yamlfile = open('pytests.yml', 'w')
yaml.safe_dump(yml, yamlfile)
