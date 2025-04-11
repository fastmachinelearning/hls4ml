import yaml

'''
Create a Gitlab CI yml file with a separate entry for each test_* file
in the pytests directory to parallelise the CI jobs.
'''


template = """
pytest.test_synthesis:
  extends: .pytest
  variables:
    PYTESTFILE: test_keras_api_temp.py
    EXAMPLEMODEL: 0
"""


def generate_fixed_test_yaml():
    yml = yaml.safe_load(template)
    return yml


if __name__ == '__main__':
    yml = generate_fixed_test_yaml()
    with open('pytests.yml', 'w') as yamlfile:
        yaml.safe_dump(yml, yamlfile, default_flow_style=False)
