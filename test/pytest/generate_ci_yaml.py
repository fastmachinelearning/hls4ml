import yaml

template = """
pytest.test_cvmfs_mount:
  extends: .pytest
  variables:
    PYTESTFILE: test_keras_api.py
    EXAMPLEMODEL: 0
"""


def generate_fixed_test_yaml():
    yml = yaml.safe_load(template)
    return yml


if __name__ == '__main__':
    yml = generate_fixed_test_yaml()
    with open('pytests.yml', 'w') as yamlfile:
        yaml.safe_dump(yml, yamlfile, default_flow_style=False)
