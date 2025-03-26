import pytest
import os


def pytest_addoption(parser):
    parser.addoption(
        "--synthesis", action="store_true", default=False, help="Enable synthesis test"
    )

@pytest.fixture
def synthesis(request):
    """
    Fixture to determine if synthesis step should be run.
    Reads the '--synthesis' command-line argument passed to pytest.
    If the argument is provided, it will return True; otherwise, False.
    """
    return request.config.getoption("--synthesis")
