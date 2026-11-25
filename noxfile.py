import os

import nox

nox.options.default_venv_backend = 'uv|virtualenv'
nox.options.sessions = ['lint']


@nox.session
def lint(session: nox.Session) -> None:
    """Run the pre-commit hooks on all files."""
    session.install('pre-commit')
    session.run('pre-commit', 'run', '--all-files', '--show-diff-on-failure', *session.posargs)


@nox.session
def tests(session: nox.Session) -> None:
    """Run pytest, e.g. ``nox -s tests -- test/pytest/test_keras_api.py``. Add extras with ``EXTRAS=testing-keras3``."""
    extras = ','.join(['testing', *filter(None, os.environ.get('EXTRAS', '').split(','))])
    session.install('-e', f'.[{extras}]')
    session.run('pytest', *session.posargs)


@nox.session
def docs(session: nox.Session) -> None:
    """Build the Sphinx documentation into docs/_build/html."""
    session.install('-r', 'docs/requirements.txt')
    session.run('make', '-C', 'docs', 'html', external=True)
