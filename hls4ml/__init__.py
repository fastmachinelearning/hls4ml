# Temporary workaround for QKeras installation requirement, will be removed after 1.0.0
def maybe_install_qkeras():
    import subprocess
    import sys

    QKERAS_PKG_NAME = 'QKeras'
    # QKERAS_PKG_SOURCE = QKERAS_PKG_NAME
    QKERAS_PKG_SOURCE = 'qkeras@git+https://github.com/fastmachinelearning/qkeras.git'

    def pip_list():
        p = subprocess.run([sys.executable, '-m', 'pip', 'list'], check=True, capture_output=True)
        return p.stdout.decode()

    def pip_install(package):
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

    all_pkgs = pip_list()
    if QKERAS_PKG_NAME not in all_pkgs:
        print('QKeras installation not found, installing one...')
        pip_install(QKERAS_PKG_SOURCE)
        print('QKeras installed.')


try:
    maybe_install_qkeras()
except Exception:
    print('Could not find QKeras installation, make sure you have QKeras installed.')

# End of workaround

from hls4ml import converters, report, utils  # noqa: F401, E402

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")


def reseed(newseed):
    print(f'\npytest-randomly: reseed with {newseed}')
    try:
        import tensorflow

        tensorflow.random.set_seed(newseed)
    except ImportError:
        print('\nTensorFlow seed not set')
    try:
        import torch

        torch.manual_seed(newseed)
    except ImportError:
        print('\nPyTorch seed not set')
