import sys

from hls4ml import converters, report, utils  # noqa: F401, E402

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = 'unknown version'
    version_tuple = (0, 0, 'unknown version')


def reseed(newseed):
    if 'tensorflow' in sys.modules:
        sys.modules['tensorflow'].random.set_seed(newseed)
    if 'torch' in sys.modules:
        sys.modules['torch'].manual_seed(newseed)
