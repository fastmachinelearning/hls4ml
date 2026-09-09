import os

WRITE_TAR_ENV_VAR = 'HLS4ML_WRITE_TAR'


class Writer:
    def __init__(self):
        pass

    def should_write_tar(self, model):
        write_tar_config = model.config.get_writer_config().get('WriteTar', False)
        env_value = os.environ.get(WRITE_TAR_ENV_VAR, '')
        write_tar_env = env_value.strip().lower() in {'1', 'true'}
        return write_tar_config or write_tar_env

    def write_hls(self, model):
        raise NotImplementedError


writer_map = {}


def register_writer(name, writer_cls):
    if name in writer_map:
        raise Exception(f'Writer {name} already registered')

    writer_map[name] = writer_cls


def get_writer(name):
    return writer_map[name]()
