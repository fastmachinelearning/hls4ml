from ..precision import FixedPointPrecision
from .common import CodegenBackend


class VitisCodegenBackend(CodegenBackend):

    @staticmethod
    def type(precision: FixedPointPrecision):
        precision = precision.make_proper()
        k, b, I = precision.k, precision.b, precision.I  # noqa: E741
        return f'ap_{"" if k else "u"}fixed<{b},{I}>'

    def get_stream_type_name(self, name: str) -> str:
        return f'{name}::value_type'
