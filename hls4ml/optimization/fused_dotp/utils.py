from .precision import FixedPointPrecision


def precision_from_const(x):
    if x > 0:
        low, high = 0, x
    elif x < 0:
        low, high = x, 0
    else:
        low, high = 0, 0.5
    return FixedPointPrecision(low, high, 2).make_proper()


class Singleton(type):
    "Singleton metaclass"
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
