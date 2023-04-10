import math
import sys

'''
A helper class for handling fixed point methods
Currently, very limited, allowing only:
    - Conversion to float
    - Exponents
    - Reciprocals
Used primarily for generating softmax look-up table
by using bit manipulation (see Vivado-equivalent implementation)
'''


class FixedPointEmulator:
    '''
    Default constructor
    Args:
        - N : Total number of bits in the fixed point number
        - I : Integer bits in the fixed point number
        - F = N-I : Fixed point bits in the number
        - signed : True/False - If True, use 2's complement when converting to float
        - self.integer_bits : Bits corresponding to the integer part of the number
        - self.decimal_bits : Bits corresponding to the decimal part of the number
    '''

    def __init__(self, N, I, signed=True, integer_bits=None, decimal_bits=None):  # noqa E741
        self.N = N
        self.I = I  # noqa E741
        self.F = N - I
        self.signed = signed
        self.integer_bits = [0] * self.I if integer_bits is None else integer_bits
        self.decimal_bits = [0] * self.F if decimal_bits is None else decimal_bits

    '''
    Converts the fixed point number stored in self.bits to a floating pont
    Args:
        - None
    Returns:
        - val : float, the floating point equivalent of the fixed point number
    Description:
        1. Check if the number is signed, and if so, set intermediate result to -2.0^(I-1) or 0.0
           otherwise, set intermediate result to +2.0^(I-1) or 0.0
        2. Traverse through integer bits, incrementing result by 2.0^(i) (using left shift)
        3. Traverse through decimal bits, incrementing result by 2.0^(-i) (using pow)
    Note:
        - This function uses left shifts instead of integer powers of 2.
    '''

    def to_float(self):
        val = float(int(self.integer_bits[0]) << (self.I - 1))
        val = -val if self.signed else val

        for i in range(self.I - 1, 0, -1):
            val += float(int(self.integer_bits[self.I - i]) << (i - 1))

        for i in range(0, self.F):
            if self.decimal_bits[i]:
                val += pow(2, -(i + 1))

        return val

    '''
    Sets the top bits of the current number
    Args:
        - bits : Values top bit should be set to
    '''

    def set_msb_bits(self, bits):
        for i in range(0, len(bits)):
            if i < self.I:
                self.integer_bits[i] = bits[i]
            elif i >= self.I and i < self.N:
                self.decimal_bits[i - self.I] = bits[i]

    '''
    Returns e^x, where x is the current fixed point number
    Args:
        - None
    Returns:
        - Float : e^x, rounded some number of decimal points
    Notice:
        - If e^x overflow, maximum value of float is used
    '''

    def exp_float(self, sig_figs=12):
        try:
            return round(math.exp(self.to_float()), sig_figs)
        except OverflowError:
            return round(sys.float_info.max, sig_figs)

    '''
    Returns 1/x, where x is the current fixed point number
    Args:
        - None
    Returns:
        - Float : 1/x, rounded some number of decimal points
    '''

    def inv_float(self, sig_figs=12):
        if self.to_float() != 0:
            return round(1.0 / self.to_float(), sig_figs)
        else:
            return round(sys.float_info.max, sig_figs)


'''
    Converts unsigned integer i to N-bit binary number
    Args:
        - i : Number to be converted
        - N : Number of bits to be used
    Note:
        - N > log2(i)+1
'''


def uint_to_binary(i, N):
    # Gets the binary representation of the number
    bits = [int(b) for b in list(f'{i:0b}')]

    # Zero padding, so exactly N bits are used
    while len(bits) < N:
        bits.insert(0, 0)

    return bits


'''
    Returns log2(i), rounding up
    Args:
        - i : Number
    Returns:
        - val : representing ceil(log2(i))
'''


def ceil_log2(i):
    return i.bit_length() - 1
