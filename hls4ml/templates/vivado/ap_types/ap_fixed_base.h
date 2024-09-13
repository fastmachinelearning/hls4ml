/*
 * Copyright 2011-2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __AP_FIXED_BASE_H__
#define __AP_FIXED_BASE_H__

#ifndef __AP_FIXED_H__
#error "Only ap_fixed.h and ap_int.h can be included directly in user code."
#endif

// for ap_int_base and its reference types.
#include <ap_int.h>
#ifndef __SYNTHESIS__
#if _AP_ENABLE_HALF_ == 1
// for half type
#include <hls_half.h>
#endif
// for std io
#include <iostream>
#endif

#ifndef __cplusplus
#error "C++ is required to include this header file"
#else // __cplusplus

// for warning on unsupported rounding mode in conversion to float/double.
#if !defined(__SYNTHESIS__) && __cplusplus >= 201103L && \
    (defined(__gnu_linux__) || defined(_WIN32))
#define AP_FIXED_ENABLE_CPP_FENV 1
#include <cfenv>
#endif

// ----------------------------------------------------------------------

/* Major TODO
  long double support: constructor, assign and other operators.
  binary operators with ap_fixed_base and const char*.
  return ap_fixed/ap_ufixed when result signedness is known.
*/

// Helper function in conversion to floating point types.

#ifdef __SYNTHESIS__
#define _AP_ctype_op_get_bit(var, index) _AP_ROOT_op_get_bit(var, index)
#define _AP_ctype_op_set_bit(var, index, x) _AP_ROOT_op_set_bit(var, index, x)
#define _AP_ctype_op_get_range(var, low, high) \
  _AP_ROOT_op_get_range(var, low, high)
#define _AP_ctype_op_set_range(var, low, high, x) \
  _AP_ROOT_op_set_range(var, low, high, x)
#else // ifdef __SYNTHESIS__
template <typename _Tp1, typename _Tp2>
inline bool _AP_ctype_op_get_bit(_Tp1& var, const _Tp2& index) {
  return !!(var & (1ull << (index)));
}
template <typename _Tp1, typename _Tp2, typename _Tp3>
inline _Tp1 _AP_ctype_op_set_bit(_Tp1& var, const _Tp2& index, const _Tp3& x) {
  var |= (((x) ? 1ull : 0ull) << (index));
  return var;
}
template <typename _Tp1, typename _Tp2, typename _Tp3>
inline _Tp1 _AP_ctype_op_get_range(_Tp1& var, const _Tp2& low,
                                   const _Tp3& high) {
  _Tp1 r = var;
  ap_ulong mask = -1ll;
  mask >>= (sizeof(_Tp1) * 8 - ((high) - (low) + 1));
  r >>= (low);
  r &= mask;
  return r;
}
template <typename _Tp1, typename _Tp2, typename _Tp3, typename _Tp4>
inline _Tp1 _AP_ctype_op_set_range(_Tp1& var, const _Tp2& low, const _Tp3& high,
                                   const _Tp4& x) {
  ap_ulong mask = -1ll;
  mask >>= (_AP_SIZE_ap_slong - ((high) - (low) + 1));
  var &= ~(mask << (low));
  var |= ((mask & x) << (low));
  return var;
}
#endif // ifdef __SYNTHESIS__


// trait for letting base class to return derived class.
// Notice that derived class template is incomplete, and we cannot use
// the member of the derived class.
template <int _AP_W2, int _AP_I2, bool _AP_S2>
struct _ap_fixed_factory;
template <int _AP_W2, int _AP_I2>
struct _ap_fixed_factory<_AP_W2, _AP_I2, true> {
  typedef ap_fixed<_AP_W2, _AP_I2> type;
};
template <int _AP_W2, int _AP_I2>
struct _ap_fixed_factory<_AP_W2, _AP_I2, false> {
  typedef ap_ufixed<_AP_W2, _AP_I2> type;
};

/// ap_fixed_base: AutoPilot fixed point.
/** partial specialization of signed.
  @tparam _AP_W width.
  @tparam _AP_I integral part width.
  @tparam _AP_S signed.
  @tparam _AP_Q quantization mode. Default is AP_TRN.
  @tparam _AP_O saturation mode. Default is AP_WRAP.
  @tparam _AP_N saturation wrap value. Default is 0.
 */
// default for _AP_Q, _AP_O and _AP_N set in ap_decl.h
template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O,
          int _AP_N>
struct ap_fixed_base : _AP_ROOT_TYPE<_AP_W, _AP_S> {
 public:
  typedef _AP_ROOT_TYPE<_AP_W, _AP_S> Base;
  static const int width = _AP_W;
  static const int iwidth = _AP_I;
  static const ap_q_mode qmode = _AP_Q;
  static const ap_o_mode omode = _AP_O;

  /// Return type trait.
  template <int _AP_W2, int _AP_I2, bool _AP_S2>
  struct RType {
    enum {
      _AP_F = _AP_W - _AP_I,
      F2 = _AP_W2 - _AP_I2,
      mult_w = _AP_W + _AP_W2,
      mult_i = _AP_I + _AP_I2,
      mult_s = _AP_S || _AP_S2,
      plus_w = AP_MAX(_AP_I + (_AP_S2 && !_AP_S), _AP_I2 + (_AP_S && !_AP_S2)) +
               1 + AP_MAX(_AP_F, F2),
      plus_i =
          AP_MAX(_AP_I + (_AP_S2 && !_AP_S), _AP_I2 + (_AP_S && !_AP_S2)) + 1,
      plus_s = _AP_S || _AP_S2,
      minus_w =
          AP_MAX(_AP_I + (_AP_S2 && !_AP_S), _AP_I2 + (_AP_S && !_AP_S2)) + 1 +
          AP_MAX(_AP_F, F2),
      minus_i =
          AP_MAX(_AP_I + (_AP_S2 && !_AP_S), _AP_I2 + (_AP_S && !_AP_S2)) + 1,
      minus_s = true,
#ifndef __SC_COMPATIBLE__
      div_w = _AP_S2 + _AP_W + AP_MAX(F2, 0),
#else
      div_w = _AP_S2 + _AP_W + AP_MAX(F2, 0) + AP_MAX(_AP_I2, 0),
#endif
      div_i = _AP_S2 + _AP_I + F2,
      div_s = _AP_S || _AP_S2,
      logic_w =
          AP_MAX(_AP_I + (_AP_S2 && !_AP_S), _AP_I2 + (_AP_S && !_AP_S2)) +
          AP_MAX(_AP_F, F2),
      logic_i = AP_MAX(_AP_I + (_AP_S2 && !_AP_S), _AP_I2 + (_AP_S && !_AP_S2)),
      logic_s = _AP_S || _AP_S2
    };

    typedef ap_fixed_base<_AP_W, _AP_I, _AP_S> lhs;
    typedef ap_fixed_base<_AP_W2, _AP_I2, _AP_S2> rhs;

    typedef ap_fixed_base<mult_w, mult_i, mult_s> mult_base;
    typedef ap_fixed_base<plus_w, plus_i, plus_s> plus_base;
    typedef ap_fixed_base<minus_w, minus_i, minus_s> minus_base;
    typedef ap_fixed_base<logic_w, logic_i, logic_s> logic_base;
    typedef ap_fixed_base<div_w, div_i, div_s> div_base;
    typedef ap_fixed_base<_AP_W, _AP_I, _AP_S> arg1_base;

    typedef typename _ap_fixed_factory<mult_w, mult_i, mult_s>::type mult;
    typedef typename _ap_fixed_factory<plus_w, plus_i, plus_s>::type plus;
    typedef typename _ap_fixed_factory<minus_w, minus_i, minus_s>::type minus;
    typedef typename _ap_fixed_factory<logic_w, logic_i, logic_s>::type logic;
    typedef typename _ap_fixed_factory<div_w, div_i, div_s>::type div;
    typedef typename _ap_fixed_factory<_AP_W, _AP_I, _AP_S>::type arg1;
  };

 private:
#ifndef __SYNTHESIS__
  // This cannot handle hex float format string.
  void fromString(const std::string& val, unsigned char radix) {
    _AP_ERROR(!(radix == 2 || radix == 8 || radix == 10 || radix == 16),
              "ap_fixed_base::fromString(%s, %d)", val.c_str(), radix);

    Base::V = 0;
    int startPos = 0;
    int endPos = val.length();
    int decPos = val.find(".");
    if (decPos == -1) decPos = endPos;

    // handle sign
    bool isNegative = false;
    if (val[0] == '-') {
      isNegative = true;
      ++startPos;
    } else if (val[0] == '+')
      ++startPos;

    // If there are no integer bits, e.g.:
    // .0000XXXX, then keep at least one bit.
    // If the width is greater than the number of integer bits, e.g.:
    // XXXX.XXXX, then we keep the integer bits
    // if the number of integer bits is greater than the width, e.g.:
    // XXX000 then we keep the integer bits.
    // Always keep one bit.
    ap_fixed_base<AP_MAX(_AP_I, 4) + 4, AP_MAX(_AP_I, 4) + 4, false>
        integer_bits = 0;

    // Figure out if we can shift instead of multiply
    unsigned shift = (radix == 16 ? 4 : radix == 8 ? 3 : radix == 2 ? 1 : 0);

    //std::cout << "\n\n" << val << "\n";
    //std::cout << startPos << " " << decPos << " " << endPos << "\n";

    bool sticky_int = false;

    // Traverse the integer digits from the MSD, multiplying by radix as we go.
    for (int i = startPos; i < decPos; i++) {
      // Get a digit
      char cdigit = val[i];
      if (cdigit == '\0') continue;
      unsigned digit = ap_private_ops::decode_digit(cdigit, radix);

      sticky_int |= integer_bits[AP_MAX(_AP_I, 4) + 4 - 1] |
                    integer_bits[AP_MAX(_AP_I, 4) + 4 - 2] |
                    integer_bits[AP_MAX(_AP_I, 4) + 4 - 3] |
                    integer_bits[AP_MAX(_AP_I, 4) + 4 - 4];
      // Shift or multiply the value by the radix
      if (shift)
        integer_bits <<= shift;
      else
        integer_bits *= radix;

      // Add in the digit we just interpreted
      integer_bits += digit;
      //std::cout << "idigit = " << digit << " " << integer_bits.to_string()
      //    << "  " << sticky_int <<  "\n";
    }
    integer_bits[AP_MAX(_AP_I, 4) + 4 - 3] =
        integer_bits[AP_MAX(_AP_I, 4) + 4 - 3] | sticky_int;

    ap_fixed_base<AP_MAX(_AP_W - _AP_I, 0) + 4 + 4, 4, false> fractional_bits = 0;
    bool sticky = false;

    // Traverse the fractional digits from the LSD, dividing by radix as we go.
    for (int i = endPos - 1; i >= decPos + 1; i--) {
      // Get a digit
      char cdigit = val[i];
      if (cdigit == '\0') continue;
      unsigned digit = ap_private_ops::decode_digit(cdigit, radix);
      // Add in the digit we just interpreted
      fractional_bits += digit;

      sticky |= fractional_bits[0] | fractional_bits[1] | fractional_bits[2] |
                fractional_bits[3];
      // Shift or divide the value by the radix
      if (shift)
        fractional_bits >>= shift;
      else
        fractional_bits /= radix;

      //std::cout << "fdigit = " << digit << " " << fractional_bits.to_string()
      //    << " " << sticky << "\n";
    }

    //std::cout << "Int =" << integer_bits.to_string() << " " <<
    //    fractional_bits.to_string() << "\n";

    fractional_bits[0] = fractional_bits[0] | sticky;

    if (isNegative)
      *this = -(integer_bits + fractional_bits);
    else
      *this = integer_bits + fractional_bits;

    //std::cout << "end = " << this->to_string(16) << "\n";
  }

  /// report invalid constrction of ap_fixed_base
  INLINE void report() {
    if (!_AP_S && _AP_O == AP_WRAP_SM) {
      fprintf(stderr, "ap_ufxied<...> cannot support AP_WRAP_SM.\n");
      exit(1);
    }
    if (_AP_W > MAX_MODE(AP_INT_MAX_W) * 1024) {
      fprintf(stderr,
              "[E] ap_%sfixed<%d, ...>: Bitwidth exceeds the "
              "default max value %d. Please use macro "
              "AP_INT_MAX_W to set a larger max value.\n",
              _AP_S ? "" : "u", _AP_W, MAX_MODE(AP_INT_MAX_W) * 1024);
      exit(1);
    }
  }
#else
  INLINE void report() {}
#endif // ifdef __SYNTHESIS__

  /// @name helper functions.
  //  @{
  INLINE void overflow_adjust(bool underflow, bool overflow, bool lD,
                              bool sign) {
    if (!underflow && !overflow) return;
    if (_AP_O == AP_WRAP) {
      if (_AP_N == 0) return;
      if (_AP_S) {
        // signed AP_WRAP
        // n_bits == 1
        Base::V = _AP_ROOT_op_set_bit(Base::V, _AP_W - 1, sign);
        if (_AP_N > 1) {
          // n_bits > 1
          ap_int_base<_AP_W, false> mask(-1);
          if (sign) mask.V = 0;
          Base::V =
              _AP_ROOT_op_set_range(Base::V, _AP_W - _AP_N, _AP_W - 2, mask.V);
        }
      } else {
        // unsigned AP_WRAP
        ap_int_base<_AP_W, false> mask(-1);
        Base::V =
            _AP_ROOT_op_set_range(Base::V, _AP_W - _AP_N, _AP_W - 1, mask.V);
      }
    } else if (_AP_O == AP_SAT_ZERO) {
      Base::V = 0;
    } else if (_AP_O == AP_WRAP_SM && _AP_S) {
      bool Ro = _AP_ROOT_op_get_bit(Base::V, _AP_W - 1);
      if (_AP_N == 0) {
        if (lD != Ro) {
          Base::V = ~Base::V;
          Base::V = _AP_ROOT_op_set_bit(Base::V, _AP_W - 1, lD);
        }
      } else {
        if (_AP_N == 1 && sign != Ro) {
          Base::V = ~Base::V;
        } else if (_AP_N > 1) {
          bool lNo = _AP_ROOT_op_get_bit(Base::V, _AP_W - _AP_N);
          if (lNo == sign) Base::V = ~Base::V;
          ap_int_base<_AP_W, false> mask(-1);
          if (sign) mask.V = 0;
          Base::V =
              _AP_ROOT_op_set_range(Base::V, _AP_W - _AP_N, _AP_W - 2, mask.V);
        }
        Base::V = _AP_ROOT_op_set_bit(Base::V, _AP_W - 1, sign);
      }
    } else {
      if (_AP_S) {
        if (overflow) {
          Base::V = 1;
          Base::V <<= _AP_W - 1;
          Base::V = ~Base::V;
        } else if (underflow) {
          Base::V = 1;
          Base::V <<= _AP_W - 1;
          if (_AP_O == AP_SAT_SYM) Base::V |= 1;
        }
      } else {
        if (overflow)
          Base::V = ~(ap_int_base<_AP_W, false>(0).V);
        else if (underflow)
          Base::V = 0;
      }
    }
  }

  INLINE bool quantization_adjust(bool qb, bool r, bool s) {
    bool carry = (bool)_AP_ROOT_op_get_bit(Base::V, _AP_W - 1);
    if (_AP_Q == AP_TRN) return false;
    if (_AP_Q == AP_RND_ZERO)
      qb &= s || r;
    else if (_AP_Q == AP_RND_MIN_INF)
      qb &= r;
    else if (_AP_Q == AP_RND_INF)
      qb &= !s || r;
    else if (_AP_Q == AP_RND_CONV)
      qb &= _AP_ROOT_op_get_bit(Base::V, 0) || r;
    else if (_AP_Q == AP_TRN_ZERO)
      qb = s && (qb || r);
    Base::V += qb;
    return carry && (!(bool)_AP_ROOT_op_get_bit(Base::V, _AP_W - 1));
  }
  //  @}

 public:
  /// @name constructors.
  //  @{
  /// default ctor.
  INLINE ap_fixed_base() {}

  /// copy ctor.
  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_fixed_base(
      const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op) {
    operator=(op);
    report();
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_fixed_base(
      const volatile ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op) {
    operator=(op);
    report();
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_fixed_base(const ap_int_base<_AP_W2, _AP_S2>& op) {
    ap_fixed_base<_AP_W2, _AP_W2, _AP_S2> tmp;
    tmp.V = op.V;
    operator=(tmp);
    report();
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_fixed_base(const volatile ap_int_base<_AP_W2, _AP_S2>& op) {
    ap_fixed_base<_AP_W2, _AP_W2, _AP_S2> tmp;
    tmp.V = op.V;
    operator=(tmp);
    report();
  }

#ifndef __SYNTHESIS__
#ifndef NON_C99STRING
  INLINE ap_fixed_base(const char* s, signed char rd = 0) {
    unsigned char radix = rd;
    std::string str = ap_private_ops::parseString(s, radix); // will guess rd, default 10
    _AP_ERROR(radix == 0, "ap_fixed_base(const char* \"%s\", %d), str=%s, radix = %d",
              s, rd, str.c_str(), radix); // TODO remove this check
    fromString(str, radix);
  }
#else
  INLINE ap_fixed_base(const char* s, signed char rd = 10) {
    ap_int_base<_AP_W, _AP_S> t(s, rd);
    Base::V = t.V;
  }
#endif // ifndef NON_C99STRING
#else // ifndef __SYNTHESIS__
  // XXX _ssdm_string2bits only takes const string and const radix.
  // It seems XFORM will do compile time processing of the string.
  INLINE ap_fixed_base(const char* s) {
    typeof(Base::V) t;
    _ssdm_string2bits((void*)(&t), (const char*)(s), 10, _AP_I, _AP_S, _AP_Q,
                      _AP_O, _AP_N, _AP_C99);
    Base::V = t;
  }
  INLINE ap_fixed_base(const char* s, signed char rd) {
    typeof(Base::V) t;
    _ssdm_string2bits((void*)(&t), (const char*)(s), rd, _AP_I, _AP_S, _AP_Q,
                      _AP_O, _AP_N, _AP_C99);
    Base::V = t;
  }
#endif // ifndef __SYNTHESIS__ else

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_fixed_base(const ap_bit_ref<_AP_W2, _AP_S2>& op) {
    *this = ((bool)op);
    report();
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_fixed_base(const ap_range_ref<_AP_W2, _AP_S2>& op) {
    *this = (ap_int_base<_AP_W2, false>(op));
    report();
  }

  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
  INLINE ap_fixed_base(
      const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>& op) {
    *this = (ap_int_base<_AP_W2 + _AP_W3, false>(op));
    report();
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_fixed_base(
      const af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op) {
    *this = (bool(op));
    report();
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_fixed_base(
      const af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op) {
    *this = (ap_int_base<_AP_W2, false>(op));
    report();
  }

  // ctors from c types.
  // make a temp ap_fixed_base first, and use ap_fixed_base.operator=
#define CTOR_FROM_INT(C_TYPE, _AP_W2, _AP_S2)        \
  INLINE ap_fixed_base(const C_TYPE x) {             \
    ap_fixed_base<(_AP_W2), (_AP_W2), (_AP_S2)> tmp; \
    tmp.V = x;                                       \
    *this = tmp;                                     \
  }

  CTOR_FROM_INT(bool, 1, false)
  CTOR_FROM_INT(char, 8, CHAR_IS_SIGNED)
  CTOR_FROM_INT(signed char, 8, true)
  CTOR_FROM_INT(unsigned char, 8, false)
  CTOR_FROM_INT(short, _AP_SIZE_short, true)
  CTOR_FROM_INT(unsigned short, _AP_SIZE_short, false)
  CTOR_FROM_INT(int, _AP_SIZE_int, true)
  CTOR_FROM_INT(unsigned int, _AP_SIZE_int, false)
  CTOR_FROM_INT(long, _AP_SIZE_long, true)
  CTOR_FROM_INT(unsigned long, _AP_SIZE_long, false)
  CTOR_FROM_INT(ap_slong, _AP_SIZE_ap_slong, true)
  CTOR_FROM_INT(ap_ulong, _AP_SIZE_ap_slong, false)
#undef CTOR_FROM_INT
/*
 * TODO:
 *Theere used to be several funtions which were AP_WEAK.
 *Now they're all INLINE expect ap_fixed_base(double d)
 *Maybe we can use '#pragma HLS inline' instead of INLINE.
 */
  AP_WEAK ap_fixed_base(double d) {
    ap_int_base<64, false> ireg;
    ireg.V = doubleToRawBits(d);
    bool isneg = _AP_ROOT_op_get_bit(ireg.V, 63);

    ap_int_base<DOUBLE_EXP + 1, true> exp;
    ap_int_base<DOUBLE_EXP, false> exp_tmp;
    exp_tmp.V =
        _AP_ROOT_op_get_range(ireg.V, DOUBLE_MAN, DOUBLE_MAN + DOUBLE_EXP - 1);
    exp = exp_tmp - DOUBLE_BIAS;
    ap_int_base<DOUBLE_MAN + 2, true> man;
    man.V = _AP_ROOT_op_get_range(ireg.V, 0, DOUBLE_MAN - 1);
    // do not support NaN
    _AP_WARNING(exp == APFX_IEEE_DOUBLE_E_MAX + 1 && man.V != 0,
                "assign NaN to fixed point value");
    man.V = _AP_ROOT_op_set_bit(man.V, DOUBLE_MAN, 1);
    if (isneg) man = -man;
    if ((ireg.V & 0x7fffffffffffffffLL) == 0) {
      Base::V = 0;
    } else {
      int _AP_W2 = DOUBLE_MAN + 2, _AP_I2 = exp.V + 2, _AP_F = _AP_W - _AP_I,
          F2 = _AP_W2 - _AP_I2;
      bool _AP_S2 = true,
           QUAN_INC = F2 > _AP_F &&
                      !(_AP_Q == AP_TRN || (_AP_Q == AP_TRN_ZERO && !_AP_S2));
      bool carry = false;
      // handle quantization
      unsigned sh_amt = (F2 > _AP_F) ? F2 - _AP_F : _AP_F - F2;
      if (F2 == _AP_F)
        Base::V = man.V;
      else if (F2 > _AP_F) {
        if (sh_amt < DOUBLE_MAN + 2)
          Base::V = man.V >> sh_amt;
        else {
          Base::V = isneg ? -1 : 0;
        }
        if ((_AP_Q != AP_TRN) && !((_AP_Q == AP_TRN_ZERO) && !_AP_S2)) {
          bool qb = (F2 - _AP_F > _AP_W2) ? isneg : (bool)_AP_ROOT_op_get_bit(
                                                        man.V, F2 - _AP_F - 1);
          bool r =
              (F2 > _AP_F + 1)
                  ? _AP_ROOT_op_get_range(man.V, 0, (F2 - _AP_F - 2 < _AP_W2)
                                                        ? (F2 - _AP_F - 2)
                                                        : (_AP_W2 - 1)) != 0
                  : false;
          carry = quantization_adjust(qb, r, isneg);
        }
      } else { // no quantization
        Base::V = man.V;
        if (sh_amt < _AP_W)
          Base::V = Base::V << sh_amt;
        else
          Base::V = 0;
      }
      // handle overflow/underflow
      if ((_AP_O != AP_WRAP || _AP_N != 0) &&
          ((!_AP_S && _AP_S2) ||
           _AP_I - _AP_S <
               _AP_I2 - _AP_S2 +
                   (QUAN_INC ||
                    (_AP_S2 && (_AP_O == AP_SAT_SYM))))) { // saturation
        bool deleted_zeros = _AP_S2 ? true : !carry, deleted_ones = true;
        bool neg_src = isneg;
        bool lD = false;
        int pos1 = F2 - _AP_F + _AP_W;
        int pos2 = F2 - _AP_F + _AP_W + 1;
        bool newsignbit = _AP_ROOT_op_get_bit(Base::V, _AP_W - 1);
        if (pos1 < _AP_W2 && pos1 >= 0)
          // lD = _AP_ROOT_op_get_bit(man.V, pos1);
          lD = (man.V >> pos1) & 1;
        if (pos1 < _AP_W2) {
          bool Range1_all_ones = true;
          bool Range1_all_zeros = true;
          bool Range2_all_ones = true;
          ap_int_base<DOUBLE_MAN + 2, false> Range2;
          ap_int_base<DOUBLE_MAN + 2, false> all_ones(-1);

          if (pos2 >= 0 && pos2 < _AP_W2) {
            // Range2.V = _AP_ROOT_op_get_range(man.V,
            //                        pos2, _AP_W2 - 1);
            Range2.V = man.V;
            Range2.V >>= pos2;
            Range2_all_ones = Range2 == (all_ones >> pos2);
          } else if (pos2 < 0)
            Range2_all_ones = false;
          if (pos1 >= 0 && pos2 < _AP_W2) {
            Range1_all_ones = Range2_all_ones && lD;
            Range1_all_zeros = !Range2.V && !lD;
          } else if (pos2 == _AP_W2) {
            Range1_all_ones = lD;
            Range1_all_zeros = !lD;
          } else if (pos1 < 0) {
            Range1_all_zeros = !man.V;
            Range1_all_ones = false;
          }

          deleted_zeros =
              deleted_zeros && (carry ? Range1_all_ones : Range1_all_zeros);
          deleted_ones =
              carry ? Range2_all_ones && (pos1 < 0 || !lD) : Range1_all_ones;
          neg_src = isneg && !(carry && Range1_all_ones);
        } else
          neg_src = isneg && newsignbit;
        bool neg_trg = _AP_S && newsignbit;
        bool overflow = (neg_trg || !deleted_zeros) && !isneg;
        bool underflow = (!neg_trg || !deleted_ones) && neg_src;
        if ((_AP_O == AP_SAT_SYM) && _AP_S2 && _AP_S)
          underflow |=
              neg_src &&
              (_AP_W > 1 ? _AP_ROOT_op_get_range(Base::V, 0, _AP_W - 2) == 0
                         : true);
        overflow_adjust(underflow, overflow, lD, neg_src);
      }
    }
    report();
  }

  // TODO more optimized implementation.
  INLINE ap_fixed_base(float d) { *this = ap_fixed_base(double(d)); }

#if _AP_ENABLE_HALF_ == 1
  // TODO more optimized implementation.
  INLINE ap_fixed_base(half d) { *this = ap_fixed_base(double(d)); }
#endif
  //  @}

  /// @name assign operator
  /// assign, using another ap_fixed_base of same template parameters.
  /*
  INLINE ap_fixed_base& operator=(
      const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op) {
    Base::V = op.V;
    return *this;
  }
  */

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_fixed_base& operator=(
      const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op) {

    const int _AP_F = _AP_W - _AP_I;
    const int F2 = _AP_W2 - _AP_I2;
    const int QUAN_INC =
          F2 > _AP_F && !(_AP_Q == AP_TRN || (_AP_Q == AP_TRN_ZERO && !_AP_S2));

    if (!op) Base::V = 0;
    bool carry = false;
    bool signbit = _AP_ROOT_op_get_bit(op.V, _AP_W2 - 1);
    bool isneg = signbit && _AP_S2;
    if (F2 == _AP_F)
      Base::V = op.V;
    else if (F2 > _AP_F) {
      unsigned int sh_amt = F2 - _AP_F;
      //  moves bits right, handle quantization.
      if (sh_amt < _AP_W2) {
        Base::V = op.V >> sh_amt;
      } else {
        Base::V = isneg ? -1 : 0;
      }
      if (_AP_Q != AP_TRN && !(_AP_Q == AP_TRN_ZERO && !_AP_S2)) {
        bool qbit = _AP_ROOT_op_get_bit(op.V, F2 - _AP_F - 1);
        // bit after LSB.
        bool qb = (F2 - _AP_F > _AP_W2) ? _AP_S2 && signbit : qbit;
        enum { hi = ((F2 - _AP_F - 2) < _AP_W2) ? (F2 - _AP_F - 2) : (_AP_W2 - 1) };
        // bits after qb.
        bool r = (F2 > _AP_F + 1) ? (_AP_ROOT_op_get_range(op.V, 0, hi) != 0) : false;
        carry = quantization_adjust(qb, r, isneg);
      }
    } else {
      unsigned  sh_amt = _AP_F - F2;
      // moves bits left, no quantization
      if (sh_amt < _AP_W) {
        if (_AP_W > _AP_W2) {
          // extend and then shift, avoid losing bits.
          Base::V = op.V;
          Base::V <<= sh_amt;
        } else {
          // shift and truncate.
          Base::V = op.V << sh_amt;
        }
      } else {
        Base::V = 0;
      }
    }
    // handle overflow/underflow
    if ((_AP_O != AP_WRAP || _AP_N != 0) &&
        ((!_AP_S && _AP_S2) ||
         _AP_I - _AP_S <
             _AP_I2 - _AP_S2 +
                 (QUAN_INC || (_AP_S2 && _AP_O == AP_SAT_SYM)))) { // saturation
      bool deleted_zeros = _AP_S2 ? true : !carry;
      bool deleted_ones = true;
      bool neg_src = isneg;
      bool newsignbit = _AP_ROOT_op_get_bit(Base::V, _AP_W - 1);
      enum { pos1 = F2 - _AP_F + _AP_W, pos2 = F2 - _AP_F + _AP_W + 1 };
      bool lD = (pos1 < _AP_W2 && pos1 >= 0) ? _AP_ROOT_op_get_bit(op.V, pos1)
                                             : false;
      if (pos1 < _AP_W2) {
        bool Range1_all_ones = true;
        bool Range1_all_zeros = true;
        bool Range2_all_ones = true;
        ap_int_base<_AP_W2, false> all_ones(-1);

        if (pos2 < _AP_W2 && pos2 >= 0) {
          ap_int_base<_AP_W2, false> Range2;
          Range2.V = _AP_ROOT_op_get_range(op.V, pos2, _AP_W2 - 1);
          Range2_all_ones = Range2 == (all_ones >> pos2);
        } else if (pos2 < 0) {
          Range2_all_ones = false;
        }

        if (pos1 >= 0 && pos2 < _AP_W2) {
          ap_int_base<_AP_W2, false> Range1;
          Range1.V = _AP_ROOT_op_get_range(op.V, pos1, _AP_W2 - 1);
          Range1_all_ones = Range1 == (all_ones >> pos1);
          Range1_all_zeros = !Range1.V;
        } else if (pos2 == _AP_W2) {
          Range1_all_ones = lD;
          Range1_all_zeros = !lD;
        } else if (pos1 < 0) {
          Range1_all_zeros = !op.V;
          Range1_all_ones = false;
        }

        deleted_zeros =
            deleted_zeros && (carry ? Range1_all_ones : Range1_all_zeros);
        deleted_ones =
            carry ? Range2_all_ones && (pos1 < 0 || !lD) : Range1_all_ones;
        neg_src = isneg && !(carry && Range1_all_ones);
      } else
        neg_src = isneg && newsignbit;
      bool neg_trg = _AP_S && newsignbit;
      bool overflow = (neg_trg || !deleted_zeros) && !isneg;
      bool underflow = (!neg_trg || !deleted_ones) && neg_src;
      if ((_AP_O == AP_SAT_SYM) && _AP_S2 && _AP_S)
        underflow |=
            neg_src &&
            (_AP_W > 1 ? _AP_ROOT_op_get_range(Base::V, 0, _AP_W - 2) == 0
                       : true);

      overflow_adjust(underflow, overflow, lD, neg_src);
    }
    return *this;
  } // operator= 

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_fixed_base& operator=(
      const volatile ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op) {
    operator=(const_cast<const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>&>(op));
    return *this;
  }

  /// Set this ap_fixed_base with ULL.
  INLINE ap_fixed_base& setBits(ap_ulong bv) {
    // TODO when ull is not be long enough...
    Base::V = bv;
    return *this;
  }

  /// Return a ap_fixed_base object whose this->V is assigned by bv.
  static INLINE ap_fixed_base bitsToFixed(ap_ulong bv) {
    // TODO fix when ull is not be long enough...
    ap_fixed_base t;
#ifdef __SYNTHESIS__
    t.V = bv;
#else
    t.V.set_bits(bv);
#endif
    return t;
  }

  // Explicit conversion functions to ap_int_base.
  /** Captures all integer bits, in truncate mode.
   *  @param[in] Cnative follow conversion from double to int.
   */
  INLINE ap_int_base<AP_MAX(_AP_I, 1), _AP_S> to_ap_int_base(
      bool Cnative = true) const {
    ap_int_base<AP_MAX(_AP_I, 1), _AP_S> ret;
    if (_AP_I == 0) {
      ret.V = 0;
    } else if (_AP_I > 0 && _AP_I <= _AP_W) {
      ret.V = _AP_ROOT_op_get_range(Base::V, _AP_W - _AP_I, _AP_W - 1);
    } else if (_AP_I > _AP_W) {
      ret.V = _AP_ROOT_op_get_range(Base::V, 0, _AP_W - 1);
      ret.V <<= (_AP_I - _AP_W);
    }
    /* Consider the following case
     *   float f = -7.5f;
     *   ap_fixed<8,4> t = f;  // -8 0 0 0 . 0.5
     *   int i = t.to_int();
     * the result should be -7 instead of -8.
     * Therefore, after truncation, the value should be increated by 1.
     * For (-1, 0), carry to MSB will happen, but result 0 is still correct.
     */
    if (Cnative && _AP_I < _AP_W) {
      // Follow C native data type, conversion from double to int
      if (_AP_S && _AP_ROOT_op_get_bit(Base::V, _AP_W - 1) && (_AP_I < _AP_W) &&
          (_AP_ROOT_op_get_range(
               Base::V, 0, _AP_I < 0 ? _AP_W - 1 : _AP_W - _AP_I - 1) != 0))
        ++ret;
    } else {
      // Follow OSCI library, conversion from sc_fixed to sc_int
    }
    return ret;
  };

 public:
  template <int _AP_W2, bool _AP_S2>
  INLINE operator ap_int_base<_AP_W2, _AP_S2>() const {
    return ap_int_base<_AP_W2, _AP_S2>(to_ap_int_base());
  }

  // Explicit conversion function to C built-in integral type.
  INLINE char to_char() const { return to_ap_int_base().to_char(); }

  INLINE int to_int() const { return to_ap_int_base().to_int(); }

  INLINE unsigned to_uint() const { return to_ap_int_base().to_uint(); }

  INLINE ap_slong to_int64() const { return to_ap_int_base().to_int64(); }

  INLINE ap_ulong to_uint64() const { return to_ap_int_base().to_uint64(); }

  /// covert function to double.
  /** only round-half-to-even mode supported, does not obey FE env. */
  INLINE double to_double() const {
#if defined(AP_FIXED_ENABLE_CPP_FENV)
    _AP_WARNING(std::fegetround() != FE_TONEAREST,
                "Only FE_TONEAREST is supported");
#endif
    enum { BITS = DOUBLE_MAN + DOUBLE_EXP + 1 };
    if (!Base::V) return 0.0f;
    bool s = _AP_S && _AP_ROOT_op_get_bit(Base::V, _AP_W - 1); ///< sign.
    ap_int_base<_AP_W, false> tmp;
    if (s)
      tmp.V = -Base::V; // may truncate one bit extra from neg in sim.
    else
      tmp.V = Base::V;
    int l = tmp.countLeadingZeros(); ///< number of leading zeros.
    int e = _AP_I - l - 1 + DOUBLE_BIAS; ///< exponent
    int lsb_index = _AP_W - l - 1 - DOUBLE_MAN;
    // more than 0.5?
    bool a = (lsb_index >=2) ?
        (_AP_ROOT_op_get_range(tmp.V, 0, lsb_index - 2) != 0) : 0;
    // round to even
    a |= (lsb_index >=0) ? _AP_ROOT_op_get_bit(tmp.V, lsb_index) : 0;
    // ull is at least 64-bit
    ap_ulong m;
    // may actually left shift, ensure buffer is wide enough.
    if (_AP_W > BITS) {
      m = (lsb_index >= 1) ? (ap_ulong)(tmp.V >> (lsb_index - 1))
                           : (ap_ulong)(tmp.V << (1 - lsb_index));
    } else {
      m = (ap_ulong)tmp.V;
      m = (lsb_index >= 1) ? (m >> (lsb_index - 1))
                           : (m << (1 - lsb_index));
    }
    m += a;
    m >>= 1;
    //std::cout << '\n' << std::hex << m << '\n'; // TODO delete this
    // carry to MSB, increase exponent
    if (_AP_ctype_op_get_bit(m, DOUBLE_MAN + 1)) {
      e += 1;
    }
    // set sign and exponent
    m = _AP_ctype_op_set_bit(m, BITS - 1, s);
    //std::cout << m << '\n'; // TODO delete this
    m = _AP_ctype_op_set_range(m, DOUBLE_MAN, DOUBLE_MAN + DOUBLE_EXP - 1, e);
    //std::cout << std::hex << m << std::dec << std::endl; // TODO delete this
    // cast to fp
    return rawBitsToDouble(m);
  }

  /// convert function to float.
  /** only round-half-to-even mode supported, does not obey FE env. */
  INLINE float to_float() const {
#if defined(AP_FIXED_ENABLE_CPP_FENV)
    _AP_WARNING(std::fegetround() != FE_TONEAREST,
                "Only FE_TONEAREST is supported");
#endif
    enum { BITS = FLOAT_MAN + FLOAT_EXP + 1 };
    if (!Base::V) return 0.0f;
    bool s = _AP_S && _AP_ROOT_op_get_bit(Base::V, _AP_W - 1); ///< sign.
    ap_int_base<_AP_W, false> tmp;
    if (s)
      tmp.V = -Base::V; // may truncate one bit extra from neg in sim.
    else
      tmp.V = Base::V;
    int l = tmp.countLeadingZeros();  ///< number of leading zeros.
    int e = _AP_I - l - 1 + FLOAT_BIAS; ///< exponent
    int lsb_index = _AP_W - l - 1 - FLOAT_MAN;
    // more than 0.5?
    bool a = (lsb_index >=2) ?
        (_AP_ROOT_op_get_range(tmp.V, 0, lsb_index - 2) != 0) : 0;
    // round to even
    a |= (lsb_index >=0) ? _AP_ROOT_op_get_bit(tmp.V, lsb_index) : 0;
    // ul is at least 32-bit
    unsigned long m;
    // may actually left shift, ensure buffer is wide enough.
    if (_AP_W > BITS) {
      m = (lsb_index >= 1) ? (unsigned long)(tmp.V >> (lsb_index - 1))
                           : (unsigned long)(tmp.V << (1 - lsb_index));
    } else {
      m = (unsigned long)tmp.V;
      m = (lsb_index >= 1) ? (m >> (lsb_index - 1))
                           : (m << (1 - lsb_index));
    }
    m += a;
    m >>= 1;
    // carry to MSB, increase exponent
    if (_AP_ctype_op_get_bit(m, FLOAT_MAN + 1)) {
      e += 1;
    }
    // set sign and exponent
    m = _AP_ctype_op_set_bit(m, BITS - 1, s);
    m = _AP_ctype_op_set_range(m, FLOAT_MAN, FLOAT_MAN + FLOAT_EXP - 1, e);
    // cast to fp
    return rawBitsToFloat(m);
  }

#if _AP_ENABLE_HALF_ == 1
  /// convert function to half.
  /** only round-half-to-even mode supported, does not obey FE env. */
  INLINE half to_half() const {
#if defined(AP_FIXED_ENABLE_CPP_FENV)
    _AP_WARNING(std::fegetround() != FE_TONEAREST,
                "Only FE_TONEAREST is supported");
#endif
    enum { BITS = HALF_MAN + HALF_EXP + 1 };
    if (!Base::V) return 0.0f;
    bool s = _AP_S && _AP_ROOT_op_get_bit(Base::V, _AP_W - 1); ///< sign.
    ap_int_base<_AP_W, false> tmp;
    if (s)
      tmp.V = -Base::V; // may truncate one bit extra from neg in sim.
    else
      tmp.V = Base::V;
    int l = tmp.countLeadingZeros();  ///< number of leading zeros.
    int e = _AP_I - l - 1 + HALF_BIAS; ///< exponent
    int lsb_index = _AP_W - l - 1 - HALF_MAN;
    // more than 0.5?
    bool a = (lsb_index >=2) ?
        (_AP_ROOT_op_get_range(tmp.V, 0, lsb_index - 2) != 0) : 0;
    // round to even
    a |= (lsb_index >=0) ? _AP_ROOT_op_get_bit(tmp.V, lsb_index) : 0;
    // short is at least 16-bit
    unsigned short m;
    // may actually left shift, ensure buffer is wide enough.
    if (_AP_W > BITS) {
      m = (lsb_index >= 1) ? (unsigned short)(tmp.V >> (lsb_index - 1))
                           : (unsigned short)(tmp.V << (1 - lsb_index));
    } else {
      m = (unsigned short)tmp.V;
      m = (lsb_index >= 1) ? (m >> (lsb_index - 1))
                           : (m << (1 - lsb_index));
    }
    m += a;
    m >>= 1;
    // carry to MSB, increase exponent
    if (_AP_ctype_op_get_bit(m, HALF_MAN + 1)) {
      e += 1;
    }
    // set sign and exponent
    m = _AP_ctype_op_set_bit(m, BITS - 1, s);
    m = _AP_ctype_op_set_range(m, HALF_MAN, HALF_MAN + HALF_EXP - 1, e);
    // cast to fp
    return rawBitsToHalf(m);
  }
#endif

  // FIXME inherited from old code, this may loose precision!
  INLINE operator long double() const { return (long double)to_double(); }

  INLINE operator double() const { return to_double(); }

  INLINE operator float() const { return to_float(); }

#if _AP_ENABLE_HALF_ == 1
  INLINE operator half() const { return to_half(); }
#endif

  INLINE operator bool() const { return (bool)Base::V != 0; }

  INLINE operator char() const { return (char)to_int(); }

  INLINE operator signed char() const { return (signed char)to_int(); }

  INLINE operator unsigned char() const { return (unsigned char)to_uint(); }

  INLINE operator short() const { return (short)to_int(); }

  INLINE operator unsigned short() const { return (unsigned short)to_uint(); }

  INLINE operator int() const { return to_int(); }

  INLINE operator unsigned int() const { return to_uint(); }

// FIXME don't assume data width...
#ifdef __x86_64__
  INLINE operator long() const { return (long)to_int64(); }

  INLINE operator unsigned long() const { return (unsigned long)to_uint64(); }
#else
  INLINE operator long() const { return (long)to_int(); }

  INLINE operator unsigned long() const { return (unsigned long)to_uint(); }
#endif // ifdef __x86_64__ else

  INLINE operator ap_ulong() const { return to_uint64(); }

  INLINE operator ap_slong() const { return to_int64(); }

  INLINE int length() const { return _AP_W; };

  // bits_to_int64 deleted.
#ifndef __SYNTHESIS__
  // Used in autowrap, when _AP_W < 64.
  INLINE ap_ulong bits_to_uint64() const {
    return (Base::V).to_uint64();
  }
#endif

  // Count the number of zeros from the most significant bit
  // to the first one bit. Note this is only for ap_fixed_base whose
  // _AP_W <= 64, otherwise will incur assertion.
  INLINE int countLeadingZeros() {
#ifdef __SYNTHESIS__
    // TODO: used llvm.ctlz intrinsic ?
    if (_AP_W <= 32) {
      ap_int_base<32, false> t(-1ULL);
      t.range(_AP_W - 1, 0) = this->range(0, _AP_W - 1);
      return __builtin_ctz(t.V);
    } else if (_AP_W <= 64) {
      ap_int_base<64, false> t(-1ULL);
      t.range(_AP_W - 1, 0) = this->range(0, _AP_W - 1);
      return __builtin_ctzll(t.V);
    } else {
      enum {__N = (_AP_W + 63) / 64};
      int NZeros = 0;
      int i = 0;
      bool hitNonZero = false;
      for (i = 0; i < __N - 1; ++i) {
        ap_int_base<64, false> t;
        t.range(0, 63) = this->range(_AP_W - i * 64 - 64, _AP_W - i * 64 - 1);
        NZeros += hitNonZero ? 0 : __builtin_clzll(t.V);
        hitNonZero |= (t != 0);
      }
      if (!hitNonZero) {
        ap_int_base<64, false> t(-1ULL);
        t.range(63 - (_AP_W - 1) % 64, 63) = this->range(0, (_AP_W - 1) % 64);
        NZeros += __builtin_clzll(t.V);
      }
      return NZeros;
    }
#else
    return Base::V.countLeadingZeros();
#endif
  }

  // Arithmetic : Binary
  // -------------------------------------------------------------------------
  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE typename RType<_AP_W2, _AP_I2, _AP_S2>::mult operator*(
      const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op2)
      const {
    typename RType<_AP_W2, _AP_I2, _AP_S2>::mult_base r, t;
    r.V = Base::V;
    t.V = op2.V;
    r.V *= op2.V;
    return r;
  }

  // multiply function deleted.

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE typename RType<_AP_W2, _AP_I2, _AP_S2>::div operator/(
      const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op2)
      const {
    typename RType<_AP_W2, _AP_I2, _AP_S2>::div_base r;
#ifndef __SYNTHESIS__
    enum {F2 = _AP_W2-_AP_I2,
              _W1=AP_MAX(_AP_W + AP_MAX(F2, 0) + ((_AP_S2 && !_AP_S) ? 1 : 0), _AP_W2 + ((_AP_S && !_AP_S2) ? 1 : 0))};
    ap_int_base<_W1,_AP_S||_AP_S2> dividend,divisior;
    ap_int_base<_W1,_AP_S> tmp1;
    ap_int_base<_W1,_AP_S2> tmp2;
    tmp1.V = Base::V;
    tmp1.V <<= AP_MAX(F2,0);
    tmp2.V = op2.V;
    dividend = tmp1;
    divisior = tmp2;
    r.V = ((_AP_S||_AP_S2) ? dividend.V.sdiv(divisior.V): dividend.V.udiv(divisior.V));
#else
    #ifndef __SC_COMPATIBLE__
        ap_fixed_base<_AP_W + AP_MAX(_AP_W2 - _AP_I2, 0),_AP_I, _AP_S> t(*this);
    #else
        ap_fixed_base<_AP_W + AP_MAX(_AP_W2 - _AP_I2, 0) + AP_MAX(_AP_I2, 0),_AP_I, _AP_S> t(*this);
    #endif
        r.V = t.V / op2.V;
#endif
/*
    enum {
      F2 = _AP_W2 - _AP_I2,
      shl = AP_MAX(F2, 0) + AP_MAX(_AP_I2, 0),
#ifndef __SC_COMPATIBLE__
      shr = AP_MAX(_AP_I2, 0),
#else
      shr = 0,
#endif
      W3 = _AP_S2 + _AP_W + shl,
      S3 = _AP_S || _AP_S2,
    };
    ap_int_base<W3, S3> dividend, t;
    dividend.V = Base::V;
    // multiply both by (1 << F2), and than do integer division.
    dividend.V <<= (int) shl;
#ifdef __SYNTHESIS__
    // .V's have right signedness, and will have right extending.
    t.V = dividend.V / op2.V;
#else
    // XXX op2 may be wider than dividend, and sdiv and udiv takes the same with
    // as left hand operand, so data might be truncated by mistake if not
    // handled here.
    t.V = S3 ? dividend.V.sdiv(op2.V) : dividend.V.udiv(op2.V);
#endif
    r.V = t.V >> (int) shr;
*/
    return r;
  }

#define OP_BIN_AF(Sym, Rty)                                                \
  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,         \
            ap_o_mode _AP_O2, int _AP_N2>                                  \
  INLINE typename RType<_AP_W2, _AP_I2, _AP_S2>::Rty operator Sym(         \
      const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& \
          op2) const {                                                     \
    typename RType<_AP_W2, _AP_I2, _AP_S2>::Rty##_base ret, lhs(*this),    \
        rhs(op2);                                                          \
    ret.V = lhs.V Sym rhs.V;                                               \
    return ret;                                                            \
  }

  OP_BIN_AF(+, plus)
  OP_BIN_AF(-, minus)
  OP_BIN_AF(&, logic)
  OP_BIN_AF(|, logic)
  OP_BIN_AF(^, logic)

// Arithmetic : assign
// -------------------------------------------------------------------------
#define OP_ASSIGN_AF(Sym)                                                  \
  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,         \
            ap_o_mode _AP_O2, int _AP_N2>                                  \
  INLINE ap_fixed_base& operator Sym##=(                                   \
      const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& \
          op2) {                                                           \
    *this = operator Sym(op2);                                             \
    return *this;                                                          \
  }

  OP_ASSIGN_AF(*)
  OP_ASSIGN_AF(/)
  OP_ASSIGN_AF(+)
  OP_ASSIGN_AF(-)
  OP_ASSIGN_AF(&)
  OP_ASSIGN_AF(|)
  OP_ASSIGN_AF(^)

  // Prefix and postfix increment and decrement.
  // -------------------------------------------------------------------------

  /// Prefix increment
  INLINE ap_fixed_base& operator++() {
    operator+=(ap_fixed_base<_AP_W - _AP_I + 1, 1, false>(1));
    return *this;
  }

  /// Prefix decrement.
  INLINE ap_fixed_base& operator--() {
    operator-=(ap_fixed_base<_AP_W - _AP_I + 1, 1, false>(1));
    return *this;
  }

  /// Postfix increment
  INLINE const ap_fixed_base operator++(int) {
    ap_fixed_base r(*this);
    operator++();
    return r;
  }

  /// Postfix decrement
  INLINE const ap_fixed_base operator--(int) {
    ap_fixed_base r(*this);
    operator--();
    return r;
  }

  // Unary arithmetic.
  // -------------------------------------------------------------------------
  INLINE ap_fixed_base operator+() { return *this; }

  INLINE ap_fixed_base<_AP_W + 1, _AP_I + 1, true> operator-() const {
    ap_fixed_base<_AP_W + 1, _AP_I + 1, true> r(*this);
    r.V = -r.V;
    return r;
  }

  INLINE ap_fixed_base<_AP_W, _AP_I, true, _AP_Q, _AP_O, _AP_N> getNeg() {
    ap_fixed_base<_AP_W, _AP_I, true, _AP_Q, _AP_O, _AP_N> r(*this);
    r.V = -r.V;
    return r;
  }

  // Not (!)
  // -------------------------------------------------------------------------
  INLINE bool operator!() const { return Base::V == 0; }

  // Bitwise complement
  // -------------------------------------------------------------------------
  // XXX different from Mentor's ac_fixed.
  INLINE ap_fixed_base<_AP_W, _AP_I, _AP_S> operator~() const {
    ap_fixed_base<_AP_W, _AP_I, _AP_S> r;
    r.V = ~Base::V;
    return r;
  }

  // Shift
  // -------------------------------------------------------------------------
  // left shift is the same as moving point right, i.e. increate I.
  template <int _AP_SHIFT>
  INLINE ap_fixed_base<_AP_W, _AP_I + _AP_SHIFT, _AP_S> lshift() const {
    ap_fixed_base<_AP_W, _AP_I + _AP_SHIFT, _AP_S> r;
    r.V = Base::V;
    return r;
  }

  template <int _AP_SHIFT>
  INLINE ap_fixed_base<_AP_W, _AP_I - _AP_SHIFT, _AP_S> rshift() const {
    ap_fixed_base<_AP_W, _AP_I - _AP_SHIFT, _AP_S> r;
    r.V = Base::V;
    return r;
  }

  // Because the return type is the type of the the first operand, shift assign
  // operators do not carry out any quantization or overflow
  // While systemc, shift assigns for sc_fixed/sc_ufixed will result in
  // quantization or overflow (depending on the mode of the first operand)
  INLINE ap_fixed_base operator<<(unsigned int sh) const {
    ap_fixed_base r;
    r.V = Base::V << sh;
// TODO check shift overflow?
#ifdef __SC_COMPATIBLE__
    if (sh == 0) return r;
    if (_AP_O != AP_WRAP || _AP_N != 0) {
      bool neg_src = _AP_S && _AP_ROOT_op_get_bit(Base::V, _AP_W - 1);
      bool allones, allzeros;
      ap_int_base<_AP_W, false> ones(-1);
      if (sh <= _AP_W) {
        ap_int_base<_AP_W, false> range1;
        range1.V = _AP_ROOT_op_get_range(
            const_cast<ap_fixed_base*>(this)->Base::V, _AP_W - sh, _AP_W - 1);
        allones = range1 == (ones >> (_AP_W - sh));
        allzeros = range1 == 0;
      } else {
        allones = false;
        allzeros = Base::V == 0;
      }
      bool overflow = !allzeros && !neg_src;
      bool underflow = !allones && neg_src;
      if ((_AP_O == AP_SAT_SYM) && _AP_S)
        underflow |=
            neg_src &&
            (_AP_W > 1 ? _AP_ROOT_op_get_range(r.V, 0, _AP_W - 2) == 0 : true);
      bool lD = false;
      if (sh < _AP_W) lD = _AP_ROOT_op_get_bit(Base::V, _AP_W - sh - 1);
      r.overflow_adjust(underflow, overflow, lD, neg_src);
    }
#endif
    return r;
  }

  INLINE ap_fixed_base operator>>(unsigned int sh) const {
    ap_fixed_base r;
    r.V = Base::V >> sh;
// TODO check shift overflow?
#ifdef __SC_COMPATIBLE__
    if (sh == 0) return r;
    if (_AP_Q != AP_TRN) {
      bool qb = false;
      if (sh <= _AP_W) qb = _AP_ROOT_op_get_bit(Base::V, sh - 1);
      bool rb = false;
      if (sh > 1 && sh <= _AP_W)
        rb = _AP_ROOT_op_get_range(const_cast<ap_fixed_base*>(this)->Base::V, 0,
                                   sh - 2) != 0;
      else if (sh > _AP_W)
        rb = Base::V != 0;
      r.quantization_adjust(qb, rb,
                            _AP_S && _AP_ROOT_op_get_bit(Base::V, _AP_W - 1));
    }
#endif
    return r;
  }

  // left and right shift for int
  INLINE ap_fixed_base operator<<(int sh) const {
    ap_fixed_base r;
    bool isNeg = sh < 0;
    unsigned int ush = isNeg ? -sh : sh;
    if (isNeg) {
      return operator>>(ush);
    } else {
      return operator<<(ush);
    }
  }

  INLINE ap_fixed_base operator>>(int sh) const {
    bool isNeg = sh < 0;
    unsigned int ush = isNeg ? -sh : sh;
    if (isNeg) {
      return operator<<(ush);
    } else {
      return operator>>(ush);
    }
  }

  // left and right shift for ap_int.
  template <int _AP_W2>
  INLINE ap_fixed_base operator<<(const ap_int_base<_AP_W2, true>& op2) const {
    // TODO the code seems not optimal. ap_fixed<8,8> << ap_int<2> needs only a
    // small mux, but integer need a big one!
    int sh = op2.to_int();
    return operator<<(sh);
  }

  template <int _AP_W2>
  INLINE ap_fixed_base operator>>(const ap_int_base<_AP_W2, true>& op2) const {
    int sh = op2.to_int();
    return operator>>(sh);
  }

  // left and right shift for ap_uint.
  template <int _AP_W2>
  INLINE ap_fixed_base operator<<(const ap_int_base<_AP_W2, false>& op2) const {
    unsigned int sh = op2.to_uint();
    return operator<<(sh);
  }

  template <int _AP_W2>
  INLINE ap_fixed_base operator>>(const ap_int_base<_AP_W2, false>& op2) const {
    unsigned int sh = op2.to_uint();
    return operator>>(sh);
  }

  // left and right shift for ap_fixed
  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_fixed_base operator<<(
      const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>&
          op2) {
    return operator<<(op2.to_ap_int_base());
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_fixed_base operator>>(
      const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>&
          op2) {
    return operator>>(op2.to_ap_int_base());
  }

  // Shift assign.
  // -------------------------------------------------------------------------

  // left shift assign.
  INLINE ap_fixed_base& operator<<=(const int sh) {
    *this = operator<<(sh);
    return *this;
  }

  INLINE ap_fixed_base& operator<<=(const unsigned int sh) {
    *this = operator<<(sh);
    return *this;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_fixed_base& operator<<=(const ap_int_base<_AP_W2, _AP_S2>& sh) {
    *this = operator<<(sh.to_int());
    return *this;
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_fixed_base& operator<<=(
      const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>&
          sh) {
    *this = operator<<(sh.to_int());
    return *this;
  }

  // right shift assign.
  INLINE ap_fixed_base& operator>>=(const int sh) {
    *this = operator>>(sh);
    return *this;
  }

  INLINE ap_fixed_base& operator>>=(const unsigned int sh) {
    *this = operator>>(sh);
    return *this;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_fixed_base& operator>>=(const ap_int_base<_AP_W2, _AP_S2>& sh) {
    *this = operator>>(sh.to_int());
    return *this;
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_fixed_base& operator>>=(
      const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>&
          sh) {
    *this = operator>>(sh.to_int());
    return *this;
  }

// Comparisons.
// -------------------------------------------------------------------------
#define OP_CMP_AF(Sym)                                                         \
  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,             \
            ap_o_mode _AP_O2, int _AP_N2>                                      \
  INLINE bool operator Sym(const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, \
                                               _AP_O2, _AP_N2>& op2) const {   \
    enum { _AP_F = _AP_W - _AP_I, F2 = _AP_W2 - _AP_I2 };                      \
    if (_AP_F == F2)                                                           \
      return Base::V Sym op2.V;                                                \
    else if (_AP_F > F2)                                                       \
      return Base::V Sym ap_fixed_base<AP_MAX(_AP_W2 + _AP_F - F2, 1), _AP_I2, \
                                       _AP_S2, _AP_Q2, _AP_O2, _AP_N2>(op2).V; \
    else                                                                       \
      return ap_fixed_base<AP_MAX(_AP_W + F2 - _AP_F + 1, 1), _AP_I + 1,       \
                           _AP_S, _AP_Q, _AP_O, _AP_N>(*this).V Sym op2.V;     \
    return false;                                                              \
  }

  OP_CMP_AF(>)
  OP_CMP_AF(<)
  OP_CMP_AF(>=)
  OP_CMP_AF(<=)
  OP_CMP_AF(==)
  OP_CMP_AF(!=)
// FIXME: Move compare with double out of struct ap_fixed_base defination
//        and combine it with compare operator(double, ap_fixed_base)
#define DOUBLE_CMP_AF(Sym) \
  INLINE bool operator Sym(double d) const { return to_double() Sym d; }

  DOUBLE_CMP_AF(>)
  DOUBLE_CMP_AF(<)
  DOUBLE_CMP_AF(>=)
  DOUBLE_CMP_AF(<=)
  DOUBLE_CMP_AF(==)
  DOUBLE_CMP_AF(!=)

  // Bit and Slice Select
  INLINE af_bit_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N> operator[](
      unsigned index) {
    _AP_WARNING(index >= _AP_W, "Attempting to read bit beyond MSB");
    return af_bit_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>(this, index);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE af_bit_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N> operator[](
      const ap_int_base<_AP_W2, _AP_S2>& index) {
    _AP_WARNING(index < 0, "Attempting to read bit with negative index");
    _AP_WARNING(index >= _AP_W, "Attempting to read bit beyond MSB");
    return af_bit_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>(this,
                                                                index.to_int());
  }

  INLINE bool operator[](unsigned index) const {
    _AP_WARNING(index >= _AP_W, "Attempting to read bit beyond MSB");
    return _AP_ROOT_op_get_bit(const_cast<ap_fixed_base*>(this)->V, index);
  }

  INLINE af_bit_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N> bit(
      unsigned index) {
    _AP_WARNING(index >= _AP_W, "Attempting to read bit beyond MSB");
    return af_bit_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>(this, index);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE af_bit_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N> bit(
      const ap_int_base<_AP_W2, _AP_S2>& index) {
    _AP_WARNING(index < 0, "Attempting to read bit with negative index");
    _AP_WARNING(index >= _AP_W, "Attempting to read bit beyond MSB");
    return af_bit_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>(this,
                                                                index.to_int());
  }

  INLINE bool bit(unsigned index) const {
    _AP_WARNING(index >= _AP_W, "Attempting to read bit beyond MSB");
    return _AP_ROOT_op_get_bit(const_cast<ap_fixed_base*>(this)->V, index);
  }

  template <int _AP_W2>
  INLINE af_bit_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N> get_bit(
      const ap_int_base<_AP_W2, true>& index) {
    _AP_WARNING(index < _AP_I - _AP_W,
                "Attempting to read bit with negative index");
    _AP_WARNING(index >= _AP_I, "Attempting to read bit beyond MSB");
    return af_bit_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>(
        this, index.to_int() + _AP_W - _AP_I);
  }

  INLINE bool get_bit(int index) const {
    _AP_WARNING(index >= _AP_I, "Attempting to read bit beyond MSB");
    _AP_WARNING(index < _AP_I - _AP_W, "Attempting to read bit beyond MSB");
    return _AP_ROOT_op_get_bit(const_cast<ap_fixed_base*>(this)->V,
                               index + _AP_W - _AP_I);
  }
#if 0
  INLINE af_bit_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N> get_bit(
      int index) {
    _AP_WARNING(index < _AP_I - _AP_W,
              "Attempting to read bit with negative index");
    _AP_WARNING(index >= _AP_I, "Attempting to read bit beyond MSB");
    return af_bit_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>(
        this, index + _AP_W - _AP_I);
  }
#endif

  template <int _AP_W2>
  INLINE bool get_bit(const ap_int_base<_AP_W2, true>& index) const {
    _AP_WARNING(index >= _AP_I, "Attempting to read bit beyond MSB");
    _AP_WARNING(index < _AP_I - _AP_W, "Attempting to read bit beyond MSB");
    return _AP_ROOT_op_get_bit(const_cast<ap_fixed_base*>(this)->V,
                               index.to_int() + _AP_W - _AP_I);
  }

  INLINE af_range_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N> range(int Hi,
                                                                      int Lo) {
    _AP_WARNING((Hi >= _AP_W) || (Lo >= _AP_W), "Out of bounds in range()");
    return af_range_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>(this, Hi, Lo);
  }

  // This is a must to strip constness to produce reference type.
  INLINE af_range_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N> range(
      int Hi, int Lo) const {
    _AP_WARNING((Hi >= _AP_W) || (Lo >= _AP_W), "Out of bounds in range()");
    return af_range_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>(
        const_cast<ap_fixed_base*>(this), Hi, Lo);
  }

  template <int _AP_W2, bool _AP_S2, int _AP_W3, bool _AP_S3>
  INLINE af_range_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N> range(
      const ap_int_base<_AP_W2, _AP_S2>& HiIdx,
      const ap_int_base<_AP_W3, _AP_S3>& LoIdx) {
    int Hi = HiIdx.to_int();
    int Lo = LoIdx.to_int();
    return this->range(Hi, Lo);
  }

  template <int _AP_W2, bool _AP_S2, int _AP_W3, bool _AP_S3>
  INLINE af_range_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N> range(
      const ap_int_base<_AP_W2, _AP_S2>& HiIdx,
      const ap_int_base<_AP_W3, _AP_S3>& LoIdx) const {
    int Hi = HiIdx.to_int();
    int Lo = LoIdx.to_int();
    return this->range(Hi, Lo);
  }

  INLINE af_range_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N> range() {
    return this->range(_AP_W - 1, 0);
  }

  INLINE af_range_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N> range() const {
    return this->range(_AP_W - 1, 0);
  }

  INLINE af_range_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N> operator()(
      int Hi, int Lo) {
    return this->range(Hi, Lo);
  }

  INLINE af_range_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N> operator()(
      int Hi, int Lo) const {
    return this->range(Hi, Lo);
  }

  template <int _AP_W2, bool _AP_S2, int _AP_W3, bool _AP_S3>
  INLINE af_range_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N> operator()(
      const ap_int_base<_AP_W2, _AP_S2>& HiIdx,
      const ap_int_base<_AP_W3, _AP_S3>& LoIdx) {
    int Hi = HiIdx.to_int();
    int Lo = LoIdx.to_int();
    return this->range(Hi, Lo);
  }

  template <int _AP_W2, bool _AP_S2, int _AP_W3, bool _AP_S3>
  INLINE af_range_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N> operator()(
      const ap_int_base<_AP_W2, _AP_S2>& HiIdx,
      const ap_int_base<_AP_W3, _AP_S3>& LoIdx) const {
    int Hi = HiIdx.to_int();
    int Lo = LoIdx.to_int();
    return this->range(Hi, Lo);
  }

  INLINE bool is_zero() const { return Base::V == 0; }

  INLINE bool is_neg() const {
    if (_AP_S && _AP_ROOT_op_get_bit(Base::V, _AP_W - 1)) return true;
    return false;
  }

  INLINE int wl() const { return _AP_W; }

  INLINE int iwl() const { return _AP_I; }

  INLINE ap_q_mode q_mode() const { return _AP_Q; }

  INLINE ap_o_mode o_mode() const { return _AP_O; }

  INLINE int n_bits() const { return _AP_N; }

  // print a string representation of this number in the given radix.
  // Radix support is 2, 8, 10, or 16.
  // The result will include a prefix indicating the radix, except for decimal,
  // where no prefix is needed.  The default is to output a signed representation
  // of signed numbers, or an unsigned representation  of unsigned numbers.  For
  // non-decimal formats, this can be changed by the 'sign' argument.
#ifndef __SYNTHESIS__
  std::string to_string(unsigned char radix = 2, bool sign = _AP_S) const {
    // XXX in autosim/autowrap.tcl "(${name}).to_string(2).c_str()" is used to
    // initialize sc_lv, which seems incapable of handling format "-0b".
    if (radix == 2) sign = false;

    std::string str;
    str.clear();
    char step = 0;
    bool isNeg = sign && (Base::V < 0);

    // Extend to take care of the -MAX case.
    ap_fixed_base<_AP_W + 1, _AP_I + 1> tmp(*this);
    if (isNeg) {
      tmp = -tmp;
      str += '-';
    }
    std::string prefix;
    switch (radix) {
      case 2:
        prefix = "0b";
        step = 1;
        break;
      case 8:
        prefix = "0o";
        step = 3;
        break;
      case 16:
        prefix = "0x";
        step = 4;
        break;
      default:
        break;
    }

    if (_AP_I > 0) {
      // Note we drop the quantization and rounding flags here.  The
      // integer part is always in range, and the fractional part we
      // want to drop.  Also, the number is always positive, because
      // of the absolute value above.
      ap_int_base<AP_MAX(_AP_I + 1, 1), false> int_part;
      //   [1] [ I ] d [ W - I ]
      //    |     |            |
      //    |    W-I           0
      //    W
      int_part.V = _AP_ROOT_op_get_range(
          tmp.V, _AP_W - _AP_I, _AP_W);
      str += int_part.to_string(radix, false);
    } else {
      str += prefix;
      str += '0';
    }

    ap_fixed_base<AP_MAX(_AP_W - _AP_I, 1), 0, false> frac_part = tmp;

    if (radix == 10) {
      if (frac_part != 0) {
        str += ".";
        while (frac_part != 0) {
          char digit = (frac_part * radix).to_char();
          str += static_cast<char>(digit + '0');
          frac_part *= radix;
        }
      }
    } else {
      if (frac_part != 0) {
        str += ".";
        for (signed i = _AP_W - _AP_I - 1; i >= 0; i -= step) {
          char digit = frac_part.range(i, AP_MAX(0, i - step + 1)).to_char();
          // If we have a partial bit pattern at the end, then we need
          // to put it in the high-order bits of 'digit'.
          int offset = AP_MIN(0, i - step + 1);
          digit <<= -offset;
          str += digit < 10 ? static_cast<char>(digit + '0')
                            : static_cast<char>(digit - 10 + 'a');
        }
        if (radix == 16)
          str += "p0"; // C99 Hex constants are required to have an exponent.
      }
    }
    return str;
  }
#else
  // XXX HLS will delete this in synthesis
  INLINE char* to_string(unsigned char radix = 2, bool sign = _AP_S) const {
    return 0;
  }
#endif
}; // struct ap_fixed_base.

template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O,
          int _AP_N>
INLINE void b_not(
    ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& ret,
    const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op) {
  ret.V = ~op.V;
}

template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O,
          int _AP_N>
INLINE void b_and(
    ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& ret,
    const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op1,
    const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op2) {
  ret.V = op1.V & op2.V;
}

template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O,
          int _AP_N>
INLINE void b_or(
    ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& ret,
    const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op1,
    const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op2) {
  ret.V = op1.V | op2.V;
}

template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O,
          int _AP_N>
INLINE void b_xor(
    ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& ret,
    const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op1,
    const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op2) {
  ret.V = op1.V ^ op2.V;
}

template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O,
          int _AP_N, int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
          ap_o_mode _AP_O2, int _AP_N2>
INLINE void neg(
    ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& ret,
    const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op) {
  ap_fixed_base<_AP_W2 + !_AP_S2, _AP_I2 + !_AP_S2, true, _AP_Q2, _AP_O2,
                _AP_N2>
      t;
  t.V = -op.V;
  ret = t;
}

template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O,
          int _AP_N, int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
          ap_o_mode _AP_O2, int _AP_N2>
INLINE void lshift(
    ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& ret,
    const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op,
    int i) {
  enum {
    F2 = _AP_W2 - _AP_I2,
    _AP_I3 = AP_MAX(_AP_I, _AP_I2),
    _AP_W3 = _AP_I3 + F2,
  };
  // wide buffer
  ap_fixed_base<_AP_W3, _AP_I3, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> t;
  t.V = op.V;
  t.V <<= i; // FIXME overflow?
  // handle quantization and overflow
  ret = t;
}

template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O,
          int _AP_N, int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
          ap_o_mode _AP_O2, int _AP_N2>
INLINE void rshift(
    ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& ret,
    const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op,
    int i) {
  enum {
    F = _AP_W - _AP_I,
    F2 = _AP_W2 - _AP_I2,
    F3 = AP_MAX(F, F2),
    _AP_W3 = _AP_I2 + F3,
    sh = F - F2,
  };
  // wide buffer
  ap_fixed_base<_AP_W3, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> t;
  t.V = op.V;
  if (sh >= 0)
    t.V <<= (int) sh;
  t.V >>= i;
  // handle quantization and overflow
  ret = t;
}

//// FIXME
//// These partial specialization ctors allow code like
////   char c = 'a';
////   ap_fixed_base<8, 8, true> x(c);
//// but what bout ap_fixed_base<9, 9, true> y(c) ?
//

#ifndef __SYNTHESIS__
INLINE std::string scientificFormat(std::string& input) {
  if (input.length() == 0) return input;

  size_t decPosition = input.find('.');
  if (decPosition == std::string::npos) decPosition = input.length();

  size_t firstNonZeroPos = 0;
  for (; input[firstNonZeroPos] > '9' || input[firstNonZeroPos] < '1';
       firstNonZeroPos++)
    ;

  int exp;
  if (firstNonZeroPos > decPosition)
    exp = decPosition - firstNonZeroPos;
  else
    exp = decPosition - firstNonZeroPos - 1;
  std::string expString = "";
  if (exp == 0)
    ;
  else if (exp < 0) {
    expString += "e-";
    exp = -exp;
  } else
    expString += "e+";

  if (exp < 10 && exp > 0) {
    expString += '0';
    expString += (char)('0' + exp);
  } else if (exp != 0) {
    std::string tmp;

    std::ostringstream oss;
    oss << exp;

    tmp = oss.str();
    expString += tmp;
  }

  int lastNonZeroPos = (int)(input.length() - 1);
  for (; lastNonZeroPos >= 0; --lastNonZeroPos)
    if (input[lastNonZeroPos] <= '9' && input[lastNonZeroPos] > '0') break;

  std::string ans = "";
  ans += input[firstNonZeroPos];
  if (firstNonZeroPos != (size_t)lastNonZeroPos) {
    ans += '.';
    for (int i = firstNonZeroPos + 1; i <= lastNonZeroPos; i++)
      if (input[i] != '.') ans += input[i];
  }

  ans += expString;
  return ans;
}

INLINE std::string reduceToPrecision(std::string& input, int precision) {
  bool isZero = true;
  size_t inputLen = input.length();
  for (size_t i = 0; i < inputLen && isZero; i++)
    if (input[i] != '.' && input[i] != '0') isZero = false;
  if (isZero) return "0";

  // Find the first valid number, skip '-'
  int FirstNonZeroPos = 0;
  int LastNonZeroPos = (int)inputLen - 1;
  int truncBitPosition = 0;
  size_t decPosition = input.find('.');
  for (; input[FirstNonZeroPos] < '1' || input[FirstNonZeroPos] > '9';
       FirstNonZeroPos++)
    ;

  for (; input[LastNonZeroPos] < '1' || input[LastNonZeroPos] > '9';
       LastNonZeroPos--)
    ;

  if (decPosition == std::string::npos) decPosition = inputLen;
  // Count the valid number, to decide whether we need to truncate
  if ((int)decPosition > LastNonZeroPos) {
    if (LastNonZeroPos - FirstNonZeroPos + 1 <= precision) return input;
    truncBitPosition = FirstNonZeroPos + precision;
  } else if ((int)decPosition < FirstNonZeroPos) { // This is pure decimal
    if (LastNonZeroPos - FirstNonZeroPos + 1 <= precision) {
      if (FirstNonZeroPos - decPosition - 1 < 4) {
        return input;
      } else {
        if (input[0] == '-') {
          std::string tmp = input.substr(1, inputLen - 1);
          return std::string("-") + scientificFormat(tmp);
        } else
          return scientificFormat(input);
      }
    }
    truncBitPosition = FirstNonZeroPos + precision;
  } else {
    if (LastNonZeroPos - FirstNonZeroPos <= precision) return input;
    truncBitPosition = FirstNonZeroPos + precision + 1;
  }

  // duplicate the input string, we want to add "0" before the valid numbers
  // This is easy for quantization, since we may change 9999 to 10000
  std::string ans = "";
  std::string dupInput = "0";
  if (input[0] == '-') {
    ans += '-';
    dupInput += input.substr(1, inputLen - 1);
  } else {
    dupInput += input.substr(0, inputLen);
    ++truncBitPosition;
  }

  // Add 'carry' after truncation, if necessary
  bool carry = dupInput[truncBitPosition] > '4';
  for (int i = truncBitPosition - 1; i >= 0 && carry; i--) {
    if (dupInput[i] == '.') continue;
    if (dupInput[i] == '9')
      dupInput[i] = '0';
    else {
      ++dupInput[i];
      carry = false;
    }
  }

  // bits outside precision range should be set to 0
  if (dupInput[0] == '1')
    FirstNonZeroPos = 0;
  else {
    FirstNonZeroPos = 0;
    while (dupInput[FirstNonZeroPos] < '1' || dupInput[FirstNonZeroPos] > '9')
      ++FirstNonZeroPos;
  }

  unsigned it = FirstNonZeroPos;
  int NValidNumber = 0;
  while (it < dupInput.length()) {
    if (dupInput[it] == '.') {
      ++it;
      continue;
    }
    ++NValidNumber;
    if (NValidNumber > precision) dupInput[it] = '0';
    ++it;
  }

  // Here we wanted to adjust the truncate position and the value
  decPosition = dupInput.find('.');
  if (decPosition == std::string::npos) // When this is integer
    truncBitPosition = (int)dupInput.length();
  else
    for (truncBitPosition = (int)(dupInput.length() - 1); truncBitPosition >= 0;
         --truncBitPosition) {
      if (dupInput[truncBitPosition] == '.') break;
      if (dupInput[truncBitPosition] != '0') {
        truncBitPosition++;
        break;
      }
    }

  if (dupInput[0] == '1')
    dupInput = dupInput.substr(0, truncBitPosition);
  else
    dupInput = dupInput.substr(1, truncBitPosition - 1);

  decPosition = dupInput.find('.');
  if (decPosition != std::string::npos) {
    size_t it = 0;
    for (it = decPosition + 1; dupInput[it] == '0'; it++)
      ;
    if (it - decPosition - 1 < 4) {
      ans += dupInput;
      return ans;
    } else {
      ans += scientificFormat(dupInput);
      return ans;
    }
  } else if ((int)(dupInput.length()) <= precision) {
    ans += dupInput;
    return ans;
  }

  ans += scientificFormat(dupInput);
  return ans;
}

template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O,
          int _AP_N>
INLINE void print(
    const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& x) {
  if (_AP_I > 0) {
    ap_int_base<_AP_I, _AP_S> p1;
    p1.V = x.V >> (_AP_W - _AP_I);
    print(p1.V); // print overlaod for .V should exit
  } else {
    printf("0");
  }
  printf(".");
  if (_AP_I < _AP_W) {
    ap_int_base<_AP_W - _AP_I, false> p2;
    p2.V = _AP_ROOT_op_get_range(x.V, 0, _AP_W - _AP_I);
    print(p2.V, false); // print overlaod for .V should exit
  }
}
#endif // ifndef __SYNTHESIS__

// XXX the following two functions have to exist in synthesis,
// as some old HLS Video Library code uses the ostream overload,
// although HLS will later delete I/O function call.

/// Output streaming
//-----------------------------------------------------------------------------
// XXX apcc cannot handle global std::ios_base::Init() brought in by <iostream>
#ifndef AP_AUTOCC
#ifndef __SYNTHESIS__
template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O,
          int _AP_N>
INLINE std::ostream& operator<<(
    std::ostream& out,
    const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& x) {
  // TODO support std::ios_base::fmtflags
  unsigned width = out.width();
  unsigned precision = out.precision();
  char fill = out.fill();
  std::string str = x.to_string(10, _AP_S);
  str = reduceToPrecision(str, precision);
  if (width > str.length()) {
    for (unsigned i = 0; i < width - str.length(); ++i)
      out << fill;
  }
  out << str;
  return out;
}
#endif // ifndef __SYNTHESIS__

/// Input streaming
// -----------------------------------------------------------------------------
#ifndef __SYNTHESIS__
template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O,
          int _AP_N>
INLINE std::istream& operator>>(
    std::istream& in,
    ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& x) {
  double d;
  in >> d;
  x = ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>(d);
  return in;
}
#endif
#endif // ifndef AP_AUTOCC

/// Operators mixing Integers with ap_fixed_base
// -----------------------------------------------------------------------------
#define AF_BIN_OP_WITH_INT_SF(BIN_OP, C_TYPE, _AP_W2, _AP_S2, RTYPE)     \
  template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q,           \
            ap_o_mode _AP_O, int _AP_N>                                  \
  INLINE typename ap_fixed_base<_AP_W, _AP_I, _AP_S>::template RType<    \
      _AP_W2, _AP_W2, _AP_S2>::RTYPE                                     \
  operator BIN_OP(                                                       \
      const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op, \
      C_TYPE i_op) {                                                     \
    return op.operator BIN_OP(ap_int_base<_AP_W2, _AP_S2>(i_op));        \
  }

#define AF_BIN_OP_WITH_INT(BIN_OP, C_TYPE, _AP_W2, _AP_S2, RTYPE)           \
  template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q,              \
            ap_o_mode _AP_O, int _AP_N>                                     \
  INLINE typename ap_fixed_base<_AP_W, _AP_I, _AP_S>::template RType<       \
      _AP_W2, _AP_W2, _AP_S2>::RTYPE                                        \
  operator BIN_OP(                                                          \
      const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op,    \
      C_TYPE i_op) {                                                        \
    return op.operator BIN_OP(ap_fixed_base<_AP_W2, _AP_W2, _AP_S2>(i_op)); \
  }                                                                         \
  template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q,              \
            ap_o_mode _AP_O, int _AP_N>                                     \
  INLINE typename ap_fixed_base<_AP_W, _AP_I, _AP_S>::template RType<       \
      _AP_W2, _AP_W2, _AP_S2>::RTYPE                                        \
  operator BIN_OP(                                                          \
      C_TYPE i_op,                                                          \
      const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op) {  \
    return ap_fixed_base<_AP_W2, _AP_W2, _AP_S2>(i_op).operator BIN_OP(op); \
  }

#define AF_REL_OP_WITH_INT(REL_OP, C_TYPE, _AP_W2, _AP_S2)                  \
  template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q,              \
            ap_o_mode _AP_O, int _AP_N>                                     \
  INLINE bool operator REL_OP(                                              \
      const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op,    \
      C_TYPE i_op) {                                                        \
    return op.operator REL_OP(ap_fixed_base<_AP_W2, _AP_W2, _AP_S2>(i_op)); \
  }                                                                         \
  template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q,              \
            ap_o_mode _AP_O, int _AP_N>                                     \
  INLINE bool operator REL_OP(                                              \
      C_TYPE i_op,                                                          \
      const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op) {  \
    return ap_fixed_base<_AP_W2, _AP_W2, _AP_S2>(i_op).operator REL_OP(op); \
  }

#define AF_ASSIGN_OP_WITH_INT(ASSIGN_OP, C_TYPE, _AP_W2, _AP_S2)               \
  template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q,                 \
            ap_o_mode _AP_O, int _AP_N>                                        \
  INLINE ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>&              \
  operator ASSIGN_OP(                                                          \
      ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op,             \
      C_TYPE i_op) {                                                           \
    return op.operator ASSIGN_OP(ap_fixed_base<_AP_W2, _AP_W2, _AP_S2>(i_op)); \
  }

#define AF_ASSIGN_OP_WITH_INT_SF(ASSIGN_OP, C_TYPE, _AP_W2, _AP_S2)  \
  template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q,       \
            ap_o_mode _AP_O, int _AP_N>                              \
  INLINE ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>&    \
  operator ASSIGN_OP(                                                \
      ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op,   \
      C_TYPE i_op) {                                                 \
    return op.operator ASSIGN_OP(ap_int_base<_AP_W2, _AP_S2>(i_op)); \
  }

#define ALL_AF_OP_WITH_INT(C_TYPE, BITS, SIGN)               \
  AF_BIN_OP_WITH_INT(+, C_TYPE, (BITS), (SIGN), plus)     \
  AF_BIN_OP_WITH_INT(-, C_TYPE, (BITS), (SIGN), minus)    \
  AF_BIN_OP_WITH_INT(*, C_TYPE, (BITS), (SIGN), mult)     \
  AF_BIN_OP_WITH_INT(/, C_TYPE, (BITS), (SIGN), div)      \
  AF_BIN_OP_WITH_INT(&, C_TYPE, (BITS), (SIGN), logic)    \
  AF_BIN_OP_WITH_INT(|, C_TYPE, (BITS), (SIGN), logic)    \
  AF_BIN_OP_WITH_INT(^, C_TYPE, (BITS), (SIGN), logic)    \
  AF_BIN_OP_WITH_INT_SF(>>, C_TYPE, (BITS), (SIGN), lhs)  \
  AF_BIN_OP_WITH_INT_SF(<<, C_TYPE, (BITS), (SIGN), lhs)  \
                                                          \
  AF_ASSIGN_OP_WITH_INT(+=, C_TYPE, (BITS), (SIGN))     \
  AF_ASSIGN_OP_WITH_INT(-=, C_TYPE, (BITS), (SIGN))     \
  AF_ASSIGN_OP_WITH_INT(*=, C_TYPE, (BITS), (SIGN))     \
  AF_ASSIGN_OP_WITH_INT(/=, C_TYPE, (BITS), (SIGN))     \
  AF_ASSIGN_OP_WITH_INT(&=, C_TYPE, (BITS), (SIGN))     \
  AF_ASSIGN_OP_WITH_INT(|=, C_TYPE, (BITS), (SIGN))     \
  AF_ASSIGN_OP_WITH_INT(^=, C_TYPE, (BITS), (SIGN))     \
  AF_ASSIGN_OP_WITH_INT_SF(>>=, C_TYPE, (BITS), (SIGN)) \
  AF_ASSIGN_OP_WITH_INT_SF(<<=, C_TYPE, (BITS), (SIGN)) \
                                                          \
  AF_REL_OP_WITH_INT(>, C_TYPE, (BITS), (SIGN))           \
  AF_REL_OP_WITH_INT(<, C_TYPE, (BITS), (SIGN))           \
  AF_REL_OP_WITH_INT(>=, C_TYPE, (BITS), (SIGN))          \
  AF_REL_OP_WITH_INT(<=, C_TYPE, (BITS), (SIGN))          \
  AF_REL_OP_WITH_INT(==, C_TYPE, (BITS), (SIGN))          \
  AF_REL_OP_WITH_INT(!=, C_TYPE, (BITS), (SIGN))

ALL_AF_OP_WITH_INT(bool, 1, false)
ALL_AF_OP_WITH_INT(char, 8, CHAR_IS_SIGNED)
ALL_AF_OP_WITH_INT(signed char, 8, true)
ALL_AF_OP_WITH_INT(unsigned char, 8, false)
ALL_AF_OP_WITH_INT(short, _AP_SIZE_short, true)
ALL_AF_OP_WITH_INT(unsigned short, _AP_SIZE_short, false)
ALL_AF_OP_WITH_INT(int, _AP_SIZE_int, true)
ALL_AF_OP_WITH_INT(unsigned int, _AP_SIZE_int, false)
ALL_AF_OP_WITH_INT(long, _AP_SIZE_long, true)
ALL_AF_OP_WITH_INT(unsigned long, _AP_SIZE_long, false)
ALL_AF_OP_WITH_INT(ap_slong, _AP_SIZE_ap_slong, true)
ALL_AF_OP_WITH_INT(ap_ulong, _AP_SIZE_ap_slong, false)

#undef ALL_AF_OP_WITH_INT
#undef AF_BIN_OP_WITH_INT
#undef AF_BIN_OP_WITH_INT_SF
#undef AF_ASSIGN_OP_WITH_INT
#undef AF_ASSIGN_OP_WITH_INT_SF
#undef AF_REL_OP_WITH_INT

/*
 * **********************************************************************
 * TODO
 * There is no operator defined with float/double/long double, so that
 * code like
 *   ap_fixed<8,4> a = 1.5f;
 *   a += 0.5f;
 * will fail in compilation.
 * Operator with warning about conversion might be wanted.
 * **********************************************************************
 */

#define AF_BIN_OP_WITH_AP_INT(BIN_OP, RTYPE)                                \
  template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q,              \
            ap_o_mode _AP_O, int _AP_N, int _AP_W2, bool _AP_S2>            \
  INLINE typename ap_fixed_base<_AP_W2, _AP_W2, _AP_S2>::template RType<    \
      _AP_W, _AP_I, _AP_S>::RTYPE                                           \
  operator BIN_OP(                                                          \
      const ap_int_base<_AP_W2, _AP_S2>& i_op,                              \
      const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op) {  \
    return ap_fixed_base<_AP_W2, _AP_W2, _AP_S2>(i_op).operator BIN_OP(op); \
  }                                                                         \
                                                                            \
  template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q,              \
            ap_o_mode _AP_O, int _AP_N, int _AP_W2, bool _AP_S2>            \
  INLINE typename ap_fixed_base<_AP_W, _AP_I, _AP_S>::template RType<       \
      _AP_W2, _AP_W2, _AP_S2>::RTYPE                                        \
  operator BIN_OP(                                                          \
      const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op,    \
      const ap_int_base<_AP_W2, _AP_S2>& i_op) {                            \
    return op.operator BIN_OP(ap_fixed_base<_AP_W2, _AP_W2, _AP_S2>(i_op)); \
  }

#define AF_REL_OP_WITH_AP_INT(REL_OP)                                       \
  template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q,              \
            ap_o_mode _AP_O, int _AP_N, int _AP_W2, bool _AP_S2>            \
  INLINE bool operator REL_OP(                                              \
      const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op,    \
      const ap_int_base<_AP_W2, _AP_S2>& i_op) {                            \
    return op.operator REL_OP(ap_fixed_base<_AP_W2, _AP_W2, _AP_S2>(i_op)); \
  }                                                                         \
                                                                            \
  template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q,              \
            ap_o_mode _AP_O, int _AP_N, int _AP_W2, bool _AP_S2>            \
  INLINE bool operator REL_OP(                                              \
      const ap_int_base<_AP_W2, _AP_S2>& i_op,                              \
      const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op) {  \
    return ap_fixed_base<_AP_W2, _AP_W2, _AP_S2>(i_op).operator REL_OP(op); \
  }

#define AF_ASSIGN_OP_WITH_AP_INT(ASSIGN_OP)                                    \
  template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q,                 \
            ap_o_mode _AP_O, int _AP_N, int _AP_W2, bool _AP_S2>               \
  INLINE ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>&              \
  operator ASSIGN_OP(                                                          \
      ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op,             \
      const ap_int_base<_AP_W2, _AP_S2>& i_op) {                               \
    return op.operator ASSIGN_OP(ap_fixed_base<_AP_W2, _AP_W2, _AP_S2>(i_op)); \
  }                                                                            \
                                                                               \
  template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q,                 \
            ap_o_mode _AP_O, int _AP_N, int _AP_W2, bool _AP_S2>               \
  INLINE ap_int_base<_AP_W2, _AP_S2>& operator ASSIGN_OP(                      \
      ap_int_base<_AP_W2, _AP_S2>& i_op,                                       \
      const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op) {     \
    return i_op.operator ASSIGN_OP(op.to_ap_int_base());                       \
  }

AF_BIN_OP_WITH_AP_INT(+, plus)
AF_BIN_OP_WITH_AP_INT(-, minus)
AF_BIN_OP_WITH_AP_INT(*, mult)
AF_BIN_OP_WITH_AP_INT(/, div)
AF_BIN_OP_WITH_AP_INT(&, logic)
AF_BIN_OP_WITH_AP_INT(|, logic)
AF_BIN_OP_WITH_AP_INT(^, logic)

#undef AF_BIN_OP_WITH_AP_INT

AF_ASSIGN_OP_WITH_AP_INT(+=)
AF_ASSIGN_OP_WITH_AP_INT(-=)
AF_ASSIGN_OP_WITH_AP_INT(*=)
AF_ASSIGN_OP_WITH_AP_INT(/=)
AF_ASSIGN_OP_WITH_AP_INT(&=)
AF_ASSIGN_OP_WITH_AP_INT(|=)
AF_ASSIGN_OP_WITH_AP_INT(^=)

#undef AF_ASSIGN_OP_WITH_AP_INT

AF_REL_OP_WITH_AP_INT(==)
AF_REL_OP_WITH_AP_INT(!=)
AF_REL_OP_WITH_AP_INT(>)
AF_REL_OP_WITH_AP_INT(>=)
AF_REL_OP_WITH_AP_INT(<)
AF_REL_OP_WITH_AP_INT(<=)

#undef AF_REL_OP_WITH_AP_INT

// Relational Operators with double
template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O,
          int _AP_N>
INLINE bool operator==(
    double op1,
    const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op2) {
  return op2.operator==(op1);
}

template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O,
          int _AP_N>
INLINE bool operator!=(
    double op1,
    const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op2) {
  return op2.operator!=(op1);
}

template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O,
          int _AP_N>
INLINE bool operator>(
    double op1,
    const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op2) {
  return op2.operator<(op1);
}

template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O,
          int _AP_N>
INLINE bool operator>=(
    double op1,
    const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op2) {
  return op2.operator<=(op1);
}

template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O,
          int _AP_N>
INLINE bool operator<(
    double op1,
    const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op2) {
  return op2.operator>(op1);
}

template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O,
          int _AP_N>
INLINE bool operator<=(
    double op1,
    const ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N>& op2) {
  return op2.operator>=(op1);
}

#endif // ifndef __cplusplus else

#endif // ifndef __AP_FIXED_BASE_H__ else

// -*- cpp -*-
