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

#ifndef __AP_INT_BASE_H__
#define __AP_INT_BASE_H__

#ifndef __AP_INT_H__
#error "Only ap_fixed.h and ap_int.h can be included directly in user code."
#endif

#ifndef __cplusplus
#error "C++ is required to include this header file"
#else

#include <ap_common.h>
#ifndef __SYNTHESIS__
#if _AP_ENABLE_HALF_ == 1
#include <hls_half.h>
#endif
#include <iostream>
#include <string.h>
#endif

/* ----------------------------------------------------------------
 * ap_int_base: AutoPilot integer/Arbitrary precision integer.
 * ----------------------------------------------------------------
 */

/* helper trait. Selecting the smallest C type that can hold the value,
 * return 64 bit C type if not possible.
 */
template <int _AP_N, bool _AP_S>
struct retval;

// at least 64 bit
template <int _AP_N>
struct retval<_AP_N, true> {
  typedef ap_slong Type;
};

template <int _AP_N>
struct retval<_AP_N, false> {
  typedef ap_ulong Type;
};

// at least 8 bit
template <>
struct retval<1, true> {
  typedef signed char Type;
};

template <>
struct retval<1, false> {
  typedef unsigned char Type;
};

// at least 16 bit
template <>
struct retval<2, true> {
  typedef short Type;
};

template <>
struct retval<2, false> {
  typedef unsigned short Type;
};

// at least 32 bit
template <>
struct retval<3, true> {
  typedef long Type;
};

template <>
struct retval<3, false> {
  typedef unsigned long Type;
};

template <>
struct retval<4, true> {
  typedef long Type;
};

template <>
struct retval<4, false> {
  typedef unsigned long Type;
};

// trait for letting base class to return derived class.
// Notice that derived class template is incomplete, and we cannot use
// the member of the derived class.
template <int _AP_W2, bool _AP_S2>
struct _ap_int_factory;
template <int _AP_W2>
struct _ap_int_factory<_AP_W2,true> { typedef ap_int<_AP_W2> type; };
template <int _AP_W2>
struct _ap_int_factory<_AP_W2,false> { typedef ap_uint<_AP_W2> type; };

template <int _AP_W, bool _AP_S>
struct ap_int_base : public _AP_ROOT_TYPE<_AP_W, _AP_S> {
 public:
  typedef _AP_ROOT_TYPE<_AP_W, _AP_S> Base;

  /* ap_int_base<_AP_W, _AP_S, true>
   * typedef typename retval<(_AP_W + 7) / 8, _AP_S>::Type RetType;
   *
   * ap_int_base<_AP_W, _AP_S, false>
   * typedef typename retval<8, _AP_S>::Type RetType;
   */
  typedef typename retval<AP_MAX((_AP_W + 7) / 8, 8), _AP_S>::Type RetType;

  static const int width = _AP_W;

  template <int _AP_W2, bool _AP_S2>
  struct RType {
    enum {
      mult_w = _AP_W + _AP_W2,
      mult_s = _AP_S || _AP_S2,
      plus_w =
          AP_MAX(_AP_W + (_AP_S2 && !_AP_S), _AP_W2 + (_AP_S && !_AP_S2)) + 1,
      plus_s = _AP_S || _AP_S2,
      minus_w =
          AP_MAX(_AP_W + (_AP_S2 && !_AP_S), _AP_W2 + (_AP_S && !_AP_S2)) + 1,
      minus_s = true,
      div_w = _AP_W + _AP_S2,
      div_s = _AP_S || _AP_S2,
      mod_w = AP_MIN(_AP_W, _AP_W2 + (!_AP_S2 && _AP_S)),
      mod_s = _AP_S,
      logic_w = AP_MAX(_AP_W + (_AP_S2 && !_AP_S), _AP_W2 + (_AP_S && !_AP_S2)),
      logic_s = _AP_S || _AP_S2
    };


    typedef ap_int_base<mult_w, mult_s> mult_base;
    typedef ap_int_base<plus_w, plus_s> plus_base;
    typedef ap_int_base<minus_w, minus_s> minus_base;
    typedef ap_int_base<logic_w, logic_s> logic_base;
    typedef ap_int_base<div_w, div_s> div_base;
    typedef ap_int_base<mod_w, mod_s> mod_base;
    typedef ap_int_base<_AP_W, _AP_S> arg1_base;

    typedef typename _ap_int_factory<mult_w, mult_s>::type mult;
    typedef typename _ap_int_factory<plus_w, plus_s>::type plus;
    typedef typename _ap_int_factory<minus_w, minus_s>::type minus;
    typedef typename _ap_int_factory<logic_w, logic_s>::type logic;
    typedef typename _ap_int_factory<div_w, div_s>::type div;
    typedef typename _ap_int_factory<mod_w, mod_s>::type mod;
    typedef typename _ap_int_factory<_AP_W, _AP_S>::type arg1;
    typedef bool reduce;
  };

  /* Constructors.
   * ----------------------------------------------------------------
   */
  /// default ctor
  INLINE ap_int_base() {
    /*
      #ifdef __SC_COMPATIBLE__
      Base::V = 0;
      #endif
    */
  }

  /// copy ctor
  template <int _AP_W2, bool _AP_S2>
  INLINE ap_int_base(const ap_int_base<_AP_W2, _AP_S2>& op) {
    Base::V = op.V;
  }

  /// volatile copy ctor
  template <int _AP_W2, bool _AP_S2>
  INLINE ap_int_base(const volatile ap_int_base<_AP_W2, _AP_S2>& op) {
    Base::V = op.V;
  }

// XXX C++11 feature.
// The explicit specifier specifies that a constructor or conversion function
// (since C++11) doesn't allow implicit conversions or copy-initialization.
//   ap_int_base<W,S> x = 1;
//   ap_int_base<W,S> foo() { return 1; }
// but allows
//   ap_int_base<W,S> x(1);
//   ap_int_base<W,S> y {1};

/// from all c types.
#define CTOR_FROM_INT(Type, Size, Signed) \
  INLINE ap_int_base(const Type op) { Base::V = op; }

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

#if _AP_ENABLE_HALF_ == 1
  /// ctor from half.
  //  TODO optimize
  INLINE ap_int_base(half op) {
    ap_int_base<_AP_W, _AP_S> t((float)op);
    Base::V = t.V;
  }
#endif

  /// ctor from float.
  INLINE ap_int_base(float op) {
    const int BITS = FLOAT_MAN + FLOAT_EXP + 1;
    ap_int_base<BITS, false> reg;
    reg.V = floatToRawBits(op);
    bool is_neg = _AP_ROOT_op_get_bit(reg.V, BITS - 1);

    ap_int_base<FLOAT_EXP + 1, true> exp = 0;
    exp.V = _AP_ROOT_op_get_range(reg.V, FLOAT_MAN, BITS - 2);
    exp = exp - FLOAT_BIAS;

    ap_int_base<FLOAT_MAN + 2, true> man;
    man.V = _AP_ROOT_op_get_range(reg.V, 0, FLOAT_MAN - 1);
    // check for NaN
    _AP_WARNING(exp == ((unsigned char)(FLOAT_BIAS + 1)) && man.V != 0,
                "assign NaN to ap integer value");
    // set leading 1.
    man.V = _AP_ROOT_op_set_bit(man.V, FLOAT_MAN, 1);
    //if (is_neg) man = -man;

    if ((reg.V & 0x7ffffffful) == 0) {
      Base::V = 0;
    } else {
      int sh_amt = FLOAT_MAN - exp.V;
      if (sh_amt == 0) {
        Base::V = man.V;
      } else if (sh_amt > 0) {
        if (sh_amt < FLOAT_MAN + 2) {
          Base::V = man.V >> sh_amt;
        } else {
          if (is_neg)
            Base::V = -1;
          else
            Base::V = 0;
        }
      } else {
        sh_amt = -sh_amt;
        if (sh_amt < _AP_W) {
          Base::V = man.V;
          Base::V <<= sh_amt;
        } else {
          Base::V = 0;
        }
      }
    }
    if (is_neg) *this = -(*this);
  }

  /// ctor from double.
  INLINE ap_int_base(double op) {
    const int BITS = DOUBLE_MAN + DOUBLE_EXP + 1;
    ap_int_base<BITS, false> reg;
    reg.V = doubleToRawBits(op);
    bool is_neg = _AP_ROOT_op_get_bit(reg.V, BITS - 1);

    ap_int_base<DOUBLE_EXP + 1, true> exp = 0;
    exp.V = _AP_ROOT_op_get_range(reg.V, DOUBLE_MAN, BITS - 2);
    exp = exp - DOUBLE_BIAS;

    ap_int_base<DOUBLE_MAN + 2, true> man;
    man.V = _AP_ROOT_op_get_range(reg.V, 0, DOUBLE_MAN - 1);
    // check for NaN
    _AP_WARNING(exp == ((unsigned char)(DOUBLE_BIAS + 1)) && man.V != 0,
                "assign NaN to ap integer value");
    // set leading 1.
    man.V = _AP_ROOT_op_set_bit(man.V, DOUBLE_MAN, 1);
    //if (is_neg) man = -man;

    if ((reg.V & 0x7fffffffffffffffull) == 0) {
      Base::V = 0;
    } else {
      int sh_amt = DOUBLE_MAN - exp.V;
      if (sh_amt == 0) {
        Base::V = man.V;
      } else if (sh_amt > 0) {
        if (sh_amt < DOUBLE_MAN + 2) {
          Base::V = man.V >> sh_amt;
        } else {
          if (is_neg)
            Base::V = -1;
          else
            Base::V = 0;
        }
      } else {
        sh_amt = -sh_amt;
        if (sh_amt < _AP_W) {
          Base::V = man.V;
          Base::V <<= sh_amt;
        } else {
          Base::V = 0;
        }
      }
    }
    if (is_neg) *this = -(*this);
  }

  /// from higer rank type.
  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_int_base(
      const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op) {
    Base::V = op.to_ap_int_base().V;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_int_base(const ap_range_ref<_AP_W2, _AP_S2>& ref) {
    Base::V = (ref.get()).V;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_int_base(const ap_bit_ref<_AP_W2, _AP_S2>& ref) {
    Base::V = ref.operator bool();
  }

  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
  INLINE ap_int_base(const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>& ref) {
    const ap_int_base<ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>::_AP_WR,
                      false>
        tmp = ref.get();
    Base::V = tmp.V;
  }

  /* radix has default value in set */

#ifndef __SYNTHESIS__
  INLINE ap_int_base(const char* s, signed char rd = 0) {
    if (rd == 0)
      rd = guess_radix(s);
    unsigned int length = strlen(s);
    Base::V.fromString(s, length, rd);
  }
#else
  // XXX __builtin_bit_from_string(...) requires const C string and radix.
  INLINE ap_int_base(const char* s) {
    typeof(Base::V) t;
    _ssdm_string2bits((void*)(&t), (const char*)(s), 10, _AP_W, _AP_S,
                      AP_TRN, AP_WRAP, 0, _AP_C99);
    Base::V = t;
  }
  INLINE ap_int_base(const char* s, signed char rd) {
    typeof(Base::V) t;
    _ssdm_string2bits((void*)(&t), (const char*)(s), rd, _AP_W, _AP_S,
                      AP_TRN, AP_WRAP, 0, _AP_C99);
    Base::V = t;
  }
#endif

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_int_base(
      const af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& val) {
    Base::V = (val.get()).V;
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_int_base(
      const af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& val) {
    Base::V = val.operator bool();
  }

  INLINE ap_int_base read() volatile {
    /*AP_DEBUG(printf("call read %d\n", Base::V););*/
    ap_int_base ret;
    ret.V = Base::V;
    return ret;
  }

  INLINE void write(const ap_int_base<_AP_W, _AP_S>& op2) volatile {
    /*AP_DEBUG(printf("call write %d\n", op2.V););*/
    Base::V = op2.V;
  }

  /* Another form of "write".*/
  template <int _AP_W2, bool _AP_S2>
  INLINE void operator=(
      const volatile ap_int_base<_AP_W2, _AP_S2>& op2) volatile {
    Base::V = op2.V;
  }

  INLINE void operator=(
      const volatile ap_int_base<_AP_W, _AP_S>& op2) volatile {
    Base::V = op2.V;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE void operator=(const ap_int_base<_AP_W2, _AP_S2>& op2) volatile {
    Base::V = op2.V;
  }

  INLINE void operator=(const ap_int_base<_AP_W, _AP_S>& op2) volatile {
    Base::V = op2.V;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_int_base& operator=(
      const volatile ap_int_base<_AP_W2, _AP_S2>& op2) {
    Base::V = op2.V;
    return *this;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_int_base& operator=(const ap_int_base<_AP_W2, _AP_S2>& op2) {
    Base::V = op2.V;
    return *this;
  }

  INLINE ap_int_base& operator=(const volatile ap_int_base<_AP_W, _AP_S>& op2) {
    Base::V = op2.V;
    return *this;
  }

  INLINE ap_int_base& operator=(const ap_int_base<_AP_W, _AP_S>& op2) {
    Base::V = op2.V;
    return *this;
  }


#define ASSIGN_OP_FROM_INT(Type, Size, Signed) \
  INLINE ap_int_base& operator=(Type op) {     \
    Base::V = op;                              \
    return *this;                              \
  }

  ASSIGN_OP_FROM_INT(bool, 1, false)
  ASSIGN_OP_FROM_INT(char, 8, CHAR_IS_SIGNED)
  ASSIGN_OP_FROM_INT(signed char, 8, true)
  ASSIGN_OP_FROM_INT(unsigned char, 8, false)
  ASSIGN_OP_FROM_INT(short, _AP_SIZE_short, true)
  ASSIGN_OP_FROM_INT(unsigned short, _AP_SIZE_short, false)
  ASSIGN_OP_FROM_INT(int, _AP_SIZE_int, true)
  ASSIGN_OP_FROM_INT(unsigned int, _AP_SIZE_int, false)
  ASSIGN_OP_FROM_INT(long, _AP_SIZE_long, true)
  ASSIGN_OP_FROM_INT(unsigned long, _AP_SIZE_long, false)
  ASSIGN_OP_FROM_INT(ap_slong, _AP_SIZE_ap_slong, true)
  ASSIGN_OP_FROM_INT(ap_ulong, _AP_SIZE_ap_slong, false)

#undef ASSIGN_OP_FROM_INT

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_int_base& operator=(const ap_bit_ref<_AP_W2, _AP_S2>& op2) {
    Base::V = (bool)op2;
    return *this;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_int_base& operator=(const ap_range_ref<_AP_W2, _AP_S2>& op2) {
    Base::V = (ap_int_base<_AP_W2, false>(op2)).V;
    return *this;
  }

  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
  INLINE ap_int_base& operator=(
      const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>& op2) {
    Base::V = op2.get().V;
    return *this;
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_int_base& operator=(
      const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op) {
    Base::V = op.to_ap_int_base().V;
    return *this;
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_int_base& operator=(
      const af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op) {
    Base::V = (bool)op;
    return *this;
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_int_base& operator=(
      const af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op) {
    Base::V = ((const ap_int_base<_AP_W2, false>)(op)).V;
    return *this;
  }

  // FIXME: UG902 has clearly required user to use to_int() to convert to built-in
  // types, but this implicit conversion is relied on in hls_cordic.h and hls_rsr.h.
  // For example:
  //     int d_exp = fps_x.exp - fps_y.exp;
  INLINE operator RetType() const { return (RetType)(Base::V); }

  /* Explicit conversions to C types.
   * ----------------------------------------------------------------
   */
  INLINE bool to_bool() const { return (bool)(Base::V); }
  INLINE char to_char() const { return (char)(Base::V); }
  INLINE signed char to_schar() const { return (signed char)(Base::V); }
  INLINE unsigned char to_uchar() const { return (unsigned char)(Base::V); }
  INLINE short to_short() const { return (short)(Base::V); }
  INLINE unsigned short to_ushort() const { return (unsigned short)(Base::V); }
  INLINE int to_int() const { return (int)(Base::V); }
  INLINE unsigned to_uint() const { return (unsigned)(Base::V); }
  INLINE long to_long() const { return (long)(Base::V); }
  INLINE unsigned long to_ulong() const { return (unsigned long)(Base::V); }
  INLINE ap_slong to_int64() const { return (ap_slong)(Base::V); }
  INLINE ap_ulong to_uint64() const { return (ap_ulong)(Base::V); }
  INLINE float to_float() const { return (float)(Base::V); }
  INLINE double to_double() const { return (double)(Base::V); }

  // TODO decide if user-defined conversion should be provided.
#if 0
  INLINE operator char() const { return (char)(Base::V); }
  INLINE operator signed char() const { return (signed char)(Base::V); }
  INLINE operator unsigned char() const { return (unsigned char)(Base::V); }
  INLINE operator short() const { return (short)(Base::V); }
  INLINE operator unsigned short() const { return (unsigned short)(Base::V); }
  INLINE operator int() const { return (int)(Base::V); }
  INLINE operator unsigned int () const { return (unsigned)(Base::V); }
  INLINE operator long () const { return (long)(Base::V); }
  INLINE operator unsigned long () const { return (unsigned long)(Base::V); }
  INLINE operator ap_slong () { return (ap_slong)(Base::V); }
  INLINE operator ap_ulong () { return (ap_ulong)(Base::V); }
#endif

  /* Helper methods.
     ----------------------------------------------------------------
  */
  /* we cannot call a non-volatile function on a volatile instance.
   * but calling a volatile function is ok.
   * XXX deleted non-volatile version.
   */
  INLINE int length() const volatile { return _AP_W; }

  /*Return true if the value of ap_int_base instance is zero*/
  INLINE bool iszero() const { return Base::V == 0; }

  /*Return true if the value of ap_int_base instance is zero*/
  INLINE bool is_zero() const { return Base::V == 0; }

  /* x < 0 */
  INLINE bool sign() const {
    if (_AP_S &&
        _AP_ROOT_op_get_bit(Base::V, _AP_W - 1))
      return true;
    else
      return false;
  }

  /* x[i] = 0 */
  INLINE void clear(int i) {
    AP_ASSERT(i >= 0 && i < _AP_W, "position out of range");
    Base::V = _AP_ROOT_op_set_bit(Base::V, i, 0);
  }

  /* x[i] = !x[i]*/
  INLINE void invert(int i) {
    AP_ASSERT(i >= 0 && i < _AP_W, "position out of range");
    bool val = _AP_ROOT_op_get_bit(Base::V, i);
    if (val)
      Base::V = _AP_ROOT_op_set_bit(Base::V, i, 0);
    else
      Base::V = _AP_ROOT_op_set_bit(Base::V, i, 1);
  }

  INLINE bool test(int i) const {
    AP_ASSERT(i >= 0 && i < _AP_W, "position out of range");
    return _AP_ROOT_op_get_bit(Base::V, i);
  }

  // Get self. For ap_concat_ref expansion.
  INLINE ap_int_base& get() { return *this; }

  // Set the ith bit into 1
  INLINE void set(int i) {
    AP_ASSERT(i >= 0 && i < _AP_W, "position out of range");
    Base::V = _AP_ROOT_op_set_bit(Base::V, i, 1);
  }

  // Set the ith bit into v
  INLINE void set(int i, bool v) {
    AP_ASSERT(i >= 0 && i < _AP_W, "position out of range");
    Base::V = _AP_ROOT_op_set_bit(Base::V, i, v);
  }

  // This is used for sc_lv and sc_bv, which is implemented by sc_uint
  // Rotate an ap_int_base object n places to the left
  INLINE ap_int_base& lrotate(int n) {
    AP_ASSERT(n >= 0 && n < _AP_W, "shift value out of range");
    // TODO unify this.
#ifdef __SYNTHESIS__
    typeof(Base::V) l_p = Base::V << n;
    typeof(Base::V) r_p = Base::V >> (_AP_W - n);
    Base::V = l_p | r_p;
#else
    Base::V.lrotate(n);
#endif
    return *this;
  }

  // This is used for sc_lv and sc_bv, which is implemented by sc_uint
  // Rotate an ap_int_base object n places to the right
  INLINE ap_int_base& rrotate(int n) {
    AP_ASSERT(n >= 0 && n < _AP_W, "shift value out of range");
    // TODO unify this.
#ifdef __SYNTHESIS__
    typeof(Base::V) l_p = Base::V << (_AP_W - n);
    typeof(Base::V) r_p = Base::V >> n;
    Base::V = l_p | r_p;
#else
    Base::V.rrotate(n);
#endif
    return *this;
  }

  // Reverse the contents of ap_int_base instance.
  // I.e. LSB becomes MSB and vise versa.
  INLINE ap_int_base& reverse() {
    Base::V = _AP_ROOT_op_get_range(Base::V, _AP_W - 1, 0);
    return *this;
  }

  // Set the ith bit into v
  INLINE void set_bit(int i, bool v) {
    Base::V = _AP_ROOT_op_set_bit(Base::V, i, v);
  }

  // Get the value of ith bit
  INLINE bool get_bit(int i) const {
    return (bool)_AP_ROOT_op_get_bit(Base::V, i);
  }

  // complements every bit
  INLINE void b_not() { Base::V = ~Base::V; }

#define OP_ASSIGN_AP(Sym)                                                    \
  template <int _AP_W2, bool _AP_S2>                                         \
  INLINE ap_int_base& operator Sym(const ap_int_base<_AP_W2, _AP_S2>& op2) { \
    Base::V Sym op2.V;                                                       \
    return *this;                                                            \
  }

  /* Arithmetic assign.
   * ----------------------------------------------------------------
   */
  OP_ASSIGN_AP(*=)
  OP_ASSIGN_AP(+=)
  OP_ASSIGN_AP(-=)
  OP_ASSIGN_AP(/=)
  OP_ASSIGN_AP(%=)
#undef OP_ASSIGN_AP

  /* Bitwise assign: and, or, xor.
   * ----------------------------------------------------------------
   */
#define OP_ASSIGN_AP_CHK(Sym)                                                \
  template <int _AP_W2, bool _AP_S2>                                         \
  INLINE ap_int_base& operator Sym(const ap_int_base<_AP_W2, _AP_S2>& op2) { \
    _AP_WARNING((_AP_W != _AP_W2),                                           \
                "Bitsize mismatch for ap_[u]int" #Sym "ap_[u]int.");         \
    Base::V Sym op2.V;                                                       \
    return *this;                                                            \
  }
  OP_ASSIGN_AP_CHK(&=)
  OP_ASSIGN_AP_CHK(|=)
  OP_ASSIGN_AP_CHK(^=)
#undef OP_ASSIGN_AP_CHK

  /* Prefix increment, decrement.
   * ----------------------------------------------------------------
   */
  INLINE ap_int_base& operator++() {
    operator+=((ap_int_base<1, false>)1);
    return *this;
  }
  INLINE ap_int_base& operator--() {
    operator-=((ap_int_base<1, false>)1);
    return *this;
  }

  /* Postfix increment, decrement
   * ----------------------------------------------------------------
   */
  INLINE const typename RType<_AP_W,_AP_S>::arg1 operator++(int) {
    ap_int_base t = *this;
    operator+=((ap_int_base<1, false>)1);
    return t;
  }
  INLINE const typename RType<_AP_W,_AP_S>::arg1 operator--(int) {
    ap_int_base t = *this;
    operator-=((ap_int_base<1, false>)1);
    return t;
  }

  /* Unary arithmetic.
   * ----------------------------------------------------------------
   */
  INLINE typename RType<_AP_W,_AP_S>::arg1 operator+() const { return *this; }

  // TODO used to be W>64 only... need check.
  INLINE typename RType<1, false>::minus operator-() const {
    return ap_int_base<1, false>(0) - *this;
  }

  /* Not (!)
   * ----------------------------------------------------------------
   */
  INLINE bool operator!() const { return Base::V == 0; }

  /* Bitwise (arithmetic) unary: complement
     ----------------------------------------------------------------
  */
  // XXX different from Mentor's ac_int!
  INLINE typename RType<_AP_W,_AP_S>::arg1 operator~() const {
    ap_int_base<_AP_W, _AP_S> r;
    r.V = ~Base::V;
    return r;
  }

  /* Shift (result constrained by left operand).
   * ----------------------------------------------------------------
   */
  template <int _AP_W2>
  INLINE typename RType<_AP_W,_AP_S>::arg1 operator<<(const ap_int_base<_AP_W2, true>& op2) const {
    bool isNeg = _AP_ROOT_op_get_bit(op2.V, _AP_W2 - 1);
    ap_int_base<_AP_W2, false> sh = op2;
    if (isNeg) {
      sh = -op2;
      return operator>>(sh);
    } else
      return operator<<(sh);
  }

  template <int _AP_W2>
  INLINE typename RType<_AP_W,_AP_S>::arg1 operator<<(const ap_int_base<_AP_W2, false>& op2) const {
    ap_int_base r;
    r.V = Base::V << op2.to_uint();
    return r;
  }

  template <int _AP_W2>
  INLINE typename RType<_AP_W,_AP_S>::arg1 operator>>(const ap_int_base<_AP_W2, true>& op2) const {
    bool isNeg = _AP_ROOT_op_get_bit(op2.V, _AP_W2 - 1);
    ap_int_base<_AP_W2, false> sh = op2;
    if (isNeg) {
      sh = -op2;
      return operator<<(sh);
    }
    return operator>>(sh);
  }

  template <int _AP_W2>
  INLINE typename RType<_AP_W,_AP_S>::arg1 operator>>(const ap_int_base<_AP_W2, false>& op2) const {
    ap_int_base r;
    r.V = Base::V >> op2.to_uint();
    return r;
  }

  // FIXME we standalone operator>> for ap_int_base and ap_range_ref.
#if 0
  template <int _AP_W2, bool _AP_S2>
  INLINE ap_int_base operator<<(const ap_range_ref<_AP_W2, _AP_S2>& op2) const {
    return *this << (op2.operator ap_int_base<_AP_W2, false>());
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_int_base operator>>(const ap_range_ref<_AP_W2, _AP_S2>& op2) const {
    return *this >> (op2.operator ap_int_base<_AP_W2, false>());
  }
#endif

  /* Shift assign
   * ----------------------------------------------------------------
   */
  template <int _AP_W2>
  INLINE ap_int_base& operator<<=(const ap_int_base<_AP_W2, true>& op2) {
    bool isNeg = _AP_ROOT_op_get_bit(op2.V, _AP_W2 - 1);
    ap_int_base<_AP_W2, false> sh = op2;
    if (isNeg) {
      sh = -op2;
      return operator>>=(sh);
    } else
      return operator<<=(sh);
  }

  template <int _AP_W2>
  INLINE ap_int_base& operator<<=(const ap_int_base<_AP_W2, false>& op2) {
    Base::V <<= op2.to_uint();
    return *this;
  }

  template <int _AP_W2>
  INLINE ap_int_base& operator>>=(const ap_int_base<_AP_W2, true>& op2) {
    bool isNeg = _AP_ROOT_op_get_bit(op2.V, _AP_W2 - 1);
    ap_int_base<_AP_W2, false> sh = op2;
    if (isNeg) {
      sh = -op2;
      return operator<<=(sh);
    }
    return operator>>=(sh);
  }

  template <int _AP_W2>
  INLINE ap_int_base& operator>>=(const ap_int_base<_AP_W2, false>& op2) {
    Base::V >>= op2.to_uint();
    return *this;
  }

  // FIXME we standalone operator>> for ap_int_base and ap_range_ref.
#if 0
  template <int _AP_W2, bool _AP_S2>
  INLINE ap_int_base& operator<<=(const ap_range_ref<_AP_W2, _AP_S2>& op2) {
    return *this <<= (op2.operator ap_int_base<_AP_W2, false>());
  }
  template <int _AP_W2, bool _AP_S2>
  INLINE ap_int_base& operator>>=(const ap_range_ref<_AP_W2, _AP_S2>& op2) {
    return *this >>= (op2.operator ap_int_base<_AP_W2, false>());
  }
#endif

  /* Equality and Relational.
   * ----------------------------------------------------------------
   */
  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator==(const ap_int_base<_AP_W2, _AP_S2>& op2) const {
    return Base::V == op2.V;
  }
  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator!=(const ap_int_base<_AP_W2, _AP_S2>& op2) const {
    return !(Base::V == op2.V);
  }
  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator<(const ap_int_base<_AP_W2, _AP_S2>& op2) const {
    return Base::V < op2.V;
  }
  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator>=(const ap_int_base<_AP_W2, _AP_S2>& op2) const {
    return Base::V >= op2.V;
  }
  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator>(const ap_int_base<_AP_W2, _AP_S2>& op2) const {
    return Base::V > op2.V;
  }
  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator<=(const ap_int_base<_AP_W2, _AP_S2>& op2) const {
    return Base::V <= op2.V;
  }

  /* Bit and Part Select
   * ----------------------------------------------------------------
   */
  INLINE ap_range_ref<_AP_W, _AP_S> range(int Hi, int Lo) {
    _AP_ERROR(Hi >= _AP_W, "Hi(%d)out of bound(%d) in range()", Hi, _AP_W);
    _AP_ERROR(Lo >= _AP_W, "Lo(%d)out of bound(%d) in range()", Lo, _AP_W);
    return ap_range_ref<_AP_W, _AP_S>(this, Hi, Lo);
  }

  // This is a must to strip constness to produce reference type.
  INLINE ap_range_ref<_AP_W, _AP_S> range(int Hi, int Lo) const {
    _AP_ERROR(Hi >= _AP_W, "Hi(%d)out of bound(%d) in range()", Hi, _AP_W);
    _AP_ERROR(Lo >= _AP_W, "Lo(%d)out of bound(%d) in range()", Lo, _AP_W);
    return ap_range_ref<_AP_W, _AP_S>(const_cast<ap_int_base*>(this), Hi, Lo);
  }

  template <int _AP_W2, bool _AP_S2, int _AP_W3, bool _AP_S3>
  INLINE ap_range_ref<_AP_W, _AP_S> range(
      const ap_int_base<_AP_W2, _AP_S2>& HiIdx,
      const ap_int_base<_AP_W3, _AP_S3>& LoIdx) {
    int Hi = HiIdx.to_int();
    int Lo = LoIdx.to_int();
    return this->range(Hi, Lo);
  }

  template <int _AP_W2, bool _AP_S2, int _AP_W3, bool _AP_S3>
  INLINE ap_range_ref<_AP_W, _AP_S> range(
      const ap_int_base<_AP_W2, _AP_S2>& HiIdx,
      const ap_int_base<_AP_W3, _AP_S3>& LoIdx) const {
    int Hi = HiIdx.to_int();
    int Lo = LoIdx.to_int();
    return this->range(Hi, Lo);
  }

  INLINE ap_range_ref<_AP_W, _AP_S> range() {
    return this->range(_AP_W - 1, 0);
  }

  INLINE ap_range_ref<_AP_W, _AP_S> range() const {
    return this->range(_AP_W - 1, 0);
  }

  INLINE ap_range_ref<_AP_W, _AP_S> operator()(int Hi, int Lo) {
    return this->range(Hi, Lo);
  }

  INLINE ap_range_ref<_AP_W, _AP_S> operator()(int Hi, int Lo) const {
    return this->range(Hi, Lo);
  }

  template <int _AP_W2, bool _AP_S2, int _AP_W3, bool _AP_S3>
  INLINE ap_range_ref<_AP_W, _AP_S> operator()(
      const ap_int_base<_AP_W2, _AP_S2>& HiIdx,
      const ap_int_base<_AP_W3, _AP_S3>& LoIdx) {
    int Hi = HiIdx.to_int();
    int Lo = LoIdx.to_int();
    return this->range(Hi, Lo);
  }

  template <int _AP_W2, bool _AP_S2, int _AP_W3, bool _AP_S3>
  INLINE ap_range_ref<_AP_W, _AP_S> operator()(
      const ap_int_base<_AP_W2, _AP_S2>& HiIdx,
      const ap_int_base<_AP_W3, _AP_S3>& LoIdx) const {
    int Hi = HiIdx.to_int();
    int Lo = LoIdx.to_int();
    return this->range(Hi, Lo);
  }

#if 0
  template<int Hi, int Lo>
  INLINE ap_int_base<Hi-Lo+1, false> slice() const {
    AP_ASSERT(Hi >= Lo && Hi < _AP_W && Lo < _AP_W, "Out of bounds in slice()");
    ap_int_base<Hi-Lo+1, false> tmp ;
    tmp.V = _AP_ROOT_op_get_range(Base::V, Lo, Hi);
    return tmp;
  }

  INLINE ap_bit_ref<_AP_W,_AP_S> operator [] ( unsigned int uindex) {
    AP_ASSERT(uindex < _AP_W, "Attempting to read bit beyond MSB");
    ap_bit_ref<_AP_W,_AP_S> bvh( this, uindex );
    return bvh;
  }
#endif

  INLINE ap_bit_ref<_AP_W, _AP_S> operator[](int index) {
    AP_ASSERT(index >= 0, "Attempting to read bit with negative index");
    AP_ASSERT(index < _AP_W, "Attempting to read bit beyond MSB");
    ap_bit_ref<_AP_W, _AP_S> bvh(this, index);
    return bvh;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_bit_ref<_AP_W, _AP_S> operator[](
      const ap_int_base<_AP_W2, _AP_S2>& index) {
    AP_ASSERT(index >= 0, "Attempting to read bit with negative index");
    AP_ASSERT(index < _AP_W, "Attempting to read bit beyond MSB");
    ap_bit_ref<_AP_W, _AP_S> bvh(this, index.to_int());
    return bvh;
  }

  INLINE bool operator[](int index) const {
    AP_ASSERT(index >= 0, "Attempting to read bit with negative index");
    AP_ASSERT(index < _AP_W, "Attempting to read bit beyond MSB");
    ap_bit_ref<_AP_W, _AP_S> br(this, index);
    return br.to_bool();
  }
  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator[](const ap_int_base<_AP_W2, _AP_S2>& index) const {
    AP_ASSERT(index < _AP_W, "Attempting to read bit beyond MSB");
    ap_bit_ref<_AP_W, _AP_S> br(this, index.to_int());
    return br.to_bool();
  }

  INLINE ap_bit_ref<_AP_W, _AP_S> bit(int index) {
    AP_ASSERT(index >= 0, "Attempting to read bit with negative index");
    AP_ASSERT(index < _AP_W, "Attempting to read bit beyond MSB");
    ap_bit_ref<_AP_W, _AP_S> bvh(this, index);
    return bvh;
  }
  template <int _AP_W2, bool _AP_S2>
  INLINE ap_bit_ref<_AP_W, _AP_S> bit(
      const ap_int_base<_AP_W2, _AP_S2>& index) {
    AP_ASSERT(index >= 0, "Attempting to read bit with negative index");
    AP_ASSERT(index < _AP_W, "Attempting to read bit beyond MSB");
    ap_bit_ref<_AP_W, _AP_S> bvh(this, index.to_int());
    return bvh;
  }

  INLINE bool bit(int index) const {
    AP_ASSERT(index >= 0, "Attempting to read bit with negative index");
    AP_ASSERT(index < _AP_W, "Attempting to read bit beyond MSB");
    ap_bit_ref<_AP_W, _AP_S> br(this, index);
    return br.to_bool();
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool bit(const ap_int_base<_AP_W2, _AP_S2>& index) const {
    return bit(index.to_int());
  }

#if 0
  template<typename _AP_T>
  INLINE bool operator[](_AP_T index) const {
    AP_ASSERT(index < _AP_W, "Attempting to read bit beyond MSB");
    ap_bit_ref<_AP_W,_AP_S> br = operator[](index);
    return br.to_bool();
  }
#endif

  // Count the number of zeros from the most significant bit
  // to the first one bit.
  INLINE int countLeadingZeros() {
#ifdef __SYNTHESIS__
    if (_AP_W <= 32) {
      ap_int_base<32, false> t(-1UL), x;
      x.V = _AP_ROOT_op_get_range(this->V, _AP_W - 1, 0); // reverse
      t.V = _AP_ROOT_op_set_range(t.V, 0, _AP_W - 1, x.V);
      return __builtin_ctz(t.V); // count trailing zeros.
    } else if (_AP_W <= 64) {
      ap_int_base<64, false> t(-1ULL);
      ap_int_base<64, false> x;
      x.V = _AP_ROOT_op_get_range(this->V, _AP_W - 1, 0); // reverse
      t.V = _AP_ROOT_op_set_range(t.V, 0, _AP_W - 1, x.V);
      return __builtin_ctzll(t.V); // count trailing zeros.
    } else {
      enum { __N = (_AP_W + 63) / 64 };
      int NZeros = 0;
      int i = 0;
      bool hitNonZero = false;
      for (i = 0; i < __N - 1; ++i) {
        ap_int_base<64, false> t;
        t.V = _AP_ROOT_op_get_range(this->V, _AP_W - i * 64 - 64, _AP_W - i * 64 - 1);
        NZeros += hitNonZero ? 0 : __builtin_clzll(t.V); // count leading zeros.
        hitNonZero |= (t.V != 0);
      }
      if (!hitNonZero) {
        ap_int_base<64, false> t(-1ULL);
        enum { REST = (_AP_W - 1) % 64 };
        ap_int_base<64, false> x;
        x.V = _AP_ROOT_op_get_range(this->V, 0, REST);
        t.V = _AP_ROOT_op_set_range(t.V, 63 - REST, 63, x.V);
        NZeros += __builtin_clzll(t.V);
      }
      return NZeros;
    }
#else
    return (Base::V).countLeadingZeros();
#endif
  } // countLeadingZeros

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_concat_ref<_AP_W, ap_int_base, _AP_W2, ap_int_base<_AP_W2, _AP_S2> >
  concat(const ap_int_base<_AP_W2, _AP_S2>& a2) const {
    return ap_concat_ref<_AP_W, ap_int_base, _AP_W2,
                         ap_int_base<_AP_W2, _AP_S2> >(
        const_cast<ap_int_base<_AP_W, _AP_S>&>(*this),
        const_cast<ap_int_base<_AP_W2, _AP_S2>&>(a2));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_concat_ref<_AP_W, ap_int_base, _AP_W2, ap_int_base<_AP_W2, _AP_S2> >
  concat(ap_int_base<_AP_W2, _AP_S2>& a2) {
    return ap_concat_ref<_AP_W, ap_int_base, _AP_W2,
                         ap_int_base<_AP_W2, _AP_S2> >(*this, a2);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE
      ap_concat_ref<_AP_W, ap_int_base, _AP_W2, ap_range_ref<_AP_W2, _AP_S2> >
      operator,(const ap_range_ref<_AP_W2, _AP_S2> &a2) const {
    return ap_concat_ref<_AP_W, ap_int_base, _AP_W2,
                         ap_range_ref<_AP_W2, _AP_S2> >(
        const_cast<ap_int_base<_AP_W, _AP_S>&>(*this),
        const_cast<ap_range_ref<_AP_W2, _AP_S2>&>(a2));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE
      ap_concat_ref<_AP_W, ap_int_base, _AP_W2, ap_range_ref<_AP_W2, _AP_S2> >
      operator,(ap_range_ref<_AP_W2, _AP_S2> &a2) {
    return ap_concat_ref<_AP_W, ap_int_base, _AP_W2,
                         ap_range_ref<_AP_W2, _AP_S2> >(*this, a2);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_concat_ref<_AP_W, ap_int_base, _AP_W2, ap_int_base<_AP_W2, _AP_S2> >
  operator,(const ap_int_base<_AP_W2, _AP_S2> &a2) {
    return ap_concat_ref<_AP_W, ap_int_base, _AP_W2,
                         ap_int_base<_AP_W2, _AP_S2> >(
        *this, const_cast<ap_int_base<_AP_W2, _AP_S2>&>(a2));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_concat_ref<_AP_W, ap_int_base, _AP_W2, ap_int_base<_AP_W2, _AP_S2> >
  operator,(ap_int_base<_AP_W2, _AP_S2> &a2) const {
    return ap_concat_ref<_AP_W, ap_int_base, _AP_W2,
                         ap_int_base<_AP_W2, _AP_S2> >(
        const_cast<ap_int_base<_AP_W, _AP_S>&>(*this), a2);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_concat_ref<_AP_W, ap_int_base, _AP_W2, ap_int_base<_AP_W2, _AP_S2> >
  operator,(const ap_int_base<_AP_W2, _AP_S2> &a2) const {
    return ap_concat_ref<_AP_W, ap_int_base, _AP_W2,
                         ap_int_base<_AP_W2, _AP_S2> >(
        const_cast<ap_int_base<_AP_W, _AP_S>&>(*this),
        const_cast<ap_int_base<_AP_W2, _AP_S2>&>(a2));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_concat_ref<_AP_W, ap_int_base, _AP_W2, ap_int_base<_AP_W2, _AP_S2> >
  operator,(ap_int_base<_AP_W2, _AP_S2> &a2) {
    return ap_concat_ref<_AP_W, ap_int_base, _AP_W2,
                         ap_int_base<_AP_W2, _AP_S2> >(*this, a2);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_concat_ref<_AP_W, ap_int_base, 1, ap_bit_ref<_AP_W2, _AP_S2> >
  operator,(const ap_bit_ref<_AP_W2, _AP_S2> &a2) const {
    return ap_concat_ref<_AP_W, ap_int_base, 1, ap_bit_ref<_AP_W2, _AP_S2> >(
        const_cast<ap_int_base<_AP_W, _AP_S>&>(*this),
        const_cast<ap_bit_ref<_AP_W2, _AP_S2>&>(a2));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_concat_ref<_AP_W, ap_int_base, 1, ap_bit_ref<_AP_W2, _AP_S2> >
  operator,(ap_bit_ref<_AP_W2, _AP_S2> &a2) {
    return ap_concat_ref<_AP_W, ap_int_base, 1, ap_bit_ref<_AP_W2, _AP_S2> >(
        *this, a2);
  }

  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
  INLINE ap_concat_ref<_AP_W, ap_int_base, _AP_W2 + _AP_W3,
                       ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> >
  operator,(const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> &a2) {
    return ap_concat_ref<_AP_W, ap_int_base, _AP_W2 + _AP_W3,
                         ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> >(
        const_cast<ap_int_base<_AP_W, _AP_S>&>(*this),
        const_cast<ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>&>(a2));
  }

  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
  INLINE ap_concat_ref<_AP_W, ap_int_base, _AP_W2 + _AP_W3,
                       ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> >
  operator,(ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> &a2) {
    return ap_concat_ref<_AP_W, ap_int_base, _AP_W2 + _AP_W3,
                         ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> >(*this,
                                                                         a2);
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_concat_ref<
      _AP_W, ap_int_base, _AP_W2,
      af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >
  operator,(const af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>
                &a2) const {
    return ap_concat_ref<
        _AP_W, ap_int_base, _AP_W2,
        af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >(
        const_cast<ap_int_base<_AP_W, _AP_S>&>(*this),
        const_cast<
            af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>&>(a2));
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_concat_ref<
      _AP_W, ap_int_base, _AP_W2,
      af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >
  operator,(af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> &a2) {
    return ap_concat_ref<
        _AP_W, ap_int_base, _AP_W2,
        af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >(*this,
                                                                       a2);
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE
      ap_concat_ref<_AP_W, ap_int_base, 1,
                    af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >
      operator,(const af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>
                    &a2) const {
    return ap_concat_ref<
        _AP_W, ap_int_base, 1,
        af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >(
        const_cast<ap_int_base<_AP_W, _AP_S>&>(*this),
        const_cast<af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>&>(
            a2));
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE
      ap_concat_ref<_AP_W, ap_int_base, 1,
                    af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >
      operator,(
          af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> &a2) {
    return ap_concat_ref<
        _AP_W, ap_int_base, 1,
        af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >(*this, a2);
  }

  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
  INLINE ap_int_base<AP_MAX(_AP_W2 + _AP_W3, _AP_W), _AP_S> operator&(
      const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>& a2) {
    return *this & a2.get();
  }

  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
  INLINE ap_int_base<AP_MAX(_AP_W2 + _AP_W3, _AP_W), _AP_S> operator|(
      const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>& a2) {
    return *this | a2.get();
  }

  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
  INLINE ap_int_base<AP_MAX(_AP_W2 + _AP_W3, _AP_W), _AP_S> operator^(
      const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>& a2) {
    return *this ^ a2.get();
  }

  template <int _AP_W3>
  INLINE void set(const ap_int_base<_AP_W3, false>& val) {
    Base::V = val.V;
  }

  /* Reduce operations.
   * ----------------------------------------------------------------
   */
  // XXX non-const version deleted.
  INLINE bool and_reduce() const { return _AP_ROOT_op_reduce(and, Base::V); }
  INLINE bool nand_reduce() const { return _AP_ROOT_op_reduce(nand, Base::V); }
  INLINE bool or_reduce() const { return _AP_ROOT_op_reduce(or, Base::V); }
  INLINE bool nor_reduce() const { return !(_AP_ROOT_op_reduce(or, Base::V)); }
  INLINE bool xor_reduce() const { return _AP_ROOT_op_reduce (xor, Base::V); }
  INLINE bool xnor_reduce() const {
    return !(_AP_ROOT_op_reduce (xor, Base::V));
  }

  /* Output as a string.
   * ----------------------------------------------------------------
   */
#ifndef __SYNTHESIS__
  std::string to_string(signed char rd = 2, bool sign = _AP_S) const {
    // XXX in autosim/autowrap.tcl "(${name}).to_string(2).c_str()" is used to
    // initialize sc_lv, which seems incapable of handling format "-0b".
    if (rd == 2) sign = false;
    return (Base::V).to_string(rd, sign);
  }
#else
  INLINE char* to_string(signed char rd = 2, bool sign = _AP_S) const {
    return 0;
  }
#endif
}; // struct ap_int_base

// XXX apcc cannot handle global std::ios_base::Init() brought in by <iostream>
#ifndef AP_AUTOCC
#ifndef __SYNTHESIS__
template <int _AP_W, bool _AP_S>
INLINE std::ostream& operator<<(std::ostream& os,
                                const ap_int_base<_AP_W, _AP_S>& x) {
  std::ios_base::fmtflags ff = std::cout.flags();
  if (ff & std::cout.hex) {
    os << x.to_string(16); // don't print sign
  } else if (ff & std::cout.oct) {
    os << x.to_string(8); // don't print sign
  } else {
    os << x.to_string(10);
  }
  return os;
}
#endif // ifndef __SYNTHESIS__

#ifndef __SYNTHESIS__
template <int _AP_W, bool _AP_S>
INLINE std::istream& operator>>(std::istream& in,
                                ap_int_base<_AP_W, _AP_S>& op) {
  std::string str;
  in >> str;
  const std::ios_base::fmtflags basefield = in.flags() & std::ios_base::basefield;
  unsigned radix = (basefield == std::ios_base::dec) ? 0 : (
                     (basefield == std::ios_base::oct) ? 8 : (
                       (basefield == std::ios_base::hex) ? 16 : 0));
  op = ap_int_base<_AP_W, _AP_S>(str.c_str(), radix);
  return in;
}
#endif // ifndef __SYNTHESIS__
#endif // ifndef AP_AUTOCC

/* Operators with another ap_int_base.
 * ----------------------------------------------------------------
 */
#define OP_BIN_AP(Sym, Rty)                                                   \
  template <int _AP_W, bool _AP_S, int _AP_W2, bool _AP_S2>                   \
  INLINE                                                                      \
      typename ap_int_base<_AP_W, _AP_S>::template RType<_AP_W2, _AP_S2>::Rty \
      operator Sym(const ap_int_base<_AP_W, _AP_S>& op,                       \
                   const ap_int_base<_AP_W2, _AP_S2>& op2) {                  \
    typename ap_int_base<_AP_W, _AP_S>::template RType<                       \
        _AP_W2, _AP_S2>::Rty##_base lhs(op);                                  \
    typename ap_int_base<_AP_W, _AP_S>::template RType<                       \
        _AP_W2, _AP_S2>::Rty##_base rhs(op2);                                 \
    typename ap_int_base<_AP_W, _AP_S>::template RType<                       \
        _AP_W2, _AP_S2>::Rty##_base ret;                                      \
    ret.V = lhs.V Sym rhs.V;                                                  \
    return ret;                                                               \
  }

OP_BIN_AP(*, mult)
OP_BIN_AP(+, plus)
OP_BIN_AP(-, minus)
OP_BIN_AP(&, logic)
OP_BIN_AP(|, logic)
OP_BIN_AP(^, logic)

#define OP_BIN_AP2(Sym, Rty)                                                  \
  template <int _AP_W, bool _AP_S, int _AP_W2, bool _AP_S2>                   \
  INLINE                                                                      \
      typename ap_int_base<_AP_W, _AP_S>::template RType<_AP_W2, _AP_S2>::Rty \
      operator Sym(const ap_int_base<_AP_W, _AP_S>& op,                       \
                   const ap_int_base<_AP_W2, _AP_S2>& op2) {                  \
    typename ap_int_base<_AP_W, _AP_S>::template RType<                       \
        _AP_W2, _AP_S2>::Rty##_base ret;                                      \
    ret.V = op.V Sym op2.V;                                                   \
    return ret;                                                               \
  }

OP_BIN_AP2(/, div)
OP_BIN_AP2(%, mod)

// shift operators are defined inside class.
// compound assignment operators are defined inside class.

/* Operators with a pointer type.
 * ----------------------------------------------------------------
 *   char a[100];
 *   char* ptr = a;
 *   ap_int<2> n = 3;
 *   char* ptr2 = ptr + n*2;
 * avoid ambiguous errors.
 */
#define OP_BIN_WITH_PTR(BIN_OP)                                           \
  template <typename PTR_TYPE, int _AP_W, bool _AP_S>                     \
  INLINE PTR_TYPE* operator BIN_OP(PTR_TYPE* i_op,                        \
                                   const ap_int_base<_AP_W, _AP_S>& op) { \
    ap_slong op2 = op.to_int64(); /* Not all implementation */            \
    return i_op BIN_OP op2;                                               \
  }                                                                       \
  template <typename PTR_TYPE, int _AP_W, bool _AP_S>                     \
  INLINE PTR_TYPE* operator BIN_OP(const ap_int_base<_AP_W, _AP_S>& op,   \
                                   PTR_TYPE* i_op) {                      \
    ap_slong op2 = op.to_int64(); /* Not all implementation */            \
    return op2 BIN_OP i_op;                                               \
  }

OP_BIN_WITH_PTR(+)
OP_BIN_WITH_PTR(-)

/* Operators with a native floating point types.
 * ----------------------------------------------------------------
 */
// float OP ap_int
// when ap_int<wa>'s width > 64, then trunc ap_int<w> to ap_int<64>
#define OP_BIN_WITH_FLOAT(BIN_OP, C_TYPE)                              \
  template <int _AP_W, bool _AP_S>                                     \
  INLINE C_TYPE operator BIN_OP(C_TYPE i_op,                           \
                                const ap_int_base<_AP_W, _AP_S>& op) { \
    typename ap_int_base<_AP_W, _AP_S>::RetType op2 = op;              \
    return i_op BIN_OP op2;                                            \
  }                                                                    \
  template <int _AP_W, bool _AP_S>                                     \
  INLINE C_TYPE operator BIN_OP(const ap_int_base<_AP_W, _AP_S>& op,   \
                                C_TYPE i_op) {                         \
    typename ap_int_base<_AP_W, _AP_S>::RetType op2 = op;              \
    return op2 BIN_OP i_op;                                            \
  }

#define ALL_OP_WITH_FLOAT(C_TYPE) \
  OP_BIN_WITH_FLOAT(*, C_TYPE) \
  OP_BIN_WITH_FLOAT(/, C_TYPE) \
  OP_BIN_WITH_FLOAT(+, C_TYPE) \
  OP_BIN_WITH_FLOAT(-, C_TYPE)

#if _AP_ENABLE_HALF_ == 1
ALL_OP_WITH_FLOAT(half)
#endif
ALL_OP_WITH_FLOAT(float)
ALL_OP_WITH_FLOAT(double)

// TODO no shift?

/* Operators with a native integral types.
 * ----------------------------------------------------------------
 */
// arithmetic and bitwise operators.
#define OP_BIN_WITH_INT(BIN_OP, C_TYPE, _AP_W2, _AP_S2, RTYPE)             \
  template <int _AP_W, bool _AP_S>                                         \
  INLINE typename ap_int_base<_AP_W, _AP_S>::template RType<_AP_W2,        \
                                                            _AP_S2>::RTYPE \
  operator BIN_OP(C_TYPE i_op, const ap_int_base<_AP_W, _AP_S>& op) {      \
    return ap_int_base<_AP_W2, _AP_S2>(i_op) BIN_OP(op);                   \
  }                                                                        \
  template <int _AP_W, bool _AP_S>                                         \
  INLINE typename ap_int_base<_AP_W, _AP_S>::template RType<_AP_W2,        \
                                                            _AP_S2>::RTYPE \
  operator BIN_OP(const ap_int_base<_AP_W, _AP_S>& op, C_TYPE i_op) {      \
    return op BIN_OP ap_int_base<_AP_W2, _AP_S2>(i_op);                    \
  }

#define ALL_OP_BIN_WITH_INT(C_TYPE, _AP_W2, _AP_S2)    \
  OP_BIN_WITH_INT(*, C_TYPE, _AP_W2, _AP_S2, mult)  \
  OP_BIN_WITH_INT(+, C_TYPE, _AP_W2, _AP_S2, plus)  \
  OP_BIN_WITH_INT(-, C_TYPE, _AP_W2, _AP_S2, minus) \
  OP_BIN_WITH_INT(/, C_TYPE, _AP_W2, _AP_S2, div)   \
  OP_BIN_WITH_INT(%, C_TYPE, _AP_W2, _AP_S2, mod)   \
  OP_BIN_WITH_INT(&, C_TYPE, _AP_W2, _AP_S2, logic) \
  OP_BIN_WITH_INT(|, C_TYPE, _AP_W2, _AP_S2, logic) \
  OP_BIN_WITH_INT(^, C_TYPE, _AP_W2, _AP_S2, logic)

ALL_OP_BIN_WITH_INT(bool, 1, false)
ALL_OP_BIN_WITH_INT(char, 8, CHAR_IS_SIGNED)
ALL_OP_BIN_WITH_INT(signed char, 8, true)
ALL_OP_BIN_WITH_INT(unsigned char, 8, false)
ALL_OP_BIN_WITH_INT(short, _AP_SIZE_short, true)
ALL_OP_BIN_WITH_INT(unsigned short, _AP_SIZE_short, false)
ALL_OP_BIN_WITH_INT(int, _AP_SIZE_int, true)
ALL_OP_BIN_WITH_INT(unsigned int, _AP_SIZE_int, false)
ALL_OP_BIN_WITH_INT(long, _AP_SIZE_long, true)
ALL_OP_BIN_WITH_INT(unsigned long, _AP_SIZE_long, false)
ALL_OP_BIN_WITH_INT(ap_slong, _AP_SIZE_ap_slong, true)
ALL_OP_BIN_WITH_INT(ap_ulong, _AP_SIZE_ap_slong, false)

#undef OP_BIN_WITH_INT
#undef ALL_OP_BIN_WITH_INT

// shift operators.
#define ALL_OP_SHIFT_WITH_INT(C_TYPE, _AP_W2, _AP_S2)    \
  template <int _AP_W, bool _AP_S>                       \
  INLINE typename ap_int_base<_AP_W, _AP_S>::template RType<_AP_W,_AP_S>::arg1 operator<<(           \
      const ap_int_base<_AP_W, _AP_S>& op, C_TYPE op2) { \
    ap_int_base<_AP_W, _AP_S> r;                         \
    if (_AP_S2)                                          \
      r.V = op2 >= 0 ? (op.V << op2) : (op.V >> (-op2)); \
    else                                                 \
      r.V = op.V << op2;                                 \
    return r;                                            \
  }                                                      \
  template <int _AP_W, bool _AP_S>                       \
  INLINE typename ap_int_base<_AP_W, _AP_S>::template RType<_AP_W,_AP_S>::arg1 operator>>(           \
      const ap_int_base<_AP_W, _AP_S>& op, C_TYPE op2) { \
    ap_int_base<_AP_W, _AP_S> r;                         \
    if (_AP_S2)                                          \
      r.V = op2 >= 0 ? (op.V >> op2) : (op.V << (-op2)); \
    else                                                 \
      r.V = op.V >> op2;                                 \
    return r;                                            \
  }

ALL_OP_SHIFT_WITH_INT(char, 8, CHAR_IS_SIGNED)
ALL_OP_SHIFT_WITH_INT(signed char, 8, true)
ALL_OP_SHIFT_WITH_INT(short, _AP_SIZE_short, true)
ALL_OP_SHIFT_WITH_INT(int, _AP_SIZE_int, true)
ALL_OP_SHIFT_WITH_INT(long, _AP_SIZE_long, true)
ALL_OP_SHIFT_WITH_INT(ap_slong, _AP_SIZE_ap_slong, true)

#undef ALL_OP_SHIFT_WITH_INT

#define ALL_OP_SHIFT_WITH_INT(C_TYPE, _AP_W2, _AP_S2)    \
  template <int _AP_W, bool _AP_S>                       \
  INLINE typename ap_int_base<_AP_W, _AP_S>::template RType<_AP_W,_AP_S>::arg1 operator<<(           \
      const ap_int_base<_AP_W, _AP_S>& op, C_TYPE op2) { \
    ap_int_base<_AP_W, _AP_S> r;                         \
    r.V = op.V << op2;                                   \
    return r;                                            \
  }                                                      \
  template <int _AP_W, bool _AP_S>                       \
  INLINE typename ap_int_base<_AP_W, _AP_S>::template RType<_AP_W,_AP_S>::arg1 operator>>(           \
      const ap_int_base<_AP_W, _AP_S>& op, C_TYPE op2) { \
    ap_int_base<_AP_W, _AP_S> r;                         \
    r.V = op.V >> op2;                                   \
    return r;                                            \
  }
ALL_OP_SHIFT_WITH_INT(bool, 1, false)
ALL_OP_SHIFT_WITH_INT(unsigned char, 8, false)
ALL_OP_SHIFT_WITH_INT(unsigned short, _AP_SIZE_short, false)
ALL_OP_SHIFT_WITH_INT(unsigned int, _AP_SIZE_int, false)
ALL_OP_SHIFT_WITH_INT(unsigned long, _AP_SIZE_long, false)
ALL_OP_SHIFT_WITH_INT(ap_ulong, _AP_SIZE_ap_slong, false)

#undef ALL_OP_SHIFT_WITH_INT

// compound assign operators.
#define OP_ASSIGN_WITH_INT(ASSIGN_OP, C_TYPE, _AP_W2, _AP_S2)       \
  template <int _AP_W, bool _AP_S>                                  \
  INLINE ap_int_base<_AP_W, _AP_S>& operator ASSIGN_OP(             \
      ap_int_base<_AP_W, _AP_S>& op, C_TYPE op2) {                  \
    return op ASSIGN_OP ap_int_base<_AP_W2, _AP_S2>(op2);           \
  }

// TODO int a; ap_int<16> b; a += b;

#define ALL_OP_ASSIGN_WITH_INT(C_TYPE, _AP_W2, _AP_S2) \
  OP_ASSIGN_WITH_INT(+=, C_TYPE, _AP_W2, _AP_S2)       \
  OP_ASSIGN_WITH_INT(-=, C_TYPE, _AP_W2, _AP_S2)       \
  OP_ASSIGN_WITH_INT(*=, C_TYPE, _AP_W2, _AP_S2)       \
  OP_ASSIGN_WITH_INT(/=, C_TYPE, _AP_W2, _AP_S2)       \
  OP_ASSIGN_WITH_INT(%=, C_TYPE, _AP_W2, _AP_S2)       \
  OP_ASSIGN_WITH_INT(&=, C_TYPE, _AP_W2, _AP_S2)       \
  OP_ASSIGN_WITH_INT(|=, C_TYPE, _AP_W2, _AP_S2)       \
  OP_ASSIGN_WITH_INT(^=, C_TYPE, _AP_W2, _AP_S2)       \
  OP_ASSIGN_WITH_INT(>>=, C_TYPE, _AP_W2, _AP_S2)      \
  OP_ASSIGN_WITH_INT(<<=, C_TYPE, _AP_W2, _AP_S2)

ALL_OP_ASSIGN_WITH_INT(bool, 1, false)
ALL_OP_ASSIGN_WITH_INT(char, 8, CHAR_IS_SIGNED)
ALL_OP_ASSIGN_WITH_INT(signed char, 8, true)
ALL_OP_ASSIGN_WITH_INT(unsigned char, 8, false)
ALL_OP_ASSIGN_WITH_INT(short, _AP_SIZE_short, true)
ALL_OP_ASSIGN_WITH_INT(unsigned short, _AP_SIZE_short, false)
ALL_OP_ASSIGN_WITH_INT(int, _AP_SIZE_int, true)
ALL_OP_ASSIGN_WITH_INT(unsigned int, _AP_SIZE_int, false)
ALL_OP_ASSIGN_WITH_INT(long, _AP_SIZE_long, true)
ALL_OP_ASSIGN_WITH_INT(unsigned long, _AP_SIZE_long, false)
ALL_OP_ASSIGN_WITH_INT(ap_slong, _AP_SIZE_ap_slong, true)
ALL_OP_ASSIGN_WITH_INT(ap_ulong, _AP_SIZE_ap_slong, false)

#undef OP_ASSIGN_WITH_INT
#undef ALL_OP_ASSIGN_WITH_INT

// equality and relational operators.
#define OP_REL_WITH_INT(REL_OP, C_TYPE, _AP_W2, _AP_S2)              \
  template <int _AP_W, bool _AP_S>                                   \
  INLINE bool operator REL_OP(C_TYPE i_op,                           \
                              const ap_int_base<_AP_W, _AP_S>& op) { \
    return ap_int_base<_AP_W2, _AP_S2>(i_op) REL_OP op;              \
  }                                                                  \
  template <int _AP_W, bool _AP_S>                                   \
  INLINE bool operator REL_OP(const ap_int_base<_AP_W, _AP_S>& op,   \
                              C_TYPE op2) {                          \
    return op REL_OP ap_int_base<_AP_W2, _AP_S2>(op2);               \
  }

#define ALL_OP_REL_WITH_INT(C_TYPE, _AP_W2, _AP_S2) \
  OP_REL_WITH_INT(>, C_TYPE, _AP_W2, _AP_S2)        \
  OP_REL_WITH_INT(<, C_TYPE, _AP_W2, _AP_S2)        \
  OP_REL_WITH_INT(>=, C_TYPE, _AP_W2, _AP_S2)       \
  OP_REL_WITH_INT(<=, C_TYPE, _AP_W2, _AP_S2)       \
  OP_REL_WITH_INT(==, C_TYPE, _AP_W2, _AP_S2)       \
  OP_REL_WITH_INT(!=, C_TYPE, _AP_W2, _AP_S2)

ALL_OP_REL_WITH_INT(bool, 1, false)
ALL_OP_REL_WITH_INT(char, 8, CHAR_IS_SIGNED)
ALL_OP_REL_WITH_INT(signed char, 8, true)
ALL_OP_REL_WITH_INT(unsigned char, 8, false)
ALL_OP_REL_WITH_INT(short, _AP_SIZE_short, true)
ALL_OP_REL_WITH_INT(unsigned short, _AP_SIZE_short, false)
ALL_OP_REL_WITH_INT(int, _AP_SIZE_int, true)
ALL_OP_REL_WITH_INT(unsigned int, _AP_SIZE_int, false)
ALL_OP_REL_WITH_INT(long, _AP_SIZE_long, true)
ALL_OP_REL_WITH_INT(unsigned long, _AP_SIZE_long, false)
ALL_OP_REL_WITH_INT(ap_slong, _AP_SIZE_ap_slong, true)
ALL_OP_REL_WITH_INT(ap_ulong, _AP_SIZE_ap_slong, false)

#undef OP_REL_WITH_INT
#undef ALL_OP_BIN_WITH_INT

#define OP_REL_WITH_DOUBLE_OR_FLOAT(Sym)                            \
  template <int _AP_W, bool _AP_S>                                  \
  INLINE bool operator Sym(const ap_int_base<_AP_W, _AP_S>& op1,    \
                           double op2) {                            \
    return op1.to_double() Sym op2 ;                                \
  }                                                                 \
  template <int _AP_W, bool _AP_S>                                  \
  INLINE bool operator Sym(double op1,                              \
                           const ap_int_base<_AP_W, _AP_S>& op2) {  \
    return op1 Sym op2.to_double() ;                                \
  }                                                                 \
  template <int _AP_W, bool _AP_S>                                  \
  INLINE bool operator Sym(const ap_int_base<_AP_W, _AP_S>& op1,    \
                           float op2) {                             \
    return op1.to_double() Sym op2 ;                                \
  }                                                                 \
  template <int _AP_W, bool _AP_S>                                  \
  INLINE bool operator Sym(float op1,                               \
                           const ap_int_base<_AP_W, _AP_S>& op2) {  \
    return op1 Sym op2.to_double() ;                                \
  }
  OP_REL_WITH_DOUBLE_OR_FLOAT(>)
  OP_REL_WITH_DOUBLE_OR_FLOAT(<)
  OP_REL_WITH_DOUBLE_OR_FLOAT(>=)
  OP_REL_WITH_DOUBLE_OR_FLOAT(<=)
  OP_REL_WITH_DOUBLE_OR_FLOAT(==)
  OP_REL_WITH_DOUBLE_OR_FLOAT(!=)

#undef OP_REL_WITH_DOUBLE_OR_FLOAT


/* Operators with ap_bit_ref.
 * ------------------------------------------------------------
 */
// arithmetic, bitwise and shift operators.
#define OP_BIN_WITH_RANGE(BIN_OP, RTYPE)                                     \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>                \
  INLINE typename ap_int_base<_AP_W1, _AP_S1>::template RType<_AP_W2,        \
                                                              _AP_S2>::RTYPE \
  operator BIN_OP(const ap_range_ref<_AP_W1, _AP_S1>& op1,                   \
                  const ap_int_base<_AP_W2, _AP_S2>& op2) {                  \
    return ap_int_base<_AP_W1, false>(op1) BIN_OP op2;                       \
  }                                                                          \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>                \
  INLINE typename ap_int_base<_AP_W1, _AP_S1>::template RType<_AP_W2,        \
                                                              _AP_S2>::RTYPE \
  operator BIN_OP(const ap_int_base<_AP_W1, _AP_S1>& op1,                    \
                  const ap_range_ref<_AP_W2, _AP_S2>& op2) {                 \
    return op1 BIN_OP ap_int_base<_AP_W2, false>(op2);                       \
  }

OP_BIN_WITH_RANGE(+, plus)
OP_BIN_WITH_RANGE(-, minus)
OP_BIN_WITH_RANGE(*, mult)
OP_BIN_WITH_RANGE(/, div)
OP_BIN_WITH_RANGE(%, mod)
OP_BIN_WITH_RANGE(&, logic)
OP_BIN_WITH_RANGE(|, logic)
OP_BIN_WITH_RANGE(^, logic)
OP_BIN_WITH_RANGE(>>, arg1)
OP_BIN_WITH_RANGE(<<, arg1)

#undef OP_BIN_WITH_RANGE

// compound assignment operators.
#define OP_ASSIGN_WITH_RANGE(ASSIGN_OP)                                      \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>                \
  INLINE ap_int_base<_AP_W1, _AP_S1>& operator ASSIGN_OP(                    \
      ap_int_base<_AP_W1, _AP_S1>& op1, ap_range_ref<_AP_W2, _AP_S2>& op2) { \
    return op1 ASSIGN_OP ap_int_base<_AP_W2, false>(op2);                    \
  }                                                                          \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>                \
  INLINE ap_range_ref<_AP_W1, _AP_S1>& operator ASSIGN_OP(                   \
      ap_range_ref<_AP_W1, _AP_S1>& op1, ap_int_base<_AP_W2, _AP_S2>& op2) { \
    ap_int_base<_AP_W1, false> tmp(op1);                                     \
    tmp ASSIGN_OP op2;                                                       \
    op1 = tmp;                                                               \
    return op1;                                                              \
  }

OP_ASSIGN_WITH_RANGE(+=)
OP_ASSIGN_WITH_RANGE(-=)
OP_ASSIGN_WITH_RANGE(*=)
OP_ASSIGN_WITH_RANGE(/=)
OP_ASSIGN_WITH_RANGE(%=)
OP_ASSIGN_WITH_RANGE(&=)
OP_ASSIGN_WITH_RANGE(|=)
OP_ASSIGN_WITH_RANGE(^=)
OP_ASSIGN_WITH_RANGE(>>=)
OP_ASSIGN_WITH_RANGE(<<=)

#undef OP_ASSIGN_WITH_RANGE

// equality and relational operators
#define OP_REL_WITH_RANGE(REL_OP)                                          \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>              \
  INLINE bool operator REL_OP(const ap_range_ref<_AP_W1, _AP_S1>& op1,     \
                              const ap_int_base<_AP_W2, _AP_S2>& op2) {    \
    return ap_int_base<_AP_W1, false>(op1).operator REL_OP(op2);           \
  }                                                                        \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>              \
  INLINE bool operator REL_OP(const ap_int_base<_AP_W1, _AP_S1>& op1,      \
                              const ap_range_ref<_AP_W2, _AP_S2>& op2) {   \
    return op1.operator REL_OP(op2.operator ap_int_base<_AP_W2, false>()); \
  }

OP_REL_WITH_RANGE(==)
OP_REL_WITH_RANGE(!=)
OP_REL_WITH_RANGE(>)
OP_REL_WITH_RANGE(>=)
OP_REL_WITH_RANGE(<)
OP_REL_WITH_RANGE(<=)

#undef OP_REL_WITH_RANGE

/* Operators with ap_bit_ref.
 * ------------------------------------------------------------
 */
// arithmetic, bitwise and shift operators.
#define OP_BIN_WITH_BIT(BIN_OP, RTYPE)                                         \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>                  \
  INLINE typename ap_int_base<_AP_W1, _AP_S1>::template RType<1, false>::RTYPE \
  operator BIN_OP(const ap_int_base<_AP_W1, _AP_S1>& op1,                      \
                  const ap_bit_ref<_AP_W2, _AP_S2>& op2) {                     \
    return op1 BIN_OP ap_int_base<1, false>(op2);                              \
  }                                                                            \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>                  \
  INLINE typename ap_int_base<1, false>::template RType<_AP_W2, _AP_S2>::RTYPE \
  operator BIN_OP(const ap_bit_ref<_AP_W1, _AP_S1>& op1,                       \
                  const ap_int_base<_AP_W2, _AP_S2>& op2) {                    \
    return ap_int_base<1, false>(op1) BIN_OP op2;                              \
  }

OP_BIN_WITH_BIT(+, plus)
OP_BIN_WITH_BIT(-, minus)
OP_BIN_WITH_BIT(*, mult)
OP_BIN_WITH_BIT(/, div)
OP_BIN_WITH_BIT(%, mod)
OP_BIN_WITH_BIT(&, logic)
OP_BIN_WITH_BIT(|, logic)
OP_BIN_WITH_BIT(^, logic)
OP_BIN_WITH_BIT(>>, arg1)
OP_BIN_WITH_BIT(<<, arg1)

#undef OP_BIN_WITH_BIT

// compound assignment operators.
#define OP_ASSIGN_WITH_BIT(ASSIGN_OP)                                      \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>              \
  INLINE ap_int_base<_AP_W1, _AP_S1>& operator ASSIGN_OP(                  \
      ap_int_base<_AP_W1, _AP_S1>& op1, ap_bit_ref<_AP_W2, _AP_S2>& op2) { \
    return op1 ASSIGN_OP ap_int_base<1, false>(op2);                       \
  }                                                                        \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>              \
  INLINE ap_bit_ref<_AP_W1, _AP_S1>& operator ASSIGN_OP(                   \
      ap_bit_ref<_AP_W1, _AP_S1>& op1, ap_int_base<_AP_W2, _AP_S2>& op2) { \
    ap_int_base<1, false> tmp(op1);                                        \
    tmp ASSIGN_OP op2;                                                     \
    op1 = tmp;                                                             \
    return op1;                                                            \
  }

OP_ASSIGN_WITH_BIT(+=)
OP_ASSIGN_WITH_BIT(-=)
OP_ASSIGN_WITH_BIT(*=)
OP_ASSIGN_WITH_BIT(/=)
OP_ASSIGN_WITH_BIT(%=)
OP_ASSIGN_WITH_BIT(&=)
OP_ASSIGN_WITH_BIT(|=)
OP_ASSIGN_WITH_BIT(^=)
OP_ASSIGN_WITH_BIT(>>=)
OP_ASSIGN_WITH_BIT(<<=)

#undef OP_ASSIGN_WITH_BIT

// equality and relational operators.
#define OP_REL_WITH_BIT(REL_OP)                                         \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>           \
  INLINE bool operator REL_OP(const ap_int_base<_AP_W1, _AP_S1>& op1,   \
                              const ap_bit_ref<_AP_W2, _AP_S2>& op2) {  \
    return op1 REL_OP ap_int_base<1, false>(op2);                       \
  }                                                                     \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>           \
  INLINE bool operator REL_OP(const ap_bit_ref<_AP_W1, _AP_S1>& op1,    \
                              const ap_int_base<_AP_W2, _AP_S2>& op2) { \
    return ap_int_base<1, false>(op1) REL_OP op2;                       \
  }

OP_REL_WITH_BIT(==)
OP_REL_WITH_BIT(!=)
OP_REL_WITH_BIT(>)
OP_REL_WITH_BIT(>=)
OP_REL_WITH_BIT(<)
OP_REL_WITH_BIT(<=)

#undef OP_REL_WITH_BIT


/* Operators with ap_concat_ref.
 * ------------------------------------------------------------
 */
// arithmetic, bitwise and shift operators.
// bitwise operators are defined in struct.
// TODO specify whether to define arithmetic and bitwise operators.
#if 0
#define OP_BIN_WITH_CONCAT(BIN_OP, RTYPE)                                      \
  template <int _AP_W1, typename _AP_T1, int _AP_W2, typename _AP_T2,          \
            int _AP_W3, bool _AP_S3>                                           \
  INLINE typename ap_int_base<_AP_W3, _AP_S3>::template RType<_AP_W1 + _AP_W2, \
                                                              false>::RTYPE    \
  operator BIN_OP(const ap_int_base<_AP_W3, _AP_S3>& op1,                      \
                  const ap_concat_ref<_AP_W1, _AP_T1, _AP_W2, _AP_T2>& op2) {  \
    /* convert ap_concat_ref to ap_int_base */                                 \
    return op1 BIN_OP op2.get();                                               \
  }                                                                            \
  template <int _AP_W1, typename _AP_T1, int _AP_W2, typename _AP_T2,          \
            int _AP_W3, bool _AP_S3>                                           \
  INLINE typename ap_int_base<_AP_W1 + _AP_W2,                                 \
                              false>::template RType<_AP_W3, _AP_S3>::RTYPE    \
  operator BIN_OP(const ap_concat_ref<_AP_W1, _AP_T1, _AP_W2, _AP_T2>& op1,    \
                  const ap_int_base<_AP_W3, _AP_S3>& op2) {                    \
    /* convert ap_concat_ref to ap_int_base */                                 \
    return op1.get() BIN_OP op2;                                               \
  }

OP_BIN_WITH_CONCAT(+, plus)
OP_BIN_WITH_CONCAT(-, minus)
OP_BIN_WITH_CONCAT(*, mult)
OP_BIN_WITH_CONCAT(/, div)
OP_BIN_WITH_CONCAT(%, mod)
OP_BIN_WITH_CONCAT(&, logic)
OP_BIN_WITH_CONCAT(|, logic)
OP_BIN_WITH_CONCAT(^, logic)
OP_BIN_WITH_CONCAT(>>, arg1)
OP_BIN_WITH_CONCAT(<<, arg1)

#undef OP_BIN_WITH_CONCAT

// compound assignment operators.
#define OP_ASSIGN_WITH_CONCAT(ASSIGN_OP)                                       \
  template <int _AP_W1, typename _AP_T1, int _AP_W2, typename _AP_T2,          \
            int _AP_W3, bool _AP_S3>                                           \
  INLINE typename ap_int_base<_AP_W3, _AP_S3>::template RType<_AP_W1 + _AP_W2, \
                                                              false>::RTYPE    \
  operator ASSIGN_OP(                                                          \
      const ap_int_base<_AP_W3, _AP_S3>& op1,                                  \
      const ap_concat_ref<_AP_W1, _AP_T1, _AP_W2, _AP_T2>& op2) {              \
    /* convert ap_concat_ref to ap_int_base */                                 \
    return op1 ASSIGN_OP op2.get();                                            \
  }                                                                            \
  template <int _AP_W1, typename _AP_T1, int _AP_W2, typename _AP_T2,          \
            int _AP_W3, bool _AP_S3>                                           \
  INLINE typename ap_int_base<_AP_W1 + _AP_W2,                                 \
                              false>::template RType<_AP_W3, _AP_S3>::RTYPE    \
  operator ASSIGN_OP(const ap_concat_ref<_AP_W1, _AP_T1, _AP_W2, _AP_T2>& op1, \
                     const ap_int_base<_AP_W3, _AP_S3>& op2) {                 \
    /* convert ap_concat_ref to ap_int_base */                                 \
    ap_int_base<_AP_W1 + _AP_W2, false> tmp = op1.get();                       \
    tmp ASSIGN_OP op2;                                                         \
    op1 = tmp;                                                                 \
    return op1;                                                                \
  }

OP_ASSIGN_WITH_CONCAT(+=)
OP_ASSIGN_WITH_CONCAT(-=)
OP_ASSIGN_WITH_CONCAT(*=)
OP_ASSIGN_WITH_CONCAT(/=)
OP_ASSIGN_WITH_CONCAT(%=)
OP_ASSIGN_WITH_CONCAT(&=)
OP_ASSIGN_WITH_CONCAT(|=)
OP_ASSIGN_WITH_CONCAT(^=)
OP_ASSIGN_WITH_CONCAT(>>=)
OP_ASSIGN_WITH_CONCAT(<<=)

#undef OP_ASSIGN_WITH_CONCAT
#endif

// equality and relational operators.
#define OP_REL_WITH_CONCAT(REL_OP)                                    \
  template <int _AP_W1, typename _AP_T1, int _AP_W2, typename _AP_T2, \
            int _AP_W3, bool _AP_S3>                                  \
  INLINE bool operator REL_OP(                                        \
      const ap_int_base<_AP_W3, _AP_S3>& op1,                         \
      const ap_concat_ref<_AP_W1, _AP_T1, _AP_W2, _AP_T2>& op2) {     \
    /* convert ap_concat_ref to ap_int_base */                        \
    return op1 REL_OP op2.get();                                      \
  }                                                                   \
  template <int _AP_W1, typename _AP_T1, int _AP_W2, typename _AP_T2, \
            int _AP_W3, bool _AP_S3>                                  \
  INLINE bool operator REL_OP(                                        \
      const ap_concat_ref<_AP_W1, _AP_T1, _AP_W2, _AP_T2>& op1,       \
      const ap_int_base<_AP_W3, _AP_S3>& op2) {                       \
    /* convert ap_concat_ref to ap_int_base */                        \
    return op1.get() REL_OP op2;                                      \
  }

OP_REL_WITH_CONCAT(==)
OP_REL_WITH_CONCAT(!=)
OP_REL_WITH_CONCAT(>)
OP_REL_WITH_CONCAT(>=)
OP_REL_WITH_CONCAT(<)
OP_REL_WITH_CONCAT(<=)

#undef OP_REL_WITH_CONCAT

#endif // ifndef __cplusplus
#endif // ifndef __AP_INT_BASE_H__

// -*- cpp -*-
