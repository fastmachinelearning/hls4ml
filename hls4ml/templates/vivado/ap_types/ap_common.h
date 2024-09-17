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

#ifndef __AP_COMMON_H__
#define __AP_COMMON_H__

// ----------------------------------------------------------------------

// Forward declaration of all AP types.
#include <ap_decl.h>


#ifdef __SYNTHESIS__
#error "The open-source version of AP types does not support synthesis."
#endif // ifdef __SYNTHESIS__
#define _AP_ENABLE_HALF_ 0


#if _AP_ENABLE_HALF_ == 1
// Before ap_private definition.
#ifdef __SYNTHESIS__
#define _HLS_HALF_DEFINED_
typedef __fp16 half;
#else
class half;
#endif // __SYNTHESIS__
#endif // _AP_ENABLE_HALF_

// ----------------------------------------------------------------------

// Macro functions
#define AP_MAX(a, b) ((a) > (b) ? (a) : (b))
#define AP_MIN(a, b) ((a) < (b) ? (a) : (b))
#define AP_ABS(a) ((a) >= 0 ? (a) : -(a))

#ifndef AP_ASSERT
#ifndef __SYNTHESIS__
#include <assert.h>
#define AP_ASSERT(cond, msg) assert((cond) && (msg))
#else
#define AP_ASSERT(cond, msg)
#endif // ifndef __SYNTHESIS__
#endif // ifndef AP_ASSERT

#ifndef __SYNTHESIS__
// for fprintf messages.
#include <stdio.h>
// for exit on error.
#include <stdlib.h>
#endif

// same disable condition as assert.
#if !defined(__SYNTHESIS__) && !defined(NDEBUG)

#define _AP_DEBUG(cond, ...)                  \
  do {                                        \
    if ((cond)) {                             \
      fprintf(stderr, "DEBUG: " __VA_ARGS__); \
      fprintf(stderr, "\n");                  \
    }                                         \
  } while (0)
#define _AP_WARNING(cond, ...)                  \
  do {                                          \
    if ((cond)) {                               \
      fprintf(stderr, "WARNING: " __VA_ARGS__); \
      fprintf(stderr, "\n");                    \
    }                                           \
  } while (0)
#define _AP_ERROR(cond, ...)                  \
  do {                                        \
    if ((cond)) {                             \
      fprintf(stderr, "ERROR: " __VA_ARGS__); \
      fprintf(stderr, "\n");                  \
      abort();                                \
    }                                         \
  } while (0)

#else // if !defined(__SYNTHESIS__) && !defined(NDEBUG)

#define __AP_VOID_CAST static_cast<void>
#define _AP_DEBUG(cond, ...) (__AP_VOID_CAST(0))
#define _AP_WARNING(cond, ...) (__AP_VOID_CAST(0))
#define _AP_ERROR(cond, ...) (__AP_VOID_CAST(0))

#endif // if !defined(__SYNTHESIS__) && !defined(NDEBUG) else

// ----------------------------------------------------------------------

// Attribute only for synthesis
#ifdef __SYNTHESIS__
#define INLINE inline __attribute__((always_inline))
//#define INLINE inline __attribute__((noinline))
#else
#define INLINE inline
#endif

#define AP_WEAK
// __attribute__((weak))

#ifndef AP_INT_MAX_W
#define AP_INT_MAX_W 1024
#endif

#define BIT_WIDTH_UPPER_LIMIT (1 << 15)
#if AP_INT_MAX_W > BIT_WIDTH_UPPER_LIMIT
#error "Bitwidth exceeds 32768 (1 << 15), the maximum allowed value"
#endif

#define MAX_MODE(BITS) ((BITS + 1023) / 1024)

// ----------------------------------------------------------------------

// XXX apcc cannot handle global std::ios_base::Init() brought in by <iostream>
#ifndef AP_AUTOCC
#ifndef __SYNTHESIS__
// for overload operator<<
#include <iostream>
#endif
#endif // ifndef AP_AUTOCC

#ifndef __SYNTHESIS__
// for string format.
#include <sstream>
// for string.
#include <string>
#endif

// for detecting if char is signed.
enum { CHAR_IS_SIGNED = (char)-1 < 0 };

// TODO we have similar traits in x_hls_utils.h, should consider unify.
namespace _ap_type {
template <typename _Tp>
struct is_signed {
  static const bool value = _Tp(-1) < _Tp(1);
};

template <typename _Tp>
struct is_integral {
  static const bool value = false;
};
#define DEF_IS_INTEGRAL(CTYPE)      \
  template <>                       \
  struct is_integral<CTYPE> {       \
    static const bool value = true; \
  };
DEF_IS_INTEGRAL(bool)
DEF_IS_INTEGRAL(char)
DEF_IS_INTEGRAL(signed char)
DEF_IS_INTEGRAL(unsigned char)
DEF_IS_INTEGRAL(short)
DEF_IS_INTEGRAL(unsigned short)
DEF_IS_INTEGRAL(int)
DEF_IS_INTEGRAL(unsigned int)
DEF_IS_INTEGRAL(long)
DEF_IS_INTEGRAL(unsigned long)
DEF_IS_INTEGRAL(ap_slong)
DEF_IS_INTEGRAL(ap_ulong)
#undef DEF_IS_INTEGRAL

template <bool, typename _Tp = void>
struct enable_if {};
// partial specialization for true
template <typename _Tp>
struct enable_if<true, _Tp> {
  typedef _Tp type;
};

template <typename _Tp>
struct remove_const {
  typedef _Tp type;
};

template <typename _Tp>
struct remove_const<_Tp const> {
  typedef _Tp type;
};
} // namespace _ap_type

// ----------------------------------------------------------------------

// Define ssdm_int and _ssdm_op.
// XXX deleted in open-source version

#ifndef NON_C99STRING
#define _AP_C99 true
#else
#define _AP_C99 false
#endif

static inline unsigned char guess_radix(const char* s) {
  unsigned char rd = 10; ///< default radix
  const char* p = s;
  // skip neg sign if it exists
  if (p[0] == '-' || p[0] == '+') ++p;
  // guess based on following two bits.
  if (p[0] == '0') {
    if (p[1] == 'b' || p[1] == 'B') {
      rd = 2;
    } else if (p[1] == 'o' || p[1] == 'O') {
      rd = 8;
    } else if (p[1] == 'x' || p[1] == 'X') {
      rd = 16;
    } else if (p[1] == 'd' || p[1] == 'D') {
      rd = 10;
    }
  }
  return rd;
}

// ----------------------------------------------------------------------

// Basic integral struct upon which ap_int and ap_fixed are defined.
#ifdef __SYNTHESIS__
// Use ssdm_int, a compiler dependent, attribute constrained integeral type as
// basic data type.
#define _AP_ROOT_TYPE ssdm_int
// Basic ops.
#define _AP_ROOT_op_concat(Ret, X, Y) _ssdm_op_concat(Ret, X, Y)
#define _AP_ROOT_op_get_bit(Val, Bit) _ssdm_op_get_bit(Val, Bit)
#define _AP_ROOT_op_set_bit(Val, Bit, Repl) _ssdm_op_set_bit(Val, Bit, Repl)
#define _AP_ROOT_op_get_range(Val, Lo, Hi) _ssdm_op_get_range(Val, Lo, Hi)
#define _AP_ROOT_op_set_range(Val, Lo, Hi, Repl) \
  _ssdm_op_set_range(Val, Lo, Hi, Repl)
#define _AP_ROOT_op_reduce(Op, Val) _ssdm_op_reduce(Op, Val)
#else // ifdef __SYNTHESIS__
// Use ap_private for compiler-independent basic data type
template <int _AP_W, bool _AP_S, bool _AP_C = _AP_W <= 64>
class ap_private;
/// model ssdm_int in standard C++ for simulation.
template <int _AP_W, bool _AP_S>
struct ssdm_int_sim {
  /// integral type with template-specified width and signedness.
  ap_private<_AP_W, _AP_S> V;
  ssdm_int_sim() {}
};
#define _AP_ROOT_TYPE ssdm_int_sim
// private's ref uses _AP_ROOT_TYPE.
#include <etc/ap_private.h>
// XXX The C-sim model cannot use GCC-extension
// Basic ops. Ret and Val are ap_private.
template <typename _Tp1, typename _Tp2, typename _Tp3>
inline _Tp1 _AP_ROOT_op_concat(const _Tp1& Ret, const _Tp2& X, const _Tp3& Y) {
  _Tp1 r = (X).operator,(Y);
  return r;
}
#define _AP_ROOT_op_get_bit(Val, Bit) (Val).get_bit((Bit))
template <typename _Tp1, typename _Tp2, typename _Tp3>
inline _Tp1& _AP_ROOT_op_set_bit(_Tp1& Val, const _Tp2& Bit, const _Tp3& Repl) {
  (Val).set_bit((Bit), (Repl));
  return Val;
}
// notice the order of high and low index is different in ssdm call and
// ap_private.range()...
#define _AP_ROOT_op_get_range(Val, Lo, Hi) (Val).range((Hi), (Lo))
template <typename _Tp1, typename _Tp2, typename _Tp3, typename _Tp4>
inline _Tp1& _AP_ROOT_op_set_range(_Tp1& Val, const _Tp2& Lo, const _Tp3& Hi,
                                   const _Tp4& Repl) {
  (Val).range((Hi), (Lo)) = Repl;
  return (Val);
}
#define _AP_ROOT_op_and_reduce(Val) (Val).and_reduce()
#define _AP_ROOT_op_nand_reduce(Val) (Val).nand_reduce()
#define _AP_ROOT_op_or_reduce(Val) (Val).or_reduce()
#define _AP_ROOT_op_xor_reduce(Val) (Val).xor_reduce()
// ## is the concatenation in preprocessor:
#define _AP_ROOT_op_reduce(Op, Val) _AP_ROOT_op_##Op##_reduce(Val)
#endif // ifdef __SYNTHESIS__ else

// ----------------------------------------------------------------------

// Constants for half, single, double pricision floating points
#define HALF_MAN 10
#define FLOAT_MAN 23
#define DOUBLE_MAN 52

#define HALF_EXP 5
#define FLOAT_EXP 8
#define DOUBLE_EXP 11

#define BIAS(e) ((1L << (e - 1L)) - 1L)
#define HALF_BIAS BIAS(HALF_EXP)
#define FLOAT_BIAS BIAS(FLOAT_EXP)
#define DOUBLE_BIAS BIAS(DOUBLE_EXP)

#define APFX_IEEE_DOUBLE_E_MAX DOUBLE_BIAS
#define APFX_IEEE_DOUBLE_E_MIN (-DOUBLE_BIAS + 1)

INLINE ap_ulong doubleToRawBits(double pf) {
  union {
    ap_ulong __L;
    double __D;
  } LD;
  LD.__D = pf;
  return LD.__L;
}

INLINE unsigned int floatToRawBits(float pf) {
  union {
    unsigned int __L;
    float __D;
  } LD;
  LD.__D = pf;
  return LD.__L;
}

#if _AP_ENABLE_HALF_ == 1
INLINE unsigned short halfToRawBits(half pf) {
#ifdef __SYNTHESIS__
  union {
    unsigned short __L;
    half __D;
  } LD;
  LD.__D = pf;
  return LD.__L;
#else
  return pf.get_bits();
#endif
}
#endif

// usigned long long is at least 64-bit
INLINE double rawBitsToDouble(ap_ulong pi) {
  union {
    ap_ulong __L;
    double __D;
  } LD;
  LD.__L = pi;
  return LD.__D;
}

// long is at least 32-bit
INLINE float rawBitsToFloat(unsigned long pi) {
  union {
    unsigned int __L;
    float __D;
  } LD;
  LD.__L = pi;
  return LD.__D;
}

#if _AP_ENABLE_HALF_ == 1
// short is at least 16-bit
INLINE half rawBitsToHalf(unsigned short pi) {
#ifdef __SYNTHESIS__
  union {
    unsigned short __L;
    half __D;
  } LD;
  LD.__L = pi;
  return LD.__D;
#else
  // sim model of half has a non-trivial constructor
  half __D;
  __D.set_bits(pi);
  return __D;
#endif
}
#endif

#endif // ifndef __AP_COMMON_H__

// -*- cpp -*-
