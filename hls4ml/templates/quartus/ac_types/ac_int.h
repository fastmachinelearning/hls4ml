/**************************************************************************
 *                                                                        *
 *  Algorithmic C (tm) Datatypes                                          *
 *                                                                        *
 *  Software Version: 4.0                                                 *
 *                                                                        *
 *  Release Date    : Sat Jun 13 12:35:18 PDT 2020                        *
 *  Release Type    : Production Release                                  *
 *  Release Build   : 4.0.0                                               *
 *                                                                        *
 *  Copyright 2004-2020, Mentor Graphics Corporation,                     *
 *                                                                        *
 *  All Rights Reserved.                                                  *
 *                                                                        *
 **************************************************************************
 *  Licensed under the Apache License, Version 2.0 (the "License");       *
 *  you may not use this file except in compliance with the License.      *
 *  You may obtain a copy of the License at                               *
 *                                                                        *
 *      http://www.apache.org/licenses/LICENSE-2.0                        *
 *                                                                        *
 *  Unless required by applicable law or agreed to in writing, software   *
 *  distributed under the License is distributed on an "AS IS" BASIS,     *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or       *
 *  implied.                                                              *
 *  See the License for the specific language governing permissions and   *
 *  limitations under the License.                                        *
 **************************************************************************
 *                                                                        *
 *  The most recent version of this package is available at github.       *
 *                                                                        *
 *************************************************************************/

/*
//  Source:         ac_int.h
//  Description:    fast arbitrary-length bit-accurate integer types:
//                    - unsigned integer of length W:  ac_int<W,false>
//                    - signed integer of length W:  ac_int<W,true>
//  Author:         Andres Takach, Ph.D.
//  Notes:
//   - C++ Runtime: important to use optimization flag (for example -O3)
//
//   - Compiler support: recent GNU compilers are required for correct
//     template compilation
//
//   - Most frequent migration issues:
//      - need to cast to common type when using question mark operator:
//          (a < 0) ? -a : a;  // a is ac_int<W,true>
//        change to:
//          (a < 0) ? -a : (ac_int<W+1,true>) a;
//        or
//          (a < 0) ? (ac_int<W+1,false>) -a : (ac_int<W+1,false>) a;
//
//      - left shift is not arithmetic ("a<<n" has same bitwidth as "a")
//          ac_int<W+1,false> b = a << 1;  // a is ac_int<W,false>
//        is not equivalent to b=2*a. In order to get 2*a behavior change to:
//          ac_int<W+1,false> b = (ac_int<W+1,false>)a << 1;
//
//      - only static length read/write slices are supported:
//         - read:  x.slc<4>(k) => returns ac_int for 4-bit slice x(4+k-1 DOWNTO k)
//         - write: x.set_slc(k,y) = writes bits of y to x starting at index k
*/

#ifndef __AC_INT_H
#define __AC_INT_H

#define AC_VERSION 3
#define AC_VERSION_MINOR 9

#ifndef __cplusplus
#error C++ is required to include this header file
#endif

#if (defined(__GNUC__) && __GNUC__ < 3 && !defined(__EDG__))
#error GCC version 3 or greater is required to include this header file
#endif

#if (defined(_MSC_VER) && _MSC_VER < 1400 && !defined(__EDG__))
#error Microsoft Visual Studio 8 or newer is required to include this header file
#endif

#if (defined(_MSC_VER) && !defined(__EDG__))
#pragma warning( push )
#pragma warning( disable: 4127 4100 4244 4307 4310 4365 4514 4554 4706 4800 )
#endif

// for safety
#if (defined(N) || defined(N2))
#error One or more of the following is defined: N, N2. Definition conflicts with their usage as template parameters.
#error DO NOT use defines before including third party header files.
#endif

// for safety
#if (defined(W) || defined(I) || defined(S) || defined(W2) || defined(I2) || defined(S2))
#error One or more of the following is defined: W, I, S, W2, I2, S2. Definition conflicts with their usage as template parameters.
#error DO NOT use defines before including third party header files.
#endif

#if defined(true)
#warning The C++ keyword true is defined which may result in subtle compilation problems. Undefining it.
#undef true
#endif
#if defined(false)
#warning The C++ keyword false is defined which may result in subtle compilation problems. Undefining it.
#undef false
#endif

#ifndef __ASSERT_H__
#define __ASSERT_H__
#include <assert.h>
#endif
#include <limits>
#ifndef AC_USER_DEFINED_ASSERT
#include <iostream>
#else
#include <ostream>
#endif
#include <math.h>
#include <string>

#ifndef __SYNTHESIS__
#ifndef __AC_INT_UTILITY_BASE
#define __AC_INT_UTILITY_BASE
#endif

#endif

#ifdef __AC_NAMESPACE
namespace __AC_NAMESPACE {
#endif

#define AC_MAX(a,b) ((a) > (b) ? (a) : (b))
#define AC_MIN(a,b) ((a) < (b) ? (a) : (b))
#define AC_ABS(a) ((a) < 0 ? -(a) : (a))

#if defined(_MSC_VER)
typedef unsigned __int64 Ulong;
typedef signed   __int64 Slong;
#else
typedef unsigned long long Ulong;
typedef signed   long long Slong;
#endif

enum ac_base_mode { AC_BIN=2, AC_OCT=8, AC_DEC=10, AC_HEX=16 };
enum ac_special_val {AC_VAL_DC, AC_VAL_0, AC_VAL_MIN, AC_VAL_MAX, AC_VAL_QUANTUM};

template <int W, bool S> class ac_int;

namespace ac_private {
#if defined(__SYNTHESIS__) && !defined(AC_IGNORE_BUILTINS)
#pragma builtin
#endif

  enum {long_w = std::numeric_limits<unsigned long>::digits};
  const unsigned int all_ones = (unsigned) ~0;

  // PRIVATE FUNCTIONS in namespace: for implementing ac_int/ac_fixed

#ifndef __SYNTHESIS__
  inline double mgc_floor(double d) { return floor(d); }
#else
  inline double mgc_floor(double d) { return 0.0; }
#endif

  #define AC_ASSERT(cond, msg) ac_private::ac_assert(cond, __FILE__, __LINE__, msg)
  inline void ac_assert(bool condition, const char *file=0, int line=0, const char *msg=0) {
  #ifndef __SYNTHESIS__
    #ifndef AC_USER_DEFINED_ASSERT
    if(!condition) {
      std::cerr << "Assert";
      if(file)
        std::cerr << " in file " << file << ":" << line;
      if(msg)
        std::cerr << " " << msg;
      std::cerr << std::endl;
      assert(0);
    }
    #else
    AC_USER_DEFINED_ASSERT(condition, file, line, msg);
    #endif
  #endif
  }

  // helper structs for statically computing log2 like functions (nbits, log2_floor, log2_ceil)
  //   using recursive templates
  template<unsigned char N>
  struct s_N {
    template<unsigned X>
    struct s_X {
      enum {
        X2 = X >> N,
        N_div_2 = N >> 1,
        nbits = X ? (X2 ? N + (int) s_N<N_div_2>::template s_X<X2>::nbits : (int) s_N<N_div_2>::template s_X<X>::nbits) : 0
      };
    };
  };
  template<> struct s_N<0> {
    template<unsigned X>
    struct s_X {
      enum {nbits = !!X };
    };
  };

  template<int N>
  inline double ldexpr32(double d) {
    double d2 = d;
    if(N < 0)
      for(int i=0; i < -N; i++)
        d2 /= (Ulong) 1 << 32;
    else
      for(int i=0; i < N; i++)
        d2 *= (Ulong) 1 << 32;
    return d2;
  }
  template<> inline double ldexpr32<0>(double d) { return d; }
  template<> inline double ldexpr32<1>(double d) { return d * ((Ulong) 1 << 32); }
  template<> inline double ldexpr32<-1>(double d) { return d / ((Ulong) 1 << 32); }
  template<> inline double ldexpr32<2>(double d) { return (d * ((Ulong) 1 << 32)) * ((Ulong) 1 << 32); }
  template<> inline double ldexpr32<-2>(double d) { return (d / ((Ulong) 1 << 32)) / ((Ulong) 1 << 32); }

  template<int N>
  inline double ldexpr(double d) {
    return ldexpr32<N/32>( N < 0 ? d/( (unsigned) 1 << (-N & 31)) : d * ( (unsigned) 1 << (N & 31)));
  }

  template<int N>
  inline void iv_copy(const int *op, int *r) {
    for(int i=0; i < N; i++)
      r[i] = op[i];
  }
  template<> inline void iv_copy<1>(const int *op, int *r) {
    r[0] = op[0];
  }
  template<> inline void iv_copy<2>(const int *op, int *r) {
    r[0] = op[0];
    r[1] = op[1];
  }

  template<int N>
  inline bool iv_equal_zero(const int *op){
    for(int i=0; i < N; i++)
      if(op[i])
        return false;
    return true;
  }
  template<> inline bool iv_equal_zero<0>(const int * /*op*/) { return true; }
  template<> inline bool iv_equal_zero<1>(const int *op) {
    return !op[0];
  }
  template<> inline bool iv_equal_zero<2>(const int *op) {
    return !(op[0] || op[1]);
  }

  template<int N>
  inline bool iv_equal_ones(const int *op){
    for(int i=0; i < N; i++)
      if(~op[i])
        return false;
    return true;
  }
  template<> inline bool iv_equal_ones<0>(const int * /*op*/) { return true; }
  template<> inline bool iv_equal_ones<1>(const int *op) {
    return !~op[0];
  }
  template<> inline bool iv_equal_ones<2>(const int *op) {
    return !(~op[0] || ~op[1]);
  }

  template<int N1, int N2>
  inline bool iv_equal(const int *op1, const int *op2){
    const int M1 = AC_MAX(N1,N2);
    const int M2 = AC_MIN(N1,N2);
    const int *OP1 = N1 >= N2 ? op1 : op2;
    const int *OP2 = N1 >= N2 ? op2 : op1;
    for(int i=0; i < M2; i++)
      if(OP1[i] != OP2[i])
        return false;
    int ext = OP2[M2-1] < 0 ? ~0 : 0;
    for(int i=M2; i < M1; i++)
      if(OP1[i] != ext)
        return false;
    return true;
  }
  template<> inline bool iv_equal<1,1>(const int *op1, const int *op2) {
    return op1[0] == op2[0];
  }

  template<int B, int N>
  inline bool iv_equal_ones_from(const int *op){
    if((B >= 32*N && op[N-1] >= 0) || (B&31 && ~(op[B/32] >> (B&31))))
      return false;
    return iv_equal_ones<N-(B+31)/32>(&op[(B+31)/32]);
  }
  template<> inline bool  iv_equal_ones_from<0,1>(const int *op){
    return iv_equal_ones<1>(op);
  }
  template<> inline bool  iv_equal_ones_from<0,2>(const int *op){
    return iv_equal_ones<2>(op);
  }

  template<int B, int N>
  inline bool iv_equal_zeros_from(const int *op){
    if((B >= 32*N && op[N-1] < 0) || (B&31 && (op[B/32] >> (B&31))))
      return false;
    return iv_equal_zero<N-(B+31)/32>(&op[(B+31)/32]);
  }
  template<> inline bool  iv_equal_zeros_from<0,1>(const int *op){
    return iv_equal_zero<1>(op);
  }
  template<> inline bool  iv_equal_zeros_from<0,2>(const int *op){
    return iv_equal_zero<2>(op);
  }

  template<int B, int N>
  inline bool iv_equal_ones_to(const int *op){
    if((B >= 32*N && op[N-1] >= 0) || (B&31 && ~(op[B/32] | (all_ones << (B&31)))))
      return false;
    return iv_equal_ones<B/32>(op);
  }
  template<> inline bool  iv_equal_ones_to<0,1>(const int *op){
    return iv_equal_ones<1>(op);
  }
  template<> inline bool  iv_equal_ones_to<0,2>(const int *op){
    return iv_equal_ones<2>(op);
  }

  template<int B, int N>
  inline bool iv_equal_zeros_to(const int *op){
    if((B >= 32*N && op[N-1] < 0) || (B&31 && (op[B/32] & ~(all_ones << (B&31)))))
      return false;
    return iv_equal_zero<B/32>(op);
  }
  template<> inline bool  iv_equal_zeros_to<0,1>(const int *op){
    return iv_equal_zero<1>(op);
  }
  template<> inline bool  iv_equal_zeros_to<0,2>(const int *op){
    return iv_equal_zero<2>(op);
  }

  template<int N1, int N2, bool greater>
  inline bool iv_compare(const int *op1, const int *op2){
    const int M1 = AC_MAX(N1,N2);
    const int M2 = AC_MIN(N1,N2);
    const int *OP1 = N1 >= N2 ? op1 : op2;
    const int *OP2 = N1 >= N2 ? op2 : op1;
    const bool b = (N1 >= N2) == greater;
    int ext = OP2[M2-1] < 0 ? ~0 : 0;
    int i2 = M1 > M2 ? ext : OP2[M1-1];
    if(OP1[M1-1] != i2)
      return b ^ (OP1[M1-1] < i2);
    for(int i=M1-2; i >= M2; i--) {
      if((unsigned) OP1[i] != (unsigned) ext)
        return b ^ ((unsigned) OP1[i] < (unsigned) ext);
    }
    for(int i=M2-1; i >= 0; i--) {
      if((unsigned) OP1[i] != (unsigned) OP2[i])
        return b ^ ((unsigned) OP1[i] < (unsigned) OP2[i]);
    }
    return false;
  }
  template<> inline bool iv_compare<1,1,true>(const int *op1, const int *op2) {
    return op1[0] > op2[0];
  }
  template<> inline bool iv_compare<1,1,false>(const int *op1, const int *op2) {
    return op1[0] < op2[0];
  }

  template<int N>
  inline void iv_extend(int *r, int ext) {
    for(int i=0; i < N; i++)
      r[i] = ext;
  }
  template<> inline void iv_extend<-2>(int * /*r*/, int /*ext*/) { }
  template<> inline void iv_extend<-1>(int * /*r*/, int /*ext*/) { }
  template<> inline void iv_extend<0>(int * /*r*/, int /*ext*/) { }
  template<> inline void iv_extend<1>(int *r, int ext) {
    r[0] = ext;
  }
  template<> inline void iv_extend<2>(int *r, int ext) {
    r[0] = ext;
    r[1] = ext;
  }

  template<int Nr>
  inline void iv_assign_int64(int *r, Slong l) {
    r[0] = (int) l;
    if(Nr > 1) {
      r[1] = (int) (l >> 32);
      iv_extend<Nr-2>(r+2, (r[1] < 0) ? ~0 : 0);
    }
  }
  template<> inline void iv_assign_int64<1>(int *r, Slong l) {
    r[0] = (int) l;
  }
  template<> inline void iv_assign_int64<2>(int *r, Slong l) {
    r[0] = (int) l;
    r[1] = (int) (l >> 32);
  }

  template<int Nr>
  inline void iv_assign_uint64(int *r, Ulong l) {
    r[0] = (int) l;
    if(Nr > 1) {
      r[1] = (int) (l >> 32);
      iv_extend<Nr-2>(r+2, 0);
    }
  }
  template<> inline void iv_assign_uint64<1>(int *r, Ulong l) {
    r[0] = (int) l;
  }
  template<> inline void iv_assign_uint64<2>(int *r, Ulong l) {
    r[0] = (int) l;
    r[1] = (int) (l >> 32);
  }

  inline Ulong mult_u_u(int a, int b) {
    return (Ulong) (unsigned) a * (Ulong) (unsigned) b;
  }
  inline Slong mult_u_s(int a, int b) {
    return (Ulong) (unsigned) a * (Slong) (signed) b;
  }
  inline Slong mult_s_u(int a, int b) {
    return (Slong) (signed) a * (Ulong) (unsigned) b;
  }
  inline Slong mult_s_s(int a, int b) {
    return (Slong) (signed) a * (Slong) (signed) b;
  }
  inline void accumulate(Ulong a, Ulong &l1, Slong &l2) {
    l1 += (Ulong) (unsigned) a;
    l2 += a >> 32;
  }
  inline void accumulate(Slong a, Ulong &l1, Slong &l2) {
    l1 += (Ulong) (unsigned) a;
    l2 += a >> 32;
  }

  template<int N1, int N2, int Nr>
  inline void iv_mult(const int *op1, const int *op2, int *r) {
    if(Nr==1)
      r[0] = op1[0] * op2[0];
    else if(N1==1 && N2==1)
      iv_assign_int64<Nr>(r, ((Slong) op1[0]) * ((Slong) op2[0]));
    else {
      const int M1 = AC_MAX(N1,N2);
      const int M2 = AC_MIN(N1,N2);
      const int *OP1 = N1 >= N2 ? op1 : op2;
      const int *OP2 = N1 >= N2 ? op2 : op1;
      const int T1 = AC_MIN(M2-1,Nr);
      const int T2 = AC_MIN(M1-1,Nr);
      const int T3 = AC_MIN(M1+M2-2,Nr);

      Ulong l1 = 0;
      Slong l2 = 0;
      for(int k=0; k < T1; k++) {
        for(int i=0; i < k+1; i++)
          accumulate(mult_u_u(OP1[k-i], OP2[i]), l1, l2);
        l2 += (Ulong) (unsigned) (l1 >> 32);
        r[k] = (int) l1;
        l1 = (unsigned) l2;
        l2 >>= 32;
      }
      for(int k=T1; k < T2; k++) {
        accumulate(mult_u_s(OP1[k-M2+1], OP2[M2-1]), l1, l2);
        for(int i=0; i < M2-1; i++)
          accumulate(mult_u_u(OP1[k-i], OP2[i]), l1, l2);
        l2 += (Ulong) (unsigned) (l1 >> 32);
        r[k] = (int) l1;
        l1 = (unsigned) l2;
        l2 >>= 32;
      }
      for(int k=T2; k < T3; k++) {
        accumulate(mult_u_s(OP1[k-M2+1], OP2[M2-1]), l1, l2);
        for(int i=k-T2+1; i < M2-1; i++)
          accumulate(mult_u_u(OP1[k-i], OP2[i]), l1, l2);
        accumulate(mult_s_u(OP1[M1-1], OP2[k-M1+1]), l1, l2);
        l2 += (Ulong) (unsigned) (l1 >> 32);
        r[k] = (int) l1;
        l1 = (unsigned) l2;
        l2 >>= 32;
      }
      if(Nr >= M1+M2-1) {
        accumulate(mult_s_s(OP1[M1-1], OP2[M2-1]), l1, l2);
        r[M1+M2-2] = (int) l1;
        if(Nr >= M1+M2) {
          l2 += (Ulong) (unsigned) (l1 >> 32);
          r[M1+M2-1] = (int) l2;
          iv_extend<Nr-(M1+M2)>(r+M1+M2, (r[M1+M2-1] < 0) ? ~0 : 0);
        }
      }
    }
  }
  template<> inline void iv_mult<1,1,1>(const int *op1, const int *op2, int *r) {
    r[0] = op1[0] * op2[0];
  }
  template<> inline void iv_mult<1,1,2>(const int *op1, const int *op2, int *r) {
    iv_assign_int64<2>(r, ((Slong) op1[0]) * ((Slong) op2[0]));
  }

  template<int N>
  inline bool iv_uadd_carry(const int *op1, bool carry, int *r) {
    Slong l = carry;
    for(int i=0; i < N; i++) {
      l += (Ulong) (unsigned) op1[i];
      r[i] = (int) l;
      l >>= 32;
    }
    return l != 0;
  }
  template<> inline bool iv_uadd_carry<0>(const int * /*op1*/, bool carry, int * /*r*/) { return carry; }
  template<> inline bool iv_uadd_carry<1>(const int *op1, bool carry, int *r) {
    Ulong l = carry + (Ulong) (unsigned) op1[0];
    r[0] = (int) l;
    return (l >> 32) & 1;
  }

  template<int N>
  inline bool iv_add_int_carry(const int *op1, int op2, bool carry, int *r) {
    if(N==0)
      return carry;
    if(N==1) {
      Ulong l = carry + (Slong) op1[0] + (Slong) op2;
      r[0] = (int) l;
      return (l >> 32) & 1;
    }
    Slong l = carry + (Ulong) (unsigned) op1[0] + (Slong) op2;
    r[0] = (int) l;
    l >>= 32;
    for(int i=1; i < N-1; i++) {
      l += (Ulong) (unsigned) op1[i];
      r[i] = (int) l;
      l >>= 32;
    }
    l += (Slong) op1[N-1];
    r[N-1] = (int) l;
    return (l >> 32) & 1;
  }
  template<> inline bool iv_add_int_carry<0>(const int * /*op1*/, int /*op2*/, bool carry, int * /*r*/) { return carry; }
  template<> inline bool iv_add_int_carry<1>(const int *op1, int op2, bool carry, int *r) {
    Ulong l = carry + (Slong) op1[0] + (Slong) op2;
    r[0] = (int) l;
    return (l >> 32) & 1;
  }

  template<int N>
  inline bool iv_uadd_n(const int *op1, const int *op2, int *r) {
    Ulong l = 0;
    for(int i=0; i < N; i++) {
      l += (Ulong)(unsigned) op1[i] + (Ulong)(unsigned) op2[i];
      r[i] = (int) l;
      l >>= 32;
    }
    return l & 1;
  }
  template<> inline bool iv_uadd_n<0>(const int * /*op1*/, const int * /*op2*/, int * /*r*/) { return false; }
  template<> inline bool iv_uadd_n<1>(const int *op1, const int *op2, int *r) {
    Ulong l = (Ulong) (unsigned) op1[0] + (Ulong) (unsigned) op2[0];
    r[0] = (int) l;
    return (l >> 32) & 1;
  }
  template<> inline bool iv_uadd_n<2>(const int *op1, const int *op2, int *r) {
    Ulong l = (Ulong) (unsigned) op1[0] + (Ulong) (unsigned) op2[0];
    r[0] = (int) l;
    l >>= 32;
    l += (Ulong) (unsigned) op1[1] + (Ulong) (unsigned) op2[1];
    r[1] = (int) l;
    return (l >> 32) & 1;
  }

  template<int N1, int N2, int Nr>
  inline void iv_add(const int *op1, const int *op2, int *r) {
    if(Nr==1)
      r[0] = op1[0] + op2[0];
    else {
      const int M1 = AC_MAX(N1,N2);
      const int M2 = AC_MIN(N1,N2);
      const int *OP1 = N1 >= N2 ? op1 : op2;
      const int *OP2 = N1 >= N2 ? op2 : op1;
      const int T1 = AC_MIN(M2-1,Nr);
      const int T2 = AC_MIN(M1,Nr);

      bool carry = iv_uadd_n<T1>(OP1, OP2, r);
      carry = iv_add_int_carry<T2-T1>(OP1+T1, OP2[T1], carry, r+T1);
      iv_extend<Nr-T2>(r+T2, carry ? ~0 : 0);
    }
  }
  template<> inline void iv_add<1,1,1>(const int *op1, const int *op2, int *r) {
    r[0] = op1[0] + op2[0];
  }
  template<> inline void iv_add<1,1,2>(const int *op1, const int *op2, int *r) {
    iv_assign_int64<2>(r, (Slong) op1[0] + (Slong) op2[0]);
  }

  template<int N>
  inline bool iv_sub_int_borrow(const int *op1, int op2, bool borrow, int *r) {
    if(N==1) {
      Ulong l = (Slong) op1[0] - (Slong) op2 - borrow;
      r[0] = (int) l;
      return (l >> 32) & 1;
    }
    Slong l = (Ulong) (unsigned) op1[0] - (Slong) op2 - borrow;
    r[0] = (int) l;
    l >>= 32;
    for(int i=1; i < N-1; i++) {
      l += (Ulong) (unsigned) op1[i];
      r[i] = (int) l;
      l >>= 32;
    }
    l += (Slong) op1[N-1];
    r[N-1] = (int) l;
    return (l >> 32) & 1;
  }
  template<> inline bool iv_sub_int_borrow<0>(const int * /*op1*/, int /*op2*/, bool borrow, int * /*r*/) { return borrow; }
  template<> inline bool iv_sub_int_borrow<1>(const int *op1, int op2, bool borrow, int *r) {
    Ulong l = (Slong) op1[0] - (Slong) op2 - borrow;
    r[0] = (int) l;
    return (l >> 32) & 1;
  }

  template<int N>
  inline bool iv_sub_int_borrow(int op1, const int *op2, bool borrow, int *r) {
    if(N==1) {
      Ulong l = (Slong) op1 - (Slong) op2[0] - borrow;
      r[0] = (int) l;
      return (l >> 32) & 1;
    }
    Slong l = (Slong) op1 - (Ulong) (unsigned) op2[0] - borrow;
    r[0] = (int) l;
    l >>= 32;
    for(int i=1; i < N-1; i++) {
      l -= (Ulong) (unsigned) op2[i];
      r[i] = (int) l;
      l >>= 32;
    }
    l -= (Slong) op2[N-1];
    r[N-1] = (int) l;
    return (l >> 32) & 1;
  }
  template<> inline bool iv_sub_int_borrow<0>(int /*op1*/, const int * /*op2*/, bool borrow, int * /*r*/) { return borrow; }
  template<> inline bool iv_sub_int_borrow<1>(int op1, const int *op2, bool borrow, int *r) {
    Ulong l = (Slong) op1 - (Slong) op2[0] - borrow;
    r[0] = (int) l;
    return (l >> 32) & 1;
  }

  template<int N>
  inline bool iv_usub_n(const int *op1, const int *op2, int *r) {
    Slong l = 0;
    for(int i=0; i < N; i++) {
      l += (Ulong)(unsigned) op1[i] - (Ulong)(unsigned) op2[i];
      r[i] = (int) l;
      l >>= 32;
    }
    return l & 1;
  }
  template<> inline bool iv_usub_n<1>(const int *op1, const int *op2, int *r) {
    Ulong l = (Ulong) (unsigned) op1[0] - (Ulong) (unsigned) op2[0];
    r[0] = (int) l;
    return (l >> 32) & 1;
  }
  template<> inline bool iv_usub_n<2>(const int *op1, const int *op2, int *r) {
    Slong l = (Ulong) (unsigned) op1[0] - (Ulong) (unsigned) op2[0];
    r[0] = (int) l;
    l >>= 32;
    l += (Ulong) (unsigned) op1[1] - (Ulong) (unsigned) op2[1];
    r[1] = (int) l;
    return (l >> 32) & 1;
  }

  template<int N1, int N2, int Nr>
  inline void iv_sub(const int *op1, const int *op2, int *r) {
    if(Nr==1)
      r[0] = op1[0] - op2[0];
    else {
      const int M1 = AC_MAX(N1,N2);
      const int M2 = AC_MIN(N1,N2);
      const int T1 = AC_MIN(M2-1,Nr);
      const int T2 = AC_MIN(M1,Nr);
      bool borrow = iv_usub_n<T1>(op1, op2, r);
      if(N1 > N2)
        borrow = iv_sub_int_borrow<T2-T1>(op1+T1, op2[T1], borrow, r+T1);
      else
        borrow = iv_sub_int_borrow<T2-T1>(op1[T1], op2+T1, borrow, r+T1);
      iv_extend<Nr-T2>(r+T2, borrow ? ~0 : 0);
    }
  }
  template<> inline void iv_sub<1,1,1>(const int *op1, const int *op2, int *r) {
    r[0] = op1[0] - op2[0];
  }
  template<> inline void iv_sub<1,1,2>(const int *op1, const int *op2, int *r) {
    iv_assign_int64<2>(r, (Slong) op1[0] - (Slong) op2[0]);
  }

  template<int N>
  inline bool iv_all_bits_same(const int *op, bool bit) {
    int t = bit ? ~0 : 0;
    for(int i=0; i < N; i++)
      if(op[i] != t)
        return false;
    return true;
  }
  template<> inline bool iv_all_bits_same<0>(const int * /*op*/, bool /*bit*/) { return true; }
  template<> inline bool iv_all_bits_same<1>(const int *op, bool bit) {
    return op[0] == (bit ? ~0 : 0);
  }

  template <int N, int Nr>
  void iv_neg(const int *op1, int *r) {
    Slong l = 0;
    for(int k = 0; k < AC_MIN(N,Nr); k++) {
      l -= (Ulong) (unsigned) op1[k];
      r[k] = (unsigned) l;
      l >>= 32;
    }
    if(Nr > N) {
      r[N] = (unsigned) (l - (op1[N-1] < 0 ? ~0 : 0));
      iv_extend<Nr-N-1>(r+N+1, r[N] < 0 ? ~0 : 0);
    }
  }

  template <int N, bool S, int Nr>
  void iv_abs(const int *op1, int *r) {
    if( S && op1[N-1] < 0) {
      iv_neg<N,Nr>(op1, r);
    } else {
      iv_copy<AC_MIN(N,Nr)>(op1, r);
      iv_extend<Nr-N>(r+N, 0);
    }
  }

  template<int N, int D, int Q, int R, typename sw2, typename uw2, typename sw4, typename uw4, int w1_length>
  void iv_udiv(const sw2 *n, const sw2 *d, sw2 *q, sw2 *r) {
    const int w2_length = 2*w1_length;
    int d_msi;  // most significant int for d
    for(d_msi = D-1; d_msi > 0 && !d[d_msi]; d_msi--) {}
    uw4 d1 = 0;
    if(!d_msi && !d[0]) {
      d1 = n[0]/d[0];  // d is zero => divide by zero
      return;
    }
    int n_msi;  // most significant int for n
    for(n_msi = N-1; n_msi > 0 && !n[n_msi]; n_msi--) {}
    for(int i=0; i < Q; i++)
      q[i] = 0;
    for(int i=0; i < R; i++)
      r[i] = n[i];
    // write most significant "words" into d1
    bool d_mss_odd = (bool) (d[d_msi] >> w1_length);
    int d_mss= 2*d_msi + d_mss_odd;  // index to most significant short (16-bit)
    d1 = (uw4) (uw2) d[d_msi] << (w1_length << (int) !d_mss_odd);
    if(d_msi)
      d1 |= (uw2) d[d_msi-1] >> (d_mss_odd ? w1_length : 0);
    bool n_mss_odd = (bool) (n[n_msi] >> w1_length);
    int n_mss = 2*n_msi + n_mss_odd;
    if(n_mss < d_mss) {
      // q already initialized to 0
      if(R) {
        int r_msi = AC_MIN(R-1, n_msi);
        for(int j = 0; j <= r_msi; j++)
          r[j] = n[j];
        for(int j = r_msi+1; j < R; j++)
          r[j] = 0;
      }
    } else {
      uw2 r1[N+1];
      r1[n_msi+1] = 0;
      for(int k = n_msi; k >= 0; k--)
        r1[k] = n[k];
      for(int k = n_mss; k >=d_mss; k--) {
        int k_msi = k >> 1;
        bool odd = k & 1;
        uw2 r1m1 = k_msi > 0 ? r1[k_msi-1] : (uw2) 0;
        uw4 n1 = odd ?
          (uw4) ((r1[k_msi+1] << w1_length) | (r1[k_msi] >> w1_length)) << w2_length | ((r1[k_msi] << w1_length) | (r1m1 >> w1_length)) :
          (uw4) r1[k_msi] << w2_length | r1m1;
        uw2 q1 = n1/d1;
        if(q1 >> w1_length)
          q1--;
        AC_ASSERT(!(q1 >> w1_length), "Problem detected in long division algorithm, Please report");
        unsigned k2 = k - d_mss;
        unsigned k2_i = k2 >> 1;
        bool odd_2 = k2 & 1;
        uw2 q2 = q1 << (odd_2 ? w1_length : 0);
        sw4 l = 0;
        for(int j = 0; j <= d_msi; j++) {
          l += r1[k2_i + j];
          bool l_sign = l < 0;
          sw4 prod = (uw4) (uw2) d[j] * (uw4) q2;
          l -= prod;
          bool ov1 = (l >= 0) & ((prod < 0) | l_sign);
          bool ov2 = (l < 0) & (prod < 0) & l_sign;
          r1[k2_i + j] = (uw2) l;
          l >>= w2_length;
          if(ov1)
            l |= ((uw4) -1 << w2_length);
          if(ov2)
            l ^= ((sw4) 1 << w2_length);
        }
        if(odd_2 | d_mss_odd) {
          l += r1[k2_i + d_msi + 1];
          r1[k2_i + d_msi + 1] = (uw2) l;
        }
        if(l < 0) {
          l = 0;
          for(int j = 0; j <= d_msi; j++) {
            l += (sw4) (uw2) d[j] << (odd_2 ? w1_length : 0);
            l += r1[k2_i + j];
            r1[k2_i + j] = (uw2) l;
            l >>= w2_length;
          }
          if(odd_2 | d_mss_odd)
            r1[k2_i + d_msi + 1] += (uw2) l;
          q1--;
        }
        if(Q && k2_i < Q) {
          if(odd_2)
            q[k2_i] = q1 << w1_length;
          else
            q[k2_i] |= q1;
        }
      }
      if(R) {
        int r_msi = AC_MIN(R-1, n_msi);
        for(int j = 0; j <= r_msi; j++)
          r[j] = r1[j];
        for(int j = r_msi+1; j < R; j++)
          r[j] = 0;
      }
    }
  }

  template<int N1, int Num_s, int N2, int Den_s, int Nr>
  inline void iv_div(const int *op1, const int *op2, int *r) {
    enum { N1_over = N1+(Den_s && (Num_s==2)) };
    if(N1_over==1 && N2==1) {
      r[0] = op1[0] / op2[0];
      iv_extend<Nr-N1>(r+1, ((Num_s || Den_s) && (r[0] < 0)) ? ~0 : 0);
    }
    else if(N1_over==1 && N2==2)
      iv_assign_int64<Nr>(r, ( (Slong) op1[0]) / (((Slong) op2[1] << 32) | (unsigned) op2[0]) );
    else if(N1_over==2 && N2==1)
      if(N1 == 1)
        iv_assign_int64<Nr>(r, ( (Slong) op1[0]) / ( (Slong) op2[0]) );
      else
        iv_assign_int64<Nr>(r, (((Slong) op1[1] << 32) | (unsigned) op1[0]) / ( (Slong) op2[0]) );
    else if(N1_over==2 && N2==2)
      if(N1 == 1)
        iv_assign_int64<Nr>(r, ( (Slong) op1[0]) / (((Slong) op2[1] << 32) | (unsigned) op2[0]) );
      else
        iv_assign_int64<Nr>(r, (((Slong) op1[1] << 32) | (unsigned) op1[0]) / (((Slong) op2[1] << 32) | (unsigned) op2[0]) );
    else if(!Num_s && !Den_s) {
      iv_udiv<N1,N2,Nr,0,int,unsigned,Slong,Ulong,16>(op1, op2, r, 0);
    }
    else {
      enum { N1_neg = N1+(Num_s==2), N2_neg = N2+(Den_s==2)};
      int numerator[N1_neg];
      int denominator[N2_neg];
      int quotient[N1_neg];
      iv_abs<N1, (bool) Num_s, N1_neg>(op1, numerator);
      iv_abs<N2, (bool) Den_s, N2_neg>(op2, denominator);
      iv_udiv<N1_neg,N2_neg,N1_neg,0,int,unsigned,Slong,Ulong,16>(numerator, denominator, quotient, 0);
      if( (Num_s && op1[N1-1] < 0) ^ (Den_s && op2[N2-1] < 0) )
        iv_neg<N1_neg, Nr>(quotient, r);
      else {
        iv_copy<AC_MIN(N1_neg,Nr)>(quotient, r);
        iv_extend<Nr-N1_neg>(r+N1_neg, (Num_s || Den_s) && r[N1_neg-1] < 0 ? ~0 : 0);
      }
    }
  }

  template<int N1, int Num_s, int N2, int Den_s, int Nr>
  inline void iv_rem(const int *op1, const int *op2, int *r) {
    enum { N1_over = N1+(Den_s && (Num_s==2)) };   // N1_over corresponds to the division
    if(N1_over==1 && N2==1) {
      r[0] = op1[0] % op2[0];
      iv_extend<Nr-1>(r+1, Num_s && r[0] < 0 ? ~0 : 0);
    }
    else if(N1_over==1 && N2==2)
      iv_assign_int64<Nr>(r, ( (Slong) op1[0]) % (((Slong) op2[1] << 32) | (unsigned) op2[0]) );
    else if(N1_over==2 && N2==1)
      if(N1 == 1)
        iv_assign_int64<Nr>(r, ( (Slong) op1[0]) % ( (Slong) op2[0]) );
      else
        iv_assign_int64<Nr>(r, (((Slong) op1[1] << 32) | (unsigned) op1[0]) % ( (Slong) op2[0]) );
    else if(N1_over==2 && N2==2)
      if(N1 == 1)
        iv_assign_int64<Nr>(r, ( (Slong) op1[0]) % (((Slong) op2[1] << 32) | (unsigned) op2[0]) );
      else
        iv_assign_int64<Nr>(r, (((Slong) op1[1] << 32) | (unsigned) op1[0]) % (((Slong) op2[1] << 32) | (unsigned) op2[0]) );
    else if(!Num_s && !Den_s) {
      iv_udiv<N1,N2,0,Nr,int,unsigned,Slong,Ulong,16>(op1, op2, 0, r);
    }
    else {
      enum { N1_neg = N1+(Num_s==2), N2_neg = N2+(Den_s==2)};
      int numerator[N1_neg];
      int denominator[N2_neg];
      int remainder[N2];
      iv_abs<N1, (bool) Num_s, N1_neg>(op1, numerator);
      iv_abs<N2, (bool) Den_s, N2_neg>(op2, denominator);
      iv_udiv<N1_neg,N2_neg,0,N2,int,unsigned,Slong,Ulong,16>(numerator, denominator, 0, remainder);
      if( (Num_s && op1[N1-1] < 0) )
        iv_neg<N2, Nr>(remainder, r);
      else {
        iv_copy<AC_MIN(N2,Nr)>(remainder, r);
        iv_extend<Nr-N2>(r+N2, Num_s && r[N2-1] < 0 ? ~0 : 0);
      }
    }
  }

  template<int N>
  inline void iv_bitwise_complement_n(const int *op, int *r) {
    for(int i=0; i < N; i++)
      r[i] = ~op[i];
  }
  template<> inline void iv_bitwise_complement_n<1>(const int *op, int *r) {
    r[0] = ~op[0];
  }
  template<> inline void iv_bitwise_complement_n<2>(const int *op, int *r) {
    r[0] = ~op[0];
    r[1] = ~op[1];
  }

  template<int N, int Nr>
  inline void iv_bitwise_complement(const int *op, int *r) {
    const int M = AC_MIN(N,Nr);
    iv_bitwise_complement_n<M>(op, r);
    iv_extend<Nr-M>(r+M, (r[M-1] < 0) ? ~0 : 0);
  }

  template<int N>
  inline void iv_bitwise_and_n(const int *op1, const int *op2, int *r) {
    for(int i=0; i < N; i++)
      r[i] = op1[i] & op2[i];
  }
  template<> inline void iv_bitwise_and_n<1>(const int *op1, const int *op2, int *r) {
    r[0] = op1[0] & op2[0];
  }
  template<> inline void iv_bitwise_and_n<2>(const int *op1, const int *op2, int *r) {
    r[0] = op1[0] & op2[0];
    r[1] = op1[1] & op2[1];
  }

  template<int N1, int N2, int Nr>
  inline void iv_bitwise_and(const int *op1, const int *op2, int *r) {
    const int M1 = AC_MIN(AC_MAX(N1,N2), Nr);
    const int M2 = AC_MIN(AC_MIN(N1,N2), Nr);
    const int *OP1 = N1 > N2 ? op1 : op2;
    const int *OP2 = N1 > N2 ? op2 : op1;

    iv_bitwise_and_n<M2>(op1, op2, r);
    if(OP2[M2-1] < 0)
      iv_copy<M1-M2>(OP1+M2, r+M2);
    else
      iv_extend<M1-M2>(r+M2, 0);
    iv_extend<Nr-M1>(r+M1, (r[M1-1] < 0) ? ~0 : 0);
  }

  template<int N>
  inline void iv_bitwise_or_n(const int *op1, const int *op2, int *r) {
    for(int i=0; i < N; i++)
      r[i] = op1[i] | op2[i];
  }
  template<> inline void iv_bitwise_or_n<1>(const int *op1, const int *op2, int *r) {
    r[0] = op1[0] | op2[0];
  }
  template<> inline void iv_bitwise_or_n<2>(const int *op1, const int *op2, int *r) {
    r[0] = op1[0] | op2[0];
    r[1] = op1[1] | op2[1];
  }

  template<int N1, int N2, int Nr>
  inline void iv_bitwise_or(const int *op1, const int *op2, int *r) {
    const int M1 = AC_MIN(AC_MAX(N1,N2), Nr);
    const int M2 = AC_MIN(AC_MIN(N1,N2), Nr);
    const int *OP1 = N1 >= N2 ? op1 : op2;
    const int *OP2 = N1 >= N2 ? op2 : op1;

    iv_bitwise_or_n<M2>(op1, op2, r);
    if(OP2[M2-1] < 0)
      iv_extend<M1-M2>(r+M2, ~0);
    else
      iv_copy<M1-M2>(OP1+M2, r+M2);
    iv_extend<Nr-M1>(r+M1, (r[M1-1] < 0) ? ~0 : 0);
  }

  template<int N>
  inline void iv_bitwise_xor_n(const int *op1, const int *op2, int *r) {
    for(int i=0; i < N; i++)
      r[i] = op1[i] ^ op2[i];
  }
  template<> inline void iv_bitwise_xor_n<1>(const int *op1, const int *op2, int *r) {
    r[0] = op1[0] ^ op2[0];
  }
  template<> inline void iv_bitwise_xor_n<2>(const int *op1, const int *op2, int *r) {
    r[0] = op1[0] ^ op2[0];
    r[1] = op1[1] ^ op2[1];
  }

  template<int N1, int N2, int Nr>
  inline void iv_bitwise_xor(const int *op1, const int *op2, int *r) {
    const int M1 = AC_MIN(AC_MAX(N1,N2), Nr);
    const int M2 = AC_MIN(AC_MIN(N1,N2), Nr);
    const int *OP1 = N1 >= N2 ? op1 : op2;
    const int *OP2 = N1 >= N2 ? op2 : op1;

    iv_bitwise_xor_n<M2>(op1, op2, r);
    if(OP2[M2-1] < 0)
      iv_bitwise_complement_n<M1-M2>(OP1+M2, r+M2);
    else
      iv_copy<M1-M2>(OP1+M2, r+M2);
    iv_extend<Nr-M1>(r+M1, (r[M1-1] < 0) ? ~0 : 0);
  }

  template<int N, int Nr>
  inline void iv_shift_l(const int *op1, unsigned op2, int *r) {
    AC_ASSERT(Nr <= N, "iv_shift_l, incorrect usage Nr > N");
    unsigned s31 = op2 & 31;
    unsigned ishift = (op2 >> 5) > Nr ? Nr : (op2 >> 5);
    if(s31 && ishift!=Nr) {
      unsigned lw = 0;
      for(unsigned i=0; i < Nr; i++) {
        unsigned hw = (i >= ishift) ? op1[i-ishift] : 0;
        r[i] = (hw << s31) | (lw >> (32-s31));
        lw = hw;
      }
    } else {
      for(unsigned i=0; i < Nr ; i++)
        r[i] = (i >= ishift) ? op1[i-ishift] : 0;
    }
  }

  template<int N, int Nr>
  inline void iv_shift_r(const int *op1, unsigned op2, int *r) {
    unsigned s31 = op2 & 31;
    unsigned ishift = (op2 >> 5) > N ? N : (op2 >> 5);
    int ext = op1[N-1] < 0 ? ~0 : 0;
    if(s31 && ishift!=N) {
      unsigned lw = (ishift < N) ? op1[ishift] : ext;
      for(unsigned i=0; i < Nr; i++) {
        unsigned hw = (i+ishift+1 < N) ? op1[i+ishift+1] : ext;
        r[i] = (lw >> s31) | (hw << (32-s31));
        lw = hw;
      }
    } else {
      for(unsigned i=0; i < Nr ; i++)
        r[i] = (i+ishift < N) ? op1[i+ishift] : ext;
    }
  }

  template<int N, int Nr, bool S>
  inline void iv_shift_l2(const int *op1, signed op2, int *r) {
    if(S && op2 < 0)
      iv_shift_r<N,Nr>(op1, -op2, r);
    else
      iv_shift_l<N,Nr>(op1, op2, r);
  }

  template<> inline void iv_shift_l2<1,1,false>(const int *op1, signed op2, int *r) {
    r[0] = (op2 < 32) ? ( (unsigned) op1[0] << op2) : 0;
  }
  template<> inline void iv_shift_l2<1,1,true>(const int *op1, signed op2, int *r) {
    r[0] = (op2 >= 0) ?
      (op2 < 32) ? ( (unsigned) op1[0] << op2) : 0 :
      (op2 > -32) ? (op1[0] >> -op2) : (op1[0] >> 31);
  }

  template<int N, int Nr, bool S>
  inline void iv_shift_r2(const int *op1, signed op2, int *r) {
    if(S && op2 < 0)
      iv_shift_l<N,Nr>(op1, -op2, r);
    else
      iv_shift_r<N,Nr>(op1, op2, r);
  }

  template<> inline void iv_shift_r2<1,1,false>(const int *op1, signed op2, int *r) {
    r[0] = (op2 < 32) ? (op1[0] >> op2) : (op1[0] >> 31);
  }
  template<> inline void iv_shift_r2<1,1,true>(const int *op1, signed op2, int *r) {
    r[0] = (op2 >= 0) ?
      (op2 < 32) ? (op1[0] >> op2) : (op1[0] >> 31) :
      (op2 > -32) ? ( (unsigned) op1[0] << -op2) : 0;
  }

  template<int N, int Nr, int B>
  inline void iv_const_shift_l(const int *op1, int *r) {
    // B >= 0
    if(!B) {
      const int M1 = AC_MIN(N,Nr);
      iv_copy<M1>(op1, r);
      iv_extend<Nr-M1>(r+M1, r[M1-1] < 0 ? -1 : 0);
    }
    else {
      const unsigned s31 = B & 31;
      const int ishift = (((B >> 5) > Nr) ? Nr : (B >> 5));
      iv_extend<ishift>(r, 0);
      const int M1 = AC_MIN(N+ishift,Nr);
      if(s31) {
        unsigned lw = 0;
        for(int i=ishift; i < M1; i++) {
          unsigned hw = op1[i-ishift];
          r[i] = (hw << s31) | (lw >> ((32-s31)&31));  // &31 is to quiet compilers
          lw = hw;
        }
        if(Nr > M1) {
          r[M1] = (signed) lw >> ((32-s31)&31);  // &31 is to quiet compilers
          iv_extend<Nr-M1-1>(r+M1+1, r[M1] < 0 ? ~0 : 0);
        }
      } else {
        for(int i=ishift; i < M1 ; i++)
          r[i] = op1[i-ishift];
        iv_extend<Nr-M1>(r+M1, r[M1-1] < 0 ? -1 : 0);
      }
    }
  }
  template<> inline void iv_const_shift_l<1,1,0>(const int *op1, int *r) {
    r[0] = op1[0];
  }
  template<> inline void iv_const_shift_l<2,1,0>(const int *op1, int *r) {
    r[0] = op1[0];
  }

  template<int N, int Nr, int B>
  inline void iv_const_shift_r(const int *op1, int *r) {
    if(!B) {
      const int M1 = AC_MIN(N,Nr);
      iv_copy<M1>(op1, r);
      iv_extend<Nr-M1>(r+M1, r[M1-1] < 0 ? ~0 : 0);
    }
    else {
      const unsigned s31 = B & 31;
      const int ishift = (((B >> 5) > N) ? N : (B >> 5));
      int ext = op1[N-1] < 0 ? ~0 : 0;
      if(s31 && ishift!=N) {
        unsigned lw = (ishift < N) ? op1[ishift] : ext;
        for(int i=0; i < Nr; i++) {
          unsigned hw = (i+ishift+1 < N) ? op1[i+ishift+1] : ext;
          r[i] = (lw >> s31) | (hw << ((32-s31)&31));  // &31 is to quiet compilers
          lw = hw;
        }
      } else {
        for(int i=0; i < Nr ; i++)
          r[i] = (i+ishift < N) ? op1[i+ishift] : ext;
      }
    }
  }
  template<> inline void iv_const_shift_r<1,1,0>(const int *op1, int *r) {
    r[0] = op1[0];
  }
  template<> inline void iv_const_shift_r<2,1,0>(const int *op1, int *r) {
    r[0] = op1[0];
  }

  template<int N>
  inline void iv_conv_from_fraction(double d, int *r, bool *qb, bool *rbits, bool *o) {
    bool b = d < 0;
    double d2 = b ? -d : d;
    double dfloor = mgc_floor(d2);
    *o = dfloor != 0.0;
    d2 = d2 - dfloor;
    for(int i=N-1; i >=0; i--) {
      d2 *= (Ulong) 1 << 32;
      unsigned k = (unsigned int) d2;
      r[i] = b ? ~k : k;
      d2 -= k;
    }
    d2 *= 2;
    bool k = ((int) d2) != 0;  // is 0 or 1
    d2 -= k;
    *rbits = d2 != 0.0;
    *qb = (b && *rbits) ^ k;
    if(b && !*rbits && !*qb)
      iv_uadd_carry<N>(r, true, r);
    *o |= b ^ (r[N-1] < 0);
  }

  template<ac_base_mode b>
  inline int to_str(int *v, int w, bool left_just, char *r) {
    const char digits[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};
    const unsigned char B = b==AC_BIN ? 1 : (b==AC_OCT ? 3 : (b==AC_HEX ? 4 : 0));
    int k = (w+B-1)/B;
    int n = (w+31) >> 5;
    int bits = 0;
    if(b != AC_BIN && left_just) {
      if( (bits = -(w % B)) )
        r[--k] = 0;
    }
    for(int i = 0; i < n; i++) {
      if (b != AC_BIN && bits < 0)
        r[k] += (unsigned char) (( (unsigned) v[i] << (B+bits)) & (b-1));
      unsigned int m = (unsigned) v[i] >> -bits;
      for(bits += 32; bits > 0 && k; bits -= B) {
        r[--k] = (char) (m & (b-1));
        m >>= B;
      }
    }
    for(int i=0; i < (w+B-1)/B; i++)
      r[i] = digits[(int)r[i]];
    return (w+B-1)/B;
  }
  template<> inline int to_str<AC_DEC>(int *v, int w, bool left_just, char *r) {
    int k = 0;
    int msw = (w-1) >> 5;
    if(left_just) {
      unsigned bits_msw = w & 31;
      if(bits_msw) {
        unsigned left_shift = 32 - bits_msw;
        for(int i=msw; i > 0; i--)
          v[i] = (unsigned) v[i] << left_shift | (unsigned) v[i-1] >> bits_msw;
        v[0] = (unsigned) v[0] << left_shift;
      }
      int lsw = 0;
      while(lsw < msw || v[msw] ) {
        Ulong l = 0;
        for(int i=lsw; i <= msw; i++) {
          l += (Ulong) (unsigned) v[i] * 10;
          v[i] = l;
          l >>= 32;
          if(i==lsw && !v[i])
            lsw++;
        }
        r[k++] = (char) ('0' + (int) l);
      }
    } else {
      const unsigned d = 1000000000;   // 10E9
      for(; msw > 0 && !v[msw]; msw--) {}
      while(msw >= 0) {
        Ulong nl = 0;
        for(int i = msw; i >= 0; i--) {
          nl <<= 32;
          nl |= (unsigned) v[i];
          unsigned q = nl/d;
          nl -= (Ulong) q * d;
          v[i] = q;
        }
        if(!v[msw])
          msw--;
        bool last = msw == -1;
        unsigned rem = (unsigned) nl;
        for(int i=0; (i < 9 && !last) || rem; i++) {
          r[k++] = (char) ('0' + (int) (rem % 10));
          rem /= 10;
        }
      }
      for(int i=0; i < k/2; i++) {
        char c = r[i];
        r[i] = r[k-1-i];
        r[k-1-i] = c;
      }
    }
    r[k] = 0;
    return k;
  }

  inline int to_string(int *v, int w, bool sign_mag, ac_base_mode base, bool left_just, char *r) {
    int n = (w+31) >> 5;
    bool neg = !sign_mag && v[n-1] < 0;
    if(!left_just) {
      while(n-- && v[n] == (neg ? ~0 : 0)) {}
      int w2 = 32*(n+1);
      if(w2) {
        int m = v[n];
        for(int i = 16; i > 0; i >>= 1) {
          if((m >> i) == (neg ? ~0 : 0))
            w2 -= i;
          else
            m >>= i;
        }
      }
      if(w2 < w)
        w = w2;
      w += !sign_mag;
    }
    if(base == AC_DEC)
      return to_str<AC_DEC>(v, w, left_just, r);
    else if (base == AC_HEX)
      return to_str<AC_HEX>(v, w, left_just, r);
    else if (base == AC_OCT)
      return to_str<AC_OCT>(v, w, left_just, r);
    else if (base == AC_BIN)
      return to_str<AC_BIN>(v, w, left_just, r);
    return 0;
  }

  template<int N>
  inline unsigned iv_leading_bits(const int *op, bool bit);

  template<> inline unsigned iv_leading_bits<1>(const int *op, bool bit) {
    const unsigned char tab[] = {4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    unsigned t = bit ? ~*op : *op;
    unsigned cnt = 0;
    if(t >> 16)
      t >>= 16;
    else
      cnt += 16;
    if(t >> 8)
      t >>= 8;
    else
      cnt += 8;
    if(t >> 4)
      t >>= 4;
    else
      cnt += 4;
    cnt += tab[t];
    return cnt;
  }

  template<int N>
  inline unsigned iv_leading_bits(const int *op, bool bit) {
    int ext_sign = bit ? -1 : 0;
    int k;
    for(k = N-1; k >= 0 && op[k] == ext_sign; k--) {}
    return 32*(N-1-k) + (k < 0 ? 0 : iv_leading_bits<1>(op+k, bit));
  }

  //////////////////////////////////////////////////////////////////////////////
  //  Integer Vector class: iv
  //////////////////////////////////////////////////////////////////////////////
  template<int N>
  class iv {
  protected:
    int v[N];
  public:
    template<int N2> friend class iv;
    iv() {}
    template<int N2>
    iv ( const iv<N2> &b ) {
      const int M = AC_MIN(N,N2);
      iv_copy<M>(b.v, v);
      iv_extend<N-M>(v+M, (v[M-1] < 0) ? ~0 : 0);
    }
    iv ( Slong t) {
      iv_assign_int64<N>(v, t);
    }
    iv ( Ulong t) {
      iv_assign_uint64<N>(v, t);
    }
    iv ( int t) {
      v[0] = t;
      iv_extend<N-1>(v+1, (t < 0) ? ~0 : 0);
    }
    iv ( unsigned int t) {
      v[0] = t;
      iv_extend<N-1>(v+1, 0);
    }
    iv ( long t) {
      if(long_w == 32) {
        v[0] = t;
        iv_extend<N-1>(v+1, (t < 0) ? ~0 : 0);
      } else
        iv_assign_int64<N>(v, t);
    }
    iv ( unsigned long t) {
      if(long_w == 32) {
        v[0] = t;
        iv_extend<N-1>(v+1, 0);
      } else
        iv_assign_uint64<N>(v, t);
    }
    iv ( double d ) {
      double d2 = ldexpr32<-N>(d);
      bool qb, rbits, o;
      iv_conv_from_fraction<N>(d2, v, &qb, &rbits, &o);
    }

    // Explicit conversion functions to C built-in types -------------
    inline Slong to_int64() const { return N==1 ? v[0] : ((Ulong)v[1] << 32) | (Ulong) (unsigned) v[0]; }
    inline Ulong to_uint64() const { return N==1 ? (Ulong) v[0] : ((Ulong)v[1] << 32) | (Ulong) (unsigned) v[0]; }
    inline double to_double() const {
      double a = v[N-1];
      for(int i=N-2; i >= 0; i--) {
        a *= (Ulong) 1 << 32;
        a += (unsigned) v[i];
      }
      return a;
    }
    inline void conv_from_fraction(double d, bool *qb, bool *rbits, bool *o) {
      iv_conv_from_fraction<N>(d, v, qb, rbits, o);
    }

    template<int N2, int Nr>
    inline void mult(const iv<N2> &op2, iv<Nr> &r) const {
      iv_mult<N,N2,Nr>(v, op2.v, r.v);
    }
    template<int N2, int Nr>
    void add(const iv<N2> &op2, iv<Nr> &r) const {
      iv_add<N,N2,Nr>(v, op2.v, r.v);
    }
    template<int N2, int Nr>
    void sub(const iv<N2> &op2, iv<Nr> &r) const {
      iv_sub<N,N2,Nr>(v, op2.v, r.v);
    }
    template<int Num_s, int N2, int Den_s, int Nr>
    void div(const iv<N2> &op2, iv<Nr> &r) const {
      iv_div<N,Num_s,N2,Den_s,Nr>(v, op2.v, r.v);
    }
    template<int Num_s, int N2, int Den_s, int Nr>
    void rem(const iv<N2> &op2, iv<Nr> &r) const {
      iv_rem<N,Num_s,N2,Den_s,Nr>(v, op2.v, r.v);
    }
    void increment() {
      iv_uadd_carry<N>(v, true, v);
    }
    void decrement() {
      iv_sub_int_borrow<N>(v, 0, true, v);
    }
    template<int Nr>
    void neg(iv<Nr> &r) const {
      iv_neg<N,Nr>(v, r.v);
    }
    template<int Nr>
    void shift_l(unsigned op2, iv<Nr> &r) const {
      iv_shift_l<N,Nr>(v, op2, r.v);
    }
    template<int Nr>
    void shift_l2(signed op2, iv<Nr> &r) const {
      iv_shift_l2<N,Nr,true>(v, op2, r.v);
    }
    template<int Nr>
    void shift_r(unsigned op2, iv<Nr> &r) const {
      iv_shift_r<N,Nr>(v, op2, r.v);
    }
    template<int Nr>
    void shift_r2(signed op2, iv<Nr> &r) const {
      iv_shift_r2<N,Nr,true>(v, op2, r.v);
    }
    template<int Nr, int B>
    void const_shift_l(iv<Nr> &r) const {
      iv_const_shift_l<N,Nr,B>(v, r.v);
    }
    template<int Nr, int B>
    void const_shift_r(iv<Nr> &r) const {
      iv_const_shift_r<N,Nr,B>(v, r.v);
    }
    template<int Nr>
    void bitwise_complement(iv<Nr> &r) const {
      iv_bitwise_complement<N,Nr>(v, r.v);
    }
    template<int N2, int Nr>
    void bitwise_and(const iv<N2> &op2, iv<Nr> &r) const {
      iv_bitwise_and<N,N2,Nr>(v, op2.v, r.v);
    }
    template<int N2, int Nr>
    void bitwise_or(const iv<N2> &op2, iv<Nr> &r) const {
      iv_bitwise_or<N,N2,Nr>(v, op2.v, r.v);
    }
    template<int N2, int Nr>
    void bitwise_xor(const iv<N2> &op2, iv<Nr> &r) const {
      iv_bitwise_xor<N,N2,Nr>(v, op2.v, r.v);
    }
    template<int N2>
    bool equal(const iv<N2> &op2) const {
      return iv_equal<N,N2>(v, op2.v);
    }
    template<int N2>
    bool greater_than(const iv<N2> &op2) const {
      return iv_compare<N,N2,true>(v, op2.v);
    }
    template<int N2>
    bool less_than(const iv<N2> &op2) const {
      return iv_compare<N,N2,false>(v, op2.v);
    }
    bool equal_zero() const {
      return iv_equal_zero<N>(v);
    }
    template<int N2>
    void set_slc(unsigned lsb, int WS, const iv<N2> &op2) {
      AC_ASSERT((31+WS)/32 == N2, "Bad usage: WS greater than length of slice");
      unsigned msb = lsb+WS-1;
      unsigned lsb_v = lsb >> 5;
      unsigned lsb_b = lsb & 31;
      unsigned msb_v = msb >> 5;
      unsigned msb_b = msb & 31;
      if(N2==1) {
        if(msb_v == lsb_v)
          v[lsb_v] ^= (v[lsb_v] ^ ((unsigned) op2.v[0] << lsb_b)) & (~(WS==32 ? 0 : all_ones<<WS) << lsb_b);
        else {
          v[lsb_v] ^= (v[lsb_v] ^ ((unsigned) op2.v[0] << lsb_b)) & (all_ones << lsb_b);
          unsigned m = (((unsigned) op2.v[0] >> 1) >> (31-lsb_b));
          v[msb_v] ^= (v[msb_v] ^ m) & ~((all_ones<<1)<<msb_b);
        }
      } else {
        v[lsb_v] ^= (v[lsb_v] ^ ((unsigned) op2.v[0] << lsb_b)) & (all_ones << lsb_b);
        for(int i = 1; i < N2-1; i++)
          v[lsb_v+i] = ((unsigned) op2.v[i] << lsb_b) | (((unsigned) op2.v[i-1] >> 1) >> (31-lsb_b));
        unsigned t = ((unsigned) op2.v[N2-1] << lsb_b) | (((unsigned) op2.v[N2-2] >> 1) >> (31-lsb_b));
        unsigned m;
        if(msb_v-lsb_v == N2) {
          v[msb_v-1] = t;
          m = (((unsigned) op2.v[N2-1] >> 1) >> (31-lsb_b));
        }
        else
          m = t;
        v[msb_v] ^= (v[msb_v] ^ m) & ~((all_ones<<1)<<msb_b);
      }
    }
    unsigned leading_bits(bool bit) const {
      return iv_leading_bits<N>(v, bit);
    }
  };

  template<> inline Slong iv<1>::to_int64() const { return v[0]; }
  template<> inline Ulong iv<1>::to_uint64() const { return v[0]; }

  template<> inline Slong iv<2>::to_int64() const {
    return ((Ulong)v[1] << 32) | (Ulong) (unsigned) v[0];
  }
  template<> inline Ulong iv<2>::to_uint64() const {
    return ((Ulong)v[1] << 32) | (Ulong) (unsigned) v[0];
  }

  template<> template<> inline void iv<1>::set_slc(unsigned lsb, int WS, const iv<1> &op2) {
    v[0] ^= (v[0] ^ ((unsigned) op2.v[0] << lsb)) & (~(WS==32 ? 0 : all_ones<<WS) << lsb);
  }
  template<> template<> inline void iv<2>::set_slc(unsigned lsb, int WS, const iv<1> &op2) {
    Ulong l = to_uint64();
    Ulong l2 = op2.to_uint64();
    l ^= (l ^ (l2 << lsb)) & (~((~(Ulong)0)<<WS) << lsb);  // WS <= 32
    *this = l;
  }
  template<> template<> inline void iv<2>::set_slc(unsigned lsb, int WS, const iv<2> &op2) {
    Ulong l = to_uint64();
    Ulong l2 = op2.to_uint64();
    l ^= (l ^ (l2 << lsb)) & (~(WS==64 ? (Ulong) 0 : ~(Ulong)0<<WS) << lsb);
    *this = l;
  }

  // add automatic conversion to Slong/Ulong depending on S and C
  template<int N, bool S, bool C>
  class iv_conv : public iv<N> {
  protected:
    iv_conv() {}
    template<class T> iv_conv(const T& t) : iv<N>(t) {}
  };

  template<int N>
  class iv_conv<N,false,true> : public iv<N> {
  public:
    operator Ulong () const { return iv<N>::to_uint64(); }
  protected:
    iv_conv() {}
    template<class T> iv_conv(const T& t) : iv<N>(t) {}
  };

  template<int N>
  class iv_conv<N,true,true> : public iv<N> {
  public:
    operator Slong () const { return iv<N>::to_int64(); }
  protected:
    iv_conv() {}
    template<class T> iv_conv(const T& t) : iv<N>(t) {}
  };

  // Set default to promote to int as this is the case for almost all types
  //  create exceptions using specializations
  template<typename T>
  struct c_prom {
    typedef int promoted_type;
  };
  template<> struct c_prom<unsigned> {
    typedef unsigned promoted_type;
  };
  template<> struct c_prom<long> {
    typedef long promoted_type;
  };
  template<> struct c_prom<unsigned long> {
    typedef unsigned long promoted_type;
  };
  template<> struct c_prom<Slong> {
    typedef Slong promoted_type;
  };
  template<> struct c_prom<Ulong> {
    typedef Ulong promoted_type;
  };
  template<> struct c_prom<float> {
    typedef float promoted_type;
  };
  template<> struct c_prom<double> {
    typedef double promoted_type;
  };

  template<typename T, typename T2>
  struct c_arith {
     // will error out for pairs of T and T2 that are not defined through specialization
  };
  template<typename T> struct c_arith<T,T> {
    typedef T arith_conv;
  };

  #define C_ARITH(C_TYPE1, C_TYPE2) \
  template<> struct c_arith<C_TYPE1, C_TYPE2> { \
    typedef C_TYPE1 arith_conv; \
  }; \
  template<> struct c_arith<C_TYPE2, C_TYPE1> { \
    typedef C_TYPE1 arith_conv; \
  };

  C_ARITH(double, float)
  C_ARITH(double, int)
  C_ARITH(double, unsigned)
  C_ARITH(double, long)
  C_ARITH(double, unsigned long)
  C_ARITH(double, Slong)
  C_ARITH(double, Ulong)
  C_ARITH(float, int)
  C_ARITH(float, unsigned)
  C_ARITH(float, long)
  C_ARITH(float, unsigned long)
  C_ARITH(float, Slong)
  C_ARITH(float, Ulong)

  C_ARITH(Slong, int)
  C_ARITH(Slong, unsigned)
  C_ARITH(Ulong, int)
  C_ARITH(Ulong, unsigned)

  template<typename T>
  struct map {
    typedef T t;
  };
  template<typename T>
  struct c_type_params {
    // will error out for T for which this template struct is not specialized
  };

  template<typename T> inline const char *c_type_name() { return "unknown"; }
  template<> inline const char *c_type_name<bool>() { return "bool";}
  template<> inline const char *c_type_name<char>() { return "char";}
  template<> inline const char *c_type_name<signed char>() { return "signed char";}
  template<> inline const char *c_type_name<unsigned char>() { return "unsigned char";}
  template<> inline const char *c_type_name<signed short>() { return "signed short";}
  template<> inline const char *c_type_name<unsigned short>() { return "unsigned short";}
  template<> inline const char *c_type_name<int>() { return "int";}
  template<> inline const char *c_type_name<unsigned>() { return "unsigned";}
  template<> inline const char *c_type_name<signed long>() { return "signed long";}
  template<> inline const char *c_type_name<unsigned long>() { return "unsigned long";}
  template<> inline const char *c_type_name<signed long long>() { return "signed long long";}
  template<> inline const char *c_type_name<unsigned long long>() { return "unsigned long long";}
  template<> inline const char *c_type_name<float>() { return "float";}
  template<> inline const char *c_type_name<double>() { return "double";}

  template<typename T> struct c_type;

  template<typename T>
  struct rt_c_type_T {
    template<typename T2>
    struct op1 {
      typedef typename T::template rt_T< c_type<T2> >::mult mult;
      typedef typename T::template rt_T< c_type<T2> >::plus plus;
      typedef typename T::template rt_T< c_type<T2> >::minus2 minus;
      typedef typename T::template rt_T< c_type<T2> >::minus minus2;
      typedef typename T::template rt_T< c_type<T2> >::logic logic;
      typedef typename T::template rt_T< c_type<T2> >::div2 div;
      typedef typename T::template rt_T< c_type<T2> >::div div2;
    };
  };
  template<typename T>
  struct c_type {
    typedef typename c_prom<T>::promoted_type c_prom_T;
    struct rt_unary {
      typedef c_prom_T neg;
      typedef c_prom_T mag_sqr;
      typedef c_prom_T mag;
      template<unsigned N>
      struct set {
        typedef c_prom_T sum;
      };
    };
    template<typename T2>
    struct rt_T {
      typedef typename rt_c_type_T<T2>::template op1<T>::mult mult;
      typedef typename rt_c_type_T<T2>::template op1<T>::plus plus;
      typedef typename rt_c_type_T<T2>::template op1<T>::minus minus;
      typedef typename rt_c_type_T<T2>::template op1<T>::minus2 minus2;
      typedef typename rt_c_type_T<T2>::template op1<T>::logic logic;
      typedef typename rt_c_type_T<T2>::template op1<T>::div div;
      typedef typename rt_c_type_T<T2>::template op1<T>::div2 div2;
    };
    inline static std::string type_name() {
      std::string r = c_type_name<T>();
      return r;
    }

  };
  // with T == c_type
  template<typename T>
  struct rt_c_type_T< c_type<T> > {
    typedef typename c_prom<T>::promoted_type c_prom_T;
    template<typename T2>
    struct op1 {
      typedef typename c_prom<T2>::promoted_type c_prom_T2;
      typedef typename c_arith< c_prom_T, c_prom_T2 >::arith_conv mult;
      typedef typename c_arith< c_prom_T, c_prom_T2 >::arith_conv plus;
      typedef typename c_arith< c_prom_T, c_prom_T2 >::arith_conv minus;
      typedef typename c_arith< c_prom_T, c_prom_T2 >::arith_conv minus2;
      typedef typename c_arith< c_prom_T, c_prom_T2 >::arith_conv logic;
      typedef typename c_arith< c_prom_T, c_prom_T2 >::arith_conv div;
      typedef typename c_arith< c_prom_T, c_prom_T2 >::arith_conv div2;
    };
  };

  #define C_TYPE_MAP(C_TYPE) \
  template<> struct map<C_TYPE> { \
    typedef c_type<C_TYPE> t; \
  };

  #define C_TYPE_PARAMS(C_TYPE, WI, SI) \
  template<> struct c_type_params<C_TYPE> { \
    enum { W = WI, I = WI, E = 0, S = SI, floating_point = 0 }; \
  };

  #define C_TYPE_MAP_INT(C_TYPE, WI, SI) \
    C_TYPE_MAP(C_TYPE) \
    C_TYPE_PARAMS(C_TYPE, WI, SI)

  #define C_TYPE_MAP_FLOAT(C_TYPE, FP, WFP, IFP, EFP) \
  C_TYPE_MAP(C_TYPE) \
  template<> struct c_type_params<C_TYPE> { \
    enum { W = WFP, I = IFP, E = EFP, S = true, floating_point = FP }; \
  };

  C_TYPE_MAP_INT(bool, 1, false)
  C_TYPE_MAP_INT(char, 8, true)
  C_TYPE_MAP_INT(signed char, 8, true)
  C_TYPE_MAP_INT(unsigned char, 8, false)
  C_TYPE_MAP_INT(signed short, 16, true)
  C_TYPE_MAP_INT(unsigned short, 16, false)
  C_TYPE_MAP_INT(signed int, 32, true)
  C_TYPE_MAP_INT(unsigned int, 32, false)
  C_TYPE_MAP_INT(signed long, ac_private::long_w, true)
  C_TYPE_MAP_INT(unsigned long, ac_private::long_w, false)
  C_TYPE_MAP_INT(signed long long, 64, true)
  C_TYPE_MAP_INT(unsigned long long, 64, false)
  C_TYPE_MAP_FLOAT(float, 1, 25, 1, 8)
  C_TYPE_MAP_FLOAT(double, 2, 54, 1, 11)

  #undef C_TYPE_INT
  #undef C_TYPE_PARAMS
  #undef C_TYPE_FLOAT
  #undef C_TYPE_MAP

  // specializations for following struct declared/defined after definition of ac_int
  template<typename T>
  struct rt_ac_int_T {
    template<int W, bool S>
    struct op1 {
      typedef typename T::template rt_T< ac_int<W,S> >::mult mult;
      typedef typename T::template rt_T< ac_int<W,S> >::plus plus;
      typedef typename T::template rt_T< ac_int<W,S> >::minus2 minus;
      typedef typename T::template rt_T< ac_int<W,S> >::minus minus2;
      typedef typename T::template rt_T< ac_int<W,S> >::logic logic;
      typedef typename T::template rt_T< ac_int<W,S> >::div2 div;
      typedef typename T::template rt_T< ac_int<W,S> >::div div2;
    };
  };
}

namespace ac {
  // compiler time constant for log2 like functions
  template<unsigned X>
  struct nbits {
    enum { val = X ? ac_private::s_N<16>::s_X<X>::nbits : 1 };
  };

  template<unsigned X>
  struct log2_floor {
    enum { val = nbits<X>::val - 1 };
  };

  // log2 of 0 is not defined: generate compiler error
  template<> struct log2_floor<0> {};

  template<unsigned X>
  struct log2_ceil {
    enum { lf = log2_floor<X>::val, val = (X == (1 << lf) ? lf : lf+1) };
  };

  // log2 of 0 is not defined: generate compiler error
  template<> struct log2_ceil<0> {};

  template<int LowerBound, int UpperBound>
  struct int_range {
    enum { l_s = (LowerBound < 0), u_s = (UpperBound < 0),
           signedness = l_s || u_s,
           l_nbits = nbits<AC_ABS(LowerBound+l_s)+l_s>::val,
           u_nbits = nbits<AC_ABS(UpperBound+u_s)+u_s>::val,
           nbits = AC_MAX(l_nbits, u_nbits + (!u_s && signedness))
         };
    typedef ac_int<nbits, signedness> type;
  };

  template<int W, int P, bool Is_MSB, bool S>
  class sliceref {
# if defined(__SYNTHESIS__) && !defined(AC_IGNORE_BUILTINS)
# pragma builtin
# endif
    int *d_iv;
    template<int W2, int P2, bool Is_MSB2, bool S2> friend class sliceref;
  public:
    sliceref( int *iv ) : d_iv(iv) {}

    inline const sliceref operator = ( const sliceref &val ) {
      return operator=<P,Is_MSB,S>(val);
    }

    template<int P2, bool Is_MSB2, bool S2>
    inline const sliceref operator = ( const sliceref<W,P2,Is_MSB2,S2> &val ) {
      const int src_lsi = P2/32;
      const int src_msi = (P2+W-1)/32;
      const int trg_lsi = P/32;
      const int trg_msi = (P+W-1)/32;
      const int trg_lsb = P&31;
      const int trg_msb = (P+W-1)&31;
      const int N = src_msi-src_lsi+1;
      const int Nr = trg_msi-trg_lsi+1;
      const int rshift = (P2&31) - (P&31);
      int shifted_src[Nr];
      int *aligned_src = val.d_iv+src_lsi;
      if(rshift) {
        if(rshift < 0)
          ac_private::iv_shift_l<N,Nr>(aligned_src, -rshift, shifted_src);
        else
          ac_private::iv_shift_r<N,Nr>(aligned_src, rshift, shifted_src);
        aligned_src = shifted_src;
      }
      unsigned mask_lsi = ac_private::all_ones << trg_lsb;
      unsigned mask_msi = ac_private::all_ones >> (31-trg_msb);
      if(Nr==1)
        mask_lsi &= mask_msi;
      int *v = d_iv+trg_lsi;
      v[0] ^= (v[0] ^ ((unsigned) aligned_src[0])) & mask_lsi;
      for(int k=1; k < Nr-1; k++)
        v[k] = aligned_src[k];
      if(Nr > 1)
        v[Nr-1] ^= (v[Nr-1] ^ ((unsigned) aligned_src[Nr-1])) & mask_msi;
      if(Is_MSB) {
        const unsigned rem = 31-trg_msb;
        if(rem) {
          v[Nr-1] =  S ? ((signed) ((unsigned) v[Nr-1]  << rem) >> rem)
                       : ((unsigned) v[Nr-1]  << rem) >> rem;
        } else if(!S) {
          v[Nr] = 0;
        }
      }
      return *this;
    }
  };
}

enum ac_q_mode { AC_TRN, AC_RND, AC_TRN_ZERO, AC_RND_ZERO, AC_RND_INF, AC_RND_MIN_INF, AC_RND_CONV, AC_RND_CONV_ODD };
enum ac_o_mode { AC_WRAP, AC_SAT, AC_SAT_ZERO, AC_SAT_SYM };
template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2> class ac_fixed;

//////////////////////////////////////////////////////////////////////////////
//  Arbitrary-Length Integer: ac_int
//////////////////////////////////////////////////////////////////////////////

template<int W, bool S=true>
class ac_int : public ac_private::iv_conv<(W+31+!S)/32, S, W<=64>
#ifndef __SYNTHESIS__
__AC_INT_UTILITY_BASE
#endif
{
#if defined(__SYNTHESIS__) && !defined(AC_IGNORE_BUILTINS)
#pragma builtin
#endif

  enum {N=(W+31+!S)/32};
  typedef ac_private::iv_conv<N, S, W <= 64> ConvBase;
  typedef ac_private::iv<N>                  Base;

  inline void bit_adjust() {
    const unsigned rem = (32-W)&31;
    Base::v[N-1] =  S ? ((signed) ((unsigned) Base::v[N-1]  << rem) >> rem) : (rem ?
                  ((unsigned) Base::v[N-1]  << rem) >> rem : 0);
  }

  inline bool is_neg() const { return S && Base::v[N-1] < 0; }

  // returns false if number is denormal
  template<int WE, bool SE>
  bool normalize_private(ac_int<WE,SE> &exp, bool reserved_min_exp=false) {
    int expt = exp;
    int lshift = leading_sign();
    bool fully_normalized = true;
    ac_int<WE, SE> min_exp;
    min_exp.template set_val<AC_VAL_MIN>();
    int max_shift = exp - min_exp - reserved_min_exp;
    if(lshift > max_shift) {
      lshift = ac_int<WE,false>(max_shift);
      expt = min_exp + reserved_min_exp;
      fully_normalized = false;
    } else {
      expt -= lshift;
    }
    if(Base::equal_zero()) {
      expt = 0;
      fully_normalized = true;
    }
    exp = expt;
    Base r;
    Base::shift_l(lshift, r);
    Base::operator=(r);
    bit_adjust();
    return fully_normalized;
  }

public:
  static const int width = W;
  static const int i_width = W;
  static const bool sign = S;
  static const ac_q_mode q_mode = AC_TRN;
  static const ac_o_mode o_mode = AC_WRAP;
  static const int e_width = 0;

  template<int W2, bool S2>
  struct rt {
    enum {
      mult_w = W+W2,
      mult_s = S||S2,
      plus_w = AC_MAX(W+(S2&&!S),W2+(S&&!S2))+1,
      plus_s = S||S2,
      minus_w = AC_MAX(W+(S2&&!S),W2+(S&&!S2))+1,
      minus_s = true,
      div_w = W+S2,
      div_s = S||S2,
      mod_w = AC_MIN(W,W2+(!S2&&S)),
      mod_s = S,
      logic_w = AC_MAX(W+(S2&&!S),W2+(S&&!S2)),
      logic_s = S||S2
    };
    typedef ac_int<mult_w, mult_s> mult;
    typedef ac_int<plus_w, plus_s> plus;
    typedef ac_int<minus_w, minus_s> minus;
    typedef ac_int<logic_w, logic_s> logic;
    typedef ac_int<div_w, div_s> div;
    typedef ac_int<mod_w, mod_s> mod;
    typedef ac_int<W, S> arg1;
  };

  template<typename T>
  struct rt_T {
    typedef typename ac_private::map<T>::t map_T;
    typedef typename ac_private::rt_ac_int_T<map_T>::template op1<W,S>::mult mult;
    typedef typename ac_private::rt_ac_int_T<map_T>::template op1<W,S>::plus plus;
    typedef typename ac_private::rt_ac_int_T<map_T>::template op1<W,S>::minus minus;
    typedef typename ac_private::rt_ac_int_T<map_T>::template op1<W,S>::minus2 minus2;
    typedef typename ac_private::rt_ac_int_T<map_T>::template op1<W,S>::logic logic;
    typedef typename ac_private::rt_ac_int_T<map_T>::template op1<W,S>::div div;
    typedef typename ac_private::rt_ac_int_T<map_T>::template op1<W,S>::div2 div2;
    typedef ac_int<W, S> arg1;
  };

  struct rt_unary {
    enum {
      neg_w = W+1,
      neg_s = true,
      mag_sqr_w = 2*W-S,
      mag_sqr_s = false,
      mag_w = W+S,
      mag_s = false,
      leading_sign_w = ac::log2_ceil<W+!S>::val,
      leading_sign_s = false
    };
    typedef ac_int<neg_w, neg_s> neg;
    typedef ac_int<mag_sqr_w, mag_sqr_s> mag_sqr;
    typedef ac_int<mag_w, mag_s> mag;
    typedef ac_int<leading_sign_w, leading_sign_s> leading_sign;
    template<unsigned N>
    struct set {
      enum { sum_w = W + ac::log2_ceil<N>::val, sum_s = S};
      typedef ac_int<sum_w, sum_s> sum;
    };
  };

  template<int W2, bool S2> friend class ac_int;
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2> friend class ac_fixed;
  ac_int() {
#if !defined(__SYNTHESIS__) && defined(AC_DEFAULT_IN_RANGE)
    bit_adjust();
#endif
  }
  template<int W2, bool S2>
  inline ac_int (const ac_int<W2,S2> &op) {
    Base::operator =(op);
    bit_adjust();
  }

  inline ac_int( bool b ) : ConvBase(b) { bit_adjust(); }
  inline ac_int( char b ) : ConvBase(b) { bit_adjust(); }
  inline ac_int( signed char b ) : ConvBase(b) { bit_adjust(); }
  inline ac_int( unsigned char b ) : ConvBase(b) { bit_adjust(); }
  inline ac_int( signed short b ) : ConvBase(b) { bit_adjust(); }
  inline ac_int( unsigned short b ) : ConvBase(b) { bit_adjust(); }
  inline ac_int( signed int b ) : ConvBase(b) { bit_adjust(); }
  inline ac_int( unsigned int b ) : ConvBase(b) { bit_adjust(); }
  inline ac_int( signed long b ) : ConvBase(b) { bit_adjust(); }
  inline ac_int( unsigned long b ) : ConvBase(b) { bit_adjust(); }
  inline ac_int( Slong b ) : ConvBase(b) { bit_adjust(); }
  inline ac_int( Ulong b ) : ConvBase(b) { bit_adjust(); }
  inline ac_int( double d ) : ConvBase(d) { bit_adjust(); }


#if (defined(_MSC_VER) && !defined(__EDG__))
#pragma warning( push )
#pragma warning( disable: 4700 )
#endif
#if (defined(__GNUC__) && ( __GNUC__ == 4 && __GNUC_MINOR__ >= 6 || __GNUC__ > 4 ) && !defined(__EDG__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
#endif
  template<ac_special_val V>
  inline ac_int &set_val() {
    const unsigned int all_ones = (unsigned) ~0;
    if(V == AC_VAL_DC) {
      ac_int r;
      Base::operator =(r);
      bit_adjust();
    }
    else if(V == AC_VAL_0 || V == AC_VAL_MIN || V == AC_VAL_QUANTUM) {
      Base::operator =(0);
      if(S && V == AC_VAL_MIN) {
        const unsigned int rem = (W-1)&31;
        Base::v[N-1] = (all_ones << rem);
      } else if(V == AC_VAL_QUANTUM)
        Base::v[0] = 1;
    }
    else {  // AC_VAL_MAX
      Base::operator =(-1);
      const unsigned int rem = (32-W - !S )&31;
      Base::v[N-1] = (all_ones >> 1) >> rem;
    }
    return *this;
  }
#if (defined(_MSC_VER) && !defined(__EDG__))
#pragma warning( pop )
#endif
#if (defined(__GNUC__) && ( __GNUC__ == 4 && __GNUC_MINOR__ >= 6 || __GNUC__ > 4 ) && !defined(__EDG__))
#pragma GCC diagnostic pop
#endif
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

  // Explicit conversion functions to C built-in types -------------
  inline int to_int() const { return Base::v[0]; }
  inline unsigned to_uint() const { return Base::v[0]; }
  inline long to_long() const {
    return ac_private::long_w == 32 ? (long) Base::v[0] : (long) Base::to_int64();
  }
  inline unsigned long to_ulong() const {
    return ac_private::long_w == 32 ? (unsigned long) Base::v[0] : (unsigned long) Base::to_uint64();
  }
  inline Slong to_int64() const { return Base::to_int64(); }
  inline Ulong to_uint64() const { return Base::to_uint64(); }
  inline double to_double() const { return Base::to_double(); }

  inline int length() const { return W; }

  inline std::string to_string(ac_base_mode base_rep, bool sign_mag = false) const {
    // base_rep == AC_DEC => sign_mag == don't care (always print decimal in sign magnitude)
    char r[N*32+4] = {0};
    int i = 0;
    if(sign_mag)
      r[i++] = is_neg() ? '-' : '+';
    else if (base_rep == AC_DEC && is_neg())
      r[i++] = '-';
    if(base_rep != AC_DEC) {
      r[i++] = '0';
      r[i++] = base_rep == AC_BIN ? 'b' : (base_rep == AC_OCT ? 'o' : 'x');
    }
    int str_w;
    if( (base_rep == AC_DEC || sign_mag) && is_neg() ) {
      ac_int<W, false>  mag = operator -();
      str_w = ac_private::to_string(mag.v, W+1, sign_mag, base_rep, false, r+i);
    } else {
      ac_int<W,S> tmp = *this;
      str_w = ac_private::to_string(tmp.v, W+!S, sign_mag, base_rep, false, r+i);
    }
    if(!str_w) {
      r[i] = '0';
      r[i+1] = 0;
    }
    return std::string(r);
  }
  inline static std::string type_name() {
    const char *tf[] = {",false>", ",true>"};
    std::string r = "ac_int<";
    r += ac_int<32,true>(W).to_string(AC_DEC);
    r += tf[S];
    return r;
  }

  // Arithmetic : Binary ----------------------------------------------------
  template<int W2, bool S2>
  typename rt<W2,S2>::mult operator *( const ac_int<W2,S2> &op2) const {
    typename rt<W2,S2>::mult r;
    Base::mult(op2, r);
    return r;
  }
  template<int W2, bool S2>
  typename rt<W2,S2>::plus operator +( const ac_int<W2,S2> &op2) const {
    typename rt<W2,S2>::plus r;
    Base::add(op2, r);
    return r;
  }
  template<int W2, bool S2>
  typename rt<W2,S2>::minus operator -( const ac_int<W2,S2> &op2) const {
    typename rt<W2,S2>::minus r;
    Base::sub(op2, r);
    return r;
  }
#if (defined(__GNUC__) && ( __GNUC__ == 4 && __GNUC_MINOR__ >= 6 || __GNUC__ > 4 ) && !defined(__EDG__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wenum-compare"
#endif
  template<int W2, bool S2>
  typename rt<W2,S2>::div operator /( const ac_int<W2,S2> &op2) const {
    typename rt<W2,S2>::div r;
    enum {Nminus = ac_int<W+S,S>::N, N2 = ac_int<W2,S2>::N, N2minus = ac_int<W2+S2,S2>::N,
          num_s = S + (Nminus > N), den_s = S2 + (N2minus > N2), Nr = rt<W2,S2>::div::N };
    Base::template div<num_s, N2, den_s, Nr>(op2, r);
    return r;
  }
  template<int W2, bool S2>
  typename rt<W2,S2>::mod operator %( const ac_int<W2,S2> &op2) const {
    typename rt<W2,S2>::mod r;
    enum {Nminus = ac_int<W+S,S>::N, N2 = ac_int<W2,S2>::N, N2minus = ac_int<W2+S2,S2>::N,
          num_s = S + (Nminus > N), den_s = S2 + (N2minus > N2), Nr = rt<W2,S2>::mod::N };
    Base::template rem<num_s, N2, den_s, Nr>(op2, r);
    return r;
  }
#if (defined(__GNUC__) && ( __GNUC__ == 4 && __GNUC_MINOR__ >= 6 || __GNUC__ > 4 ) && !defined(__EDG__))
#pragma GCC diagnostic pop
#endif
  // Arithmetic assign  ------------------------------------------------------
  template<int W2, bool S2>
  ac_int &operator *=( const ac_int<W2,S2> &op2) {
    Base r;
    Base::mult(op2, r);
    Base::operator=(r);
    bit_adjust();
    return *this;
  }
  template<int W2, bool S2>
  ac_int &operator +=( const ac_int<W2,S2> &op2) {
    Base r;
    Base::add(op2, r);
    Base::operator=(r);
    bit_adjust();
    return *this;
  }
  template<int W2, bool S2>
  ac_int &operator -=( const ac_int<W2,S2> &op2) {
    Base r;
    Base::sub(op2, r);
    Base::operator=(r);
    bit_adjust();
    return *this;
  }
#if (defined(__GNUC__) && ( __GNUC__ == 4 && __GNUC_MINOR__ >= 6 || __GNUC__ > 4 ) && !defined(__EDG__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wenum-compare"
#endif
  template<int W2, bool S2>
  ac_int &operator /=( const ac_int<W2,S2> &op2) {
    enum {Nminus = ac_int<W+S,S>::N, N2 = ac_int<W2,S2>::N, N2minus = ac_int<W2+S2,S2>::N,
          num_s = S + (Nminus > N), den_s = S2 + (N2minus > N2), Nr = N };
    Base r;
    Base::template div<num_s, N2, den_s, Nr>(op2, r);
    Base::operator=(r);
    bit_adjust();
    return *this;
  }
  template<int W2, bool S2>
  ac_int &operator %=( const ac_int<W2,S2> &op2) {
    enum {Nminus = ac_int<W+S,S>::N, N2 = ac_int<W2,S2>::N, N2minus = ac_int<W2+S2,S2>::N,
          num_s = S + (Nminus > N), den_s = S2 + (N2minus > N2), Nr = N };
    Base r;
    Base::template rem<num_s, N2, den_s, Nr>(op2, r);
    Base::operator=(r);
    bit_adjust();
    return *this;
  }
#if (defined(__GNUC__) && ( __GNUC__ == 4 && __GNUC_MINOR__ >= 6 || __GNUC__ > 4 ) && !defined(__EDG__))
#pragma GCC diagnostic pop
#endif
  // Arithmetic prefix increment, decrement ----------------------------------
  ac_int &operator ++() {
    Base::increment();
    bit_adjust();
    return *this;
  }
  ac_int &operator --() {
    Base::decrement();
    bit_adjust();
    return *this;
  }
  // Arithmetic postfix increment, decrement ---------------------------------
  const ac_int operator ++(int) {
    ac_int t = *this;
    Base::increment();
    bit_adjust();
    return t;
  }
  const ac_int operator --(int) {
    ac_int t = *this;
    Base::decrement();
    bit_adjust();
    return t;
  }
  // Arithmetic Unary --------------------------------------------------------
  ac_int operator +() {
    return *this;
  }
  typename rt_unary::neg operator -() const {
    typename rt_unary::neg r;
    Base::neg(r);
    r.bit_adjust();
    return r;
  }
  // ! ------------------------------------------------------------------------
  bool operator ! () const {
    return Base::equal_zero();
  }

  // Bitwise (arithmetic) unary: complement  -----------------------------
  ac_int<W+!S, true> operator ~() const {
    ac_int<W+!S, true> r;
    Base::bitwise_complement(r);
    return r;
  }
  // Bitwise (non-arithmetic) bit_complement  -----------------------------
  ac_int<W, false> bit_complement() const {
    ac_int<W, false> r;
    Base::bitwise_complement(r);
    r.bit_adjust();
    return r;
  }
  // Bitwise (arithmetic): and, or, xor ----------------------------------
  template<int W2, bool S2>
  typename rt<W2,S2>::logic operator & ( const ac_int<W2,S2> &op2) const {
    typename rt<W2,S2>::logic r;
    Base::bitwise_and(op2, r);
    return r;
  }
  template<int W2, bool S2>
  typename rt<W2,S2>::logic operator | ( const ac_int<W2,S2> &op2) const {
    typename rt<W2,S2>::logic r;
    Base::bitwise_or(op2, r);
    return r;
  }
  template<int W2, bool S2>
  typename rt<W2,S2>::logic operator ^ ( const ac_int<W2,S2> &op2) const {
    typename rt<W2,S2>::logic r;
    Base::bitwise_xor(op2, r);
    return r;
  }
  // Bitwise assign (not arithmetic): and, or, xor ----------------------------
  template<int W2, bool S2>
  ac_int &operator &= ( const ac_int<W2,S2> &op2 ) {
    Base r;
    Base::bitwise_and(op2, r);
    Base::operator=(r);
    bit_adjust();
    return *this;
  }
  template<int W2, bool S2>
  ac_int &operator |= ( const ac_int<W2,S2> &op2 ) {
    Base r;
    Base::bitwise_or(op2, r);
    Base::operator=(r);
    bit_adjust();
    return *this;
  }
  template<int W2, bool S2>
  ac_int &operator ^= ( const ac_int<W2,S2> &op2 ) {
    Base r;
    Base::bitwise_xor(op2, r);
    Base::operator=(r);
    bit_adjust();
    return *this;
  }
  // Shift (result constrained by left operand) -------------------------------
  template<int W2>
  ac_int operator << ( const ac_int<W2,true> &op2 ) const {
    ac_int r;
    Base::shift_l2(op2.to_int(), r);
    r.bit_adjust();
    return r;
  }
  template<int W2>
  ac_int operator << ( const ac_int<W2,false> &op2 ) const {
    ac_int r;
    Base::shift_l(op2.to_uint(), r);
    r.bit_adjust();
    return r;
  }
  template<int W2>
  ac_int operator >> ( const ac_int<W2,true> &op2 ) const {
    ac_int r;
    Base::shift_r2(op2.to_int(), r);
    r.bit_adjust();
    return r;
  }
  template<int W2>
  ac_int operator >> ( const ac_int<W2,false> &op2 ) const {
    ac_int r;
    Base::shift_r(op2.to_uint(), r);
    r.bit_adjust();
    return r;
  }
  // Shift assign ------------------------------------------------------------
  template<int W2>
  ac_int &operator <<= ( const ac_int<W2,true> &op2 ) {
    Base r;
    Base::shift_l2(op2.to_int(), r);
    Base::operator=(r);
    bit_adjust();
    return *this;
  }
  template<int W2>
  ac_int &operator <<= ( const ac_int<W2,false> &op2 ) {
    Base r;
    Base::shift_l(op2.to_uint(), r);
    Base::operator=(r);
    bit_adjust();
    return *this;
  }
  template<int W2>
  ac_int &operator >>= ( const ac_int<W2,true> &op2 ) {
    Base r;
    Base::shift_r2(op2.to_int(), r);
    Base::operator=(r);
    bit_adjust();
    return *this;
  }
  template<int W2>
  ac_int &operator >>= ( const ac_int<W2,false> &op2 ) {
    Base r;
    Base::shift_r(op2.to_uint(), r);
    Base::operator=(r);
    bit_adjust();
    return *this;
  }
  // Relational ---------------------------------------------------------------
  template<int W2, bool S2>
  bool operator == ( const ac_int<W2,S2> &op2) const {
    return Base::equal(op2);
  }
  template<int W2, bool S2>
  bool operator != ( const ac_int<W2,S2> &op2) const {
    return !Base::equal(op2);
  }
  template<int W2, bool S2>
  bool operator < ( const ac_int<W2,S2> &op2) const {
    return Base::less_than(op2);
  }
  template<int W2, bool S2>
  bool operator >= ( const ac_int<W2,S2> &op2) const {
    return !Base::less_than(op2);
  }
  template<int W2, bool S2>
  bool operator > ( const ac_int<W2,S2> &op2) const {
    return Base::greater_than(op2);
  }
  template<int W2, bool S2>
  bool operator <= ( const ac_int<W2,S2> &op2) const {
    return !Base::greater_than(op2);
  }

  // Bit and Slice Select -----------------------------------------------------
  template<int WS, int WX, bool SX>
  inline const ac_int<WS,S> slc(const ac_int<WX,SX> &index) const {
    ac_int<WS,S> r;
    AC_ASSERT(index.to_int() >= 0, "Attempting to read slc with negative indeces");
    unsigned uindex = ac_int<WX-SX, false>(index).to_uint();
    Base::shift_r(uindex, r);
    r.bit_adjust();
    return r;
  }

  template<int WS>
  inline const ac_int<WS,S> slc(signed index) const {
    ac_int<WS,S> r;
    AC_ASSERT(index >= 0, "Attempting to read slc with negative indeces");
    unsigned uindex = index & ((unsigned)~0 >> 1);
    Base::shift_r(uindex, r);
    r.bit_adjust();
    return r;
  }
  template<int WS>
  inline const ac_int<WS,S> slc(unsigned uindex) const {
    ac_int<WS,S> r;
    Base::shift_r(uindex, r);
    r.bit_adjust();
    return r;
  }

  template<int W2, bool S2, int WX, bool SX>
  inline ac_int &set_slc(const ac_int<WX,SX> lsb, const ac_int<W2,S2> &slc) {
    AC_ASSERT(lsb.to_int() + W2 <= W && lsb.to_int() >= 0, "Out of bounds set_slc");
    if(W == W2)
      Base::operator =(slc);
    else {
      unsigned ulsb = ac_int<WX-SX, false>(lsb).to_uint();
      Base::set_slc(ulsb, W2, (ac_int<W2,true>) slc);
    }
    bit_adjust();   // in case sign bit was assigned
    return *this;
  }
  template<int W2, bool S2>
  inline ac_int &set_slc(signed lsb, const ac_int<W2,S2> &slc) {
    AC_ASSERT(lsb + W2 <= W && lsb >= 0, "Out of bounds set_slc");
    if(W == W2)
      Base::operator =(slc);
    else {
      unsigned ulsb = lsb & ((unsigned)~0 >> 1);
      Base::set_slc(ulsb, W2, (ac_int<W2,true>) slc);
    }
    bit_adjust();   // in case sign bit was assigned
    return *this;
  }
  template<int W2, bool S2>
  inline ac_int &set_slc(unsigned ulsb, const ac_int<W2,S2> &slc) {
    AC_ASSERT(ulsb + W2 <= W, "Out of bounds set_slc");
    if(W == W2)
      Base::operator =(slc);
    else
      Base::set_slc(ulsb, W2, (ac_int<W2,true>) slc);
    bit_adjust();   // in case sign bit was assigned
    return *this;
  }

  template<int Msb, int Lsb>
  inline ac::sliceref<Msb-Lsb+1,Lsb,Msb==W-1,S> range() {
    #if __cplusplus > 199711L
    static_assert(Msb-Lsb+1 > 0, "Range length not positive: MSB < LSB");
    static_assert(Lsb >= 0, "LSB is negative");
    static_assert(Msb < W, "MSB >= W");
    #endif
    return ac::sliceref<Msb-Lsb+1,Lsb,Msb==W-1,S>(Base::v);
  }

  class ac_bitref {
# if defined(__SYNTHESIS__) && !defined(AC_IGNORE_BUILTINS)
# pragma builtin
# endif
    ac_int &d_bv;
    unsigned d_index;
  public:
    ac_bitref( ac_int *bv, unsigned index=0 ) : d_bv(*bv), d_index(index) {}
    operator bool () const { return (d_index < W) ? (d_bv.v[d_index>>5]>>(d_index&31) & 1) : 0; }

    template<int W2, bool S2>
    operator ac_int<W2,S2> () const { return operator bool (); }

    inline ac_bitref operator = ( int val ) {
      // lsb of int (val&1) is written to bit
      if(d_index < W) {
        int *pval = &d_bv.v[d_index>>5];
        *pval ^= (*pval ^ ( (unsigned) val << (d_index&31) )) & 1 << (d_index&31);
        d_bv.bit_adjust();   // in case sign bit was assigned
      }
      return *this;
    }
    template<int W2, bool S2>
    inline ac_bitref operator = ( const ac_int<W2,S2> &val ) {
      return operator =(val.to_int());
    }
    inline ac_bitref operator = ( const ac_bitref &val ) {
      return operator =((int) (bool) val);
    }
  };

  ac_bitref operator [] ( unsigned int uindex) {
    AC_ASSERT(uindex < W, "Attempting to read bit beyond MSB");
    ac_bitref bvh( this, uindex );
    return bvh;
  }
  ac_bitref operator [] ( int index) {
    AC_ASSERT(index >= 0, "Attempting to read bit with negative index");
    unsigned uindex = index & ((unsigned)~0 >> 1);
    AC_ASSERT(uindex < W, "Attempting to read bit beyond MSB");
    ac_bitref bvh( this, uindex );
    return bvh;
  }
  template<int W2, bool S2>
  ac_bitref operator [] ( const ac_int<W2,S2> &index) {
    AC_ASSERT(index.to_int() >= 0, "Attempting to read bit with negative index");
    unsigned uindex = ac_int<W2-S2,false>(index).to_uint();
    AC_ASSERT(uindex < W, "Attempting to read bit beyond MSB");
    ac_bitref bvh( this, uindex );
    return bvh;
  }
  bool operator [] ( unsigned int uindex) const {
    AC_ASSERT(uindex < W, "Attempting to read bit beyond MSB");
    return (uindex < W) ? (Base::v[uindex>>5]>>(uindex&31) & 1) : 0;
  }
  bool operator [] ( int index) const {
    AC_ASSERT(index >= 0, "Attempting to read bit with negative index");
    unsigned uindex = index & ((unsigned)~0 >> 1);
    AC_ASSERT(uindex < W, "Attempting to read bit beyond MSB");
    return (uindex < W) ? (Base::v[uindex>>5]>>(uindex&31) & 1) : 0;
  }
  template<int W2, bool S2>
  bool operator [] ( const ac_int<W2,S2> &index) const {
    AC_ASSERT(index.to_int() >= 0, "Attempting to read bit with negative index");
    unsigned uindex = ac_int<W2-S2,false>(index).to_uint();
    AC_ASSERT(uindex < W, "Attempting to read bit beyond MSB");
    return (uindex < W) ? (Base::v[uindex>>5]>>(uindex&31) & 1) : 0;
  }

  typename rt_unary::leading_sign leading_sign() const {
    unsigned ls = Base::leading_bits(S & (Base::v[N-1] < 0)) - (32*N - W)-S;
    return ls;
  }
  typename rt_unary::leading_sign leading_sign(bool &all_sign) const {
    unsigned ls = Base::leading_bits(S & (Base::v[N-1] < 0)) - (32*N - W)-S;
    all_sign = (ls == W-S);
    return ls;
  }
  // returns false if number is denormal
  template<int WE, bool SE>
  bool normalize(ac_int<WE,SE> &exp) {
    return normalize_private(exp);
  }
  // returns false if number is denormal, minimum exponent is reserved (usually for encoding special values/errors)
  template<int WE, bool SE>
  bool normalize_RME(ac_int<WE,SE> &exp) {
    return normalize_private(exp, true);
  }
  bool and_reduce() const {
    return ac_private::iv_equal_ones_to<W,N>(Base::v);
  }
  bool or_reduce() const {
    return !Base::equal_zero();
  }
  bool xor_reduce() const {
    unsigned r = Base::v[N-1];
    if(S) {
      const unsigned rem = (32-W)&31;
      r = (r << rem) >> rem;
    }
    if(N > 1)
      r ^= Base::v[N-2];
    if(N > 2) {
      for(int i=0; i<N-2; i++)
        r ^= Base::v[i];
    }
    if(W > 16)
      r ^= r >> 16;
    if(W > 8)
      r ^= r >> 8;
    if(W > 4)
      r ^= r >> 4;
    if(W > 2)
      r ^= r >> 2;
    if(W > 1)
    r ^= r >> 1;
    return r&1;
  }

  inline void bit_fill_hex(const char *str) {
    // Zero Pads if str is too short, throws ms bits away if str is too long
    // Asserts if anything other than 0-9a-fA-F is encountered
    ac_int<W,S> res = 0;
    while(*str) {
      char c = *str;
      int h = 0;
      if(c >= '0' && c <= '9')
        h = c - '0';
      else if(c >= 'A' && c <= 'F')
        h = c - 'A' + 10;
      else if(c >= 'a' && c <= 'f')
        h = c - 'a' + 10;
      else {
        AC_ASSERT(!c, "Invalid hex digit");
        break;
      }
      res <<= ac_int<3,false>(4);
      res |= ac_int<4,false>(h);
      str++;
    }
    *this = res;
  }

  template<int Na>
  inline void bit_fill(const int (&ivec)[Na], bool bigendian=true) {
    // bit_fill from integer vector
    //   if W > N*32, missing most significant bits are zeroed
    //   if W < N*32, additional bits in ivec are ignored (no overflow checking)
    // Example:
    //   ac_int<80,false> x;    int vec[] = { 0xffffa987, 0x6543210f, 0xedcba987 };
    //   x.bit_fill(vec);   // vec[0] fill bits 79-64
    enum { N0 = (W+31)/32, M = AC_MIN(N0,Na) };
    ac_int<M*32,false> res = 0;
    for(int i=0; i < M; i++)
      res.set_slc(i*32, ac_int<32>(ivec[bigendian ? M-1-i : i]));
    *this = res;
  }
};

namespace ac {
  template<typename T, typename T2>
  struct rt_2T {
    typedef typename ac_private::map<T>::t map_T;
    typedef typename ac_private::map<T2>::t map_T2;
    typedef typename map_T::template rt_T< map_T2 >::mult mult;
    typedef typename map_T::template rt_T< map_T2 >::plus plus;
    typedef typename map_T::template rt_T< map_T2 >::minus minus;
    typedef typename map_T::template rt_T< map_T2 >::minus2 minus2;
    typedef typename map_T::template rt_T< map_T2 >::logic logic;
    typedef typename map_T::template rt_T< map_T2 >::div div;
    typedef typename map_T::template rt_T< map_T2 >::div2 div2;
  };
}

namespace ac {
  template<typename T>
  struct ac_int_represent {
    enum { t_w = ac_private::c_type_params<T>::W, t_s = ac_private::c_type_params<T>::S };
    typedef ac_int<t_w,t_s> type;
  };
  template<> struct ac_int_represent<float> {};
  template<> struct ac_int_represent<double> {};
  template<int W, bool S>
  struct ac_int_represent< ac_int<W,S> > {
    typedef ac_int<W,S> type;
  };
}

namespace ac_private {
  template<int W2, bool S2>
  struct rt_ac_int_T< ac_int<W2,S2> > {
    typedef ac_int<W2,S2> i2_t;
    template<int W, bool S>
    struct op1 {
      typedef ac_int<W,S> i_t;
      typedef typename i_t::template rt<W2,S2>::mult mult;
      typedef typename i_t::template rt<W2,S2>::plus plus;
      typedef typename i_t::template rt<W2,S2>::minus minus;
      typedef typename i2_t::template rt<W,S>::minus minus2;
      typedef typename i_t::template rt<W2,S2>::logic logic;
      typedef typename i_t::template rt<W2,S2>::div div;
      typedef typename i2_t::template rt<W,S>::div div2;
      typedef typename i_t::template rt<W2,S2>::mod mod;
      typedef typename i2_t::template rt<W,S>::mod mod2;
    };
  };

  template<typename T>
  struct rt_ac_int_T< c_type<T> > {
    typedef typename ac::ac_int_represent<T>::type i2_t;
    enum { W2 = i2_t::width, S2 = i2_t::sign };
    template<int W, bool S>
    struct op1 {
      typedef ac_int<W,S> i_t;
      typedef typename i_t::template rt<W2,S2>::mult mult;
      typedef typename i_t::template rt<W2,S2>::plus plus;
      typedef typename i_t::template rt<W2,S2>::minus minus;
      typedef typename i2_t::template rt<W,S>::minus minus2;
      typedef typename i_t::template rt<W2,S2>::logic logic;
      typedef typename i_t::template rt<W2,S2>::div div;
      typedef typename i2_t::template rt<W,S>::div div2;
      typedef typename i_t::template rt<W2,S2>::mod mod;
      typedef typename i2_t::template rt<W,S>::mod mod2;
    };
  };
}


// Specializations for constructors on integers that bypass bit adjusting
//  and are therefore more efficient
template<> inline ac_int<1,true>::ac_int( bool b ) { v[0] = b ? -1 : 0; }

template<> inline ac_int<1,false>::ac_int( bool b ) { v[0] = b; }
template<> inline ac_int<1,false>::ac_int( signed char b ) { v[0] = b&1; }
template<> inline ac_int<1,false>::ac_int( unsigned char b ) { v[0] = b&1; }
template<> inline ac_int<1,false>::ac_int( signed short b ) { v[0] = b&1; }
template<> inline ac_int<1,false>::ac_int( unsigned short b ) { v[0] = b&1; }
template<> inline ac_int<1,false>::ac_int( signed int b ) { v[0] = b&1; }
template<> inline ac_int<1,false>::ac_int( unsigned int b ) { v[0] = b&1; }
template<> inline ac_int<1,false>::ac_int( signed long b ) { v[0] = b&1; }
template<> inline ac_int<1,false>::ac_int( unsigned long b ) { v[0] = b&1; }
template<> inline ac_int<1,false>::ac_int( Ulong b ) { v[0] = (int) b&1; }
template<> inline ac_int<1,false>::ac_int( Slong b ) { v[0] = (int) b&1; }

template<> inline ac_int<8,true>::ac_int( bool b ) { v[0] = b; }
template<> inline ac_int<8,false>::ac_int( bool b ) { v[0] = b; }
template<> inline ac_int<8,true>::ac_int( signed char b ) { v[0] = b; }
template<> inline ac_int<8,false>::ac_int( unsigned char b ) { v[0] = b; }
template<> inline ac_int<8,true>::ac_int( unsigned char b ) { v[0] = (signed char) b; }
template<> inline ac_int<8,false>::ac_int( signed char b ) { v[0] = (unsigned char) b; }

template<> inline ac_int<16,true>::ac_int( bool b ) { v[0] = b; }
template<> inline ac_int<16,false>::ac_int( bool b ) { v[0] = b; }
template<> inline ac_int<16,true>::ac_int( signed char b ) { v[0] = b; }
template<> inline ac_int<16,false>::ac_int( unsigned char b ) { v[0] = b; }
template<> inline ac_int<16,true>::ac_int( unsigned char b ) { v[0] = b; }
template<> inline ac_int<16,false>::ac_int( signed char b ) { v[0] = (unsigned short) b; }
template<> inline ac_int<16,true>::ac_int( signed short b ) { v[0] = b; }
template<> inline ac_int<16,false>::ac_int( unsigned short b ) { v[0] = b; }
template<> inline ac_int<16,true>::ac_int( unsigned short b ) { v[0] = (signed short) b; }
template<> inline ac_int<16,false>::ac_int( signed short b ) { v[0] = (unsigned short) b; }

template<> inline ac_int<32,true>::ac_int( signed int b ) { v[0] = b; }
template<> inline ac_int<32,true>::ac_int( unsigned int b ) { v[0] = b; }
template<> inline ac_int<32,false>::ac_int( signed int b ) { v[0] = b; v[1] = 0;}
template<> inline ac_int<32,false>::ac_int( unsigned int b ) { v[0] = b; v[1] = 0;}

template<> inline ac_int<32,true>::ac_int( Slong b ) { v[0] = (int) b; }
template<> inline ac_int<32,true>::ac_int( Ulong b ) { v[0] = (int) b; }
template<> inline ac_int<32,false>::ac_int( Slong b ) { v[0] = (int) b; v[1] = 0;}
template<> inline ac_int<32,false>::ac_int( Ulong b ) { v[0] = (int) b; v[1] = 0;}

template<> inline ac_int<64,true>::ac_int( Slong b ) { v[0] = (int) b; v[1] = (int) (b >> 32); }
template<> inline ac_int<64,true>::ac_int( Ulong b ) { v[0] = (int) b; v[1] = (int) (b >> 32);}
template<> inline ac_int<64,false>::ac_int( Slong b ) { v[0] = (int) b; v[1] = (int) ((Ulong) b >> 32); v[2] = 0; }
template<> inline ac_int<64,false>::ac_int( Ulong b ) { v[0] = (int) b; v[1] = (int) (b >> 32); v[2] = 0; }

// Stream --------------------------------------------------------------------

template<int W, bool S>
inline std::ostream& operator << (std::ostream &os, const ac_int<W,S> &x) {
#ifndef __SYNTHESIS__
  if ((os.flags() & std::ios::hex) != 0) {
    os << x.to_string(AC_HEX);
  } else if ((os.flags() & std::ios::oct) != 0) {
    os << x.to_string(AC_OCT);
  } else {
    os << x.to_string(AC_DEC);
  }
#endif
  return os;
}

// Macros for Binary Operators with Integers --------------------------------------------

#define BIN_OP_WITH_INT(BIN_OP, C_TYPE, WI, SI, RTYPE)  \
  template<int W, bool S> \
  inline typename ac_int<WI,SI>::template rt<W,S>::RTYPE operator BIN_OP ( C_TYPE i_op, const ac_int<W,S> &op) {  \
    return ac_int<WI,SI>(i_op).operator BIN_OP (op);  \
  } \
  template<int W, bool S>   \
  inline typename ac_int<W,S>::template rt<WI,SI>::RTYPE operator BIN_OP ( const ac_int<W,S> &op, C_TYPE i_op) {  \
    return op.operator BIN_OP (ac_int<WI,SI>(i_op));  \
  }

#define REL_OP_WITH_INT(REL_OP, C_TYPE, W2, S2)  \
  template<int W, bool S>   \
  inline bool operator REL_OP ( const ac_int<W,S> &op, C_TYPE op2) {  \
    return op.operator REL_OP (ac_int<W2,S2>(op2));  \
  }  \
  template<int W, bool S> \
  inline bool operator REL_OP ( C_TYPE op2, const ac_int<W,S> &op) {  \
    return ac_int<W2,S2>(op2).operator REL_OP (op);  \
  }

#define ASSIGN_OP_WITH_INT(ASSIGN_OP, C_TYPE, W2, S2)  \
  template<int W, bool S>   \
  inline ac_int<W,S> &operator ASSIGN_OP ( ac_int<W,S> &op, C_TYPE op2) {  \
    return op.operator ASSIGN_OP (ac_int<W2,S2>(op2));  \
  }

#define OPS_WITH_INT(C_TYPE, WI, SI) \
  BIN_OP_WITH_INT(*, C_TYPE, WI, SI, mult) \
  BIN_OP_WITH_INT(+, C_TYPE, WI, SI, plus) \
  BIN_OP_WITH_INT(-, C_TYPE, WI, SI, minus) \
  BIN_OP_WITH_INT(/, C_TYPE, WI, SI, div) \
  BIN_OP_WITH_INT(%, C_TYPE, WI, SI, mod) \
  BIN_OP_WITH_INT(>>, C_TYPE, WI, SI, arg1) \
  BIN_OP_WITH_INT(<<, C_TYPE, WI, SI, arg1) \
  BIN_OP_WITH_INT(&, C_TYPE, WI, SI, logic) \
  BIN_OP_WITH_INT(|, C_TYPE, WI, SI, logic) \
  BIN_OP_WITH_INT(^, C_TYPE, WI, SI, logic) \
  \
  REL_OP_WITH_INT(==, C_TYPE, WI, SI) \
  REL_OP_WITH_INT(!=, C_TYPE, WI, SI) \
  REL_OP_WITH_INT(>, C_TYPE, WI, SI) \
  REL_OP_WITH_INT(>=, C_TYPE, WI, SI) \
  REL_OP_WITH_INT(<, C_TYPE, WI, SI) \
  REL_OP_WITH_INT(<=, C_TYPE, WI, SI) \
  \
  ASSIGN_OP_WITH_INT(+=, C_TYPE, WI, SI) \
  ASSIGN_OP_WITH_INT(-=, C_TYPE, WI, SI) \
  ASSIGN_OP_WITH_INT(*=, C_TYPE, WI, SI) \
  ASSIGN_OP_WITH_INT(/=, C_TYPE, WI, SI) \
  ASSIGN_OP_WITH_INT(%=, C_TYPE, WI, SI) \
  ASSIGN_OP_WITH_INT(>>=, C_TYPE, WI, SI) \
  ASSIGN_OP_WITH_INT(<<=, C_TYPE, WI, SI) \
  ASSIGN_OP_WITH_INT(&=, C_TYPE, WI, SI) \
  ASSIGN_OP_WITH_INT(|=, C_TYPE, WI, SI) \
  ASSIGN_OP_WITH_INT(^=, C_TYPE, WI, SI)

// ------------------------------------- End of Macros for Binary Operators with Integers

// for backward compatability with v3.9.0 and earlier define following macro
#ifdef AC_INT_NS_FOR_MIXED_OPERATORS
namespace ac {
  namespace ops_with_other_types {
#endif
//  Mixed Operators with Integers  -----------------------------------------------
OPS_WITH_INT(bool, 1, false)
OPS_WITH_INT(char, 8, true)
OPS_WITH_INT(signed char, 8, true)
OPS_WITH_INT(unsigned char, 8, false)
OPS_WITH_INT(short, 16, true)
OPS_WITH_INT(unsigned short, 16, false)
OPS_WITH_INT(int, 32, true)
OPS_WITH_INT(unsigned int, 32, false)
OPS_WITH_INT(long, ac_private::long_w, true)
OPS_WITH_INT(unsigned long, ac_private::long_w, false)
OPS_WITH_INT(Slong, 64, true)
OPS_WITH_INT(Ulong, 64, false)
// -----------------------------------------  End of Mixed Operators with Integers
#ifdef AC_INT_NS_FOR_MIXED_OPERATORS
  }  // ops_with_other_types namespace
}
using namespace ac::ops_with_other_types;
#endif

namespace ac {
  // Functions to fill bits

  template<typename T>
  inline T bit_fill_hex(const char *str) {
    T res;
    res.bit_fill_hex(str);
    return res;
  }

  // returns bit_fill for type
  //   example:
  //   ac_int<80,false> x = ac::bit_fill< ac_int<80,false> > ((int [3]) {0xffffa987, 0x6543210f, 0xedcba987 });
  template<typename T, int N>
  inline T bit_fill(const int (&ivec)[N], bool bigendian=true) {
    T res;
    res.bit_fill(ivec, bigendian);
    return res;
  }

}  // ac namespace

//  Mixed Operators with Pointers  -----------------------------------------------

// Addition of ac_int and  pointer
template<typename T, int W, bool S>
T *operator +(T *ptr, const ac_int<W,S> &op2) {
  return ptr + op2.to_int64();
}
template<typename T, int W, bool S>
T *operator +(const ac_int<W,S> &op2, T *ptr) {
  return ptr + op2.to_int64();
}
// Subtraction of ac_int from pointer
template<typename T, int W, bool S>
T *operator -(T *ptr, const ac_int<W,S> &op2) {
  return ptr - op2.to_int64();
}
// -----------------------------------------  End of Mixed Operators with Pointers

namespace ac_intN {
  ///////////////////////////////////////////////////////////////////////////////
  //  Predefined for ease of use
  ///////////////////////////////////////////////////////////////////////////////
  typedef ac_int<1,          true>   int1;
  typedef ac_int<1,          false>  uint1;
  typedef ac_int<2,          true>   int2;
  typedef ac_int<2,          false>  uint2;
  typedef ac_int<3,          true>   int3;
  typedef ac_int<3,          false>  uint3;
  typedef ac_int<4,          true>   int4;
  typedef ac_int<4,          false>  uint4;
  typedef ac_int<5,          true>   int5;
  typedef ac_int<5,          false>  uint5;
  typedef ac_int<6,          true>   int6;
  typedef ac_int<6,          false>  uint6;
  typedef ac_int<7,          true>   int7;
  typedef ac_int<7,          false>  uint7;
  typedef ac_int<8,          true>   int8;
  typedef ac_int<8,          false>  uint8;
  typedef ac_int<9,          true>   int9;
  typedef ac_int<9,          false>  uint9;
  typedef ac_int<10,         true>   int10;
  typedef ac_int<10,         false>  uint10;
  typedef ac_int<11,         true>   int11;
  typedef ac_int<11,         false>  uint11;
  typedef ac_int<12,         true>   int12;
  typedef ac_int<12,         false>  uint12;
  typedef ac_int<13,         true>   int13;
  typedef ac_int<13,         false>  uint13;
  typedef ac_int<14,         true>   int14;
  typedef ac_int<14,         false>  uint14;
  typedef ac_int<15,         true>   int15;
  typedef ac_int<15,         false>  uint15;
  typedef ac_int<16,         true>   int16;
  typedef ac_int<16,         false>  uint16;
  typedef ac_int<17,         true>   int17;
  typedef ac_int<17,         false>  uint17;
  typedef ac_int<18,         true>   int18;
  typedef ac_int<18,         false>  uint18;
  typedef ac_int<19,         true>   int19;
  typedef ac_int<19,         false>  uint19;
  typedef ac_int<20,         true>   int20;
  typedef ac_int<20,         false>  uint20;
  typedef ac_int<21,         true>   int21;
  typedef ac_int<21,         false>  uint21;
  typedef ac_int<22,         true>   int22;
  typedef ac_int<22,         false>  uint22;
  typedef ac_int<23,         true>   int23;
  typedef ac_int<23,         false>  uint23;
  typedef ac_int<24,         true>   int24;
  typedef ac_int<24,         false>  uint24;
  typedef ac_int<25,         true>   int25;
  typedef ac_int<25,         false>  uint25;
  typedef ac_int<26,         true>   int26;
  typedef ac_int<26,         false>  uint26;
  typedef ac_int<27,         true>   int27;
  typedef ac_int<27,         false>  uint27;
  typedef ac_int<28,         true>   int28;
  typedef ac_int<28,         false>  uint28;
  typedef ac_int<29,         true>   int29;
  typedef ac_int<29,         false>  uint29;
  typedef ac_int<30,         true>   int30;
  typedef ac_int<30,         false>  uint30;
  typedef ac_int<31,         true>   int31;
  typedef ac_int<31,         false>  uint31;
  typedef ac_int<32,         true>   int32;
  typedef ac_int<32,         false>  uint32;
  typedef ac_int<33,         true>   int33;
  typedef ac_int<33,         false>  uint33;
  typedef ac_int<34,         true>   int34;
  typedef ac_int<34,         false>  uint34;
  typedef ac_int<35,         true>   int35;
  typedef ac_int<35,         false>  uint35;
  typedef ac_int<36,         true>   int36;
  typedef ac_int<36,         false>  uint36;
  typedef ac_int<37,         true>   int37;
  typedef ac_int<37,         false>  uint37;
  typedef ac_int<38,         true>   int38;
  typedef ac_int<38,         false>  uint38;
  typedef ac_int<39,         true>   int39;
  typedef ac_int<39,         false>  uint39;
  typedef ac_int<40,         true>   int40;
  typedef ac_int<40,         false>  uint40;
  typedef ac_int<41,         true>   int41;
  typedef ac_int<41,         false>  uint41;
  typedef ac_int<42,         true>   int42;
  typedef ac_int<42,         false>  uint42;
  typedef ac_int<43,         true>   int43;
  typedef ac_int<43,         false>  uint43;
  typedef ac_int<44,         true>   int44;
  typedef ac_int<44,         false>  uint44;
  typedef ac_int<45,         true>   int45;
  typedef ac_int<45,         false>  uint45;
  typedef ac_int<46,         true>   int46;
  typedef ac_int<46,         false>  uint46;
  typedef ac_int<47,         true>   int47;
  typedef ac_int<47,         false>  uint47;
  typedef ac_int<48,         true>   int48;
  typedef ac_int<48,         false>  uint48;
  typedef ac_int<49,         true>   int49;
  typedef ac_int<49,         false>  uint49;
  typedef ac_int<50,         true>   int50;
  typedef ac_int<50,         false>  uint50;
  typedef ac_int<51,         true>   int51;
  typedef ac_int<51,         false>  uint51;
  typedef ac_int<52,         true>   int52;
  typedef ac_int<52,         false>  uint52;
  typedef ac_int<53,         true>   int53;
  typedef ac_int<53,         false>  uint53;
  typedef ac_int<54,         true>   int54;
  typedef ac_int<54,         false>  uint54;
  typedef ac_int<55,         true>   int55;
  typedef ac_int<55,         false>  uint55;
  typedef ac_int<56,         true>   int56;
  typedef ac_int<56,         false>  uint56;
  typedef ac_int<57,         true>   int57;
  typedef ac_int<57,         false>  uint57;
  typedef ac_int<58,         true>   int58;
  typedef ac_int<58,         false>  uint58;
  typedef ac_int<59,         true>   int59;
  typedef ac_int<59,         false>  uint59;
  typedef ac_int<60,         true>   int60;
  typedef ac_int<60,         false>  uint60;
  typedef ac_int<61,         true>   int61;
  typedef ac_int<61,         false>  uint61;
  typedef ac_int<62,         true>   int62;
  typedef ac_int<62,         false>  uint62;
  typedef ac_int<63,         true>   int63;
  typedef ac_int<63,         false>  uint63;
}  // namespace ac_intN

#ifndef AC_NOT_USING_INTN
using namespace ac_intN;
#endif

///////////////////////////////////////////////////////////////////////////////

#if (defined(_MSC_VER) && !defined(__EDG__))
#pragma warning( disable: 4700 )
#endif
#if (defined(__GNUC__) && ( __GNUC__ == 4 && __GNUC_MINOR__ >= 6 || __GNUC__ > 4 ) && !defined(__EDG__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
#endif

// Global templatized functions for easy initialization to special values
template<ac_special_val V, int W, bool S>
inline ac_int<W,S> value(ac_int<W,S>) {
  ac_int<W,S> r;
  return r.template set_val<V>();
}
// forward declaration, otherwise GCC errors when calling init_array
template<ac_special_val V, int W, int I, bool S, ac_q_mode Q, ac_o_mode O>
inline ac_fixed<W,I,S,Q,O> value(ac_fixed<W,I,S,Q,O>);

#define SPECIAL_VAL_FOR_INTS_DC(C_TYPE, WI, SI) \
template<> inline C_TYPE value<AC_VAL_DC>(C_TYPE) { C_TYPE x; return x; }

// -- C int types -----------------------------------------------------------------
#define SPECIAL_VAL_FOR_INTS(C_TYPE, WI, SI) \
template<ac_special_val val> inline C_TYPE value(C_TYPE); \
template<> inline C_TYPE value<AC_VAL_0>(C_TYPE) { return (C_TYPE)0; } \
SPECIAL_VAL_FOR_INTS_DC(C_TYPE, WI, SI) \
template<> inline C_TYPE value<AC_VAL_QUANTUM>(C_TYPE) { return (C_TYPE)1; } \
template<> inline C_TYPE value<AC_VAL_MAX>(C_TYPE) { return (C_TYPE)(SI ? ~(((C_TYPE) 1) << (WI-1)) : (C_TYPE) -1); } \
template<> inline C_TYPE value<AC_VAL_MIN>(C_TYPE) { return (C_TYPE)(SI ? ((C_TYPE) 1) << (WI-1) : (C_TYPE) 0); }

SPECIAL_VAL_FOR_INTS(bool, 1, false)
SPECIAL_VAL_FOR_INTS(char, 8, true)
SPECIAL_VAL_FOR_INTS(signed char, 8, true)
SPECIAL_VAL_FOR_INTS(unsigned char, 8, false)
SPECIAL_VAL_FOR_INTS(short, 16, true)
SPECIAL_VAL_FOR_INTS(unsigned short, 16, false)
SPECIAL_VAL_FOR_INTS(int, 32, true)
SPECIAL_VAL_FOR_INTS(unsigned int, 32, false)
SPECIAL_VAL_FOR_INTS(long, ac_private::long_w, true)
SPECIAL_VAL_FOR_INTS(unsigned long, ac_private::long_w, false)
SPECIAL_VAL_FOR_INTS(Slong, 64, true)
SPECIAL_VAL_FOR_INTS(Ulong, 64, false)

#define INIT_ARRAY_SPECIAL_VAL_FOR_INTS(C_TYPE) \
  template<ac_special_val V> \
  inline bool init_array(C_TYPE *a, int n) { \
    C_TYPE t = value<V>((C_TYPE) 0); \
    for(int i=0; i < n; i++) \
      a[i] = t; \
    return true; \
  }

namespace ac {
// PUBLIC FUNCTIONS
// function to initialize (or uninitialize) arrays
  template<ac_special_val V, int W, bool S>
  inline bool init_array(ac_int<W,S> *a, int n) {
    ac_int<W,S> t;
    t.template set_val<V>();
    for(int i=0; i < n; i++)
      a[i] = t;
    return true;
  }

  INIT_ARRAY_SPECIAL_VAL_FOR_INTS(bool)
  INIT_ARRAY_SPECIAL_VAL_FOR_INTS(char)
  INIT_ARRAY_SPECIAL_VAL_FOR_INTS(signed char)
  INIT_ARRAY_SPECIAL_VAL_FOR_INTS(unsigned char)
  INIT_ARRAY_SPECIAL_VAL_FOR_INTS(signed short)
  INIT_ARRAY_SPECIAL_VAL_FOR_INTS(unsigned short)
  INIT_ARRAY_SPECIAL_VAL_FOR_INTS(signed int)
  INIT_ARRAY_SPECIAL_VAL_FOR_INTS(unsigned int)
  INIT_ARRAY_SPECIAL_VAL_FOR_INTS(signed long)
  INIT_ARRAY_SPECIAL_VAL_FOR_INTS(unsigned long)
  INIT_ARRAY_SPECIAL_VAL_FOR_INTS(signed long long)
  INIT_ARRAY_SPECIAL_VAL_FOR_INTS(unsigned long long)
}

#if (defined(_MSC_VER) && !defined(__EDG__))
#pragma warning( pop )
#endif
#if (defined(__GNUC__) && ( __GNUC__ == 4 && __GNUC_MINOR__ >= 6 || __GNUC__ > 4 ) && !defined(__EDG__))
#pragma GCC diagnostic pop
#endif
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#ifdef __AC_NAMESPACE
}
#endif

#endif // __AC_INT_H
