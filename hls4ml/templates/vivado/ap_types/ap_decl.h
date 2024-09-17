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

#ifndef __AP_DECL_H__
#define __AP_DECL_H__

// ----------------------------------------------------------------------

#if !defined(__AP_FIXED_H__) && !defined(__AP_INT_H__) && !defined(__AUTOPILOT_CBE_H__) && !defined(__HLS_HALF_H__)
#error "Only ap_fixed.h and ap_int.h can be included directly in user code."
#endif

// Test __SYNTHESIS__ only for mode
#if !defined(__SYNTHESIS__) && (defined(AESL_SYN) || defined(__HLS_SYN__))
//#pragma message "AESL_SYN and __HLS_SYN__ should be replaced by __SYNTHESIS__"
#define __SYNTHESIS__
#endif

/* for safety*/
#if (defined(_AP_N) || defined(_AP_C))
#error One or more of the following is defined: _AP_N, _AP_C. Definition conflicts with their usage as template parameters.
#endif

/* for safety*/
#if (defined(_AP_W) || defined(_AP_I) || defined(_AP_S) || defined(_AP_Q) || \
     defined(_AP_O) || defined(_AP_W2) || defined(_AP_I2) ||                 \
     defined(_AP_S2) || defined(_AP_Q2) || defined(_AP_O2) ||                \
     defined(_AP_N) || defined(_AP_N2))
#error \
    "One or more of the following is defined: _AP_W, _AP_I, _AP_S, _AP_Q, _AP_O,  _AP_N, _AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2. Definition conflicts with their usage as template parameters."
#endif

/*for safety*/
#if (defined(_AP_W3) || defined(_AP_S3) || defined(_AP_W4) || defined(_AP_S4))
#error \
    "One or more of the following is defined: _AP_W3, _AP_S3, _AP_W4,_AP_S4. Definition conflicts with their usage as template parameters."
#endif

#if (defined(_AP_W1) || defined(_AP_S1) || defined(_AP_T) || \
     defined(_AP_T1) || defined(_AP_T2) || defined(_AP_T3) || defined(_AP_T4))
#error \
    "One or more of the following is defined: _AP_W1, _AP_S1, _AP_T,  _AP_T1, _AP_T2, _AP_T3, _AP_T4. Definition conflicts with their usage as template parameters."
#endif

#ifndef __cplusplus
#error "AP data type can only be used in C++"
#endif

// ----------------------------------------------------------------------

#ifndef __SC_COMPATIBLE__
/// ap_fixed quantification mode
enum ap_q_mode {
  AP_RND,         //< rounding to plus infinity
  AP_RND_ZERO,    //< rounding to zero
  AP_RND_MIN_INF, //< rounding to minus infinity
  AP_RND_INF,     //< rounding to infinity
  AP_RND_CONV,    //< convergent rounding
  AP_TRN,         //< truncation
  AP_TRN_ZERO,    //< truncation to zero
};

// FIXME for legacy code
#ifndef SYSTEMC_INCLUDED
#define SC_RND AP_RND
#define SC_RND_ZERO AP_RND_ZERO
#define SC_RND_MIN_INF AP_RND_MIN_INF
#define SC_RND_INF AP_RND_INF
#define SC_RND_CONV AP_RND_CONV
#define SC_TRN AP_TRN
#define SC_TRN_ZERO AP_TRN_ZERO
#endif // !defined(SYSTEMC_INCLUDED)

/// ap_fixed saturation mode
enum ap_o_mode {
  AP_SAT,      //< saturation
  AP_SAT_ZERO, //< saturation to zero
  AP_SAT_SYM,  //< symmetrical saturation
  AP_WRAP,     //< wrap-around (*)
  AP_WRAP_SM,  //< sign magnitude wrap-around (*)
};

// FIXME for legacy code
#ifndef SYSTEMC_INCLUDED
#define SC_SAT AP_SAT
#define SC_SAT_ZERO AP_SAT_ZERO
#define SC_SAT_SYM AP_SAT_SYM
#define SC_WRAP AP_WRAP
#define SC_WRAP_SM AP_WRAP_SM
#endif // !defined(SYSTEMC_INCLUDED)

#else // defined(__SC_COMPATIBLE__)

// There will not be sc_fxdefs.h, and the emu should be defined by ap_fixed.

/// ap_fixed quantification mode
enum ap_q_mode {
  SC_RND,         //< rounding to plus infinity
  SC_RND_ZERO,    //< rounding to zero
  SC_RND_MIN_INF, //< rounding to minus infinity
  SC_RND_INF,     //< rounding to infinity
  SC_RND_CONV,    //< convergent rounding
  SC_TRN,         //< truncation
  SC_TRN_ZERO,    //< truncation to zero
};

#define AP_RND SC_RND
#define AP_RND_ZERO SC_RND_ZERO
#define AP_RND_MIN_INF SC_RND_MIN_INF
#define AP_RND_INF SC_RND_INF
#define AP_RND_CONV SC_RND_CONV
#define AP_TRN SC_TRN
#define AP_TRN_ZERO SC_TRN_ZERO

/// ap_fixed saturation mode
enum ap_o_mode {
  SC_SAT,      //< saturation
  SC_SAT_ZERO, //< saturation to zero
  SC_SAT_SYM,  //< symmetrical saturation
  SC_WRAP,     //< wrap-around (*)
  SC_WRAP_SM,  //< sign magnitude wrap-around (*)
};

#define AP_SAT SC_SAT
#define AP_SAT_ZERO SC_SAT_ZERO
#define AP_SAT_SYM SC_SAT_SYM
#define AP_WRAP SC_WRAP
#define AP_WRAP_SM SC_WRAP_SM

#endif // defined(__SC_COMPATIBLE__)

template <int _AP_W, bool _AP_S>
struct ap_int_base;

template <int _AP_W>
struct ap_int;

template <int _AP_W>
struct ap_uint;

template <int _AP_W, bool _AP_S>
struct ap_range_ref;

template <int _AP_W, bool _AP_S>
struct ap_bit_ref;

template <int _AP_W1, typename _AP_T1, int _AP_W2, typename _AP_T2>
struct ap_concat_ref;

template <int _AP_W, int _AP_I, bool _AP_S = true, ap_q_mode _AP_Q = AP_TRN,
          ap_o_mode _AP_O = AP_WRAP, int _AP_N = 0>
struct ap_fixed_base;

template <int _AP_W, int _AP_I, ap_q_mode _AP_Q = AP_TRN,
          ap_o_mode _AP_O = AP_WRAP, int _AP_N = 0>
struct ap_fixed;

template <int _AP_W, int _AP_I, ap_q_mode _AP_Q = AP_TRN,
          ap_o_mode _AP_O = AP_WRAP, int _AP_N = 0>
struct ap_ufixed;

template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O,
          int _AP_N>
struct af_range_ref;

template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O,
          int _AP_N>
struct af_bit_ref;

/// string base mode
enum BaseMode { AP_BIN = 2, AP_OCT = 8, AP_DEC = 10, AP_HEX = 16 };

#ifndef SYSTEMC_INCLUDED
#define SC_BIN 2
#define SC_OCT 8
#define SC_DEC 10
#define SC_HEX 16
#endif // !defined(SYSTEMC_INCLUDED)

// Alias C data types
#ifdef _MSC_VER
typedef signed __int64 ap_slong;
typedef unsigned __int64 ap_ulong;
#else  // !defined(_MSC_VER)
typedef signed long long ap_slong;
typedef unsigned long long ap_ulong;
#endif // !defined(_MSC_VER)

enum {
  _AP_SIZE_char = 8,
  _AP_SIZE_short = sizeof(short) * 8,
  _AP_SIZE_int = sizeof(int) * 8,
  _AP_SIZE_long = sizeof(long) * 8,
  _AP_SIZE_ap_slong = sizeof(ap_slong) * 8
};

#endif // !defined(__AP_DECL_H__)

// -*- cpp -*-
