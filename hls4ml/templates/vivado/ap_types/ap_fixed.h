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

#ifndef __AP_FIXED_H__
#define __AP_FIXED_H__

#include <ap_common.h>
#include <ap_fixed_base.h>
#include <ap_fixed_ref.h>

//---------------------------------------------------------------

/// Signed Arbitrary Precision Fixed-Point Type.
// default for _AP_Q, _AP_O and _AP_N set in ap_decl.h
template <int _AP_W, int _AP_I, ap_q_mode _AP_Q, ap_o_mode _AP_O, int _AP_N>
struct ap_fixed : ap_fixed_base<_AP_W, _AP_I, true, _AP_Q, _AP_O, _AP_N> {
  typedef ap_fixed_base<_AP_W, _AP_I, true, _AP_Q, _AP_O, _AP_N> Base;
  // Constructor
  /// default ctor
  INLINE ap_fixed() : Base() {}

  /// default copy ctor
  INLINE ap_fixed(const ap_fixed& op) { Base::V = op.V; }

  /// copy ctor from ap_fixed_base.
  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_fixed(const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2,
                                      _AP_O2, _AP_N2>& op)
      : Base(op) {}

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_fixed(const volatile ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2,
                                               _AP_O2, _AP_N2>& op)
      : Base(op) {}

  //// from ap_fixed
  //template <int _AP_W2, int _AP_I2, ap_q_mode _AP_Q2, ap_o_mode _AP_O2,
  //          int _AP_N2>
  //INLINE ap_fixed(
  //    const ap_fixed<_AP_W2, _AP_I2, _AP_Q2, _AP_O2, _AP_N2>& op)
  //    : Base(ap_fixed_base<_AP_W2, _AP_I2, true, _AP_Q2, _AP_O2, _AP_N2>(op)) {}

  //template <int _AP_W2, int _AP_I2, ap_q_mode _AP_Q2, ap_o_mode _AP_O2,
  //          int _AP_N2>
  //INLINE ap_fixed(
  //    const volatile ap_fixed<_AP_W2, _AP_I2, _AP_Q2, _AP_O2, _AP_N2>& op)
  //    : Base(ap_fixed_base<_AP_W2, _AP_I2, true, _AP_Q2, _AP_O2, _AP_N2>(op)) {}

  //// from ap_ufixed.
  //template <int _AP_W2, int _AP_I2, ap_q_mode _AP_Q2, ap_o_mode _AP_O2,
  //          int _AP_N2>
  //INLINE ap_fixed(
  //    const ap_ufixed<_AP_W2, _AP_I2, _AP_Q2, _AP_O2, _AP_N2>& op)
  //    : Base(ap_fixed_base<_AP_W2, _AP_I2, false, _AP_Q2, _AP_O2, _AP_N2>(op)) {
  //}

  //template <int _AP_W2, int _AP_I2, ap_q_mode _AP_Q2, ap_o_mode _AP_O2,
  //          int _AP_N2>
  //INLINE ap_fixed(
  //    const volatile ap_ufixed<_AP_W2, _AP_I2, _AP_Q2, _AP_O2, _AP_N2>& op)
  //    : Base(ap_fixed_base<_AP_W2, _AP_I2, false, _AP_Q2, _AP_O2, _AP_N2>(op)) {
  //}

  /// copy ctor from ap_int_base.
  template <int _AP_W2, bool _AP_S2>
  INLINE ap_fixed(const ap_int_base<_AP_W2, _AP_S2>& op) : Base(op) {}

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_fixed(const volatile ap_int_base<_AP_W2, _AP_S2>& op) : Base(op) {}

  //// from ap_int.
  //template <int _AP_W2>
  //INLINE ap_fixed(const ap_int<_AP_W2>& op)
  //    : Base(ap_int_base<_AP_W2, true>(op)) {}

  //template <int _AP_W2>
  //INLINE ap_fixed(const volatile ap_int<_AP_W2>& op)
  //    : Base(ap_int_base<_AP_W2, true>(op)) {}

  //// from ap_uint.
  //template <int _AP_W2>
  //INLINE ap_fixed(const ap_uint<_AP_W2>& op)
  //    : Base(ap_int_base<_AP_W2, false>(op)) {}

  //template <int _AP_W2>
  //INLINE ap_fixed(const volatile ap_uint<_AP_W2>& op)
  //    : Base(ap_int_base<_AP_W2, false>(op)) {}

  // from ap_bit_ref.
  template <int _AP_W2, bool _AP_S2>
  INLINE ap_fixed(const ap_bit_ref<_AP_W2, _AP_S2>& op) : Base(op) {}

  // from ap_range_ref.
  template <int _AP_W2, bool _AP_S2>
  INLINE ap_fixed(const ap_range_ref<_AP_W2, _AP_S2>& op) : Base(op) {}

  // from ap_concat_ref.
  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
  INLINE ap_fixed(const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>& op)
      : Base(op) {}

  // from af_bit_ref.
  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_fixed(
      const af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op)
      : Base(op) {}

  // from af_range_ref.
  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_fixed(
      const af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op)
      : Base(op) {}

// from c types.
#define CTOR(TYPE) \
  INLINE ap_fixed(TYPE v) : Base(v) {}

  CTOR(bool)
  CTOR(char)
  CTOR(signed char)
  CTOR(unsigned char)
  CTOR(short)
  CTOR(unsigned short)
  CTOR(int)
  CTOR(unsigned int)
  CTOR(long)
  CTOR(unsigned long)
  CTOR(ap_slong)
  CTOR(ap_ulong)
#if _AP_ENABLE_HALF_ == 1
  CTOR(half)
#endif
  CTOR(float)
  CTOR(double)
#undef CTOR

  INLINE ap_fixed(const char* s) : Base(s) {}

  INLINE ap_fixed(const char* s, signed char rd) : Base(s, rd) {}

  // Assignment
  // The assignment operator is technically inherited; however, it is always
  // hidden by an explicitly or implicitly defined assignment operator for the
  // derived class.
  /* XXX ctor will be used when right is not of proper type. */
  INLINE ap_fixed& operator=(
      const ap_fixed<_AP_W, _AP_I, _AP_Q, _AP_O, _AP_N>& op) {
    Base::V = op.V;
    return *this;
  }

  INLINE void operator=(
      const ap_fixed<_AP_W, _AP_I, _AP_Q, _AP_O, _AP_N>& op) volatile {
    Base::V = op.V;
  }

  INLINE ap_fixed& operator=(
      const volatile ap_fixed<_AP_W, _AP_I, _AP_Q, _AP_O, _AP_N>& op) {
    Base::V = op.V;
    return *this;
  }

  INLINE void operator=(
      const volatile ap_fixed<_AP_W, _AP_I, _AP_Q, _AP_O, _AP_N>& op) volatile {
    Base::V = op.V;
  }
}; // struct ap_fixed.

//-------------------------------------------------------------------

// Unsigned Arbitrary Precision Fixed-Point Type.
// default for _AP_Q, _AP_O and _AP_N set in ap_decl.h
template <int _AP_W, int _AP_I, ap_q_mode _AP_Q, ap_o_mode _AP_O, int _AP_N>
struct ap_ufixed : ap_fixed_base<_AP_W, _AP_I, false, _AP_Q, _AP_O, _AP_N> {
  typedef ap_fixed_base<_AP_W, _AP_I, false, _AP_Q, _AP_O, _AP_N> Base;
  // Constructor
  /// default ctor
  INLINE ap_ufixed() : Base() {}

  /// default copy ctor
  INLINE ap_ufixed(const ap_ufixed& op) { Base::V = op.V; }

  /// copy ctor from ap_fixed_base
  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_ufixed(const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2,
                                       _AP_O2, _AP_N2>& op)
      : Base(op) {}

  /// copy ctor from ap_fixed_base
  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_ufixed(const volatile ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2,
                                                _AP_O2, _AP_N2>& op)
      : Base(op) {}

  //template <int _AP_W2, int _AP_I2, ap_q_mode _AP_Q2, ap_o_mode _AP_O2,
  //          int _AP_N2>
  //INLINE ap_ufixed(
  //    const ap_fixed<_AP_W2, _AP_I2, _AP_Q2, _AP_O2, _AP_N2>& op)
  //    : Base(ap_fixed_base<_AP_W2, _AP_I2, true, _AP_Q2, _AP_O2, _AP_N2>(op)) {}

  //template <int _AP_W2, int _AP_I2, ap_q_mode _AP_Q2, ap_o_mode _AP_O2,
  //          int _AP_N2>
  //INLINE ap_ufixed(
  //    const volatile ap_fixed<_AP_W2, _AP_I2, _AP_Q2, _AP_O2, _AP_N2>& op)
  //    : Base(ap_fixed_base<_AP_W2, _AP_I2, true, _AP_Q2, _AP_O2, _AP_N2>(op)) {}

  //template <int _AP_W2, int _AP_I2, ap_q_mode _AP_Q2, ap_o_mode _AP_O2,
  //          int _AP_N2>
  //INLINE ap_ufixed(
  //    const ap_ufixed<_AP_W2, _AP_I2, _AP_Q2, _AP_O2, _AP_N2>& op)
  //    : Base(ap_fixed_base<_AP_W2, _AP_I2, false, _AP_Q2, _AP_O2, _AP_N2>(op)) {
  //}

  //template <int _AP_W2, int _AP_I2, ap_q_mode _AP_Q2, ap_o_mode _AP_O2,
  //          int _AP_N2>
  //INLINE ap_ufixed(
  //    const volatile ap_ufixed<_AP_W2, _AP_I2, _AP_Q2, _AP_O2, _AP_N2>& op)
  //    : Base(ap_fixed_base<_AP_W2, _AP_I2, false, _AP_Q2, _AP_O2, _AP_N2>(op)) {
  //}

  /// copy ctor from ap_int_base.
  template <int _AP_W2, bool _AP_S2>
  INLINE ap_ufixed(const ap_int_base<_AP_W2, _AP_S2>& op) : Base(op) {}

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_ufixed(const volatile ap_int_base<_AP_W2, _AP_S2>& op) : Base(op) {}

  //template <int _AP_W2>
  //INLINE ap_ufixed(const ap_int<_AP_W2>& op)
  //    : Base(ap_int_base<_AP_W2, true>(op)) {}

  //template <int _AP_W2>
  //INLINE ap_ufixed(const volatile ap_int<_AP_W2>& op)
  //    : Base(ap_int_base<_AP_W2, true>(op)) {}

  //template <int _AP_W2>
  //INLINE ap_ufixed(const ap_uint<_AP_W2>& op)
  //    : Base(ap_int_base<_AP_W2, false>(op)) {}

  //template <int _AP_W2>
  //INLINE ap_ufixed(const volatile ap_uint<_AP_W2>& op)
  //    : Base(ap_int_base<_AP_W2, false>(op)) {}

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_ufixed(const ap_bit_ref<_AP_W2, _AP_S2>& op) : Base(op) {}

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_ufixed(const ap_range_ref<_AP_W2, _AP_S2>& op) : Base(op) {}

  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
  INLINE ap_ufixed(const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>& op)
      : Base(op) {}

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_ufixed(
      const af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op)
      : Base(op) {}

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_ufixed(
      const af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& op)
      : Base(op) {}

#define CTOR(TYPE) \
  INLINE ap_ufixed(TYPE v) : Base(v) {}

  CTOR(bool)
  CTOR(char)
  CTOR(signed char)
  CTOR(unsigned char)
  CTOR(short)
  CTOR(unsigned short)
  CTOR(int)
  CTOR(unsigned int)
  CTOR(long)
  CTOR(unsigned long)
  CTOR(ap_slong)
  CTOR(ap_ulong)
#if _AP_ENABLE_HALF_ == 1
  CTOR(half)
#endif
  CTOR(float)
  CTOR(double)
#undef CTOR

  INLINE ap_ufixed(const char* s) : Base(s) {}

  INLINE ap_ufixed(const char* s, signed char rd) : Base(s, rd) {}

  // Assignment
  INLINE ap_ufixed& operator=(
      const ap_ufixed<_AP_W, _AP_I, _AP_Q, _AP_O, _AP_N>& op) {
    Base::V = op.V;
    return *this;
  }

  INLINE void operator=(
      const ap_ufixed<_AP_W, _AP_I, _AP_Q, _AP_O, _AP_N>& op) volatile {
    Base::V = op.V;
  }

  INLINE ap_ufixed& operator=(
      const volatile ap_ufixed<_AP_W, _AP_I, _AP_Q, _AP_O, _AP_N>& op) {
    Base::V = op.V;
    return *this;
  }

  INLINE void operator=(const volatile ap_ufixed<_AP_W, _AP_I, _AP_Q, _AP_O,
                                                 _AP_N>& op) volatile {
    Base::V = op.V;
  }
}; // struct ap_ufixed


#if !defined(__SYNTHESIS__) && (defined(SYSTEMC_H) || defined(SYSTEMC_INCLUDED))
// XXX sc_trace overload for ap_fixed is already included in
// "ap_sysc/ap_sc_extras.h", so do not define in synthesis.
template <int _AP_W, int _AP_I, ap_q_mode _AP_Q, ap_o_mode _AP_O, int _AP_N>
INLINE void sc_trace(sc_core::sc_trace_file* tf,
                     const ap_fixed<_AP_W, _AP_I, _AP_Q, _AP_O, _AP_N>& op,
                     const std::string& name) {
  tf->trace(sc_dt::sc_lv<_AP_W>(op.to_string(2).c_str()), name);
}

template <int _AP_W, int _AP_I, ap_q_mode _AP_Q, ap_o_mode _AP_O, int _AP_N>
INLINE void sc_trace(sc_core::sc_trace_file* tf,
                     const ap_ufixed<_AP_W, _AP_I, _AP_Q, _AP_O, _AP_N>& op,
                     const std::string& name) {
  tf->trace(sc_dt::sc_lv<_AP_W>(op.to_string(2).c_str()), name);
}
#endif // System C sim

// Specialization of std containers, so that std::complex<ap_fixed> can have its
// image part automatically zero-initialized when only real part is provided.
#include <ap_fixed_special.h>

#endif // ifndef __AP_FIXED_H__

// -*- cpp -*-
