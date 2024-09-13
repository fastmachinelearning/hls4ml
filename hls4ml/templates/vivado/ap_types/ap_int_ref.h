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

#ifndef __AP_INT_REF_H__
#define __AP_INT_REF_H__

#ifndef __AP_INT_H__
#error "Only ap_fixed.h and ap_int.h can be included directly in user code."
#endif

#ifndef __cplusplus
#error "C++ is required to include this header file"

#else

#ifndef __SYNTHESIS__
#include <iostream>
#endif

/* Concatination reference.
   ----------------------------------------------------------------
*/
template <int _AP_W1, typename _AP_T1, int _AP_W2, typename _AP_T2>
struct ap_concat_ref {
  enum {
    _AP_WR = _AP_W1 + _AP_W2,
  };

  _AP_T1& mbv1;
  _AP_T2& mbv2;

  INLINE ap_concat_ref(const ap_concat_ref<_AP_W1, _AP_T1, _AP_W2, _AP_T2>& ref)
      : mbv1(ref.mbv1), mbv2(ref.mbv2) {}

  INLINE ap_concat_ref(_AP_T1& bv1, _AP_T2& bv2) : mbv1(bv1), mbv2(bv2) {}

  template <int _AP_W3, bool _AP_S3>
  INLINE ap_concat_ref& operator=(const ap_int_base<_AP_W3, _AP_S3>& val) {
    ap_int_base<_AP_W1 + _AP_W2, false> vval(val);
    int W_ref1 = mbv1.length();
    int W_ref2 = mbv2.length();
    ap_int_base<_AP_W1, false> Part1;
    Part1.V = _AP_ROOT_op_get_range(vval.V, W_ref2, W_ref1 + W_ref2 - 1);
    mbv1.set(Part1);
    ap_int_base<_AP_W2, false> Part2;
    Part2.V = _AP_ROOT_op_get_range(vval.V, 0, W_ref2 - 1);
    mbv2.set(Part2);
    return *this;
  }

  // assign op from hls supported C integral types.
  // FIXME disabled to support legacy code directly assign from sc_signal<T>
  //template <typename T>
  //INLINE typename _ap_type::enable_if<_ap_type::is_integral<T>::value,
  //                                    ap_concat_ref&>::type
  //operator=(T val) {
  //  ap_int_base<_AP_W1 + _AP_W2, false> tmpVal(val);
  //  return operator=(tmpVal);
  //}
#define ASSIGN_WITH_CTYPE(_Tp)                       \
  INLINE ap_concat_ref& operator=(_Tp val) {         \
    ap_int_base<_AP_W1 + _AP_W2, false> tmpVal(val); \
    return operator=(tmpVal);                        \
  }

  ASSIGN_WITH_CTYPE(bool)
  ASSIGN_WITH_CTYPE(char)
  ASSIGN_WITH_CTYPE(signed char)
  ASSIGN_WITH_CTYPE(unsigned char)
  ASSIGN_WITH_CTYPE(short)
  ASSIGN_WITH_CTYPE(unsigned short)
  ASSIGN_WITH_CTYPE(int)
  ASSIGN_WITH_CTYPE(unsigned int)
  ASSIGN_WITH_CTYPE(long)
  ASSIGN_WITH_CTYPE(unsigned long)
  ASSIGN_WITH_CTYPE(ap_slong)
  ASSIGN_WITH_CTYPE(ap_ulong)
#if _AP_ENABLE_HALF_ == 1
  ASSIGN_WITH_CTYPE(half)
#endif
  ASSIGN_WITH_CTYPE(float)
  ASSIGN_WITH_CTYPE(double)

#undef ASSIGN_WITH_CTYPE

  // Be explicit to prevent it from being deleted, as field d_bv
  // is of reference type.
  INLINE ap_concat_ref& operator=(
      const ap_concat_ref<_AP_W1, _AP_T1, _AP_W2, _AP_T2>& val) {
    ap_int_base<_AP_W1 + _AP_W2, false> tmpVal(val);
    return operator=(tmpVal);
  }

  template <int _AP_W3, typename _AP_T3, int _AP_W4, typename _AP_T4>
  INLINE ap_concat_ref& operator=(
      const ap_concat_ref<_AP_W3, _AP_T3, _AP_W4, _AP_T4>& val) {
    ap_int_base<_AP_W1 + _AP_W2, false> tmpVal(val);
    return operator=(tmpVal);
  }

  template <int _AP_W3, bool _AP_S3>
  INLINE ap_concat_ref& operator=(const ap_bit_ref<_AP_W3, _AP_S3>& val) {
    ap_int_base<_AP_W1 + _AP_W2, false> tmpVal(val);
    return operator=(tmpVal);
  }
  template <int _AP_W3, bool _AP_S3>
  INLINE ap_concat_ref& operator=(const ap_range_ref<_AP_W3, _AP_S3>& val) {
    ap_int_base<_AP_W1 + _AP_W2, false> tmpVal(val);
    return operator=(tmpVal);
  }

  template <int _AP_W3, int _AP_I3, bool _AP_S3, ap_q_mode _AP_Q3,
            ap_o_mode _AP_O3, int _AP_N3>
  INLINE ap_concat_ref& operator=(
      const af_range_ref<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3, _AP_N3>& val) {
    return operator=((const ap_int_base<_AP_W3, false>)(val));
  }

  template <int _AP_W3, int _AP_I3, bool _AP_S3, ap_q_mode _AP_Q3,
            ap_o_mode _AP_O3, int _AP_N3>
  INLINE ap_concat_ref& operator=(
      const ap_fixed_base<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3, _AP_N3>&
          val) {
    return operator=(val.to_ap_int_base());
  }

  template <int _AP_W3, int _AP_I3, bool _AP_S3, ap_q_mode _AP_Q3,
            ap_o_mode _AP_O3, int _AP_N3>
  INLINE ap_concat_ref& operator=(
      const af_bit_ref<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3, _AP_N3>& val) {
    return operator=((ap_ulong)(bool)(val));
  }

  INLINE operator ap_int_base<_AP_WR, false>() const { return get(); }

  INLINE operator ap_ulong() const { return get().to_uint64(); }

  template <int _AP_W3, bool _AP_S3>
  INLINE ap_concat_ref<_AP_WR, ap_concat_ref, _AP_W3,
                       ap_range_ref<_AP_W3, _AP_S3> >
  operator,(const ap_range_ref<_AP_W3, _AP_S3> &a2) {
    return ap_concat_ref<_AP_WR, ap_concat_ref, _AP_W3,
                         ap_range_ref<_AP_W3, _AP_S3> >(
        *this, const_cast<ap_range_ref<_AP_W3, _AP_S3>&>(a2));
  }

  template <int _AP_W3, bool _AP_S3>
  INLINE
      ap_concat_ref<_AP_WR, ap_concat_ref, _AP_W3, ap_int_base<_AP_W3, _AP_S3> >
      operator,(ap_int_base<_AP_W3, _AP_S3> &a2) {
    return ap_concat_ref<_AP_WR, ap_concat_ref, _AP_W3,
                         ap_int_base<_AP_W3, _AP_S3> >(*this, a2);
  }

  template <int _AP_W3, bool _AP_S3>
  INLINE
      ap_concat_ref<_AP_WR, ap_concat_ref, _AP_W3, ap_int_base<_AP_W3, _AP_S3> >
      operator,(volatile ap_int_base<_AP_W3, _AP_S3> &a2) {
    return ap_concat_ref<_AP_WR, ap_concat_ref, _AP_W3,
                         ap_int_base<_AP_W3, _AP_S3> >(
        *this, const_cast<ap_int_base<_AP_W3, _AP_S3>&>(a2));
  }

  template <int _AP_W3, bool _AP_S3>
  INLINE
      ap_concat_ref<_AP_WR, ap_concat_ref, _AP_W3, ap_int_base<_AP_W3, _AP_S3> >
      operator,(const ap_int_base<_AP_W3, _AP_S3> &a2) {
    return ap_concat_ref<_AP_WR, ap_concat_ref, _AP_W3,
                         ap_int_base<_AP_W3, _AP_S3> >(
        *this, const_cast<ap_int_base<_AP_W3, _AP_S3>&>(a2));
  }

  template <int _AP_W3, bool _AP_S3>
  INLINE
      ap_concat_ref<_AP_WR, ap_concat_ref, _AP_W3, ap_int_base<_AP_W3, _AP_S3> >
      operator,(const volatile ap_int_base<_AP_W3, _AP_S3> &a2) {
    // FIXME op's life does not seem long enough
    ap_int_base<_AP_W3, _AP_S3> op(a2);
    return ap_concat_ref<_AP_WR, ap_concat_ref, _AP_W3,
                         ap_int_base<_AP_W3, _AP_S3> >(
        *this, const_cast<ap_int_base<_AP_W3, _AP_S3>&>(op));
  }

  template <int _AP_W3, bool _AP_S3>
  INLINE ap_concat_ref<_AP_WR, ap_concat_ref, 1, ap_bit_ref<_AP_W3, _AP_S3> >
  operator,(const ap_bit_ref<_AP_W3, _AP_S3> &a2) {
    return ap_concat_ref<_AP_WR, ap_concat_ref, 1, ap_bit_ref<_AP_W3, _AP_S3> >(
        *this, const_cast<ap_bit_ref<_AP_W3, _AP_S3>&>(a2));
  }

  template <int _AP_W3, typename _AP_T3, int _AP_W4, typename _AP_T4>
  INLINE ap_concat_ref<_AP_WR, ap_concat_ref, _AP_W3 + _AP_W4,
                       ap_concat_ref<_AP_W3, _AP_T3, _AP_W4, _AP_T4> >
  operator,(const ap_concat_ref<_AP_W3, _AP_T3, _AP_W4, _AP_T4> &a2) {
    return ap_concat_ref<_AP_WR, ap_concat_ref, _AP_W3 + _AP_W4,
                         ap_concat_ref<_AP_W3, _AP_T3, _AP_W4, _AP_T4> >(
        *this, const_cast<ap_concat_ref<_AP_W3, _AP_T3, _AP_W4, _AP_T4>&>(a2));
  }

  template <int _AP_W3, int _AP_I3, bool _AP_S3, ap_q_mode _AP_Q3,
            ap_o_mode _AP_O3, int _AP_N3>
  INLINE ap_concat_ref<
      _AP_WR, ap_concat_ref, _AP_W3,
      af_range_ref<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3, _AP_N3> >
  operator,(
      const af_range_ref<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3, _AP_N3> &a2) {
    return ap_concat_ref<
        _AP_WR, ap_concat_ref, _AP_W3,
        af_range_ref<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3, _AP_N3> >(
        *this,
        const_cast<
            af_range_ref<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3, _AP_N3>&>(a2));
  }

  template <int _AP_W3, int _AP_I3, bool _AP_S3, ap_q_mode _AP_Q3,
            ap_o_mode _AP_O3, int _AP_N3>
  INLINE
      ap_concat_ref<_AP_WR, ap_concat_ref, 1,
                    af_bit_ref<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3, _AP_N3> >
      operator,(const af_bit_ref<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3, _AP_N3>
                    &a2) {
    return ap_concat_ref<
        _AP_WR, ap_concat_ref, 1,
        af_bit_ref<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3, _AP_N3> >(
        *this,
        const_cast<af_bit_ref<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3, _AP_N3>&>(
            a2));
  }

  template <int _AP_W3, bool _AP_S3>
  INLINE ap_int_base<AP_MAX(_AP_WR, _AP_W3), _AP_S3> operator&(
      const ap_int_base<_AP_W3, _AP_S3>& a2) {
    return get() & a2;
  }

  template <int _AP_W3, bool _AP_S3>
  INLINE ap_int_base<AP_MAX(_AP_WR, _AP_W3), _AP_S3> operator|(
      const ap_int_base<_AP_W3, _AP_S3>& a2) {
    return get() | a2;
  }

  template <int _AP_W3, bool _AP_S3>
  INLINE ap_int_base<AP_MAX(_AP_WR, _AP_W3), _AP_S3> operator^(
      const ap_int_base<_AP_W3, _AP_S3>& a2) {
    return get() ^ a2;
  }

#if 0
  template<int Hi, int Lo>
  INLINE ap_int_base<Hi-Lo+1, false> slice() {
    ap_int_base<_AP_WR, false> bv = get();
    return bv.slice<Hi,Lo>();
  }
#endif

  INLINE ap_int_base<_AP_WR, false> get() const {
    ap_int_base<_AP_WR, false> tmpVal(0);
    int W_ref1 = mbv1.length();
    int W_ref2 = mbv2.length();
    ap_int_base<_AP_W2, false> v2(mbv2);
    ap_int_base<_AP_W1, false> v1(mbv1);
    tmpVal.V = _AP_ROOT_op_set_range(tmpVal.V, 0, W_ref2 - 1, v2.V);
    tmpVal.V =
        _AP_ROOT_op_set_range(tmpVal.V, W_ref2, W_ref1 + W_ref2 - 1, v1.V);
    return tmpVal;
  }

  template <int _AP_W3>
  INLINE void set(const ap_int_base<_AP_W3, false>& val) {
    ap_int_base<_AP_W1 + _AP_W2, false> vval(val);
    int W_ref1 = mbv1.length();
    int W_ref2 = mbv2.length();
    ap_int_base<_AP_W1, false> tmpVal1;
    tmpVal1.V = _AP_ROOT_op_get_range(vval.V, W_ref2, W_ref1 + W_ref2 - 1);
    mbv1.set(tmpVal1);
    ap_int_base<_AP_W2, false> tmpVal2;
    tmpVal2.V = _AP_ROOT_op_get_range(vval.V, 0, W_ref2 - 1);
    mbv2.set(tmpVal2);
  }

  INLINE int length() const { return mbv1.length() + mbv2.length(); }
}; // struct ap_concat_ref

/* Range (slice) reference.
   ----------------------------------------------------------------
*/
template <int _AP_W, bool _AP_S>
struct ap_range_ref {
  // struct ssdm_int or its sim model.
  // TODO make it possible to reference to ap_fixed_base/ap_fixed/ap_ufixed
  //      and then we can retire af_range_ref.
  typedef ap_int_base<_AP_W, _AP_S> ref_type;
  ref_type& d_bv;
  int l_index;
  int h_index;

 public:
  INLINE ap_range_ref(const ap_range_ref<_AP_W, _AP_S>& ref)
      : d_bv(ref.d_bv), l_index(ref.l_index), h_index(ref.h_index) {}

  INLINE ap_range_ref(ref_type* bv, int h, int l)
      : d_bv(*bv), l_index(l), h_index(h) {}

  INLINE ap_range_ref(const ref_type* bv, int h, int l)
      : d_bv(*const_cast<ref_type*>(bv)), l_index(l), h_index(h) {}

  INLINE operator ap_int_base<_AP_W, false>() const {
    ap_int_base<_AP_W, false> ret;
    ret.V = _AP_ROOT_op_get_range(d_bv.V, l_index, h_index);
    return ret;
  }

  INLINE operator ap_ulong() const { return to_uint64(); }

  /// @name assign operators
  //  @{

  // FIXME disabled to work-around lagacy code assigning from sc_signal<T>,
  // which dependes on implicit type conversion.
  //
  //   /// assign from hls supported C integral types.
  //   template <typename T>
  //   INLINE typename _ap_type::enable_if<_ap_type::is_integral<T>::value,
  //                                       ap_range_ref&>::type
  //   operator=(T val) {
  //     ap_int_base<_AP_W, false> tmp(val);
  //     d_bv.V = _AP_ROOT_op_set_range(d_bv.V, l_index, h_index, tmp.V);
  //     return *this;
  //   }
#define ASSIGN_WITH_CTYPE(_Tp)                                       \
  INLINE ap_range_ref& operator=(_Tp val) {                          \
    ap_int_base<_AP_W, false> tmp(val);                              \
    d_bv.V = _AP_ROOT_op_set_range(d_bv.V, l_index, h_index, tmp.V); \
    return *this;                                                    \
  }

  ASSIGN_WITH_CTYPE(bool)
  ASSIGN_WITH_CTYPE(char)
  ASSIGN_WITH_CTYPE(signed char)
  ASSIGN_WITH_CTYPE(unsigned char)
  ASSIGN_WITH_CTYPE(short)
  ASSIGN_WITH_CTYPE(unsigned short)
  ASSIGN_WITH_CTYPE(int)
  ASSIGN_WITH_CTYPE(unsigned int)
  ASSIGN_WITH_CTYPE(long)
  ASSIGN_WITH_CTYPE(unsigned long)
  ASSIGN_WITH_CTYPE(ap_slong)
  ASSIGN_WITH_CTYPE(ap_ulong)
#if _AP_ENABLE_HALF_ == 1
  ASSIGN_WITH_CTYPE(half)
#endif
  ASSIGN_WITH_CTYPE(float)
  ASSIGN_WITH_CTYPE(double)

#undef ASSIGN_WITH_CTYPE

  /// assign using string. XXX crucial for cosim.
  INLINE ap_range_ref& operator=(const char* val) {
    const ap_int_base<_AP_W, false> tmp(val); // XXX figure out radix
    d_bv.V = _AP_ROOT_op_set_range(d_bv.V, l_index, h_index, tmp.V);
    return *this;
  }

  /// assign from ap_int_base.
  template <int _AP_W2, bool _AP_S2>
  INLINE ap_range_ref& operator=(const ap_int_base<_AP_W2, _AP_S2>& val) {
    ap_int_base<_AP_W, false> tmp(val);
    d_bv.V = _AP_ROOT_op_set_range(d_bv.V, l_index, h_index, tmp.V);
    return *this;
  }

  /// copy assign operator
  // XXX Be explicit to prevent it from being deleted, as field d_bv
  // is of reference type.
  INLINE ap_range_ref& operator=(const ap_range_ref& val) {
    return operator=((const ap_int_base<_AP_W, false>)val);
  }

  /// assign from range reference to ap_int_base.
  template <int _AP_W2, bool _AP_S2>
  INLINE ap_range_ref& operator=(const ap_range_ref<_AP_W2, _AP_S2>& val) {
    return operator=((const ap_int_base<_AP_W2, false>)val);
  }

  /// assign from bit reference to ap_int_base.
  template <int _AP_W2, bool _AP_S2>
  INLINE ap_range_ref& operator=(const ap_bit_ref<_AP_W2, _AP_S2>& val) {
    return operator=((ap_ulong)(bool)(val));
  }

  /// assign from ap_fixed_base.
  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_range_ref& operator=(
      const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>&
          val) {
    return operator=(val.to_ap_int_base());
  }

  /// assign from range reference to ap_fixed_base.
  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_range_ref& operator=(
      const af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& val) {
    return operator=((const ap_int_base<_AP_W2, false>)val);
  }

  /// assign from bit reference to ap_fixed_base.
  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_range_ref& operator=(
      const af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& val) {
    return operator=((ap_ulong)(bool)(val));
  }

  /// assign from compound reference.
  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
  INLINE ap_range_ref& operator=(
      const ap_concat_ref<_AP_W2, _AP_T3, _AP_W3, _AP_T3>& val) {
    return operator=((const ap_int_base<_AP_W2 + _AP_W3, false>)(val));
  }
  //  @}

  template <int _AP_W2, bool _AP_S2>
  INLINE
      ap_concat_ref<_AP_W, ap_range_ref, _AP_W2, ap_range_ref<_AP_W2, _AP_S2> >
      operator,(const ap_range_ref<_AP_W2, _AP_S2> &a2) {
    return ap_concat_ref<_AP_W, ap_range_ref, _AP_W2,
                         ap_range_ref<_AP_W2, _AP_S2> >(
        *this, const_cast<ap_range_ref<_AP_W2, _AP_S2>&>(a2));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE
      ap_concat_ref<_AP_W, ap_range_ref, _AP_W2, ap_int_base<_AP_W2, _AP_S2> >
      operator,(ap_int_base<_AP_W2, _AP_S2> &a2) {
    return ap_concat_ref<_AP_W, ap_range_ref, _AP_W2,
                         ap_int_base<_AP_W2, _AP_S2> >(*this, a2);
  }

  INLINE
  ap_concat_ref<_AP_W, ap_range_ref, _AP_W, ap_int_base<_AP_W, _AP_S> >
  operator,(ap_int_base<_AP_W, _AP_S>& a2) {
    return ap_concat_ref<_AP_W, ap_range_ref, _AP_W,
                         ap_int_base<_AP_W, _AP_S> >(*this, a2);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE
      ap_concat_ref<_AP_W, ap_range_ref, _AP_W2, ap_int_base<_AP_W2, _AP_S2> >
      operator,(volatile ap_int_base<_AP_W2, _AP_S2> &a2) {
    return ap_concat_ref<_AP_W, ap_range_ref, _AP_W2,
                         ap_int_base<_AP_W2, _AP_S2> >(
        *this, const_cast<ap_int_base<_AP_W2, _AP_S2>&>(a2));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE
      ap_concat_ref<_AP_W, ap_range_ref, _AP_W2, ap_int_base<_AP_W2, _AP_S2> >
      operator,(const ap_int_base<_AP_W2, _AP_S2> &a2) {
    return ap_concat_ref<_AP_W, ap_range_ref, _AP_W2,
                         ap_int_base<_AP_W2, _AP_S2> >(
        *this, const_cast<ap_int_base<_AP_W2, _AP_S2>&>(a2));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE
      ap_concat_ref<_AP_W, ap_range_ref, _AP_W2, ap_int_base<_AP_W2, _AP_S2> >
      operator,(const volatile ap_int_base<_AP_W2, _AP_S2> &a2) {
    return ap_concat_ref<_AP_W, ap_range_ref, _AP_W2,
                         ap_int_base<_AP_W2, _AP_S2> >(
        *this, const_cast<ap_int_base<_AP_W2, _AP_S2>&>(a2));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_concat_ref<_AP_W, ap_range_ref, 1, ap_bit_ref<_AP_W2, _AP_S2> >
  operator,(const ap_bit_ref<_AP_W2, _AP_S2> &a2) {
    return ap_concat_ref<_AP_W, ap_range_ref, 1, ap_bit_ref<_AP_W2, _AP_S2> >(
        *this, const_cast<ap_bit_ref<_AP_W2, _AP_S2>&>(a2));
  }

  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
  INLINE ap_concat_ref<_AP_W, ap_range_ref, _AP_W2 + _AP_W3,
                       ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> >
  operator,(const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> &a2) {
    return ap_concat_ref<_AP_W, ap_range_ref, _AP_W2 + _AP_W3,
                         ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> >(
        *this, const_cast<ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>&>(a2));
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_concat_ref<
      _AP_W, ap_range_ref, _AP_W2,
      af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >
  operator,(
      const af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> a2) {
    return ap_concat_ref<
        _AP_W, ap_range_ref, _AP_W2,
        af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >(
        *this,
        const_cast<
            af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>&>(a2));
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE
      ap_concat_ref<_AP_W, ap_range_ref, 1,
                    af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >
      operator,(const af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>
                    &a2) {
    return ap_concat_ref<
        _AP_W, ap_range_ref, 1,
        af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >(
        *this,
        const_cast<af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>&>(
            a2));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator==(const ap_range_ref<_AP_W2, _AP_S2>& op2) {
    ap_int_base<_AP_W, false> lop(*this);
    ap_int_base<_AP_W2, false> hop(op2);
    return lop == hop;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator!=(const ap_range_ref<_AP_W2, _AP_S2>& op2) {
    return !(operator==(op2));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator<(const ap_range_ref<_AP_W2, _AP_S2>& op2) {
    ap_int_base<_AP_W, false> lop(*this);
    ap_int_base<_AP_W2, false> hop(op2);
    return lop < hop;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator<=(const ap_range_ref<_AP_W2, _AP_S2>& op2) {
    ap_int_base<_AP_W, false> lop(*this);
    ap_int_base<_AP_W2, false> hop(op2);
    return lop <= hop;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator>(const ap_range_ref<_AP_W2, _AP_S2>& op2) {
    return !(operator<=(op2));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator>=(const ap_range_ref<_AP_W2, _AP_S2>& op2) {
    return !(operator<(op2));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_range_ref<_AP_W, _AP_S>& operator|=(
      const ap_range_ref<_AP_W2, _AP_S2>& op2) {
    (this->d_bv).V |= (op2.d_bv).V;
    return *this;
  };

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_range_ref<_AP_W, _AP_S>& operator|=(
      const ap_int_base<_AP_W2, _AP_S2>& op2) {
    (this->d_bv).V |= op2.V;
    return *this;
  };

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_range_ref<_AP_W, _AP_S>& operator&=(
      const ap_range_ref<_AP_W2, _AP_S2>& op2) {
    (this->d_bv).V &= (op2.d_bv).V;
    return *this;
  };

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_range_ref<_AP_W, _AP_S>& operator&=(
      const ap_int_base<_AP_W2, _AP_S2>& op2) {
    (this->d_bv).V &= op2.V;
    return *this;
  };

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_range_ref<_AP_W, _AP_S>& operator^=(
      const ap_range_ref<_AP_W2, _AP_S2>& op2) {
    (this->d_bv).V ^= (op2.d_bv).V;
    return *this;
  };

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_range_ref<_AP_W, _AP_S>& operator^=(
      const ap_int_base<_AP_W2, _AP_S2>& op2) {
    (this->d_bv).V ^= op2.V;
    return *this;
  };

  INLINE ap_int_base<_AP_W, false> get() const {
    ap_int_base<_AP_W, false> ret;
    ret.V = _AP_ROOT_op_get_range(d_bv.V, l_index, h_index);
    return ret;
  }

  template <int _AP_W2>
  INLINE void set(const ap_int_base<_AP_W2, false>& val) {
    d_bv.V = _AP_ROOT_op_set_range(d_bv.V, l_index, h_index, val.V);
  }

  INLINE int length() const {
    return h_index >= l_index ? h_index - l_index + 1 : l_index - h_index + 1;
  }

  INLINE int to_int() const {
    return (int)(_AP_ROOT_op_get_range(d_bv.V, l_index, h_index));
  }

  INLINE unsigned to_uint() const {
    return (unsigned)(_AP_ROOT_op_get_range(d_bv.V, l_index, h_index));
  }

  INLINE long to_long() const {
    return (long)(_AP_ROOT_op_get_range(d_bv.V, l_index, h_index));
  }

  INLINE unsigned long to_ulong() const {
    return (unsigned long)(_AP_ROOT_op_get_range(d_bv.V, l_index, h_index));
  }

  INLINE ap_slong to_int64() const {
    return (ap_slong)(_AP_ROOT_op_get_range(d_bv.V, l_index, h_index));
  }

  INLINE ap_ulong to_uint64() const {
    return (ap_ulong)(_AP_ROOT_op_get_range(d_bv.V, l_index, h_index));
  }

  INLINE bool and_reduce() const {
    bool ret = true;
    bool reverse = l_index > h_index;
    unsigned low = reverse ? h_index : l_index;
    unsigned high = reverse ? l_index : h_index;
    for (unsigned i = low; i != high; ++i) {
#ifdef __SYNTHESIS__
#pragma HLS unroll
#endif
      ret &= _AP_ROOT_op_get_bit(d_bv.V, i);
    }
    return ret;
  }

  INLINE bool or_reduce() const {
    bool ret = false;
    bool reverse = l_index > h_index;
    unsigned low = reverse ? h_index : l_index;
    unsigned high = reverse ? l_index : h_index;
    for (unsigned i = low; i != high; ++i) {
#ifdef __SYNTHESIS__
#pragma HLS unroll
#endif
      ret |= _AP_ROOT_op_get_bit(d_bv.V, i);
    }
    return ret;
  }

  INLINE bool xor_reduce() const {
    bool ret = false;
    bool reverse = l_index > h_index;
    unsigned low = reverse ? h_index : l_index;
    unsigned high = reverse ? l_index : h_index;
    for (unsigned i = low; i != high; ++i) {
#ifdef __SYNTHESIS__
#pragma HLS unroll
#endif
      ret ^= _AP_ROOT_op_get_bit(d_bv.V, i);
    }
    return ret;
  }
#ifndef __SYNTHESIS__
  std::string to_string(signed char radix = 2) const {
    ap_int_base<_AP_W, false> ret;
    ret.V = _AP_ROOT_op_get_range(d_bv.V, l_index, h_index);
    return ret.to_string(radix);
  }
#else
  // XXX HLS will delete this in synthesis
  INLINE char* to_string(signed char radix = 2) const {
    return 0;
  }
#endif
}; // struct ap_range_ref

// XXX apcc cannot handle global std::ios_base::Init() brought in by <iostream>
#ifndef AP_AUTOCC
#ifndef __SYNTHESIS__
template <int _AP_W, bool _AP_S>
INLINE std::ostream& operator<<(std::ostream& os,
                                const ap_range_ref<_AP_W, _AP_S>& x) {
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
                                ap_range_ref<_AP_W, _AP_S>& op) {
  std::string str;
  in >> str;
  op = ap_int_base<_AP_W, _AP_S>(str.c_str());
  return in;
}
#endif // ifndef __SYNTHESIS__
#endif // ifndef AP_AUTOCC

/* Bit reference.
   ----------------------------------------------------------------
*/
template <int _AP_W, bool _AP_S>
struct ap_bit_ref {
  // struct ssdm_int or its sim model.
  // TODO make it possible to reference to ap_fixed_base/ap_fixed/ap_ufixed
  //      and then we can retire af_bit_ref.
  typedef ap_int_base<_AP_W, _AP_S> ref_type;
  ref_type& d_bv;
  int d_index;

 public:
  // copy ctor
  INLINE ap_bit_ref(const ap_bit_ref<_AP_W, _AP_S>& ref)
      : d_bv(ref.d_bv), d_index(ref.d_index) {}

  INLINE ap_bit_ref(ref_type* bv, int index = 0) : d_bv(*bv), d_index(index) {}

  INLINE ap_bit_ref(const ref_type* bv, int index = 0)
      : d_bv(*const_cast<ref_type*>(bv)), d_index(index) {}

  INLINE operator bool() const { return _AP_ROOT_op_get_bit(d_bv.V, d_index); }
  INLINE bool to_bool() const { return _AP_ROOT_op_get_bit(d_bv.V, d_index); }

  // assign op from hls supported C integral types.
  // FIXME disabled to support sc_signal<bool>.
  // NOTE this used to be unsigned long long.
  //template <typename T>
  //INLINE typename _ap_type::enable_if<_ap_type::is_integral<T>::value,
  //                                    ap_bit_ref&>::type
  //operator=(T val) {
  //  d_bv.V = _AP_ROOT_op_set_bit(d_bv.V, d_index, val);
  //  return *this;
  //}
#define ASSIGN_WITH_CTYPE(_Tp)                          \
  INLINE ap_bit_ref& operator=(_Tp val) {               \
    d_bv.V = _AP_ROOT_op_set_bit(d_bv.V, d_index, val); \
    return *this;                                       \
  }

  ASSIGN_WITH_CTYPE(bool)
  ASSIGN_WITH_CTYPE(char)
  ASSIGN_WITH_CTYPE(signed char)
  ASSIGN_WITH_CTYPE(unsigned char)
  ASSIGN_WITH_CTYPE(short)
  ASSIGN_WITH_CTYPE(unsigned short)
  ASSIGN_WITH_CTYPE(int)
  ASSIGN_WITH_CTYPE(unsigned int)
  ASSIGN_WITH_CTYPE(long)
  ASSIGN_WITH_CTYPE(unsigned long)
  ASSIGN_WITH_CTYPE(ap_slong)
  ASSIGN_WITH_CTYPE(ap_ulong)

#undef ASSIGN_WITH_CTYPE

#define ASSIGN_WITH_CTYPE_FP(_Tp)                           \
  INLINE ap_bit_ref& operator=(_Tp val) {                   \
    bool tmp_val = val;                                     \
    d_bv.V = _AP_ROOT_op_set_bit(d_bv.V, d_index,tmp_val);  \
    return *this;                                           \
  }

#if _AP_ENABLE_HALF_ == 1
  ASSIGN_WITH_CTYPE_FP(half)
#endif
  ASSIGN_WITH_CTYPE_FP(float)
  ASSIGN_WITH_CTYPE_FP(double)

#undef ASSIGN_WITH_CTYPE_FP


  template <int _AP_W2, bool _AP_S2>
  INLINE ap_bit_ref& operator=(const ap_int_base<_AP_W2, _AP_S2>& val) {
    return operator=((ap_ulong)(val.V != 0));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_bit_ref& operator=(const ap_range_ref<_AP_W2, _AP_S2>& val) {
    return operator=((ap_int_base<_AP_W2, false>)val);
  }

  // Be explicit to prevent it from being deleted, as field d_bv
  // is of reference type.
  INLINE ap_bit_ref& operator=(const ap_bit_ref& val) {
    return operator=((ap_ulong)(bool)val);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_bit_ref& operator=(const ap_bit_ref<_AP_W2, _AP_S2>& val) {
    return operator=((ap_ulong)(bool)val);
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_bit_ref& operator=(
      const af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& val) {
    return operator=((const ap_int_base<_AP_W2, false>)val);
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_bit_ref& operator=(
      const af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& val) {
    return operator=((ap_ulong)(bool)val);
  }

  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
  INLINE ap_bit_ref& operator=(
      const ap_concat_ref<_AP_W2, _AP_T3, _AP_W3, _AP_T3>& val) {
    return operator=((const ap_int_base<_AP_W2 + _AP_W3, false>)val);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_concat_ref<1, ap_bit_ref, _AP_W2, ap_int_base<_AP_W2, _AP_S2> >
  operator,(ap_int_base<_AP_W2, _AP_S2> &a2) {
    return ap_concat_ref<1, ap_bit_ref, _AP_W2, ap_int_base<_AP_W2, _AP_S2> >(
        *this, a2);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_concat_ref<1, ap_bit_ref, _AP_W2, ap_int_base<_AP_W2, _AP_S2> >
  operator,(volatile ap_int_base<_AP_W2, _AP_S2> &a2) {
    return ap_concat_ref<1, ap_bit_ref, _AP_W2, ap_int_base<_AP_W2, _AP_S2> >(
        *this, const_cast<ap_int_base<_AP_W2, _AP_S2>&>(a2));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_concat_ref<1, ap_bit_ref, _AP_W2, ap_int_base<_AP_W2, _AP_S2> >
  operator,(const ap_int_base<_AP_W2, _AP_S2> &a2) {
    ap_int_base<_AP_W2, _AP_S2> op(a2);
    return ap_concat_ref<1, ap_bit_ref, _AP_W2, ap_int_base<_AP_W2, _AP_S2> >(
        *this, const_cast<ap_int_base<_AP_W2, _AP_S2>&>(op));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_concat_ref<1, ap_bit_ref, _AP_W2, ap_int_base<_AP_W2, _AP_S2> >
  operator,(const volatile ap_int_base<_AP_W2, _AP_S2> &a2) {
    ap_int_base<_AP_W2, _AP_S2> op(a2);
    return ap_concat_ref<1, ap_bit_ref, _AP_W2, ap_int_base<_AP_W2, _AP_S2> >(
        *this, const_cast<ap_int_base<_AP_W2, _AP_S2>&>(op));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_concat_ref<1, ap_bit_ref, _AP_W2, ap_range_ref<_AP_W2, _AP_S2> >
  operator,(const ap_range_ref<_AP_W2, _AP_S2> &a2) {
    return ap_concat_ref<1, ap_bit_ref, _AP_W2, ap_range_ref<_AP_W2, _AP_S2> >(
        *this, const_cast<ap_range_ref<_AP_W2, _AP_S2>&>(a2));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_concat_ref<1, ap_bit_ref, 1, ap_bit_ref<_AP_W2, _AP_S2> > operator,(
      const ap_bit_ref<_AP_W2, _AP_S2> &a2) {
    return ap_concat_ref<1, ap_bit_ref, 1, ap_bit_ref<_AP_W2, _AP_S2> >(
        *this, const_cast<ap_bit_ref<_AP_W2, _AP_S2>&>(a2));
  }

  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
  INLINE ap_concat_ref<1, ap_bit_ref, _AP_W2 + _AP_W3,
                       ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> >
  operator,(const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> &a2) {
    return ap_concat_ref<1, ap_bit_ref, _AP_W2 + _AP_W3,
                         ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> >(
        *this, const_cast<ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>&>(a2));
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_concat_ref<
      1, ap_bit_ref, _AP_W2,
      af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >
  operator,(
      const af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> &a2) {
    return ap_concat_ref<
        1, ap_bit_ref, _AP_W2,
        af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >(
        *this,
        const_cast<
            af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>&>(a2));
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE ap_concat_ref<1, ap_bit_ref, 1, af_bit_ref<_AP_W2, _AP_I2, _AP_S2,
                                                    _AP_Q2, _AP_O2, _AP_N2> >
  operator,(
      const af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> &a2) {
    return ap_concat_ref<1, ap_bit_ref, 1, af_bit_ref<_AP_W2, _AP_I2, _AP_S2,
                                                      _AP_Q2, _AP_O2, _AP_N2> >(
        *this,
        const_cast<af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>&>(
            a2));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator==(const ap_bit_ref<_AP_W2, _AP_S2>& op) {
    return get() == op.get();
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator!=(const ap_bit_ref<_AP_W2, _AP_S2>& op) {
    return get() != op.get();
  }

  INLINE bool get() const { return _AP_ROOT_op_get_bit(d_bv.V, d_index); }

  INLINE bool get() { return _AP_ROOT_op_get_bit(d_bv.V, d_index); }

  template <int _AP_W3>
  INLINE void set(const ap_int_base<_AP_W3, false>& val) {
    operator=(val);
  }

  INLINE bool operator~() const {
    bool bit = _AP_ROOT_op_get_bit(d_bv.V, d_index);
    return bit ? false : true;
  }

  INLINE int length() const { return 1; }

#ifndef __SYNTHESIS__
  std::string to_string() const { return get() ? "1" : "0"; }
#else
  // XXX HLS will delete this in synthesis
  INLINE char* to_string() const { return 0; }
#endif
}; // struct ap_bit_ref

/* ap_range_ref with int.
 * ------------------------------------------------------------
 */
// equality and relational operators.
#define REF_REL_OP_WITH_INT(REL_OP, C_TYPE, _AP_W2, _AP_S2)                \
  template <int _AP_W, bool _AP_S>                                         \
  INLINE bool operator REL_OP(const ap_range_ref<_AP_W, _AP_S>& op,        \
                              C_TYPE op2) {                                \
    return ap_int_base<_AP_W, false>(op)                                   \
        REL_OP ap_int_base<_AP_W2, _AP_S2>(op2);                           \
  }                                                                        \
  template <int _AP_W, bool _AP_S>                                         \
  INLINE bool operator REL_OP(const ap_bit_ref<_AP_W, _AP_S>& op,          \
                              C_TYPE op2) {                                \
    return bool(op) REL_OP op2;                                            \
  }                                                                        \
  template <int _AP_W, bool _AP_S>                                         \
  INLINE bool operator REL_OP(C_TYPE op2,                                  \
                              const ap_bit_ref<_AP_W, _AP_S>& op) {        \
    return op2 REL_OP bool(op);                                            \
  }                                                                        \
  template <int _AP_W, typename _AP_T, int _AP_W1, typename _AP_T1>        \
  INLINE bool operator REL_OP(                                             \
      const ap_concat_ref<_AP_W, _AP_T, _AP_W1, _AP_T1>& op, C_TYPE op2) { \
    return ap_int_base<_AP_W + _AP_W1, false>(op)                          \
        REL_OP ap_int_base<_AP_W2, _AP_S2>(op2);                           \
  }

// Make the line shorter than 5000 chars
#define REF_REL_WITH_INT_1(C_TYPE, _AP_WI, _AP_SI) \
  REF_REL_OP_WITH_INT(>, C_TYPE, _AP_WI, _AP_SI)   \
  REF_REL_OP_WITH_INT(<, C_TYPE, _AP_WI, _AP_SI)   \
  REF_REL_OP_WITH_INT(>=, C_TYPE, _AP_WI, _AP_SI)  \
  REF_REL_OP_WITH_INT(<=, C_TYPE, _AP_WI, _AP_SI)

REF_REL_WITH_INT_1(bool, 1, false)
REF_REL_WITH_INT_1(char, 8, CHAR_IS_SIGNED)
REF_REL_WITH_INT_1(signed char, 8, true)
REF_REL_WITH_INT_1(unsigned char, 8, false)
REF_REL_WITH_INT_1(short, _AP_SIZE_short, true)
REF_REL_WITH_INT_1(unsigned short, _AP_SIZE_short, false)
REF_REL_WITH_INT_1(int, _AP_SIZE_int, true)
REF_REL_WITH_INT_1(unsigned int, _AP_SIZE_int, false)
REF_REL_WITH_INT_1(long, _AP_SIZE_long, true)
REF_REL_WITH_INT_1(unsigned long, _AP_SIZE_long, false)
REF_REL_WITH_INT_1(ap_slong, _AP_SIZE_ap_slong, true)
REF_REL_WITH_INT_1(ap_ulong, _AP_SIZE_ap_slong, false)

// Make the line shorter than 5000 chars
#define REF_REL_WITH_INT_2(C_TYPE, _AP_WI, _AP_SI) \
  REF_REL_OP_WITH_INT(==, C_TYPE, _AP_WI, _AP_SI)  \
  REF_REL_OP_WITH_INT(!=, C_TYPE, _AP_WI, _AP_SI)

REF_REL_WITH_INT_2(bool, 1, false)
REF_REL_WITH_INT_2(char, 8, CHAR_IS_SIGNED)
REF_REL_WITH_INT_2(signed char, 8, true)
REF_REL_WITH_INT_2(unsigned char, 8, false)
REF_REL_WITH_INT_2(short, _AP_SIZE_short, true)
REF_REL_WITH_INT_2(unsigned short, _AP_SIZE_short, false)
REF_REL_WITH_INT_2(int, _AP_SIZE_int, true)
REF_REL_WITH_INT_2(unsigned int, _AP_SIZE_int, false)
REF_REL_WITH_INT_2(long, _AP_SIZE_long, true)
REF_REL_WITH_INT_2(unsigned long, _AP_SIZE_long, false)
REF_REL_WITH_INT_2(ap_slong, _AP_SIZE_ap_slong, true)
REF_REL_WITH_INT_2(ap_ulong, _AP_SIZE_ap_slong, false)

#undef REF_REL_OP_WITH_INT
#undef REF_REL_WITH_INT_1
#undef REF_REL_WITH_INT_2

#define REF_BIN_OP_WITH_INT(BIN_OP, RTYPE, C_TYPE, _AP_W2, _AP_S2)          \
  template <int _AP_W, bool _AP_S>                                          \
  INLINE typename ap_int_base<_AP_W, false>::template RType<_AP_W2,         \
                                                            _AP_S2>::RTYPE  \
  operator BIN_OP(const ap_range_ref<_AP_W, _AP_S>& op, C_TYPE op2) {       \
    return ap_int_base<_AP_W, false>(op)                                    \
        BIN_OP ap_int_base<_AP_W2, _AP_S2>(op2);                            \
  }                                                                         \
  template <int _AP_W, bool _AP_S>                                          \
  INLINE typename ap_int_base<_AP_W2, _AP_S2>::template RType<_AP_W,        \
                                                              false>::RTYPE \
  operator BIN_OP(C_TYPE op2, const ap_range_ref<_AP_W, _AP_S>& op) {       \
    return ap_int_base<_AP_W2, _AP_S2>(op2)                                 \
        BIN_OP ap_int_base<_AP_W, false>(op);                               \
  }

// arithmetic operators.
#define REF_BIN_OP_WITH_INT_ARITH(C_TYPE, _AP_W2, _AP_S2)   \
  REF_BIN_OP_WITH_INT(+, plus, C_TYPE, (_AP_W2), (_AP_S2))  \
  REF_BIN_OP_WITH_INT(-, minus, C_TYPE, (_AP_W2), (_AP_S2)) \
  REF_BIN_OP_WITH_INT(*, mult, C_TYPE, (_AP_W2), (_AP_S2))  \
  REF_BIN_OP_WITH_INT(/, div, C_TYPE, (_AP_W2), (_AP_S2))   \
  REF_BIN_OP_WITH_INT(%, mod, C_TYPE, (_AP_W2), (_AP_S2))

REF_BIN_OP_WITH_INT_ARITH(bool, 1, false)
REF_BIN_OP_WITH_INT_ARITH(char, 8, CHAR_IS_SIGNED)
REF_BIN_OP_WITH_INT_ARITH(signed char, 8, true)
REF_BIN_OP_WITH_INT_ARITH(unsigned char, 8, false)
REF_BIN_OP_WITH_INT_ARITH(short, _AP_SIZE_short, true)
REF_BIN_OP_WITH_INT_ARITH(unsigned short, _AP_SIZE_short, false)
REF_BIN_OP_WITH_INT_ARITH(int, _AP_SIZE_int, true)
REF_BIN_OP_WITH_INT_ARITH(unsigned int, _AP_SIZE_int, false)
REF_BIN_OP_WITH_INT_ARITH(long, _AP_SIZE_long, true)
REF_BIN_OP_WITH_INT_ARITH(unsigned long, _AP_SIZE_long, false)
REF_BIN_OP_WITH_INT_ARITH(ap_slong, _AP_SIZE_ap_slong, true)
REF_BIN_OP_WITH_INT_ARITH(ap_ulong, _AP_SIZE_ap_slong, false)

#undef REF_BIN_OP_WITH_INT_ARITH

// bitwise and shift operators
#define REF_BIN_OP_WITH_INT_BITS(C_TYPE, _AP_W2, _AP_S2)     \
  REF_BIN_OP_WITH_INT(&, logic, C_TYPE, (_AP_W2), (_AP_S2))  \
  REF_BIN_OP_WITH_INT(|, logic, C_TYPE, (_AP_W2), (_AP_S2))  \
  REF_BIN_OP_WITH_INT(^, logic, C_TYPE, (_AP_W2), (_AP_S2)) \
  REF_BIN_OP_WITH_INT(>>, arg1, C_TYPE, (_AP_W2), (_AP_S2))  \
  REF_BIN_OP_WITH_INT(<<, arg1, C_TYPE, (_AP_W2), (_AP_S2))

REF_BIN_OP_WITH_INT_BITS(bool, 1, false)
REF_BIN_OP_WITH_INT_BITS(char, 8, CHAR_IS_SIGNED)
REF_BIN_OP_WITH_INT_BITS(signed char, 8, true)
REF_BIN_OP_WITH_INT_BITS(unsigned char, 8, false)
REF_BIN_OP_WITH_INT_BITS(short, _AP_SIZE_short, true)
REF_BIN_OP_WITH_INT_BITS(unsigned short, _AP_SIZE_short, false)
REF_BIN_OP_WITH_INT_BITS(int, _AP_SIZE_int, true)
REF_BIN_OP_WITH_INT_BITS(unsigned int, _AP_SIZE_int, false)
REF_BIN_OP_WITH_INT_BITS(long, _AP_SIZE_long, true)
REF_BIN_OP_WITH_INT_BITS(unsigned long, _AP_SIZE_long, false)
REF_BIN_OP_WITH_INT_BITS(ap_slong, _AP_SIZE_ap_slong, true)
REF_BIN_OP_WITH_INT_BITS(ap_ulong, _AP_SIZE_ap_slong, false)

#undef REF_BIN_OP_WITH_INT_BITS

/* ap_range_ref with ap_range_ref
 *  ------------------------------------------------------------
 */
#define REF_BIN_OP(BIN_OP, RTYPE)                                              \
  template <int _AP_W, bool _AP_S, int _AP_W2, bool _AP_S2>                    \
  INLINE                                                                       \
      typename ap_int_base<_AP_W, false>::template RType<_AP_W2, false>::RTYPE \
      operator BIN_OP(const ap_range_ref<_AP_W, _AP_S>& lhs,                   \
                      const ap_range_ref<_AP_W2, _AP_S2>& rhs) {               \
    return (lhs.operator ap_int_base<_AP_W, false>())BIN_OP(                   \
        rhs.operator ap_int_base<_AP_W2, false>());                            \
  }

REF_BIN_OP(+, plus)
REF_BIN_OP(-, minus)
REF_BIN_OP(*, mult)
REF_BIN_OP(/, div)
REF_BIN_OP(%, mod)
REF_BIN_OP(&, logic)
REF_BIN_OP(|, logic)
REF_BIN_OP(^, logic)
REF_BIN_OP(>>, arg1)
REF_BIN_OP(<<, arg1)

/* ap_concat_ref with ap_concat_ref.
 *  ------------------------------------------------------------
 */

//************************************************************************
//  Implement
//      ap_int_base<M+N> = ap_concat_ref<M> OP ap_concat_ref<N>
//  for operators  +, -, *, /, %, >>, <<, &, |, ^
//  Without these operators the operands are converted to int64 and
//  larger results lose informations (higher order bits).
//
//                       operand OP
//                      /          |
//              left-concat         right-concat
//                /     |            /         |
//         <LW1,LT1>  <LW2,LT2>   <RW1,RT1>    <RW2,RT2>
//
//      _AP_LW1, _AP_LT1 (width and type of left-concat's left side)
//      _AP_LW2, _AP_LT2 (width and type of left-concat's right side)
//  Similarly for RHS of operand OP: _AP_RW1, AP_RW2, _AP_RT1, _AP_RT2
//
//  In Verilog 2001 result of concatenation is always unsigned even
//  when both sides are signed.
//************************************************************************

#undef SYN_CONCAT_REF_BIN_OP

#define SYN_CONCAT_REF_BIN_OP(BIN_OP, RTYPE)                              \
  template <int _AP_LW1, typename _AP_LT1, int _AP_LW2, typename _AP_LT2, \
            int _AP_RW1, typename _AP_RT1, int _AP_RW2, typename _AP_RT2> \
  INLINE typename ap_int_base<_AP_LW1 + _AP_LW2, false>::template RType<  \
      _AP_RW1 + _AP_RW2, false>::RTYPE                                    \
  operator BIN_OP(                                                        \
      const ap_concat_ref<_AP_LW1, _AP_LT1, _AP_LW2, _AP_LT2>& lhs,       \
      const ap_concat_ref<_AP_RW1, _AP_RT1, _AP_RW2, _AP_RT2>& rhs) {     \
    return lhs.get() BIN_OP rhs.get();                                    \
  }

SYN_CONCAT_REF_BIN_OP(+, plus)
SYN_CONCAT_REF_BIN_OP(-, minus)
SYN_CONCAT_REF_BIN_OP(*, mult)
SYN_CONCAT_REF_BIN_OP(/, div)
SYN_CONCAT_REF_BIN_OP(%, mod)
SYN_CONCAT_REF_BIN_OP(&, logic)
SYN_CONCAT_REF_BIN_OP(|, logic)
SYN_CONCAT_REF_BIN_OP(^, logic)
SYN_CONCAT_REF_BIN_OP(>>, arg1)
SYN_CONCAT_REF_BIN_OP(<<, arg1)

#undef SYN_CONCAT_REF_BIN_OP

#define CONCAT_OP_WITH_INT(C_TYPE, _AP_WI, _AP_SI)                          \
  template <int _AP_W, bool _AP_S>                                          \
  INLINE ap_int_base<_AP_W + _AP_WI, false> operator,(                      \
      const ap_int_base<_AP_W, _AP_S> &op1, C_TYPE op2) {                   \
    ap_int_base<_AP_WI + _AP_W, false> val(op2);                            \
    ap_int_base<_AP_WI + _AP_W, false> ret(op1);                            \
    ret <<= _AP_WI;                                                         \
    if (_AP_SI) {                                                           \
      val <<= _AP_W;                                                        \
      val >>= _AP_W;                                                        \
    }                                                                       \
    ret |= val;                                                             \
    return ret;                                                             \
  }                                                                         \
  template <int _AP_W, bool _AP_S>                                          \
  INLINE ap_int_base<_AP_W + _AP_WI, false> operator,(                      \
      C_TYPE op1, const ap_int_base<_AP_W, _AP_S> &op2) {                   \
    ap_int_base<_AP_WI + _AP_W, false> val(op1);                            \
    ap_int_base<_AP_WI + _AP_W, false> ret(op2);                            \
    if (_AP_S) {                                                            \
      ret <<= _AP_WI;                                                       \
      ret >>= _AP_WI;                                                       \
    }                                                                       \
    ret |= val << _AP_W;                                                    \
    return ret;                                                             \
  }                                                                         \
  template <int _AP_W, bool _AP_S>                                          \
  INLINE ap_int_base<_AP_W + _AP_WI, false> operator,(                      \
      const ap_range_ref<_AP_W, _AP_S> &op1, C_TYPE op2) {                  \
    ap_int_base<_AP_WI + _AP_W, false> val(op2);                            \
    ap_int_base<_AP_WI + _AP_W, false> ret(op1);                            \
    ret <<= _AP_WI;                                                         \
    if (_AP_SI) {                                                           \
      val <<= _AP_W;                                                        \
      val >>= _AP_W;                                                        \
    }                                                                       \
    ret |= val;                                                             \
    return ret;                                                             \
  }                                                                         \
  template <int _AP_W, bool _AP_S>                                          \
  INLINE ap_int_base<_AP_W + _AP_WI, false> operator,(                      \
      C_TYPE op1, const ap_range_ref<_AP_W, _AP_S> &op2) {                  \
    ap_int_base<_AP_WI + _AP_W, false> val(op1);                            \
    ap_int_base<_AP_WI + _AP_W, false> ret(op2);                            \
    int len = op2.length();                                                 \
    val <<= len;                                                            \
    ret |= val;                                                             \
    return ret;                                                             \
  }                                                                         \
  template <int _AP_W, bool _AP_S>                                          \
  INLINE ap_int_base<_AP_WI + 1, false> operator,(                          \
      const ap_bit_ref<_AP_W, _AP_S> &op1, C_TYPE op2) {                    \
    ap_int_base<_AP_WI + 1, false> val(op2);                                \
    val[_AP_WI] = op1;                                                      \
    return val;                                                             \
  }                                                                         \
  template <int _AP_W, bool _AP_S>                                          \
  INLINE ap_int_base<_AP_WI + 1, false> operator,(                          \
      C_TYPE op1, const ap_bit_ref<_AP_W, _AP_S> &op2) {                    \
    ap_int_base<_AP_WI + 1, false> val(op1);                                \
    val <<= 1;                                                              \
    val[0] = op2;                                                           \
    return val;                                                             \
  }                                                                         \
  template <int _AP_W, typename _AP_T, int _AP_W2, typename _AP_T2>         \
  INLINE ap_int_base<_AP_W + _AP_W2 + _AP_WI, false> operator,(             \
      const ap_concat_ref<_AP_W, _AP_T, _AP_W2, _AP_T2> &op1, C_TYPE op2) { \
    ap_int_base<_AP_WI + _AP_W + _AP_W2, _AP_SI> val(op2);                  \
    ap_int_base<_AP_WI + _AP_W + _AP_W2, _AP_SI> ret(op1);                  \
    if (_AP_SI) {                                                           \
      val <<= _AP_W + _AP_W2;                                               \
      val >>= _AP_W + _AP_W2;                                               \
    }                                                                       \
    ret <<= _AP_WI;                                                         \
    ret |= val;                                                             \
    return ret;                                                             \
  }                                                                         \
  template <int _AP_W, typename _AP_T, int _AP_W2, typename _AP_T2>         \
  INLINE ap_int_base<_AP_W + _AP_W2 + _AP_WI, false> operator,(             \
      C_TYPE op1, const ap_concat_ref<_AP_W, _AP_T, _AP_W2, _AP_T2> &op2) { \
    ap_int_base<_AP_WI + _AP_W + _AP_W2, _AP_SI> val(op1);                  \
    ap_int_base<_AP_WI + _AP_W + _AP_W2, _AP_SI> ret(op2);                  \
    int len = op2.length();                                                 \
    val <<= len;                                                            \
    ret |= val;                                                             \
    return ret;                                                             \
  }                                                                         \
  template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q,              \
            ap_o_mode _AP_O, int _AP_N>                                     \
  INLINE ap_int_base<_AP_W + _AP_WI, false> operator,(                      \
      const af_range_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N> &op1,    \
      C_TYPE op2) {                                                         \
    ap_int_base<_AP_WI + _AP_W, false> val(op2);                            \
    ap_int_base<_AP_WI + _AP_W, false> ret(op1);                            \
    if (_AP_SI) {                                                           \
      val <<= _AP_W;                                                        \
      val >>= _AP_W;                                                        \
    }                                                                       \
    ret <<= _AP_WI;                                                         \
    ret |= val;                                                             \
    return ret;                                                             \
  }                                                                         \
  template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q,              \
            ap_o_mode _AP_O, int _AP_N>                                     \
  INLINE ap_int_base<_AP_W + _AP_WI, false> operator,(                      \
      C_TYPE op1,                                                           \
      const af_range_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N> &op2) {  \
    ap_int_base<_AP_WI + _AP_W, false> val(op1);                            \
    ap_int_base<_AP_WI + _AP_W, false> ret(op2);                            \
    int len = op2.length();                                                 \
    val <<= len;                                                            \
    ret |= val;                                                             \
    return ret;                                                             \
  }                                                                         \
  template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q,              \
            ap_o_mode _AP_O, int _AP_N>                                     \
  INLINE ap_int_base<1 + _AP_WI, false> operator,(                          \
      const af_bit_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N> &op1,      \
      C_TYPE op2) {                                                         \
    ap_int_base<_AP_WI + 1, _AP_SI> val(op2);                               \
    val[_AP_WI] = op1;                                                      \
    return val;                                                             \
  }                                                                         \
  template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q,              \
            ap_o_mode _AP_O, int _AP_N>                                     \
  INLINE ap_int_base<1 + _AP_WI, false> operator,(                          \
      C_TYPE op1,                                                           \
      const af_bit_ref<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, _AP_N> &op2) {    \
    ap_int_base<_AP_WI + 1, _AP_SI> val(op1);                               \
    val <<= 1;                                                              \
    val[0] = op2;                                                           \
    return val;                                                             \
  }

CONCAT_OP_WITH_INT(bool, 1, false)
CONCAT_OP_WITH_INT(char, 8, CHAR_IS_SIGNED)
CONCAT_OP_WITH_INT(signed char, 8, true)
CONCAT_OP_WITH_INT(unsigned char, 8, false)
CONCAT_OP_WITH_INT(short, _AP_SIZE_short, true)
CONCAT_OP_WITH_INT(unsigned short, _AP_SIZE_short, false)
CONCAT_OP_WITH_INT(int, _AP_SIZE_int, true)
CONCAT_OP_WITH_INT(unsigned int, _AP_SIZE_int, false)
CONCAT_OP_WITH_INT(long, _AP_SIZE_long, true)
CONCAT_OP_WITH_INT(unsigned long, _AP_SIZE_long, false)
CONCAT_OP_WITH_INT(ap_slong, _AP_SIZE_ap_slong, true)
CONCAT_OP_WITH_INT(ap_ulong, _AP_SIZE_ap_slong, false)

#undef CONCAT_OP_WITH_INT

#define CONCAT_SHIFT_WITH_INT(C_TYPE, OP)                                  \
  template <int _AP_W, typename _AP_T, int _AP_W1, typename _AP_T1>        \
  INLINE ap_uint<_AP_W + _AP_W1> operator OP(                              \
      const ap_concat_ref<_AP_W, _AP_T, _AP_W1, _AP_T1> lhs, C_TYPE rhs) { \
    return ap_uint<_AP_W + _AP_W1>(lhs).get() OP int(rhs);                 \
  }

// FIXME int(rhs) may loose precision.

CONCAT_SHIFT_WITH_INT(int, <<)
CONCAT_SHIFT_WITH_INT(unsigned int, <<)
CONCAT_SHIFT_WITH_INT(long, <<)
CONCAT_SHIFT_WITH_INT(unsigned long, <<)
CONCAT_SHIFT_WITH_INT(ap_slong, <<)
CONCAT_SHIFT_WITH_INT(ap_ulong, <<)

CONCAT_SHIFT_WITH_INT(int, >>)
CONCAT_SHIFT_WITH_INT(unsigned int, >>)
CONCAT_SHIFT_WITH_INT(long, >>)
CONCAT_SHIFT_WITH_INT(unsigned long, >>)
CONCAT_SHIFT_WITH_INT(ap_slong, >>)
CONCAT_SHIFT_WITH_INT(ap_ulong, >>)

#endif // ifndef __cplusplus
#endif // ifndef __AP_INT_REF_H__

// -*- cpp -*-
