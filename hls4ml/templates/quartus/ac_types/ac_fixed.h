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
 *  Copyright 2005-2020, Mentor Graphics Corporation,                     *
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
//  Source:         ac_fixed.h
//  Description:    class for fixed point operation handling in C++
//  Author:         Andres Takach, Ph.D.
*/

#ifndef __AC_FIXED_H
#define __AC_FIXED_H

#include "ac_int.h"

#if (defined(__GNUC__) && __GNUC__ < 3 && !defined(__EDG__))
#error GCC version 3 or greater is required to include this header file
#endif

#if (defined(_MSC_VER) && _MSC_VER < 1400 && !defined(__EDG__))
#error Microsoft Visual Studio 8 or newer is required to include this header file
#endif

#if (defined(_MSC_VER) && !defined(__EDG__))
#pragma warning( push )
#pragma warning( disable: 4127 4308 4365 4514 4800 )
#endif

#ifndef __SYNTHESIS__
#ifndef __AC_FIXED_UTILITY_BASE
#define __AC_FIXED_UTILITY_BASE
#endif

#endif

#ifdef __SYNTHESIS__
#ifdef __AC_FIXED_NUMERICAL_ANALYSIS_BASE
#undef __AC_FIXED_NUMERICAL_ANALYSIS_BASE
#endif
#endif

#ifdef __AC_NAMESPACE
namespace __AC_NAMESPACE {
#endif

namespace ac_private {
  template<typename T>
  struct rt_ac_fixed_T {
    template<int W, int I, bool S>
    struct op1 {
      typedef typename T::template rt_T< ac_fixed<W,I,S,AC_TRN,AC_WRAP> >::mult mult;
      typedef typename T::template rt_T< ac_fixed<W,I,S,AC_TRN,AC_WRAP> >::plus plus;
      typedef typename T::template rt_T< ac_fixed<W,I,S,AC_TRN,AC_WRAP> >::minus2 minus;
      typedef typename T::template rt_T< ac_fixed<W,I,S,AC_TRN,AC_WRAP> >::minus minus2;
      typedef typename T::template rt_T< ac_fixed<W,I,S,AC_TRN,AC_WRAP> >::logic logic;
      typedef typename T::template rt_T< ac_fixed<W,I,S,AC_TRN,AC_WRAP> >::div2 div;
      typedef typename T::template rt_T< ac_fixed<W,I,S,AC_TRN,AC_WRAP> >::div div2;
    };
  };
  // specializations after definition of ac_fixed
}

namespace ac {
  template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O>
  class basic_num_ovf_base {
    bool d_enable;
  public:
    basic_num_ovf_base() : d_enable(true) {}
    void enable_ovf(bool a) { d_enable = a; }
    bool is_enabled() const { return d_enable; }
    template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
    void update(bool overflow, bool neg, const basic_num_ovf_base<W2,I2,S2,Q2,O2> &op2) {
#ifndef __AC_OVERRIDE_OVF_UPDATE_BODY
      if(d_enable) {
        if(overflow) {
          std::cerr << (neg ? "-" : "+") << "OVF: ";
          std::cerr << type_name() << " ( " << basic_num_ovf_base<W2,I2,S2,Q2,O2>::type_name(); 
          std::cerr << " ( " << op2.value().to_double() << " ) )" << std::endl;
        }
      }
#else
      __AC_OVERRIDE_OVF_UPDATE_BODY
#endif
    }
    void update(bool overflow, bool neg, double op2) {
#ifndef __AC_OVERRIDE_OVF_UPDATE2_BODY
      if(d_enable) {
        if(overflow) {
          std::cerr << (neg ? "-" : "+") << "OVF: ";
          std::cerr << type_name() << " ( " << "double"; 
          std::cerr << " ( " << op2 << " ) )" << std::endl;
        }
      }
#else
      __AC_OVERRIDE_OVF_UPDATE2_BODY
#endif
    }
    const ac_fixed<W,I,S,Q,O> &value() const;
    static std::string type_name();
  };
}

//////////////////////////////////////////////////////////////////////////////
//  ac_fixed
//////////////////////////////////////////////////////////////////////////////

//enum ac_q_mode { AC_TRN, AC_RND, AC_TRN_ZERO, AC_RND_ZERO, AC_RND_INF, AC_RND_MIN_INF, AC_RND_CONV, AC_RND_CONV_ODD };
//enum ac_o_mode { AC_WRAP, AC_SAT, AC_SAT_ZERO, AC_SAT_SYM };

template<int W, int I, bool S=true, ac_q_mode Q=AC_TRN, ac_o_mode O=AC_WRAP>
class ac_fixed : private ac_private::iv<(W+31+!S)/32>
#ifndef __SYNTHESIS__
__AC_FIXED_UTILITY_BASE
#endif
#ifdef __AC_FIXED_NUMERICAL_ANALYSIS_BASE
, public __AC_FIXED_NUMERICAL_ANALYSIS_BASE
#endif
{
#if defined(__SYNTHESIS__) && !defined(AC_IGNORE_BUILTINS)
#pragma builtin
#endif

  enum {N=(W+31+!S)/32};

  template<int W2>
  struct rt_priv {
    enum {w_shiftl = AC_MAX(W+W2,1) };
    typedef ac_fixed<w_shiftl, I, S> shiftl;
  };

  typedef ac_private::iv<N> Base;

  inline void bit_adjust() {
    const unsigned rem = (32-W)&31;
    Base::v[N-1] =  S ? ((signed) ((unsigned)Base::v[N-1]  << rem) >> rem) : (rem ?
                  ((unsigned) Base::v[N-1]  << rem) >> rem : 0);
  }
  inline Base &base() { return *this; }
  inline const Base &base() const { return *this; }

  inline void overflow_adjust(bool overflow, bool neg) {
    if(O==AC_WRAP) {
      bit_adjust();
      return;
    }
    else if(O==AC_SAT_ZERO) {
      if(overflow)
        ac_private::iv_extend<N>(Base::v, 0);
      else
        bit_adjust();
    }
    else if(S) {
      if(overflow) {
        if(!neg) {
          ac_private::iv_extend<N-1>(Base::v, ~0);
          Base::v[N-1] = ~((unsigned)~0 << ((W-1)&31));
        } else {
          ac_private::iv_extend<N-1>(Base::v, 0);
          Base::v[N-1] = ((unsigned)~0 << ((W-1)&31));
          if(O==AC_SAT_SYM)
            Base::v[0] |= 1;
        }
      } else
        bit_adjust();
    }
    else {
      if(overflow) {
        if(!neg) {
          ac_private::iv_extend<N-1>(Base::v, ~0);
          Base::v[N-1] = ~((unsigned)~0 << (W&31));
        } else
          ac_private::iv_extend<N>(Base::v, 0);
      } else
        bit_adjust();
    }
  }

  inline bool quantization_adjust(bool qb, bool r, bool s) {
    if(Q==AC_TRN)
      return false;
    if(Q==AC_RND_ZERO)
      qb &= s || r;
    else if(Q==AC_RND_MIN_INF)
      qb &= r;
    else if(Q==AC_RND_INF)
      qb &= !s || r;
    else if(Q==AC_RND_CONV)
      qb &= (Base::v[0] & 1) || r;
    else if(Q==AC_RND_CONV_ODD)
      qb &= (!(Base::v[0] & 1)) || r;
    else if(Q==AC_TRN_ZERO)
      qb = s && ( qb || r );
    return ac_private::iv_uadd_carry<N>(Base::v, qb, Base::v);
  }

  inline bool is_neg() const { return S && Base::v[N-1] < 0; }

public:
  static const int width = W;
  static const int i_width = I;
  static const bool sign = S;
  static const ac_o_mode o_mode = O;
  static const ac_q_mode q_mode = Q;
  static const int e_width = 0;
#ifdef __AC_FIXED_NUMERICAL_ANALYSIS_BASE
  static const bool compute_overflow_for_wrap = true;
#else
  static const bool compute_overflow_for_wrap = false;
#endif

  template<int W2, int I2, bool S2>
  struct rt {
    enum {
      F=W-I,
      F2=W2-I2,
      mult_w = W+W2,
      mult_i = I+I2,
      mult_s = S||S2,
      plus_w = AC_MAX(I+(S2&&!S),I2+(S&&!S2))+1+AC_MAX(F,F2),
      plus_i = AC_MAX(I+(S2&&!S),I2+(S&&!S2))+1,
      plus_s = S||S2,
      minus_w = AC_MAX(I+(S2&&!S),I2+(S&&!S2))+1+AC_MAX(F,F2),
      minus_i = AC_MAX(I+(S2&&!S),I2+(S&&!S2))+1,
      minus_s = true,
      div_w = W+AC_MAX(W2-I2,0)+S2,
      div_i = I+(W2-I2)+S2,
      div_s = S||S2,
      logic_w = AC_MAX(I+(S2&&!S),I2+(S&&!S2))+AC_MAX(F,F2),
      logic_i = AC_MAX(I+(S2&&!S),I2+(S&&!S2)),
      logic_s = S||S2
    };
    typedef ac_fixed<mult_w, mult_i, mult_s> mult;
    typedef ac_fixed<plus_w, plus_i, plus_s> plus;
    typedef ac_fixed<minus_w, minus_i, minus_s> minus;
    typedef ac_fixed<logic_w, logic_i, logic_s> logic;
    typedef ac_fixed<div_w, div_i, div_s> div;
    typedef ac_fixed<W, I, S> arg1;
  };

  template<typename T>
  struct rt_T {
    typedef typename ac_private::map<T>::t map_T;
    typedef typename ac_private::rt_ac_fixed_T<map_T>::template op1<W,I,S>::mult mult;
    typedef typename ac_private::rt_ac_fixed_T<map_T>::template op1<W,I,S>::plus plus;
    typedef typename ac_private::rt_ac_fixed_T<map_T>::template op1<W,I,S>::minus minus;
    typedef typename ac_private::rt_ac_fixed_T<map_T>::template op1<W,I,S>::minus2 minus2;
    typedef typename ac_private::rt_ac_fixed_T<map_T>::template op1<W,I,S>::logic logic;
    typedef typename ac_private::rt_ac_fixed_T<map_T>::template op1<W,I,S>::div div;
    typedef typename ac_private::rt_ac_fixed_T<map_T>::template op1<W,I,S>::div2 div2;
    typedef ac_fixed<W, I, S> arg1;
  };

  struct rt_unary {
    enum {
      neg_w = W+1,
      neg_i = I+1,
      neg_s = true,
      mag_sqr_w = 2*W-S,
      mag_sqr_i = 2*I-S,
      mag_sqr_s = false,
      mag_w = W+S,
      mag_i = I+S,
      mag_s = false,
      leading_sign_w = ac::log2_ceil<W+!S>::val,
      leading_sign_s = false
    };
    typedef ac_int<leading_sign_w, leading_sign_s> leading_sign;
    typedef ac_fixed<neg_w, neg_i, neg_s> neg;
    typedef ac_fixed<mag_sqr_w, mag_sqr_i, mag_sqr_s> mag_sqr;
    typedef ac_fixed<mag_w, mag_i, mag_s> mag;
    template<unsigned N>
    struct set {
      enum { sum_w = W + ac::log2_ceil<N>::val, sum_i = (sum_w-W) + I, sum_s = S};
      typedef ac_fixed<sum_w, sum_i, sum_s> sum;
    };
  };

  ac_fixed(const ac_fixed &op): Base(op) { }

  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2> friend class ac_fixed;
  ac_fixed() {
#if !defined(__SYNTHESIS__) && defined(AC_DEFAULT_IN_RANGE)
    bit_adjust();
    if( O==AC_SAT_SYM && S && Base::v[N-1] < 0 && (W > 1 ? ac_private::iv_equal_zeros_to<W-1,N>(Base::v) : true) )
      Base::v[0] |= 1;
#endif
  }
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  inline ac_fixed (const ac_fixed<W2,I2,S2,Q2,O2> &op) {
    enum {N2=(W2+31+!S2)/32, F=W-I, F2=W2-I2, QUAN_INC = F2>F && !(Q==AC_TRN || (Q==AC_TRN_ZERO && !S2)) };
    bool carry = false;
    // handle quantization
    if(F2 == F)
      Base::operator =(op);
    else if(F2 > F) {
      op.template const_shift_r<N,F2-F>(*this);
//      ac_private::iv_const_shift_r<N2,N,F2-F>(op.v, Base::v);
      if(Q!=AC_TRN && !(Q==AC_TRN_ZERO && !S2)) {
        bool qb = (F2-F > W2) ? (op.v[N2-1] < 0) : (bool) op[F2-F-1];
        bool r = (F2 > F+1) ? !ac_private::iv_equal_zeros_to<F2-F-1,N2>(op.v) : false;
        carry = quantization_adjust(qb, r, S2 && op.v[N2-1] < 0);
      }
    }
    else  // no quantization
      op.template const_shift_l<N,F-F2>(*this);
//      ac_private::iv_const_shift_l<N2,N,F-F2>(op.v, Base::v);
    // handle overflow
    if((O!=AC_WRAP || compute_overflow_for_wrap)
       && ((!S && S2) || I-S < I2-S2+(QUAN_INC || (S2 && O==AC_SAT_SYM && (O2 != AC_SAT_SYM || F2 > F) )))
    ) { // saturation
      bool deleted_bits_zero = !(W&31)&S || !(Base::v[N-1] >> (W&31));
      bool deleted_bits_one = !(W&31)&S || !~(Base::v[N-1] >> (W&31));
      bool neg_src;
      if(F2-F+32*N < W2) {
        bool all_ones = ac_private::iv_equal_ones_from<F2-F+32*N,N2>(op.v);
        deleted_bits_zero = deleted_bits_zero && (carry ? all_ones : ac_private::iv_equal_zeros_from<F2-F+32*N,N2>(op.v));
        deleted_bits_one = deleted_bits_one && (carry ? ac_private::iv_equal_ones_from<1+F2-F+32*N,N2>(op.v) && !op[F2-F+32*N] : all_ones);
        neg_src = S2 && op.v[N2-1] < 0 && !(carry & all_ones);
      }
      else
        neg_src = S2 && op.v[N2-1] < 0 && Base::v[N-1] < 0;
      bool neg_trg = S && (bool) this->operator[](W-1);
      bool overflow = !neg_src && (neg_trg || !deleted_bits_zero);
      overflow |= neg_src && (!neg_trg || !deleted_bits_one);
      if(O==AC_SAT_SYM && S && S2)
        overflow |= neg_src && (W > 1 ? ac_private::iv_equal_zeros_to<W-1,N>(Base::v) : true);
      overflow_adjust(overflow, neg_src);
#ifdef __AC_FIXED_NUMERICAL_ANALYSIS_BASE
    __AC_FIXED_NUMERICAL_ANALYSIS_BASE::update(overflow,neg_src,op);
#endif
    }
    else
      bit_adjust();
  }

  template<int W2, bool S2>
  inline ac_fixed (const ac_int<W2,S2> &op) {
    ac_fixed<W2,W2,S2> f_op;
    f_op.base().operator =(op);
    *this = f_op;
  }

  template<int W2>
  typename rt_priv<W2>::shiftl shiftl() const {
    typedef typename rt_priv<W2>::shiftl shiftl_t;
    shiftl_t r;
    Base::template const_shift_l<shiftl_t::N,W2>(r);
    return r;
  }

  inline ac_fixed( bool b ) { *this = (ac_int<1,false>) b; }
  inline ac_fixed( char b ) { *this = (ac_int<8,true>) b; }
  inline ac_fixed( signed char b ) { *this = (ac_int<8,true>) b; }
  inline ac_fixed( unsigned char b ) { *this = (ac_int<8,false>) b; }
  inline ac_fixed( signed short b ) { *this = (ac_int<16,true>) b; }
  inline ac_fixed( unsigned short b ) { *this = (ac_int<16,false>) b; }
  inline ac_fixed( signed int b ) { *this = (ac_int<32,true>) b; }
  inline ac_fixed( unsigned int b ) { *this = (ac_int<32,false>) b; }
  inline ac_fixed( signed long b ) { *this = (ac_int<ac_private::long_w,true>) b; }
  inline ac_fixed( unsigned long b ) { *this = (ac_int<ac_private::long_w,false>) b; }
  inline ac_fixed( Slong b ) { *this = (ac_int<64,true>) b; }
  inline ac_fixed( Ulong b ) { *this = (ac_int<64,false>) b; }

  inline ac_fixed( double d ) {
    double di = ac_private::ldexpr<-(I+!S+((32-W-!S)&31))>(d);
    bool o, qb, r;
    bool neg_src = d < 0;
    Base::conv_from_fraction(di, &qb, &r, &o);
    quantization_adjust(qb, r, neg_src);
    // a neg number may become non neg (0) after quantization
    neg_src &= o || Base::v[N-1] < 0;

    if(O!=AC_WRAP || compute_overflow_for_wrap) { // saturation
      bool overflow;
      bool neg_trg = S && (bool) this->operator[](W-1);
      if(o) {
        overflow = true;
      } else {
        bool deleted_bits_zero = !(W&31)&S || !(Base::v[N-1] >> (W&31));
        bool deleted_bits_one = !(W&31)&S || !~(Base::v[N-1] >> (W&31));
        overflow = !neg_src && (neg_trg || !deleted_bits_zero);
        overflow |= neg_src && (!neg_trg || !deleted_bits_one);
      }
      if(O==AC_SAT_SYM && S)
        overflow |= neg_src && (W > 1 ? ac_private::iv_equal_zeros_to<W-1,N>(Base::v) : true);
      overflow_adjust(overflow, neg_src);
#ifdef __AC_FIXED_NUMERICAL_ANALYSIS_BASE
      __AC_FIXED_NUMERICAL_ANALYSIS_BASE::update(overflow,neg_src,d);
#endif
    } else
      bit_adjust();
  }

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
  inline ac_fixed &set_val() {
    if(V == AC_VAL_DC) {
      ac_fixed r;
      Base::operator =(r);
      bit_adjust();
    }
    else if(V == AC_VAL_0 || V == AC_VAL_MIN || V == AC_VAL_QUANTUM) {
      Base::operator =(0);
      if(S && V == AC_VAL_MIN) {
        const unsigned rem = (W-1)&31;
        Base::v[N-1] = ((unsigned)~0 << rem);
        if(O == AC_SAT_SYM) {
          if(W == 1)
            Base::v[0] = 0;
          else
            Base::v[0] |= 1;
        }
      } else if(V == AC_VAL_QUANTUM)
        Base::v[0] = 1;
    }
    else {  // AC_VAL_MAX
      Base::operator =(-1);
      const unsigned int rem = (32-W - (unsigned) !S )&31;
      Base::v[N-1] = ((unsigned) (-1) >> 1) >> rem;
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

  // Explicit conversion functions to ac_int that captures all integer bits (bits are truncated)
  inline ac_int<AC_MAX(I,1),S> to_ac_int() const { return ((ac_fixed<AC_MAX(I,1),AC_MAX(I,1),S>) *this).template slc<AC_MAX(I,1)>(0); }

  // Explicit conversion functions to C built-in types -------------
  inline int to_int() const { return ((I-W) >= 32) ? 0 : (signed int) to_ac_int(); }
  inline unsigned to_uint() const { return ((I-W) >= 32) ? 0 : (unsigned int) to_ac_int(); }
  inline long to_long() const { return ((I-W) >= ac_private::long_w) ? 0 : (signed long) to_ac_int(); }
  inline unsigned long to_ulong() const { return ((I-W) >= ac_private::long_w) ? 0 : (unsigned long) to_ac_int(); }
  inline Slong to_int64() const { return ((I-W) >= 64) ? 0 : (Slong) to_ac_int(); }
  inline Ulong to_uint64() const { return ((I-W) >= 64) ? 0 : (Ulong) to_ac_int(); }
  inline double to_double() const { return ac_private::ldexpr<I-W>(Base::to_double()); }

  inline int length() const { return W; }

  inline std::string to_string(ac_base_mode base_rep, bool sign_mag = false) const {
    // base_rep == AC_DEC => sign_mag == don't care (always print decimal in sign magnitude)
    char r[(W-AC_MIN(AC_MIN(W-I,I),0)+31)/32*32+5] = {0};
    int i = 0;
    if(sign_mag)
      r[i++] = is_neg() ? '-' : '+';
    else if (base_rep == AC_DEC && is_neg())
      r[i++] = '-';
    if(base_rep != AC_DEC) {
      r[i++] = '0';
      r[i++] = base_rep == AC_BIN ? 'b' : (base_rep == AC_OCT ? 'o' : 'x');
    }
    ac_fixed<W+1, I+1, true> t;
    if( (base_rep == AC_DEC || sign_mag) && is_neg() )
      t = operator -();
    else
      t = *this;
    ac_fixed<AC_MAX(I+1,1),AC_MAX(I+1,1),true> i_part = t;
    ac_fixed<AC_MAX(W-I,1),0,false> f_part = t;
    i += ac_private::to_string(i_part.v, AC_MAX(I+1,1), sign_mag, base_rep, false, r+i);
    if(W-I > 0) {
      r[i++] = '.';
      if(!ac_private::to_string(f_part.v, W-I, false, base_rep, true, r+i))
        r[--i] = 0;
    }
    if(!i) {
      r[0] = '0';
      r[1] = 0;
    }
    return std::string(r);
  }
  inline static std::string type_name() {
    const char *tf[] = {"false", "true" };
    const char *q[] = {"AC_TRN", "AC_RND", "AC_TRN_ZERO", "AC_RND_ZERO", "AC_RND_INF", "AC_RND_MIN_INF", "AC_RND_CONV", "AC_RND_CONV_ODD" };
    const char *o[] = {"AC_WRAP", "AC_SAT", "AC_SAT_ZERO", "AC_SAT_SYM" };
    std::string r = "ac_fixed<";
    r += ac_int<32,true>(W).to_string(AC_DEC) + ',';
    r += ac_int<32,true>(I).to_string(AC_DEC) + ',';
    r += tf[S];
    r += ',';
    r += q[Q];
    r += ',';
    r += o[O];
    r += '>';
    return r;
  }

  // Arithmetic : Binary ----------------------------------------------------
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  typename rt<W2,I2,S2>::mult operator *( const ac_fixed<W2,I2,S2,Q2,O2> &op2) const {
    typename rt<W2,I2,S2>::mult r;
    Base::mult(op2, r);
    return r;
  }
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  typename rt<W2,I2,S2>::plus operator +( const ac_fixed<W2,I2,S2,Q2,O2> &op2) const {
    enum { F=W-I, F2=W2-I2 };
    typename rt<W2,I2,S2>::plus r;
    if(F == F2)
      Base::add(op2, r);
    else if(F > F2)
      Base::add(op2.template shiftl<F-F2>(), r);
    else
      shiftl<F2-F>().add(op2, r);
    return r;
  }
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  typename rt<W2,I2,S2>::minus operator -( const ac_fixed<W2,I2,S2,Q2,O2> &op2) const {
    enum { F=W-I, F2=W2-I2 };
    typename rt<W2,I2,S2>::minus r;
    if(F == F2)
      Base::sub(op2, r);
    else if(F > F2)
      Base::sub(op2.template shiftl<F-F2>(), r);
    else
      shiftl<F2-F>().sub(op2, r);
    return r;
  }
#if (defined(__GNUC__) && ( __GNUC__ == 4 && __GNUC_MINOR__ >= 6 || __GNUC__ > 4 ) && !defined(__EDG__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wenum-compare"
#endif
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  typename rt<W2,I2,S2>::div operator /( const ac_fixed<W2,I2,S2,Q2,O2> &op2) const {
    typename rt<W2,I2,S2>::div r;
    enum { Num_w = W+AC_MAX(W2-I2,0), Num_i = I, Num_w_minus = Num_w+S, Num_i_minus = Num_i+S,
          N1 = ac_fixed<Num_w,Num_i,S>::N, N1minus = ac_fixed<Num_w_minus,Num_i_minus,S>::N,
          N2 = ac_fixed<W2,I2,S2>::N, N2minus = ac_fixed<W2+S2,I2+S2,S2>::N,
          num_s = S + (N1minus > N1), den_s = S2 + (N2minus > N2), Nr = rt<W2,I2,S2>::div::N };
    ac_fixed<Num_w, Num_i, S> t = *this;
    t.template div<num_s, N2, den_s, Nr>(op2, r);
    return r;
  }
#if (defined(__GNUC__) && ( __GNUC__ == 4 && __GNUC_MINOR__ >= 6 || __GNUC__ > 4 ) && !defined(__EDG__))
#pragma GCC diagnostic pop
#endif
  // Arithmetic assign  ------------------------------------------------------
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  ac_fixed &operator *=( const ac_fixed<W2,I2,S2,Q2,O2> &op2) {
    *this = this->operator *(op2);
    return *this;
  }
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  ac_fixed &operator +=( const ac_fixed<W2,I2,S2,Q2,O2> &op2) {
    *this = this->operator +(op2);
    return *this;
  }
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  ac_fixed &operator -=( const ac_fixed<W2,I2,S2,Q2,O2> &op2) {
    *this = this->operator -(op2);
    return *this;
  }
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  ac_fixed &operator /=( const ac_fixed<W2,I2,S2,Q2,O2> &op2) {
    *this = this->operator /(op2);
    return *this;
  }
  // increment/decrement by quantum (smallest difference that can be represented)
  // Arithmetic prefix increment, decrement ---------------------------------
  ac_fixed &operator ++() {
    ac_fixed<1,I-W+1,false> q;
    q.template set_val<AC_VAL_QUANTUM>();
    operator += (q);
    return *this;
  }
  ac_fixed &operator --() {
    ac_fixed<1,I-W+1,false> q;
    q.template set_val<AC_VAL_QUANTUM>();
    operator -= (q);
    return *this;
  }
  // Arithmetic postfix increment, decrement ---------------------------------
  const ac_fixed operator ++(int) {
    ac_fixed t = *this;
    ac_fixed<1,I-W+1,false> q;
    q.template set_val<AC_VAL_QUANTUM>();
    operator += (q);
    return t;
  }
  const ac_fixed operator --(int) {
    ac_fixed t = *this;
    ac_fixed<1,I-W+1,false> q;
    q.template set_val<AC_VAL_QUANTUM>();
    operator -= (q);
    return t;
  }
  // Arithmetic Unary --------------------------------------------------------
  ac_fixed operator +() {
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
  ac_fixed<W+!S, I+!S, true> operator ~() const {
    ac_fixed<W+!S, I+!S, true> r;
    Base::bitwise_complement(r);
    return r;
  }
  // Bitwise (not arithmetic) bit complement  -----------------------------
  ac_fixed<W, I, false> bit_complement() const {
    ac_fixed<W, I, false> r;
    Base::bitwise_complement(r);
    r.bit_adjust();
    return r;
  }
  // Bitwise (not arithmetic): and, or, xor ----------------------------------
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  typename rt<W2,I2,S2>::logic operator &( const ac_fixed<W2,I2,S2,Q2,O2> &op2) const {
    enum { F=W-I, F2=W2-I2 };
    typename rt<W2,I2,S2>::logic r;
    if(F == F2)
      Base::bitwise_and(op2, r);
    else if(F > F2)
      Base::bitwise_and(op2.template shiftl<F-F2>(), r);
    else
      shiftl<F2-F>().bitwise_and(op2, r);
    return r;
  }
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  typename rt<W2,I2,S2>::logic operator |( const ac_fixed<W2,I2,S2,Q2,O2> &op2) const {
    enum { F=W-I, F2=W2-I2 };
    typename rt<W2,I2,S2>::logic r;
    if(F == F2)
      Base::bitwise_or(op2, r);
    else if(F > F2)
      Base::bitwise_or(op2.template shiftl<F-F2>(), r);
    else
      shiftl<F2-F>().bitwise_or(op2, r);
    return r;
  }
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  typename rt<W2,I2,S2>::logic operator ^( const ac_fixed<W2,I2,S2,Q2,O2> &op2) const {
    enum { F=W-I, F2=W2-I2 };
    typename rt<W2,I2,S2>::logic r;
    if(F == F2)
      Base::bitwise_xor(op2, r);
    else if(F > F2)
      Base::bitwise_xor(op2.template shiftl<F-F2>(), r);
    else
      shiftl<F2-F>().bitwise_xor(op2, r);
    return r;
  }
  // Bitwise assign (not arithmetic): and, or, xor ----------------------------
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  ac_fixed &operator &= ( const ac_fixed<W2,I2,S2,Q2,O2> &op2 ) {
    *this = this->operator &(op2);
    return *this;
  }
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  ac_fixed &operator |= ( const ac_fixed<W2,I2,S2,Q2,O2> &op2 ) {
    *this = this->operator |(op2);
    return *this;
  }
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  ac_fixed &operator ^= ( const ac_fixed<W2,I2,S2,Q2,O2> &op2 ) {
    *this = this->operator ^(op2);
    return *this;
  }
  // Shift (result constrained by left operand) -------------------------------
  template<int W2>
  ac_fixed operator << ( const ac_int<W2,true> &op2 ) const {
    // currently not written to overflow or quantize (neg shift)
    ac_fixed r;
    Base::shift_l2(op2.to_int(), r);
    r.bit_adjust();
    return r;
  }
  template<int W2>
  ac_fixed operator << ( const ac_int<W2,false> &op2 ) const {
    // currently not written to overflow
    ac_fixed r;
    Base::shift_l(op2.to_uint(), r);
    r.bit_adjust();
    return r;
  }
  template<int W2>
  ac_fixed operator >> ( const ac_int<W2,true> &op2 ) const {
    // currently not written to quantize or overflow (neg shift)
    ac_fixed r;
    Base::shift_r2(op2.to_int(), r);
    r.bit_adjust();
    return r;
  }
  template<int W2>
  ac_fixed operator >> ( const ac_int<W2,false> &op2 ) const {
    // currently not written to quantize
    ac_fixed r;
    Base::shift_r(op2.to_uint(), r);
    r.bit_adjust();
    return r;
  }
  // Shift assign ------------------------------------------------------------
  template<int W2>
  ac_fixed operator <<= ( const ac_int<W2,true> &op2 ) {
    // currently not written to overflow or quantize (neg shift)
    Base r;
    Base::shift_l2(op2.to_int(), r);
    Base::operator=(r);
    bit_adjust();
    return *this;
  }
  template<int W2>
  ac_fixed operator <<= ( const ac_int<W2,false> &op2 ) {
    // currently not written to overflow
    Base r;
    Base::shift_l(op2.to_uint(), r);
    Base::operator=(r);
    bit_adjust();
    return *this;
  }
  template<int W2>
  ac_fixed operator >>= ( const ac_int<W2,true> &op2 ) {
    // currently not written to quantize or overflow (neg shift)
    Base r;
    Base::shift_r2(op2.to_int(), r);
    Base::operator=(r);
    bit_adjust();
    return *this;
  }
  template<int W2>
  ac_fixed operator >>= ( const ac_int<W2,false> &op2 ) {
    // currently not written to quantize
    Base r;
    Base::shift_r(op2.to_uint(), r);
    Base::operator=(r);
    bit_adjust();
    return *this;
  }
  // Relational ---------------------------------------------------------------
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  bool operator == ( const ac_fixed<W2,I2,S2,Q2,O2> &op2) const {
    enum { F=W-I, F2=W2-I2 };
    if(F == F2)
      return Base::equal(op2);
    else if(F > F2)
      return Base::equal(op2.template shiftl<F-F2>());
    else
      return shiftl<F2-F>().equal(op2);
  }
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  bool operator != ( const ac_fixed<W2,I2,S2,Q2,O2> &op2) const {
    enum { F=W-I, F2=W2-I2 };
    if(F == F2)
      return ! Base::equal(op2);
    else if(F > F2)
      return ! Base::equal(op2.template shiftl<F-F2>());
    else
      return ! shiftl<F2-F>().equal(op2);
  }
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  bool operator < ( const ac_fixed<W2,I2,S2,Q2,O2> &op2) const {
    enum { F=W-I, F2=W2-I2 };
    if(F == F2)
      return Base::less_than(op2);
    else if(F > F2)
      return Base::less_than(op2.template shiftl<F-F2>());
    else
      return shiftl<F2-F>().less_than(op2);
  }
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  bool operator >= ( const ac_fixed<W2,I2,S2,Q2,O2> &op2) const {
    enum { F=W-I, F2=W2-I2 };
    if(F == F2)
      return ! Base::less_than(op2);
    else if(F > F2)
      return ! Base::less_than(op2.template shiftl<F-F2>());
    else
      return ! shiftl<F2-F>().less_than(op2);
  }
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  bool operator > ( const ac_fixed<W2,I2,S2,Q2,O2> &op2) const {
    enum { F=W-I, F2=W2-I2 };
    if(F == F2)
      return Base::greater_than(op2);
    else if(F > F2)
      return Base::greater_than(op2.template shiftl<F-F2>());
    else
      return shiftl<F2-F>().greater_than(op2);
  }
  template<int W2, int I2, bool S2, ac_q_mode Q2, ac_o_mode O2>
  bool operator <= ( const ac_fixed<W2,I2,S2,Q2,O2> &op2) const {
    enum { F=W-I, F2=W2-I2 };
    if(F == F2)
      return ! Base::greater_than(op2);
    else if(F > F2)
      return ! Base::greater_than(op2.template shiftl<F-F2>());
    else
      return ! shiftl<F2-F>().greater_than(op2);
  }
  bool operator == ( double d) const {
    if(is_neg() != (d < 0.0))
      return false;
    double di = ac_private::ldexpr<-(I+!S+((32-W-!S)&31))>(d);
    bool overflow, qb, r;
    ac_fixed<W,I,S> t;
    t.conv_from_fraction(di, &qb, &r, &overflow);
    if(qb || r || overflow)
      return false;
    return operator == (t);
  }
  bool operator != ( double d) const {
    return !operator == ( d );
  }
  bool operator < ( double d) const {
    if(is_neg() != (d < 0.0))
      return is_neg();
    double di = ac_private::ldexpr<-(I+!S+((32-W-!S)&31))>(d);
    bool overflow, qb, r;
    ac_fixed<W,I,S> t;
    t.conv_from_fraction(di, &qb, &r, &overflow);
    if(is_neg() && overflow)
      return false;
    return (!is_neg() && overflow) || ((qb || r) && operator <= (t)) || operator < (t);
  }
  bool operator >= ( double d) const {
    return !operator < ( d );
  }
  bool operator > ( double d) const {
    if(is_neg() != (d < 0.0))
      return !is_neg();
    double di = ac_private::ldexpr<-(I+!S+((32-W-!S)&31))>(d);
    bool overflow, qb, r;
    ac_fixed<W,I,S> t;
    t.conv_from_fraction(di, &qb, &r, &overflow);
    if(!is_neg() && overflow )
      return false;
    return (is_neg() && overflow) || operator > (t);
  }
  bool operator <= ( double d) const {
    return !operator > ( d );
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
  inline ac_fixed &set_slc(const ac_int<WX,SX> lsb, const ac_int<W2,S2> &slc) {
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
  inline ac_fixed &set_slc(signed lsb, const ac_int<W2,S2> &slc) {
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
  inline ac_fixed &set_slc(unsigned ulsb, const ac_int<W2,S2> &slc) {
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
    ac_fixed &d_bv;
    unsigned d_index;
  public:
    ac_bitref( ac_fixed *bv, unsigned index=0 ) : d_bv(*bv), d_index(index) {}
    operator bool () const { return (d_index < W) ? (d_bv.v[d_index>>5]>>(d_index&31) & 1) : 0; }

    inline ac_bitref operator = ( int val ) {
      // lsb of int (val&1) is written to bit
      if(d_index < W) {
        int *pval = &d_bv.v[d_index>>5];
        *pval ^= (*pval ^ ((unsigned) val << (d_index&31) )) & 1 << (d_index&31);
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
    ac_int<W,S> m = this->template slc<W>(0);
    bool r = m.normalize(exp);
    this->set_slc(0,m);
    return r;
  }
  // returns false if number is denormal, minimum exponent is reserved (usually for encoding special values/errors)
  template<int WE, bool SE>
  bool normalize_RME(ac_int<WE,SE> &exp) {
    ac_int<W,S> m = this->template slc<W>(0);
    bool r = m.normalize_RME(exp);
    this->set_slc(0,m);
    return r;
  }
  inline void bit_fill_hex(const char *str) {
    // Zero Pads if str is too short, throws ms bits away if str is too long
    // Asserts if anything other than 0-9a-fA-F is encountered
    ac_int<W,S> x;
    x.bit_fill_hex(str);
    set_slc(0, x);
  }
  template<int N>
  inline void bit_fill(const int (&ivec)[N], bool bigendian=true) {
    // bit_fill from integer vector
    //   if W > N*32, missing most significant bits are zeroed
    //   if W < N*32, additional bits in ivec are ignored (no overflow checking)
    //
    // Example:
    //   ac_fixed<80,40,false> x;    int vec[] = { 0xffffa987, 0x6543210f, 0xedcba987 };
    //   x.bit_fill(vec);   // vec[0] fill bits 79-64
    ac_int<W,S> x;
    x.bit_fill(ivec, bigendian);
    set_slc(0, x);
  }
};

namespace ac {
  template<typename T>
  struct ac_fixed_represent {
    enum { t_w = ac_private::c_type_params<T>::W, t_i = t_w, t_s = ac_private::c_type_params<T>::S };
    typedef ac_fixed<t_w,t_i,t_s> type;
  };
  template<> struct ac_fixed_represent<float> {};
  template<> struct ac_fixed_represent<double> {};
  template<int W, bool S>
  struct ac_fixed_represent< ac_int<W,S> > {
    typedef ac_fixed<W,W,S> type;
  };
  template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O>
  struct ac_fixed_represent< ac_fixed<W,I,S,Q,O> > {
    typedef ac_fixed<W,I,S,Q,O> type;
  };
}

namespace ac_private {
  // with T == ac_fixed
  template<int W2, int I2, bool S2>
  struct rt_ac_fixed_T< ac_fixed<W2,I2,S2> > {
    typedef ac_fixed<W2,I2,S2> fx2_t;
    template<int W, int I, bool S>
    struct op1 {
      typedef ac_fixed<W,I,S> fx_t;
      typedef typename fx_t::template rt<W2,I2,S2>::mult mult;
      typedef typename fx_t::template rt<W2,I2,S2>::plus plus;
      typedef typename fx_t::template rt<W2,I2,S2>::minus minus;
      typedef typename fx2_t::template rt<W,I,S>::minus minus2;
      typedef typename fx_t::template rt<W2,I2,S2>::logic logic;
      typedef typename fx_t::template rt<W2,I2,S2>::div div;
      typedef typename fx2_t::template rt<W,I,S>::div div2;
    };
  };
  // with T == ac_int
  template<int W2, bool S2>
  struct rt_ac_fixed_T< ac_int<W2,S2> > {
    typedef ac_fixed<W2,W2,S2> fx2_t;
    template<int W, int I, bool S>
    struct op1 {
      typedef ac_fixed<W,I,S> fx_t;
      typedef typename fx_t::template rt<W2,W2,S2>::mult mult;
      typedef typename fx_t::template rt<W2,W2,S2>::plus plus;
      typedef typename fx_t::template rt<W2,W2,S2>::minus minus;
      typedef typename fx2_t::template rt<W,I,S>::minus minus2;
      typedef typename fx_t::template rt<W2,W2,S2>::logic logic;
      typedef typename fx_t::template rt<W2,W2,S2>::div div;
      typedef typename fx2_t::template rt<W,I,S>::div div2;
    };
  };

  template<typename T>
  struct rt_ac_fixed_T< c_type<T> > {
    typedef typename ac::ac_fixed_represent<T>::type fx2_t;
    enum { W2 = fx2_t::width, I2 = W2, S2 = fx2_t::sign };
    template<int W, int I, bool S>
    struct op1 {
      typedef ac_fixed<W,I,S> fx_t;
      typedef typename fx_t::template rt<W2,W2,S2>::mult mult;
      typedef typename fx_t::template rt<W2,W2,S2>::plus plus;
      typedef typename fx_t::template rt<W2,W2,S2>::minus minus;
      typedef typename fx2_t::template rt<W,I,S>::minus minus2;
      typedef typename fx_t::template rt<W2,W2,S2>::logic logic;
      typedef typename fx_t::template rt<W2,W2,S2>::div div;
      typedef typename fx2_t::template rt<W,I,S>::div div2;
    };
  };
}


// Specializations for constructors on integers that bypass bit adjusting
//  and are therefore more efficient
template<> inline ac_fixed<1,1,true,AC_TRN,AC_WRAP>::ac_fixed( bool b ) { v[0] = b ? -1 : 0; }

template<> inline ac_fixed<1,1,false,AC_TRN,AC_WRAP>::ac_fixed( bool b ) { v[0] = b; }
template<> inline ac_fixed<1,1,false,AC_TRN,AC_WRAP>::ac_fixed( signed char b ) { v[0] = b&1; }
template<> inline ac_fixed<1,1,false,AC_TRN,AC_WRAP>::ac_fixed( unsigned char b ) { v[0] = b&1; }
template<> inline ac_fixed<1,1,false,AC_TRN,AC_WRAP>::ac_fixed( signed short b ) { v[0] = b&1; }
template<> inline ac_fixed<1,1,false,AC_TRN,AC_WRAP>::ac_fixed( unsigned short b ) { v[0] = b&1; }
template<> inline ac_fixed<1,1,false,AC_TRN,AC_WRAP>::ac_fixed( signed int b ) { v[0] = b&1; }
template<> inline ac_fixed<1,1,false,AC_TRN,AC_WRAP>::ac_fixed( unsigned int b ) { v[0] = b&1; }
template<> inline ac_fixed<1,1,false,AC_TRN,AC_WRAP>::ac_fixed( signed long b ) { v[0] = b&1; }
template<> inline ac_fixed<1,1,false,AC_TRN,AC_WRAP>::ac_fixed( unsigned long b ) { v[0] = b&1; }
template<> inline ac_fixed<1,1,false,AC_TRN,AC_WRAP>::ac_fixed( Ulong b ) { v[0] = (int) b&1; }
template<> inline ac_fixed<1,1,false,AC_TRN,AC_WRAP>::ac_fixed( Slong b ) { v[0] = (int) b&1; }

template<> inline ac_fixed<8,8,true,AC_TRN,AC_WRAP>::ac_fixed( bool b ) { v[0] = b; }
template<> inline ac_fixed<8,8,false,AC_TRN,AC_WRAP>::ac_fixed( bool b ) { v[0] = b; }
template<> inline ac_fixed<8,8,true,AC_TRN,AC_WRAP>::ac_fixed( signed char b ) { v[0] = b; }
template<> inline ac_fixed<8,8,false,AC_TRN,AC_WRAP>::ac_fixed( unsigned char b ) { v[0] = b; }
template<> inline ac_fixed<8,8,true,AC_TRN,AC_WRAP>::ac_fixed( unsigned char b ) { v[0] = (signed char) b; }
template<> inline ac_fixed<8,8,false,AC_TRN,AC_WRAP>::ac_fixed( signed char b ) { v[0] = (unsigned char) b; }

template<> inline ac_fixed<16,16,true,AC_TRN,AC_WRAP>::ac_fixed( bool b ) { v[0] = b; }
template<> inline ac_fixed<16,16,false,AC_TRN,AC_WRAP>::ac_fixed( bool b ) { v[0] = b; }
template<> inline ac_fixed<16,16,true,AC_TRN,AC_WRAP>::ac_fixed( signed char b ) { v[0] = b; }
template<> inline ac_fixed<16,16,false,AC_TRN,AC_WRAP>::ac_fixed( unsigned char b ) { v[0] = b; }
template<> inline ac_fixed<16,16,true,AC_TRN,AC_WRAP>::ac_fixed( unsigned char b ) { v[0] = b; }
template<> inline ac_fixed<16,16,false,AC_TRN,AC_WRAP>::ac_fixed( signed char b ) { v[0] = (unsigned short) b; }
template<> inline ac_fixed<16,16,true,AC_TRN,AC_WRAP>::ac_fixed( signed short b ) { v[0] = b; }
template<> inline ac_fixed<16,16,false,AC_TRN,AC_WRAP>::ac_fixed( unsigned short b ) { v[0] = b; }
template<> inline ac_fixed<16,16,true,AC_TRN,AC_WRAP>::ac_fixed( unsigned short b ) { v[0] = (signed short) b; }
template<> inline ac_fixed<16,16,false,AC_TRN,AC_WRAP>::ac_fixed( signed short b ) { v[0] = (unsigned short) b; }

template<> inline ac_fixed<32,32,true,AC_TRN,AC_WRAP>::ac_fixed( signed int b ) { v[0] = b; }
template<> inline ac_fixed<32,32,true,AC_TRN,AC_WRAP>::ac_fixed( unsigned int b ) { v[0] = b; }
template<> inline ac_fixed<32,32,false,AC_TRN,AC_WRAP>::ac_fixed( signed int b ) { v[0] = b; v[1] = 0;}
template<> inline ac_fixed<32,32,false,AC_TRN,AC_WRAP>::ac_fixed( unsigned int b ) { v[0] = b; v[1] = 0;}

template<> inline ac_fixed<32,32,true,AC_TRN,AC_WRAP>::ac_fixed( Slong b ) { v[0] = (int) b; }
template<> inline ac_fixed<32,32,true,AC_TRN,AC_WRAP>::ac_fixed( Ulong b ) { v[0] = (int) b; }
template<> inline ac_fixed<32,32,false,AC_TRN,AC_WRAP>::ac_fixed( Slong b ) { v[0] = (int) b; v[1] = 0;}
template<> inline ac_fixed<32,32,false,AC_TRN,AC_WRAP>::ac_fixed( Ulong b ) { v[0] = (int) b; v[1] = 0;}

template<> inline ac_fixed<64,64,true,AC_TRN,AC_WRAP>::ac_fixed( Slong b ) { v[0] = (int) b; v[1] = (int) (b >> 32); }
template<> inline ac_fixed<64,64,true,AC_TRN,AC_WRAP>::ac_fixed( Ulong b ) { v[0] = (int) b; v[1] = (int) (b >> 32);}
template<> inline ac_fixed<64,64,false,AC_TRN,AC_WRAP>::ac_fixed( Slong b ) { v[0] = (int) b; v[1] = (int) ((Ulong) b >> 32); v[2] = 0; }
template<> inline ac_fixed<64,64,false,AC_TRN,AC_WRAP>::ac_fixed( Ulong b ) { v[0] = (int) b; v[1] = (int) (b >> 32); v[2] = 0; }


// Stream --------------------------------------------------------------------

template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O>
inline std::ostream& operator << (std::ostream &os, const ac_fixed<W,I,S,Q,O> &x) {
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


// Macros for Binary Operators with C Integers --------------------------------------------

#define FX_BIN_OP_WITH_INT_2I(BIN_OP, C_TYPE, WI, SI)  \
  template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O> \
  inline ac_fixed<W,I,S,Q,O> operator BIN_OP ( const ac_fixed<W,I,S,Q,O> &op, C_TYPE i_op) {  \
    return op.operator BIN_OP (ac_int<WI,SI>(i_op));  \
  }

#define FX_BIN_OP_WITH_INT(BIN_OP, C_TYPE, WI, SI, RTYPE)  \
  template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O> \
  inline typename ac_fixed<WI,WI,SI>::template rt<W,I,S>::RTYPE operator BIN_OP ( C_TYPE i_op, const ac_fixed<W,I,S,Q,O> &op) {  \
    return ac_fixed<WI,WI,SI>(i_op).operator BIN_OP (op);  \
  } \
  template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O> \
  inline typename ac_fixed<W,I,S>::template rt<WI,WI,SI>::RTYPE operator BIN_OP ( const ac_fixed<W,I,S,Q,O> &op, C_TYPE i_op) {  \
    return op.operator BIN_OP (ac_fixed<WI,WI,SI>(i_op));  \
  }

#define FX_REL_OP_WITH_INT(REL_OP, C_TYPE, W2, S2)  \
  template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O> \
  inline bool operator REL_OP ( const ac_fixed<W,I,S,Q,O> &op, C_TYPE op2) {  \
    return op.operator REL_OP (ac_fixed<W2,W2,S2>(op2));  \
  }  \
  template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O> \
  inline bool operator REL_OP ( C_TYPE op2, const ac_fixed<W,I,S,Q,O> &op) {  \
    return ac_fixed<W2,W2,S2>(op2).operator REL_OP (op);  \
  }

#define FX_ASSIGN_OP_WITH_INT_2(ASSIGN_OP, C_TYPE, W2, S2)  \
  template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O> \
  inline ac_fixed<W,I,S,Q,O> &operator ASSIGN_OP ( ac_fixed<W,I,S,Q,O> &op, C_TYPE op2) {  \
    return op.operator ASSIGN_OP (ac_fixed<W2,W2,S2>(op2));  \
  }

#define FX_ASSIGN_OP_WITH_INT_2I(ASSIGN_OP, C_TYPE, W2, S2)  \
  template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O> \
  inline ac_fixed<W,I,S> operator ASSIGN_OP ( ac_fixed<W,I,S,Q,O> &op, C_TYPE op2) {  \
    return op.operator ASSIGN_OP (ac_int<W2,S2>(op2));  \
  }

#define FX_OPS_WITH_INT(C_TYPE, WI, SI) \
  FX_BIN_OP_WITH_INT(*, C_TYPE, WI, SI, mult) \
  FX_BIN_OP_WITH_INT(+, C_TYPE, WI, SI, plus) \
  FX_BIN_OP_WITH_INT(-, C_TYPE, WI, SI, minus) \
  FX_BIN_OP_WITH_INT(/, C_TYPE, WI, SI, div) \
  FX_BIN_OP_WITH_INT_2I(>>, C_TYPE, WI, SI) \
  FX_BIN_OP_WITH_INT_2I(<<, C_TYPE, WI, SI) \
  FX_BIN_OP_WITH_INT(&, C_TYPE, WI, SI, logic) \
  FX_BIN_OP_WITH_INT(|, C_TYPE, WI, SI, logic) \
  FX_BIN_OP_WITH_INT(^, C_TYPE, WI, SI, logic) \
  \
  FX_REL_OP_WITH_INT(==, C_TYPE, WI, SI) \
  FX_REL_OP_WITH_INT(!=, C_TYPE, WI, SI) \
  FX_REL_OP_WITH_INT(>, C_TYPE, WI, SI) \
  FX_REL_OP_WITH_INT(>=, C_TYPE, WI, SI) \
  FX_REL_OP_WITH_INT(<, C_TYPE, WI, SI) \
  FX_REL_OP_WITH_INT(<=, C_TYPE, WI, SI) \
  \
  FX_ASSIGN_OP_WITH_INT_2(+=, C_TYPE, WI, SI) \
  FX_ASSIGN_OP_WITH_INT_2(-=, C_TYPE, WI, SI) \
  FX_ASSIGN_OP_WITH_INT_2(*=, C_TYPE, WI, SI) \
  FX_ASSIGN_OP_WITH_INT_2(/=, C_TYPE, WI, SI) \
  FX_ASSIGN_OP_WITH_INT_2I(>>=, C_TYPE, WI, SI) \
  FX_ASSIGN_OP_WITH_INT_2I(<<=, C_TYPE, WI, SI) \
  FX_ASSIGN_OP_WITH_INT_2(&=, C_TYPE, WI, SI) \
  FX_ASSIGN_OP_WITH_INT_2(|=, C_TYPE, WI, SI) \
  FX_ASSIGN_OP_WITH_INT_2(^=, C_TYPE, WI, SI)

// --------------------------------------- End of Macros for Binary Operators with C Integers

#ifdef AC_FIXED_NS_FOR_MIXED_OPERATORS
namespace ac {
  namespace ops_with_other_types {
#endif
    // Binary Operators with C Integers --------------------------------------------
    FX_OPS_WITH_INT(bool, 1, false)
    FX_OPS_WITH_INT(char, 8, true)
    FX_OPS_WITH_INT(signed char, 8, true)
    FX_OPS_WITH_INT(unsigned char, 8, false)
    FX_OPS_WITH_INT(short, 16, true)
    FX_OPS_WITH_INT(unsigned short, 16, false)
    FX_OPS_WITH_INT(int, 32, true)
    FX_OPS_WITH_INT(unsigned int, 32, false)
    FX_OPS_WITH_INT(long, ac_private::long_w, true)
    FX_OPS_WITH_INT(unsigned long, ac_private::long_w, false)
    FX_OPS_WITH_INT(Slong, 64, true)
    FX_OPS_WITH_INT(Ulong, 64, false)
    // -------------------------------------- End of Binary Operators with Integers
#ifdef AC_FIXED_NS_FOR_MIXED_OPERATORS
  }  // ops_with_other_types namespace
} // ac namespace
#endif


// Macros for Binary Operators with ac_int --------------------------------------------

#define FX_BIN_OP_WITH_AC_INT_1(BIN_OP, RTYPE)  \
  template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O, int WI, bool SI> \
  inline typename ac_fixed<WI,WI,SI>::template rt<W,I,S>::RTYPE operator BIN_OP ( const ac_int<WI,SI> &i_op, const ac_fixed<W,I,S,Q,O> &op) {  \
    return ac_fixed<WI,WI,SI>(i_op).operator BIN_OP (op);  \
  }

#define FX_BIN_OP_WITH_AC_INT_2(BIN_OP, RTYPE)  \
  template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O, int WI, bool SI> \
  inline typename ac_fixed<W,I,S>::template rt<WI,WI,SI>::RTYPE operator BIN_OP ( const ac_fixed<W,I,S,Q,O> &op, const ac_int<WI,SI> &i_op) {  \
    return op.operator BIN_OP (ac_fixed<WI,WI,SI>(i_op));  \
  }

#define FX_BIN_OP_WITH_AC_INT(BIN_OP, RTYPE)  \
  FX_BIN_OP_WITH_AC_INT_1(BIN_OP, RTYPE) \
  FX_BIN_OP_WITH_AC_INT_2(BIN_OP, RTYPE)

#define FX_REL_OP_WITH_AC_INT(REL_OP)  \
  template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O, int WI, bool SI> \
  inline bool operator REL_OP ( const ac_fixed<W,I,S,Q,O> &op, const ac_int<WI,SI> &op2) {  \
    return op.operator REL_OP (ac_fixed<WI,WI,SI>(op2));  \
  }  \
  template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O, int WI, bool SI> \
  inline bool operator REL_OP ( ac_int<WI,SI> &op2, const ac_fixed<W,I,S,Q,O> &op) {  \
    return ac_fixed<WI,WI,SI>(op2).operator REL_OP (op);  \
  }

#define FX_ASSIGN_OP_WITH_AC_INT(ASSIGN_OP)  \
  template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O, int WI, bool SI> \
  inline ac_fixed<W,I,S,Q,O> &operator ASSIGN_OP ( ac_fixed<W,I,S,Q,O> &op, const ac_int<WI,SI> &op2) {  \
    return op.operator ASSIGN_OP (ac_fixed<WI,WI,SI>(op2));  \
  }  \
  template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O, int WI, bool SI> \
  inline ac_int<WI,SI> &operator ASSIGN_OP ( ac_int<WI,SI> &op, const ac_fixed<W,I,S,Q,O> &op2) {  \
    return op.operator ASSIGN_OP (op2.to_ac_int());  \
  }

// -------------------------------------------- End of Macros for Binary Operators with ac_int

#ifdef AC_FIXED_NS_FOR_MIXED_OPERATORS
namespace ac {
  namespace ops_with_other_types {
#endif
    // Binary Operators with ac_int --------------------------------------------
    FX_BIN_OP_WITH_AC_INT(*, mult)
    FX_BIN_OP_WITH_AC_INT(+, plus)
    FX_BIN_OP_WITH_AC_INT(-, minus)
    FX_BIN_OP_WITH_AC_INT(/, div)
    FX_BIN_OP_WITH_AC_INT(&, logic)
    FX_BIN_OP_WITH_AC_INT(|, logic)
    FX_BIN_OP_WITH_AC_INT(^, logic)

    FX_REL_OP_WITH_AC_INT(==)
    FX_REL_OP_WITH_AC_INT(!=)
    FX_REL_OP_WITH_AC_INT(>)
    FX_REL_OP_WITH_AC_INT(>=)
    FX_REL_OP_WITH_AC_INT(<)
    FX_REL_OP_WITH_AC_INT(<=)

    FX_ASSIGN_OP_WITH_AC_INT(+=)
    FX_ASSIGN_OP_WITH_AC_INT(-=)
    FX_ASSIGN_OP_WITH_AC_INT(*=)
    FX_ASSIGN_OP_WITH_AC_INT(/=)
    FX_ASSIGN_OP_WITH_AC_INT(&=)
    FX_ASSIGN_OP_WITH_AC_INT(|=)
    FX_ASSIGN_OP_WITH_AC_INT(^=)
    // -------------------------------------- End of Binary Operators with ac_int

    // Relational Operators with double --------------------------------------
    template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O>
    inline bool operator == ( double op, const ac_fixed<W,I,S,Q,O> &op2) {
      return op2.operator == (op);
    }
    template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O>
    inline bool operator != ( double op, const ac_fixed<W,I,S,Q,O> &op2) {
      return op2.operator != (op);
    }
    template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O>
    inline bool operator > ( double op, const ac_fixed<W,I,S,Q,O> &op2) {
      return op2.operator < (op);
    }
    template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O>
    inline bool operator < ( double op, const ac_fixed<W,I,S,Q,O> &op2) {
      return op2.operator > (op);
    }
    template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O>
    inline bool operator <= ( double op, const ac_fixed<W,I,S,Q,O> &op2) {
      return op2.operator >= (op);
    }
    template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O>
    inline bool operator >= ( double op, const ac_fixed<W,I,S,Q,O> &op2) {
      return op2.operator <= (op);
    }
    // -------------------------------------- End of Relational Operators with double
#ifdef AC_FIXED_NS_FOR_MIXED_OPERATORS
  }  // ops_with_other_types namespace
} // ac namespace
using namespace ac::ops_with_other_types;
#endif


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
template<ac_special_val V, int W, int I, bool S, ac_q_mode Q, ac_o_mode O>
inline ac_fixed<W,I,S,Q,O> value(ac_fixed<W,I,S,Q,O>) {
  ac_fixed<W,I,S> r;
  return r.template set_val<V>();
}

namespace ac {
// PUBLIC FUNCTIONS
// function to initialize (or uninitialize) arrays
  template<ac_special_val V, int W, int I, bool S, ac_q_mode Q, ac_o_mode O>
  inline bool init_array(ac_fixed<W,I,S,Q,O> *a, int n) {
    ac_fixed<W,I,S> t;
    t.template set_val<V>();
    for(int i=0; i < n; i++)
      a[i] = t;
    return true;
  }

  inline ac_fixed<54,2,true> frexp_d(double d, ac_int<11,true> &exp) {
    enum {Min_Exp = -1022, Max_Exp = 1023, Mant_W = 52, Denorm_Min_Exp = Min_Exp - Mant_W};
    if(!d) {
      exp = 0;
      return 0;
    }
    int exp_i;
    double f0 = frexp(d, &exp_i);
    AC_ASSERT(exp_i <= Max_Exp+1, "Exponent greater than standard double-precision float exponent max (+1024). It is probably an extended double");
    AC_ASSERT(exp_i >= Denorm_Min_Exp+1, "Exponent less than standard double-precision float exponent min (-1021). It is probably an extended double");
    exp_i--;
    int rshift = exp_i < Min_Exp ? Min_Exp - exp_i : (exp_i > Min_Exp && f0 < 0 && f0 >= -0.5) ? -1 : 0;
    exp = exp_i + rshift;
    ac_int<Mant_W+2,true> f_i = f0 * ((Ulong) 1 << (Mant_W + 1 -rshift));
    ac_fixed<Mant_W+2,2,true> r;
    r.set_slc(0, f_i);
    return r;
  }
  inline ac_fixed<25,2,true> frexp_f(float f, ac_int<8,true> &exp) {
    enum {Min_Exp = -126, Max_Exp = 127, Mant_W = 23, Denorm_Min_Exp = Min_Exp - Mant_W};
    if(!f) {
      exp = 0;
      return 0;
    }
    int exp_i;
    float f0 = frexpf(f, &exp_i);
    AC_ASSERT(exp_i <= Max_Exp+1, "Exponent greater than standard single-precision float exponent max (+128). It is probably an extended float");
    AC_ASSERT(exp_i >= Denorm_Min_Exp+1, "Exponent less than standard single-precision float exponent min (-125). It is probably an extended float");
    exp_i--;
    int rshift = exp_i < Min_Exp ? Min_Exp - exp_i : (exp_i >= Min_Exp && f0 < 0 && f0 >= -0.5) ? -1 : 0;
    exp = exp_i + rshift;
    ac_int<Mant_W+2,true> f_i = f0 * (1 << (Mant_W + 1 - rshift));
    ac_fixed<Mant_W+2,2,true> r;
    r.set_slc(0, f_i);
    return r;
  }

  inline ac_fixed<53,1,false> frexp_sm_d(double d, ac_int<11,true> &exp, bool &sign) {
    enum {Min_Exp = -1022, Max_Exp = 1023, Mant_W = 52, Denorm_Min_Exp = Min_Exp - Mant_W};
    if(!d) {
      exp = 0;
      sign = false;
      return 0;
    }
    int exp_i;
    bool s = d < 0;
    double f0 = frexp(s ? -d : d, &exp_i);
    AC_ASSERT(exp_i <= Max_Exp+1, "Exponent greater than standard double-precision float exponent max (+1024). It is probably an extended double");
    AC_ASSERT(exp_i >= Denorm_Min_Exp+1, "Exponent less than standard double-precision float exponent min (-1021). It is probably an extended double");
    exp_i--;
    int rshift = exp_i < Min_Exp ? Min_Exp - exp_i : 0;
    exp = exp_i + rshift;
    ac_int<Mant_W+1,false> f_i = f0 * ((Ulong) 1 << (Mant_W + 1 -rshift));
    ac_fixed<Mant_W+1,1,false> r;
    r.set_slc(0, f_i);
    sign = s;
    return r;
  }
  inline ac_fixed<24,1,false> frexp_sm_f(float f, ac_int<8,true> &exp, bool &sign) {
    enum {Min_Exp = -126, Max_Exp = 127, Mant_W = 23, Denorm_Min_Exp = Min_Exp - Mant_W};
    if(!f) {
      exp = 0;
      sign = false;
      return 0;
    }
    int exp_i;
    bool s = f < 0;
    float f0 = frexp(s ? -f : f, &exp_i);
    AC_ASSERT(exp_i <= Max_Exp+1, "Exponent greater than standard single-precision float exponent max (+128). It is probably an extended float");
    AC_ASSERT(exp_i >= Denorm_Min_Exp+1, "Exponent less than standard single-precision float exponent min (-125). It is probably an extended float");
    exp_i--;
    int rshift = exp_i < Min_Exp ? Min_Exp - exp_i : 0;
    exp = exp_i + rshift;
    ac_int<24,false> f_i = f0 * (1 << (Mant_W + 1 - rshift));
    ac_fixed<24,1,false> r;
    r.set_slc(0, f_i);
    sign = s;
    return r;
  }

  template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O>
  const ac_fixed<W,I,S,Q,O> &basic_num_ovf_base<W,I,S,Q,O>::value() const {
    return (const ac_fixed<W,I,S,Q,O> &) *this;
  }

  template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O> std::string basic_num_ovf_base<W,I,S,Q,O>::type_name() {
    return ac_fixed<W,I,S,Q,O>::type_name();
  }
}


///////////////////////////////////////////////////////////////////////////////

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

#endif // __AC_FIXED_H
