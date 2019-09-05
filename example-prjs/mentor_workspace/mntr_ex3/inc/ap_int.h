#ifndef __AESL_AP_INT_H__
#define __AESL_AP_INT_H__

#include <ac_int.h>
#include <ac_fixed.h>

template<int W>
using ap_int = ac_int<W, true>;

template<int W>
using ap_uint = ac_int<W, false>;

template<int W, int I>
using ap_fixed = ac_fixed<W,I,true>;

template<int W, int I>
using ap_ufixed = ac_fixed<W,I,false>;

#endif
