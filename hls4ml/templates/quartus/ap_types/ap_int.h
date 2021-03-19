#ifndef __REMAP_AP_INT_H__
#define __REMAP_AP_INT_H__

#ifdef __INTELFPGA_COMPILER__
#include "HLS/ac_int.h"
#include "HLS/ac_fixed.h"
#else
#include <ac_int.h>
#include <ac_fixed.h>
#endif

template<int W>
using ap_int = ac_int<W, true>;

template<int W>
using ap_uint = ac_int<W, false>;

template<int W, int I>
using ap_fixed = ac_fixed<W,I,true>;

template<int W, int I>
using ap_ufixed = ac_fixed<W,I,false>;

#endif
