#ifndef __AESL_AP_INT_H__
#define __AESL_AP_INT_H__

#define SC_INCLUDE_FX
#include <systemc.h>
template<int W>
using ap_int = sc_bigint<W>;

template<int W>
using ap_uint = sc_biguint<W>;

template<int W, int I, sc_q_mode Q, sc_o_mode O>
using ap_fixed = sc_fixed<W,I,Q,O>;

template<int W, int I, sc_q_mode Q, sc_o_mode O>
using ap_ufixed = sc_ufixed<W,I,Q,O>;

#ifndef AP_TRN
#define AP_TRN SC_TRN
#endif
#ifndef AP_RND
#define AP_RND SC_RND
#endif
#ifndef AP_TRN_ZERO
#define AP_TRN_ZERO SC_TRN_ZERO
#endif
#ifndef AP_RND_ZERO
#define AP_RND_ZERO SC_RND_ZERO
#endif
#ifndef AP_RND_INF
#define AP_RND_INF SC_RND_INF
#endif
#ifndef AP_RND_MIN_INF
#define AP_RND_MIN_INF SC_RND_MIN_INF
#endif
#ifndef AP_RND_CONV
#define AP_RND_CONV SC_RND_CONV
#endif
#ifndef AP_WRAP
#define AP_WRAP SC_WRAP
#endif
#ifndef AP_SAT
#define AP_SAT SC_SAT
#endif
#ifndef AP_SAT_ZERO
#define AP_SAT_ZERO SC_SAT_ZERO
#endif
#ifndef AP_SAT_SYM
#define AP_SAT_SYM SC_SAT_SYM
#endif
#ifndef AP_WRSC_SM
#define AP_WRSC_SM SC_WRSC_SM 
#endif
#ifndef AP_BIN
#define AP_BIN  SC_BIN
#endif
#ifndef AP_OCT
#define AP_OCT  SC_OCT
#endif
#ifndef AP_DEC
#define AP_DEC SC_DEC
#endif
#ifndef AP_HEX
#define AP_HEX SC_HEX
#endif

#endif
