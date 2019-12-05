//
// Created with the ESP Memory Generator
//
// Copyright (c) 2011-2019 Columbia University, System Level Design Group
// SPDX-License-Identifier: Apache-2.0
//
// @author Paolo Mantovani <paolo@cs.columbia.edu>
//

#ifndef __W6_DATA2304_1W8R_HPP__
#define __W6_DATA2304_1W8R_HPP__
#include "w6_data2304_1w8r.h"
template<class T, unsigned S, typename ioConfig=CYN::PIN>
class w6_data2304_1w8r_t : public sc_module
{

  HLS_INLINE_MODULE;
public:
  w6_data2304_1w8r_t(const sc_module_name& name = sc_gen_unique_name("w6_data2304_1w8r"))
  : sc_module(name)
  , clk("clk")
  , port1("port1")
  , port2("port2")
  , port3("port3")
  , port4("port4")
  , port5("port5")
  , port6("port6")
  , port7("port7")
  , port8("port8")
  , port9("port9")
  {
    m_m0.clk_rst(clk);
    port1(m_m0.if1);
    port2(m_m0.if2);
    port3(m_m0.if3);
    port4(m_m0.if4);
    port5(m_m0.if5);
    port6(m_m0.if6);
    port7(m_m0.if7);
    port8(m_m0.if8);
    port9(m_m0.if9);
  }

  sc_in<bool> clk;

  w6_data2304_1w8r::wrapper<ioConfig> m_m0;

  typedef w6_data2304_1w8r::port_1<ioConfig, T[1][S]> Port1_t;
  typedef w6_data2304_1w8r::port_2<ioConfig, T[1][S]> Port2_t;
  typedef w6_data2304_1w8r::port_3<ioConfig, T[1][S]> Port3_t;
  typedef w6_data2304_1w8r::port_4<ioConfig, T[1][S]> Port4_t;
  typedef w6_data2304_1w8r::port_5<ioConfig, T[1][S]> Port5_t;
  typedef w6_data2304_1w8r::port_6<ioConfig, T[1][S]> Port6_t;
  typedef w6_data2304_1w8r::port_7<ioConfig, T[1][S]> Port7_t;
  typedef w6_data2304_1w8r::port_8<ioConfig, T[1][S]> Port8_t;
  typedef w6_data2304_1w8r::port_9<ioConfig, T[1][S]> Port9_t;

  Port1_t port1;
  Port2_t port2;
  Port3_t port3;
  Port4_t port4;
  Port5_t port5;
  Port6_t port6;
  Port7_t port7;
  Port8_t port8;
  Port9_t port9;
};
#endif
