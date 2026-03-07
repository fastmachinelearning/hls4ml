// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.

// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

/*
 * This file contains the definition of the data types for AXI streaming.
 * ap_axi_s is a signed interpretation of the AXI stream
 * ap_axi_u is an unsigned interpretation of the AXI stream
 */

#ifndef __AP__AXI_SDATA__
#define __AP__AXI_SDATA__

#include "ap_int.h"
#include "hls_stream.h"
#include <cassert>
#include <climits>
#include <cstdint>
#include <type_traits>
//#include "ap_fixed.h"
template <int _AP_W, int _AP_I, ap_q_mode _AP_Q, ap_o_mode _AP_O, int _AP_N>
struct ap_fixed;
template <int _AP_W, int _AP_I, ap_q_mode _AP_Q, ap_o_mode _AP_O, int _AP_N>
struct ap_ufixed;

namespace hls {

template <typename T> constexpr std::size_t bitwidth = sizeof(T) * CHAR_BIT;
template <> constexpr std::size_t bitwidth<void> = 1 * CHAR_BIT;

template <std::size_t W> constexpr std::size_t bitwidth<ap_int<W>> = W;
template <std::size_t W> constexpr std::size_t bitwidth<ap_uint<W>> = W;
template <int _AP_W, int _AP_I, ap_q_mode _AP_Q, ap_o_mode _AP_O, int _AP_N>
constexpr std::size_t bitwidth<ap_fixed<_AP_W, _AP_I, _AP_Q, _AP_O, _AP_N>> =
    _AP_W;
template <int _AP_W, int _AP_I, ap_q_mode _AP_Q, ap_o_mode _AP_O, int _AP_N>
constexpr std::size_t bitwidth<ap_ufixed<_AP_W, _AP_I, _AP_Q, _AP_O, _AP_N>> =
    _AP_W;

template <typename T>
constexpr std::size_t bytewidth = (bitwidth<T> + CHAR_BIT - 1) / CHAR_BIT;
template <> constexpr std::size_t bytewidth<void> = 1;

struct axis_disabled_signal {};

// Enablement for axis signals
#define AXIS_ENABLE_DATA 0b00000001
#define AXIS_ENABLE_DEST 0b00000010
#define AXIS_ENABLE_ID 0b00000100
#define AXIS_ENABLE_KEEP 0b00001000
#define AXIS_ENABLE_LAST 0b00010000
#define AXIS_ENABLE_STRB 0b00100000
#define AXIS_ENABLE_USER 0b01000000

// clang-format off
// Disablement mask for DATA axis signals
#define AXIS_DISABLE_DATA (0b11111111 ^ AXIS_ENABLE_DATA) & \
                          (0b11111111 ^ AXIS_ENABLE_KEEP) & \
                          (0b11111111 ^ AXIS_ENABLE_STRB)

// Enablement/disablement of all axis signals
#define AXIS_ENABLE_ALL  0b01111111
#define AXIS_DISABLE_ALL 0b00000000

// Struct: axis - struct that has one or more member 'signals'
//   Signals: DATA, DEST, ID, KEEP, LAST, STRB, USER
//   All signals are optional:
//     LAST is enabled by default
//     DEST, ID, & USER are disabled by default
//     DATA, KEEP, & STRB are enabled by default for non-void DATA type
//   Template parameters:
//     T                : type of the DATA signal
//     WUser            : size of the USER signal, if zero signal will be disabled
//     WId              : size of the ID signal, if zero signal will be disabled
//     WDest            : size of the DEST signal, if zero signal will be disabled
//     EnableSignals    : bit field to enable signals, see AXIS_ENABLE_*
//     StrictEnablement : when true check that EnableSignals matches other parameters
// clang-format on
template <typename T, std::size_t WUser = 0, std::size_t WId = 0,
          std::size_t WDest = 0,
          uint8_t EnableSignals =
              (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB),
          bool StrictEnablement = false>
struct axis {
  static_assert((EnableSignals & 0b10000000) == 0,
                "Template parameter 'EnableSignals' is invalid only "
                "low 7 bits can be set!");
  friend class stream<
      axis<T, WUser, WId, WDest, EnableSignals, StrictEnablement>>;

  static constexpr bool has_data = !std::is_void<T>::value;
  static constexpr bool has_user = WUser > 0;
  static constexpr bool has_id = WId > 0;
  static constexpr bool has_dest = WDest > 0;
  static constexpr bool has_keep = EnableSignals & AXIS_ENABLE_KEEP;
  static constexpr bool has_strb = EnableSignals & AXIS_ENABLE_STRB;
  static constexpr bool has_last = EnableSignals & AXIS_ENABLE_LAST;

  static constexpr std::size_t width_user = has_user ? WUser : 1;
  static constexpr std::size_t width_id = has_id ? WId : 1;
  static constexpr std::size_t width_dest = has_dest ? WDest : 1;
  static constexpr std::size_t width_keep = bytewidth<T>;
  static constexpr std::size_t width_strb = bytewidth<T>;
  static constexpr std::size_t width_last = 1;

  static_assert(has_data || has_user || has_id || has_dest || has_keep ||
                    has_strb || has_last,
                "No axis signals are enabled");

  static_assert(StrictEnablement
                    ? has_data == (bool)(EnableSignals & AXIS_ENABLE_DATA)
                    : true,
                "Found mismatched enablement for DATA signal");
  static_assert(StrictEnablement
                    ? has_user == (bool)(EnableSignals & AXIS_ENABLE_USER)
                    : true,
                "Found mismatched enablement for USER signal");
  static_assert(StrictEnablement
                    ? has_id == (bool)(EnableSignals & AXIS_ENABLE_ID)
                    : true,
                "Found mismatched enablement for ID signal");
  static_assert(StrictEnablement
                    ? has_dest == (bool)(EnableSignals & AXIS_ENABLE_DEST)
                    : true,
                "Found mismatched enablement for DEST signal");

  typedef typename std::conditional<has_data, T, axis_disabled_signal>::type
      Type_data;
  Type_data data;

#ifdef AESL_SYN

  NODEBUG Type_data get_data() const {
#pragma HLS inline
    assert(has_data);
    return data;
  }
  NODEBUG void set_data(Type_data d) {
#pragma HLS inline
    assert(has_data);
    data = d;
  }

#define _AXIS_CHANNEL_API(CHAN_NAME)                                           \
  typedef                                                                      \
      typename std::conditional<has_##CHAN_NAME, ap_uint<width_##CHAN_NAME>,   \
                                axis_disabled_signal>::type Type_##CHAN_NAME;  \
  Type_##CHAN_NAME CHAN_NAME;                                                  \
  __attribute__((nodebug)) __attribute__((always_inline))                      \
      Type_##CHAN_NAME get_##CHAN_NAME() const {                               \
    assert(has_##CHAN_NAME);                                                   \
    return CHAN_NAME;                                                          \
  }                                                                            \
  __attribute__((nodebug)) __attribute__(                                      \
      (always_inline)) void set_##CHAN_NAME(Type_##CHAN_NAME value) {          \
    assert(has_##CHAN_NAME);                                                   \
    CHAN_NAME = value;                                                         \
  }

#else

  Type_data get_data() const {
    if (!has_data)
      throw std::runtime_error("CHAN_NAME is not enabled");
    return data;
  }
  void set_data(Type_data d) {
    if (!has_data)
      throw std::runtime_error("CHAN_NAME is not enabled");
    data = d;
  }

#define _AXIS_CHANNEL_API(CHAN_NAME)                                           \
  typedef                                                                      \
      typename std::conditional<has_##CHAN_NAME, ap_uint<width_##CHAN_NAME>,   \
                                axis_disabled_signal>::type Type_##CHAN_NAME;  \
  Type_##CHAN_NAME CHAN_NAME;                                                  \
  Type_##CHAN_NAME get_##CHAN_NAME() const {                                   \
    if (!has_##CHAN_NAME)                                                      \
      throw std::runtime_error("CHAN_NAME is not enabled");                    \
    return CHAN_NAME;                                                          \
  }                                                                            \
  void set_##CHAN_NAME(Type_##CHAN_NAME value) {                               \
    if (!has_##CHAN_NAME)                                                      \
      throw std::runtime_error("CHAN_NAME is not enabled");                    \
    CHAN_NAME = value;                                                         \
  }

#endif

  _AXIS_CHANNEL_API(keep)
  _AXIS_CHANNEL_API(strb)
  _AXIS_CHANNEL_API(user)
  _AXIS_CHANNEL_API(last)
  _AXIS_CHANNEL_API(id)
  _AXIS_CHANNEL_API(dest)
#undef _AXIS_CHANNEL_API

// For original `qdma_axis`
#ifdef AESL_SYN
  NODEBUG
#endif
  void keep_all() {
#pragma HLS inline
#ifdef AESL_SYN
    assert(has_keep);
#else
    if (!has_data)
      throw std::runtime_error("CHAN_NAME is not enabled");
#endif
    ap_uint<width_keep> k = 0;
    keep = ~k;
  }

private:
#ifdef AESL_SYN
#define _AXIS_CHANNEL_INTERNAL_API(CHAN_NAME)                                  \
  __attribute__((nodebug)) __attribute__((always_inline))                      \
      Type_##CHAN_NAME *get_##CHAN_NAME##_ptr() {                              \
    return (!has_##CHAN_NAME) ? nullptr : &CHAN_NAME;                          \
  }

  _AXIS_CHANNEL_INTERNAL_API(data)
  _AXIS_CHANNEL_INTERNAL_API(keep)
  _AXIS_CHANNEL_INTERNAL_API(strb)
  _AXIS_CHANNEL_INTERNAL_API(user)
  _AXIS_CHANNEL_INTERNAL_API(last)
  _AXIS_CHANNEL_INTERNAL_API(id)
  _AXIS_CHANNEL_INTERNAL_API(dest)
#undef _AXIS_CHANNEL_INTERNAL_API
#endif
};

// clang-format off
// Struct: axis_data (alternative to axis)
//   DATA signal always enabled
//   All other signals are optional, disabled by default
// Example usage:
//   hls::axis_data<int, AXIS_ENABLE_LAST> A; // DATA and LAST signals only
//   hls::axis_data<int, AXIS_ENABLE_LAST | AXIS_ENABLE_USER, 32> B; // DATA, LAST, and USER signals only (USER width is 32)
//   hls::axis_data<int, AXIS_ENABLE_ALL, 32, 8, 8> C; // All signals enabled
//   hls::axis_data<int, AXIS_ENABLE_ALL> D; // All signals enabled, this throw an exception due to zero size for WUser/WId/WDest
// clang-format on
template <typename TData, uint8_t EnableSignals = AXIS_ENABLE_DATA,
          std::size_t WUser = 0, std::size_t WId = 0, std::size_t WDest = 0,
          bool StrictEnablement = true>
using axis_data = axis<TData, WUser, WId, WDest,
                       (EnableSignals | AXIS_ENABLE_DATA), StrictEnablement>;

// Struct: axis_user (alternative to axis)
//   USER signal always enabled
//   DATA signal always disabled
//   All other signals are optional, disabled by default
// Example usage:
//   hls::axis_user<32> C; // USER signal only
//   hls::axis_user<32, AXIS_ENABLE_LAST> D; // USER and LAST signals only
template <std::size_t WUser, uint8_t EnableSignals = AXIS_ENABLE_USER,
          std::size_t WId = 0, std::size_t WDest = 0,
          bool StrictEnablement = true>
using axis_user = axis<void, WUser, WId, WDest,
                       (EnableSignals & AXIS_DISABLE_DATA), StrictEnablement>;

} // namespace hls

template <std::size_t WData, std::size_t WUser, std::size_t WId,
          std::size_t WDest,
          uint8_t EnableSignals =
              (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB),
          bool StrictEnablement = false>
using ap_axis = hls::axis<ap_int<WData>, WUser, WId, WDest, EnableSignals,
                          StrictEnablement>;

template <std::size_t WData, std::size_t WUser, std::size_t WId,
          std::size_t WDest,
          uint8_t EnableSignals =
              (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB),
          bool StrictEnablement = false>
using ap_axiu = hls::axis<ap_uint<WData>, WUser, WId, WDest, EnableSignals,
                          StrictEnablement>;

// original usage: qdma_axis<WData, 0, 0, 0>, and TSTRB is omitted.
template <std::size_t WData, std::size_t WUser, std::size_t WId,
          std::size_t WDest>
using qdma_axis = hls::axis<ap_uint<WData>, WUser, WId, WDest,
                            AXIS_ENABLE_ALL ^ AXIS_ENABLE_STRB, false>;

#ifdef AESL_SYN
#if ((__clang_major__ != 3) || (__clang_minor__ != 1))
namespace hls {

template <typename T, std::size_t WUser, std::size_t WId, std::size_t WDest,
          uint8_t EnableSignals, bool StrictEnablement>
class stream<axis<T, WUser, WId, WDest, EnableSignals, StrictEnablement>>
    final {
  typedef axis<T, WUser, WId, WDest, EnableSignals, StrictEnablement>
      __STREAM_T__;

public:
  /// Constructors
  INLINE NODEBUG stream() {}

  INLINE NODEBUG stream(const char *name) { (void)name; }

  /// Make copy constructor and assignment operator private
private:
  INLINE NODEBUG stream(const stream<__STREAM_T__> &chn) : V(chn.V) {}

public:
  /// Overload >> and << operators to implement read() and write()
  INLINE NODEBUG void operator>>(__STREAM_T__ &rdata) { read(rdata); }

  INLINE NODEBUG void operator<<(const __STREAM_T__ &wdata) { write(wdata); }

  /// empty & full
  NODEBUG bool empty() {
#pragma HLS inline
    bool tmp = __fpga_axis_valid(
        V.get_data_ptr(), V.get_keep_ptr(), V.get_strb_ptr(), V.get_user_ptr(),
        V.get_last_ptr(), V.get_id_ptr(), V.get_dest_ptr());
    return !tmp;
  }

  NODEBUG bool full() {
#pragma HLS inline
    bool tmp = __fpga_axis_ready(
        V.get_data_ptr(), V.get_keep_ptr(), V.get_strb_ptr(), V.get_user_ptr(),
        V.get_last_ptr(), V.get_id_ptr(), V.get_dest_ptr());
    return !tmp;
  }

  /// Blocking read
  NODEBUG void read(__STREAM_T__ &dout) {
#pragma HLS inline
    __STREAM_T__ tmp;
    __fpga_axis_pop(V.get_data_ptr(), V.get_keep_ptr(), V.get_strb_ptr(),
                    V.get_user_ptr(), V.get_last_ptr(), V.get_id_ptr(),
                    V.get_dest_ptr(), tmp.get_data_ptr(), tmp.get_keep_ptr(),
                    tmp.get_strb_ptr(), tmp.get_user_ptr(), tmp.get_last_ptr(),
                    tmp.get_id_ptr(), tmp.get_dest_ptr());
    dout = tmp;
  }

  NODEBUG __STREAM_T__ read() {
#pragma HLS inline
    __STREAM_T__ tmp;
    __fpga_axis_pop(V.get_data_ptr(), V.get_keep_ptr(), V.get_strb_ptr(),
                    V.get_user_ptr(), V.get_last_ptr(), V.get_id_ptr(),
                    V.get_dest_ptr(), tmp.get_data_ptr(), tmp.get_keep_ptr(),
                    tmp.get_strb_ptr(), tmp.get_user_ptr(), tmp.get_last_ptr(),
                    tmp.get_id_ptr(), tmp.get_dest_ptr());
    return tmp;
  }

  /// Blocking write
  NODEBUG void write(const __STREAM_T__ &din) {
#pragma HLS inline
    __STREAM_T__ tmp = din;
    __fpga_axis_push(V.get_data_ptr(), V.get_keep_ptr(), V.get_strb_ptr(),
                     V.get_user_ptr(), V.get_last_ptr(), V.get_id_ptr(),
                     V.get_dest_ptr(), tmp.get_data_ptr(), tmp.get_keep_ptr(),
                     tmp.get_strb_ptr(), tmp.get_user_ptr(), tmp.get_last_ptr(),
                     tmp.get_id_ptr(), tmp.get_dest_ptr());
  }

  /// Non-Blocking read
  NODEBUG bool read_nb(__STREAM_T__ &dout) {
#pragma HLS inline
    __STREAM_T__ tmp;
    if (__fpga_axis_nb_pop(V.get_data_ptr(), V.get_keep_ptr(), V.get_strb_ptr(),
                           V.get_user_ptr(), V.get_last_ptr(), V.get_id_ptr(),
                           V.get_dest_ptr(), tmp.get_data_ptr(),
                           tmp.get_keep_ptr(), tmp.get_strb_ptr(),
                           tmp.get_user_ptr(), tmp.get_last_ptr(),
                           tmp.get_id_ptr(), tmp.get_dest_ptr())) {
      dout = tmp;
      return true;
    } else {
      return false;
    }
  }

  /// Non-Blocking write
  NODEBUG bool write_nb(const __STREAM_T__ &in) {
#pragma HLS inline
    __STREAM_T__ tmp = in;
    bool full_n = __fpga_axis_nb_push(
        V.get_data_ptr(), V.get_keep_ptr(), V.get_strb_ptr(), V.get_user_ptr(),
        V.get_last_ptr(), V.get_id_ptr(), V.get_dest_ptr(), tmp.get_data_ptr(),
        tmp.get_keep_ptr(), tmp.get_strb_ptr(), tmp.get_user_ptr(),
        tmp.get_last_ptr(), tmp.get_id_ptr(), tmp.get_dest_ptr());
    return full_n;
  }

private:
  __STREAM_T__ V NO_CTOR;
};

} // namespace hls
#endif
#endif
#endif
