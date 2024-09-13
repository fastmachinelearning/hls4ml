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

#ifndef __AP_PRIVATE_H__
#define __AP_PRIVATE_H__

// common macros and type declarations are now defined in ap_common.h, and
// ap_private becomes part of it.
#ifndef __AP_COMMON_H__
#error "etc/ap_private.h cannot be included directly."
#endif

// forward declarations
//template <int _AP_W, bool _AP_S, bool _AP_C = _AP_W <= 64>
//class ap_private; // moved to ap_common.h
template <int _AP_W, bool _AP_S>
struct _private_range_ref;
template <int _AP_W, bool _AP_S>
struct _private_bit_ref;

// TODO clean up this part.
#ifndef LLVM_SUPPORT_MATHEXTRAS_H
#define LLVM_SUPPORT_MATHEXTRAS_H

#ifdef _MSC_VER
#if _MSC_VER <= 1500
typedef __int8 int8_t;
typedef unsigned __int8 uint8_t;
typedef __int16 int16_t;
typedef unsigned __int16 uint16_t;
typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
#else
#include <stdint.h>
#endif
#else
#include <stdint.h>
#endif

#ifndef INLINE
#define INLINE inline
// Enable to debug ap_int/ap_fixed
// #define INLINE  __attribute__((weak))
#endif

// NOTE: The following support functions use the _32/_64 extensions instead of
// type overloading so that signed and unsigned integers can be used without
// ambiguity.
namespace AESL_std {
template <class DataType>
DataType INLINE min(DataType a, DataType b) {
  return (a >= b) ? b : a;
}

template <class DataType>
DataType INLINE max(DataType a, DataType b) {
  return (a >= b) ? a : b;
}
} // namespace AESL_std

// TODO clean up included headers.
#include <math.h>
#include <stdio.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

namespace ap_private_ops {
/// Hi_32 - This function returns the high 32 bits of a 64 bit value.
static INLINE uint32_t Hi_32(uint64_t Value) {
  return static_cast<uint32_t>(Value >> 32);
}

/// Lo_32 - This function returns the low 32 bits of a 64 bit value.
static INLINE uint32_t Lo_32(uint64_t Value) {
  return static_cast<uint32_t>(Value);
}

template <int _AP_W>
INLINE bool isNegative(const ap_private<_AP_W, false>& a) {
  return false;
}

template <int _AP_W>
INLINE bool isNegative(const ap_private<_AP_W, true>& a) {
  enum {
    APINT_BITS_PER_WORD = 64,
    _AP_N = (_AP_W + APINT_BITS_PER_WORD - 1) / APINT_BITS_PER_WORD
  };
  static const uint64_t sign_mask = 1ULL << ((_AP_W - 1) % APINT_BITS_PER_WORD);
  return (sign_mask & a.get_pVal(_AP_N - 1)) != 0;
}

/// CountLeadingZeros_32 - this function performs the platform optimal form of
/// counting the number of zeros from the most significant bit to the first one
/// bit.  Ex. CountLeadingZeros_32(0x00F000FF) == 8.
/// Returns 32 if the word is zero.
static INLINE unsigned CountLeadingZeros_32(uint32_t Value) {
  unsigned Count; // result
#if __GNUC__ >= 4
// PowerPC is defined for __builtin_clz(0)
#if !defined(__ppc__) && !defined(__ppc64__)
  if (Value == 0) return 32;
#endif
  Count = __builtin_clz(Value);
#else
  if (Value == 0) return 32;
  Count = 0;
  // bisecton method for count leading zeros
  for (unsigned Shift = 32 >> 1; Shift; Shift >>= 1) {
    uint32_t Tmp = (Value) >> (Shift);
    if (Tmp) {
      Value = Tmp;
    } else {
      Count |= Shift;
    }
  }
#endif
  return Count;
}

/// CountLeadingZeros_64 - This function performs the platform optimal form
/// of counting the number of zeros from the most significant bit to the first
/// one bit (64 bit edition.)
/// Returns 64 if the word is zero.
static INLINE unsigned CountLeadingZeros_64(uint64_t Value) {
  unsigned Count; // result
#if __GNUC__ >= 4
// PowerPC is defined for __builtin_clzll(0)
#if !defined(__ppc__) && !defined(__ppc64__)
  if (!Value) return 64;
#endif
  Count = __builtin_clzll(Value);
#else
  if (sizeof(long) == sizeof(int64_t)) {
    if (!Value) return 64;
    Count = 0;
    // bisecton method for count leading zeros
    for (unsigned Shift = 64 >> 1; Shift; Shift >>= 1) {
      uint64_t Tmp = (Value) >> (Shift);
      if (Tmp) {
        Value = Tmp;
      } else {
        Count |= Shift;
      }
    }
  } else {
    // get hi portion
    uint32_t Hi = Hi_32(Value);

    // if some bits in hi portion
    if (Hi) {
      // leading zeros in hi portion plus all bits in lo portion
      Count = CountLeadingZeros_32(Hi);
    } else {
      // get lo portion
      uint32_t Lo = Lo_32(Value);
      // same as 32 bit value
      Count = CountLeadingZeros_32(Lo) + 32;
    }
  }
#endif
  return Count;
}

/// CountTrailingZeros_64 - This function performs the platform optimal form
/// of counting the number of zeros from the least significant bit to the first
/// one bit (64 bit edition.)
/// Returns 64 if the word is zero.
static INLINE unsigned CountTrailingZeros_64(uint64_t Value) {
#if __GNUC__ >= 4
  return (Value != 0) ? __builtin_ctzll(Value) : 64;
#else
  static const unsigned Mod67Position[] = {
      64, 0,  1,  39, 2,  15, 40, 23, 3,  12, 16, 59, 41, 19, 24, 54, 4,
      64, 13, 10, 17, 62, 60, 28, 42, 30, 20, 51, 25, 44, 55, 47, 5,  32,
      65, 38, 14, 22, 11, 58, 18, 53, 63, 9,  61, 27, 29, 50, 43, 46, 31,
      37, 21, 57, 52, 8,  26, 49, 45, 36, 56, 7,  48, 35, 6,  34, 33, 0};
  return Mod67Position[(uint64_t)(-(int64_t)Value & (int64_t)Value) % 67];
#endif
}

/// CountPopulation_64 - this function counts the number of set bits in a value,
/// (64 bit edition.)
static INLINE unsigned CountPopulation_64(uint64_t Value) {
#if __GNUC__ >= 4
  return __builtin_popcountll(Value);
#else
  uint64_t v = Value - (((Value) >> 1) & 0x5555555555555555ULL);
  v = (v & 0x3333333333333333ULL) + (((v) >> 2) & 0x3333333333333333ULL);
  v = (v + ((v) >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
  return unsigned((uint64_t)(v * 0x0101010101010101ULL) >> 56);
#endif
}

static INLINE uint32_t countLeadingOnes_64(uint64_t __V, uint32_t skip) {
  uint32_t Count = 0;
  if (skip) (__V) <<= (skip);
  while (__V && (__V & (1ULL << 63))) {
    Count++;
    (__V) <<= 1;
  }
  return Count;
}

static INLINE std::string oct2Bin(char oct) {
  switch (oct) {
    case '\0': {
      return "";
    }
    case '.': {
      return ".";
    }
    case '0': {
      return "000";
    }
    case '1': {
      return "001";
    }
    case '2': {
      return "010";
    }
    case '3': {
      return "011";
    }
    case '4': {
      return "100";
    }
    case '5': {
      return "101";
    }
    case '6': {
      return "110";
    }
    case '7': {
      return "111";
    }
  }
  assert(0 && "Invalid character in digit string");
  return "";
}

static INLINE std::string hex2Bin(char hex) {
  switch (hex) {
    case '\0': {
      return "";
    }
    case '.': {
      return ".";
    }
    case '0': {
      return "0000";
    }
    case '1': {
      return "0001";
    }
    case '2': {
      return "0010";
    }
    case '3': {
      return "0011";
    }
    case '4': {
      return "0100";
    }
    case '5': {
      return "0101";
    }
    case '6': {
      return "0110";
    }
    case '7': {
      return "0111";
    }
    case '8': {
      return "1000";
    }
    case '9': {
      return "1001";
    }
    case 'A':
    case 'a': {
      return "1010";
    }
    case 'B':
    case 'b': {
      return "1011";
    }
    case 'C':
    case 'c': {
      return "1100";
    }
    case 'D':
    case 'd': {
      return "1101";
    }
    case 'E':
    case 'e': {
      return "1110";
    }
    case 'F':
    case 'f': {
      return "1111";
    }
  }
  assert(0 && "Invalid character in digit string");
  return "";
}

static INLINE uint32_t decode_digit(char cdigit, int radix) {
  uint32_t digit = 0;
  if (radix == 16) {
#define isxdigit(c)                                            \
  (((c) >= '0' && (c) <= '9') || ((c) >= 'a' && (c) <= 'f') || \
   ((c) >= 'A' && (c) <= 'F'))
#define isdigit(c) ((c) >= '0' && (c) <= '9')
    if (!isxdigit(cdigit)) assert(0 && "Invalid hex digit in string");
    if (isdigit(cdigit))
      digit = cdigit - '0';
    else if (cdigit >= 'a')
      digit = cdigit - 'a' + 10;
    else if (cdigit >= 'A')
      digit = cdigit - 'A' + 10;
    else
      assert(0 && "huh? we shouldn't get here");
  } else if (isdigit(cdigit)) {
    digit = cdigit - '0';
  } else {
    assert(0 && "Invalid character in digit string");
  }
#undef isxdigit
#undef isdigit
  return digit;
}

// Determine the radix of "val".
static INLINE std::string parseString(const std::string& input, unsigned char& radix) {
  size_t len = input.length();
  if (len == 0) {
    if (radix == 0) radix = 10;
    return input;
  }

  size_t startPos = 0;
  // Trim whitespace
  while (input[startPos] == ' ' && startPos < len) startPos++;
  while (input[len - 1] == ' ' && startPos < len) len--;

  std::string val = input.substr(startPos, len - startPos);
  // std::cout << "val = " << val << "\n";
  len = val.length();
  startPos = 0;

  // If the length of the string is less than 2, then radix
  // is decimal and there is no exponent.
  if (len < 2) {
    if (radix == 0) radix = 10;
    return val;
  }

  bool isNegative = false;
  std::string ans;

  // First check to see if we start with a sign indicator
  if (val[0] == '-') {
    ans = "-";
    ++startPos;
    isNegative = true;
  } else if (val[0] == '+')
    ++startPos;

  if (len - startPos < 2) {
    if (radix == 0) radix = 10;
    return val;
  }

  if (val.substr(startPos, 2) == "0x" || val.substr(startPos, 2) == "0X") {
    // If we start with "0x", then the radix is hex.
    radix = 16;
    startPos += 2;
  } else if (val.substr(startPos, 2) == "0b" ||
             val.substr(startPos, 2) == "0B") {
    // If we start with "0b", then the radix is binary.
    radix = 2;
    startPos += 2;
  } else if (val.substr(startPos, 2) == "0o" ||
             val.substr(startPos, 2) == "0O") {
    // If we start with "0o", then the radix is octal.
    radix = 8;
    startPos += 2;
  } else if (radix == 0) {
    radix = 10;
  }

  int exp = 0;
  if (radix == 10) {
    // If radix is decimal, then see if there is an
    // exponent indicator.
    size_t expPos = val.find('e');
    bool has_exponent = true;
    if (expPos == std::string::npos) expPos = val.find('E');
    if (expPos == std::string::npos) {
      // No exponent indicator, so the mantissa goes to the end.
      expPos = len;
      has_exponent = false;
    }
    // std::cout << "startPos = " << startPos << " " << expPos << "\n";

    ans += val.substr(startPos, expPos - startPos);
    if (has_exponent) {
      // Parse the exponent.
      std::istringstream iss(val.substr(expPos + 1, len - expPos - 1));
      iss >> exp;
    }
  } else {
    // Check for a binary exponent indicator.
    size_t expPos = val.find('p');
    bool has_exponent = true;
    if (expPos == std::string::npos) expPos = val.find('P');
    if (expPos == std::string::npos) {
      // No exponent indicator, so the mantissa goes to the end.
      expPos = len;
      has_exponent = false;
    }

    // std::cout << "startPos = " << startPos << " " << expPos << "\n";

    assert(startPos <= expPos);
    // Convert to binary as we go.
    for (size_t i = startPos; i < expPos; ++i) {
      if (radix == 16) {
        ans += hex2Bin(val[i]);
      } else if (radix == 8) {
        ans += oct2Bin(val[i]);
      } else { // radix == 2
        ans += val[i];
      }
    }
    // End in binary
    radix = 2;
    if (has_exponent) {
      // Parse the exponent.
      std::istringstream iss(val.substr(expPos + 1, len - expPos - 1));
      iss >> exp;
    }
  }
  if (exp == 0) return ans;

  size_t decPos = ans.find('.');
  if (decPos == std::string::npos) decPos = ans.length();
  if ((int)decPos + exp >= (int)ans.length()) {
    int i = decPos;
    for (; i < (int)ans.length() - 1; ++i) ans[i] = ans[i + 1];
    for (; i < (int)ans.length(); ++i) ans[i] = '0';
    for (; i < (int)decPos + exp; ++i) ans += '0';
    return ans;
  } else if ((int)decPos + exp < (int)isNegative) {
    std::string dupAns = "0.";
    if (ans[0] == '-') dupAns = "-0.";
    for (int i = 0; i < isNegative - (int)decPos - exp; ++i) dupAns += '0';
    for (size_t i = isNegative; i < ans.length(); ++i)
      if (ans[i] != '.') dupAns += ans[i];
    return dupAns;
  }

  if (exp > 0)
    for (size_t i = decPos; i < decPos + exp; ++i) ans[i] = ans[i + 1];
  else {
    if (decPos == ans.length()) ans += ' ';
    for (int i = decPos; i > (int)decPos + exp; --i) ans[i] = ans[i - 1];
  }
  ans[decPos + exp] = '.';
  return ans;
}

/// sub_1 - This function subtracts a single "digit" (64-bit word), y, from
/// the multi-digit integer array, x[], propagating the borrowed 1 value until
/// no further borrowing is neeeded or it runs out of "digits" in x.  The result
/// is 1 if "borrowing" exhausted the digits in x, or 0 if x was not exhausted.
/// In other words, if y > x then this function returns 1, otherwise 0.
/// @returns the borrow out of the subtraction
static INLINE bool sub_1(uint64_t x[], uint32_t len, uint64_t y) {
  for (uint32_t i = 0; i < len; ++i) {
    uint64_t __X = x[i];
    x[i] -= y;
    if (y > __X)
      y = 1; // We have to "borrow 1" from next "digit"
    else {
      y = 0; // No need to borrow
      break; // Remaining digits are unchanged so exit early
    }
  }
  return (y != 0);
}

/// add_1 - This function adds a single "digit" integer, y, to the multiple
/// "digit" integer array,  x[]. x[] is modified to reflect the addition and
/// 1 is returned if there is a carry out, otherwise 0 is returned.
/// @returns the carry of the addition.
static INLINE bool add_1(uint64_t dest[], uint64_t x[], uint32_t len,
                         uint64_t y) {
  for (uint32_t i = 0; i < len; ++i) {
    dest[i] = y + x[i];
    if (dest[i] < y)
      y = 1; // Carry one to next digit.
    else {
      y = 0; // No need to carry so exit early
      break;
    }
  }
  return (y != 0);
}

/// add - This function adds the integer array x to the integer array Y and
/// places the result in dest.
/// @returns the carry out from the addition
/// @brief General addition of 64-bit integer arrays
static INLINE bool add(uint64_t* dest, const uint64_t* x, const uint64_t* y,
                       uint32_t destlen, uint32_t xlen, uint32_t ylen,
                       bool xsigned, bool ysigned) {
  bool carry = false;
  uint32_t len = AESL_std::min(xlen, ylen);
  uint32_t i;
  for (i = 0; i < len && i < destlen; ++i) {
    uint64_t limit =
        AESL_std::min(x[i], y[i]); // must come first in case dest == x
    dest[i] = x[i] + y[i] + carry;
    carry = dest[i] < limit || (carry && dest[i] == limit);
  }
  if (xlen > ylen) {
    const uint64_t yext = ysigned && int64_t(y[ylen - 1]) < 0 ? -1 : 0;
    for (i = ylen; i < xlen && i < destlen; i++) {
      uint64_t limit = AESL_std::min(x[i], yext);
      dest[i] = x[i] + yext + carry;
      carry = (dest[i] < limit) || (carry && dest[i] == limit);
    }
  } else if (ylen > xlen) {
    const uint64_t xext = xsigned && int64_t(x[xlen - 1]) < 0 ? -1 : 0;
    for (i = xlen; i < ylen && i < destlen; i++) {
      uint64_t limit = AESL_std::min(xext, y[i]);
      dest[i] = xext + y[i] + carry;
      carry = (dest[i] < limit) || (carry && dest[i] == limit);
    }
  }
  return carry;
}

/// @returns returns the borrow out.
/// @brief Generalized subtraction of 64-bit integer arrays.
static INLINE bool sub(uint64_t* dest, const uint64_t* x, const uint64_t* y,
                       uint32_t destlen, uint32_t xlen, uint32_t ylen,
                       bool xsigned, bool ysigned) {
  bool borrow = false;
  uint32_t i;
  uint32_t len = AESL_std::min(xlen, ylen);
  for (i = 0; i < len && i < destlen; ++i) {
    uint64_t x_tmp = borrow ? x[i] - 1 : x[i];
    borrow = y[i] > x_tmp || (borrow && x[i] == 0);
    dest[i] = x_tmp - y[i];
  }
  if (xlen > ylen) {
    const uint64_t yext = ysigned && int64_t(y[ylen - 1]) < 0 ? -1 : 0;
    for (i = ylen; i < xlen && i < destlen; i++) {
      uint64_t x_tmp = borrow ? x[i] - 1 : x[i];
      borrow = yext > x_tmp || (borrow && x[i] == 0);
      dest[i] = x_tmp - yext;
    }
  } else if (ylen > xlen) {
    const uint64_t xext = xsigned && int64_t(x[xlen - 1]) < 0 ? -1 : 0;
    for (i = xlen; i < ylen && i < destlen; i++) {
      uint64_t x_tmp = borrow ? xext - 1 : xext;
      borrow = y[i] > x_tmp || (borrow && xext == 0);
      dest[i] = x_tmp - y[i];
    }
  }
  return borrow;
}

/// Subtracts the RHS ap_private from this ap_private
/// @returns this, after subtraction
/// @brief Subtraction assignment operator.

/// Multiplies an integer array, x by a a uint64_t integer and places the result
/// into dest.
/// @returns the carry out of the multiplication.
/// @brief Multiply a multi-digit ap_private by a single digit (64-bit) integer.
static INLINE uint64_t mul_1(uint64_t dest[], const uint64_t x[], uint32_t len,
                             uint64_t y) {
  // Split y into high 32-bit part (hy)  and low 32-bit part (ly)
  uint64_t ly = y & 0xffffffffULL, hy = (y) >> 32;
  uint64_t carry = 0;
  static const uint64_t two_power_32 = 1ULL << 32;
  // For each digit of x.
  for (uint32_t i = 0; i < len; ++i) {
    // Split x into high and low words
    uint64_t lx = x[i] & 0xffffffffULL;
    uint64_t hx = (x[i]) >> 32;
    // hasCarry - A flag to indicate if there is a carry to the next digit.
    // hasCarry == 0, no carry
    // hasCarry == 1, has carry
    // hasCarry == 2, no carry and the calculation result == 0.
    uint8_t hasCarry = 0;
    dest[i] = carry + lx * ly;
    // Determine if the add above introduces carry.
    hasCarry = (dest[i] < carry) ? 1 : 0;
    carry = hx * ly + ((dest[i]) >> 32) + (hasCarry ? two_power_32 : 0);
    // The upper limit of carry can be (2^32 - 1)(2^32 - 1) +
    // (2^32 - 1) + 2^32 = 2^64.
    hasCarry = (!carry && hasCarry) ? 1 : (!carry ? 2 : 0);

    carry += (lx * hy) & 0xffffffffULL;
    dest[i] = ((carry) << 32) | (dest[i] & 0xffffffffULL);
    carry = (((!carry && hasCarry != 2) || hasCarry == 1) ? two_power_32 : 0) +
            ((carry) >> 32) + ((lx * hy) >> 32) + hx * hy;
  }
  return carry;
}

/// Multiplies integer array x by integer array y and stores the result into
/// the integer array dest. Note that dest's size must be >= xlen + ylen in
/// order to
/// do a full precision computation. If it is not, then only the low-order words
/// are returned.
/// @brief Generalized multiplicate of integer arrays.
static INLINE void mul(uint64_t dest[], const uint64_t x[], uint32_t xlen,
                       const uint64_t y[], uint32_t ylen, uint32_t destlen) {
  assert(xlen > 0);
  assert(ylen > 0);
  assert(destlen >= xlen + ylen);
  if (xlen < destlen) dest[xlen] = mul_1(dest, x, xlen, y[0]);
  for (uint32_t i = 1; i < ylen; ++i) {
    uint64_t ly = y[i] & 0xffffffffULL, hy = (y[i]) >> 32;
    uint64_t carry = 0, lx = 0, hx = 0;
    for (uint32_t j = 0; j < xlen; ++j) {
      lx = x[j] & 0xffffffffULL;
      hx = (x[j]) >> 32;
      // hasCarry - A flag to indicate if has carry.
      // hasCarry == 0, no carry
      // hasCarry == 1, has carry
      // hasCarry == 2, no carry and the calculation result == 0.
      uint8_t hasCarry = 0;
      uint64_t resul = carry + lx * ly;
      hasCarry = (resul < carry) ? 1 : 0;
      carry = (hasCarry ? (1ULL << 32) : 0) + hx * ly + ((resul) >> 32);
      hasCarry = (!carry && hasCarry) ? 1 : (!carry ? 2 : 0);
      carry += (lx * hy) & 0xffffffffULL;
      resul = ((carry) << 32) | (resul & 0xffffffffULL);
      if (i + j < destlen) dest[i + j] += resul;
      carry =
          (((!carry && hasCarry != 2) || hasCarry == 1) ? (1ULL << 32) : 0) +
          ((carry) >> 32) + (dest[i + j] < resul ? 1 : 0) + ((lx * hy) >> 32) +
          hx * hy;
    }
    if (i + xlen < destlen) dest[i + xlen] = carry;
  }
}

/// Implementation of Knuth's Algorithm D (Division of nonnegative integers)
/// from "Art of Computer Programming, Volume 2", section 4.3.1, p. 272. The
/// variables here have the same names as in the algorithm. Comments explain
/// the algorithm and any deviation from it.
static INLINE void KnuthDiv(uint32_t* u, uint32_t* v, uint32_t* q, uint32_t* r,
                            uint32_t m, uint32_t n) {
  assert(u && "Must provide dividend");
  assert(v && "Must provide divisor");
  assert(q && "Must provide quotient");
  assert(u != v && u != q && v != q && "Must us different memory");
  assert(n > 1 && "n must be > 1");

  // Knuth uses the value b as the base of the number system. In our case b
  // is 2^31 so we just set it to -1u.
  uint64_t b = uint64_t(1) << 32;

  // DEBUG(cerr << "KnuthDiv: m=" << m << " n=" << n << '\n');
  // DEBUG(cerr << "KnuthDiv: original:");
  // DEBUG(for (int i = m+n; i >=0; i--) cerr << " " << std::setbase(16) <<
  // u[i]);
  // DEBUG(cerr << " by");
  // DEBUG(for (int i = n; i >0; i--) cerr << " " << std::setbase(16) <<
  // v[i-1]);
  // DEBUG(cerr << '\n');
  // D1. [Normalize.] Set d = b / (v[n-1] + 1) and multiply all the digits of
  // u and v by d. Note that we have taken Knuth's advice here to use a power
  // of 2 value for d such that d * v[n-1] >= b/2 (b is the base). A power of
  // 2 allows us to shift instead of multiply and it is easy to determine the
  // shift amount from the leading zeros.  We are basically normalizing the u
  // and v so that its high bits are shifted to the top of v's range without
  // overflow. Note that this can require an extra word in u so that u must
  // be of length m+n+1.
  uint32_t shift = CountLeadingZeros_32(v[n - 1]);
  uint32_t v_carry = 0;
  uint32_t u_carry = 0;
  if (shift) {
    for (uint32_t i = 0; i < m + n; ++i) {
      uint32_t u_tmp = (u[i]) >> (32 - shift);
      u[i] = ((u[i]) << (shift)) | u_carry;
      u_carry = u_tmp;
    }
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t v_tmp = (v[i]) >> (32 - shift);
      v[i] = ((v[i]) << (shift)) | v_carry;
      v_carry = v_tmp;
    }
  }
  u[m + n] = u_carry;
  // DEBUG(cerr << "KnuthDiv:   normal:");
  // DEBUG(for (int i = m+n; i >=0; i--) cerr << " " << std::setbase(16) <<
  // u[i]);
  // DEBUG(cerr << " by");
  // DEBUG(for (int i = n; i >0; i--) cerr << " " << std::setbase(16) <<
  // v[i-1]);
  // DEBUG(cerr << '\n');

  // D2. [Initialize j.]  Set j to m. This is the loop counter over the places.
  int j = m;
  do {
    // DEBUG(cerr << "KnuthDiv: quotient digit #" << j << '\n');
    // D3. [Calculate q'.].
    //     Set qp = (u[j+n]*b + u[j+n-1]) / v[n-1]. (qp=qprime=q')
    //     Set rp = (u[j+n]*b + u[j+n-1]) % v[n-1]. (rp=rprime=r')
    // Now test if qp == b or qp*v[n-2] > b*rp + u[j+n-2]; if so, decrease
    // qp by 1, inrease rp by v[n-1], and repeat this test if rp < b. The test
    // on v[n-2] determines at high speed most of the cases in which the trial
    // value qp is one too large, and it eliminates all cases where qp is two
    // too large.
    uint64_t dividend = ((uint64_t(u[j + n]) << 32) + u[j + n - 1]);
    // DEBUG(cerr << "KnuthDiv: dividend == " << dividend << '\n');
    uint64_t qp = dividend / v[n - 1];
    uint64_t rp = dividend % v[n - 1];
    if (qp == b || qp * v[n - 2] > b * rp + u[j + n - 2]) {
      qp--;
      rp += v[n - 1];
      if (rp < b && (qp == b || qp * v[n - 2] > b * rp + u[j + n - 2])) qp--;
    }
    // DEBUG(cerr << "KnuthDiv: qp == " << qp << ", rp == " << rp << '\n');

    // D4. [Multiply and subtract.] Replace (u[j+n]u[j+n-1]...u[j]) with
    // (u[j+n]u[j+n-1]..u[j]) - qp * (v[n-1]...v[1]v[0]). This computation
    // consists of a simple multiplication by a one-place number, combined with
    // a subtraction.
    bool isNeg = false;
    for (uint32_t i = 0; i < n; ++i) {
      uint64_t u_tmp = uint64_t(u[j + i]) | ((uint64_t(u[j + i + 1])) << 32);
      uint64_t subtrahend = uint64_t(qp) * uint64_t(v[i]);
      bool borrow = subtrahend > u_tmp;
      /*DEBUG(cerr << "KnuthDiv: u_tmp == " << u_tmp
        << ", subtrahend == " << subtrahend
        << ", borrow = " << borrow << '\n');*/

      uint64_t result = u_tmp - subtrahend;
      uint32_t k = j + i;
      u[k++] = (uint32_t)(result & (b - 1)); // subtract low word
      u[k++] = (uint32_t)((result) >> 32);   // subtract high word
      while (borrow && k <= m + n) {         // deal with borrow to the left
        borrow = u[k] == 0;
        u[k]--;
        k++;
      }
      isNeg |= borrow;
      /*DEBUG(cerr << "KnuthDiv: u[j+i] == " << u[j+i] << ",  u[j+i+1] == " <<
        u[j+i+1] << '\n');*/
    }
    /*DEBUG(cerr << "KnuthDiv: after subtraction:");
      DEBUG(for (int i = m+n; i >=0; i--) cerr << " " << u[i]);
      DEBUG(cerr << '\n');*/
    // The digits (u[j+n]...u[j]) should be kept positive; if the result of
    // this step is actually negative, (u[j+n]...u[j]) should be left as the
    // true value plus b**(n+1), namely as the b's complement of
    // the true value, and a "borrow" to the left should be remembered.
    //
    if (isNeg) {
      bool carry = true; // true because b's complement is "complement + 1"
      for (uint32_t i = 0; i <= m + n; ++i) {
        u[i] = ~u[i] + carry; // b's complement
        carry = carry && u[i] == 0;
      }
    }
    /*DEBUG(cerr << "KnuthDiv: after complement:");
      DEBUG(for (int i = m+n; i >=0; i--) cerr << " " << u[i]);
      DEBUG(cerr << '\n');*/

    // D5. [Test remainder.] Set q[j] = qp. If the result of step D4 was
    // negative, go to step D6; otherwise go on to step D7.
    q[j] = (uint32_t)qp;
    if (isNeg) {
      // D6. [Add back]. The probability that this step is necessary is very
      // small, on the order of only 2/b. Make sure that test data accounts for
      // this possibility. Decrease q[j] by 1
      q[j]--;
      // and add (0v[n-1]...v[1]v[0]) to (u[j+n]u[j+n-1]...u[j+1]u[j]).
      // A carry will occur to the left of u[j+n], and it should be ignored
      // since it cancels with the borrow that occurred in D4.
      bool carry = false;
      for (uint32_t i = 0; i < n; i++) {
        uint32_t limit = AESL_std::min(u[j + i], v[i]);
        u[j + i] += v[i] + carry;
        carry = u[j + i] < limit || (carry && u[j + i] == limit);
      }
      u[j + n] += carry;
    }
    /*DEBUG(cerr << "KnuthDiv: after correction:");
      DEBUG(for (int i = m+n; i >=0; i--) cerr <<" " << u[i]);
      DEBUG(cerr << "\nKnuthDiv: digit result = " << q[j] << '\n');*/

    // D7. [Loop on j.]  Decrease j by one. Now if j >= 0, go back to D3.
  } while (--j >= 0);

  /*DEBUG(cerr << "KnuthDiv: quotient:");
    DEBUG(for (int i = m; i >=0; i--) cerr <<" " << q[i]);
    DEBUG(cerr << '\n');*/

  // D8. [Unnormalize]. Now q[...] is the desired quotient, and the desired
  // remainder may be obtained by dividing u[...] by d. If r is non-null we
  // compute the remainder (urem uses this).
  if (r) {
    // The value d is expressed by the "shift" value above since we avoided
    // multiplication by d by using a shift left. So, all we have to do is
    // shift right here. In order to mak
    if (shift) {
      uint32_t carry = 0;
      // DEBUG(cerr << "KnuthDiv: remainder:");
      for (int i = n - 1; i >= 0; i--) {
        r[i] = ((u[i]) >> (shift)) | carry;
        carry = (u[i]) << (32 - shift);
        // DEBUG(cerr << " " << r[i]);
      }
    } else {
      for (int i = n - 1; i >= 0; i--) {
        r[i] = u[i];
        // DEBUG(cerr << " " << r[i]);
      }
    }
    // DEBUG(cerr << '\n');
  }
  // DEBUG(cerr << std::setbase(10) << '\n');
}

template <int _AP_W, bool _AP_S>
void divide(const ap_private<_AP_W, _AP_S>& LHS, uint32_t lhsWords,
            const ap_private<_AP_W, _AP_S>& RHS, uint32_t rhsWords,
            ap_private<_AP_W, _AP_S>* Quotient,
            ap_private<_AP_W, _AP_S>* Remainder) {
  assert(lhsWords >= rhsWords && "Fractional result");
  enum { APINT_BITS_PER_WORD = 64 };
  // First, compose the values into an array of 32-bit words instead of
  // 64-bit words. This is a necessity of both the "short division" algorithm
  // and the the Knuth "classical algorithm" which requires there to be native
  // operations for +, -, and * on an m bit value with an m*2 bit result. We
  // can't use 64-bit operands here because we don't have native results of
  // 128-bits. Furthremore, casting the 64-bit values to 32-bit values won't
  // work on large-endian machines.
  uint64_t mask = ~0ull >> (sizeof(uint32_t) * 8);
  uint32_t n = rhsWords * 2;
  uint32_t m = (lhsWords * 2) - n;

  // Allocate space for the temporary values we need either on the stack, if
  // it will fit, or on the heap if it won't.
  uint32_t SPACE[128];
  uint32_t* __U = 0;
  uint32_t* __V = 0;
  uint32_t* __Q = 0;
  uint32_t* __R = 0;
  if ((Remainder ? 4 : 3) * n + 2 * m + 1 <= 128) {
    __U = &SPACE[0];
    __V = &SPACE[m + n + 1];
    __Q = &SPACE[(m + n + 1) + n];
    if (Remainder) __R = &SPACE[(m + n + 1) + n + (m + n)];
  } else {
    __U = new uint32_t[m + n + 1];
    __V = new uint32_t[n];
    __Q = new uint32_t[m + n];
    if (Remainder) __R = new uint32_t[n];
  }

  // Initialize the dividend
  memset(__U, 0, (m + n + 1) * sizeof(uint32_t));
  for (unsigned i = 0; i < lhsWords; ++i) {
    uint64_t tmp = LHS.get_pVal(i);
    __U[i * 2] = (uint32_t)(tmp & mask);
    __U[i * 2 + 1] = (tmp) >> (sizeof(uint32_t) * 8);
  }
  __U[m + n] = 0; // this extra word is for "spill" in the Knuth algorithm.

  // Initialize the divisor
  memset(__V, 0, (n) * sizeof(uint32_t));
  for (unsigned i = 0; i < rhsWords; ++i) {
    uint64_t tmp = RHS.get_pVal(i);
    __V[i * 2] = (uint32_t)(tmp & mask);
    __V[i * 2 + 1] = (tmp) >> (sizeof(uint32_t) * 8);
  }

  // initialize the quotient and remainder
  memset(__Q, 0, (m + n) * sizeof(uint32_t));
  if (Remainder) memset(__R, 0, n * sizeof(uint32_t));

  // Now, adjust m and n for the Knuth division. n is the number of words in
  // the divisor. m is the number of words by which the dividend exceeds the
  // divisor (i.e. m+n is the length of the dividend). These sizes must not
  // contain any zero words or the Knuth algorithm fails.
  for (unsigned i = n; i > 0 && __V[i - 1] == 0; i--) {
    n--;
    m++;
  }
  for (unsigned i = m + n; i > 0 && __U[i - 1] == 0; i--) m--;

  // If we're left with only a single word for the divisor, Knuth doesn't work
  // so we implement the short division algorithm here. This is much simpler
  // and faster because we are certain that we can divide a 64-bit quantity
  // by a 32-bit quantity at hardware speed and short division is simply a
  // series of such operations. This is just like doing short division but we
  // are using base 2^32 instead of base 10.
  assert(n != 0 && "Divide by zero?");
  if (n == 1) {
    uint32_t divisor = __V[0];
    uint32_t remainder = 0;
    for (int i = m + n - 1; i >= 0; i--) {
      uint64_t partial_dividend = (uint64_t(remainder)) << 32 | __U[i];
      if (partial_dividend == 0) {
        __Q[i] = 0;
        remainder = 0;
      } else if (partial_dividend < divisor) {
        __Q[i] = 0;
        remainder = (uint32_t)partial_dividend;
      } else if (partial_dividend == divisor) {
        __Q[i] = 1;
        remainder = 0;
      } else {
        __Q[i] = (uint32_t)(partial_dividend / divisor);
        remainder = (uint32_t)(partial_dividend - (__Q[i] * divisor));
      }
    }
    if (__R) __R[0] = remainder;
  } else {
    // Now we're ready to invoke the Knuth classical divide algorithm. In this
    // case n > 1.
    KnuthDiv(__U, __V, __Q, __R, m, n);
  }

  // If the caller wants the quotient
  if (Quotient) {
    // Set up the Quotient value's memory.
    if (Quotient->BitWidth != LHS.BitWidth) {
      if (Quotient->isSingleWord()) Quotient->set_VAL(0);
    } else
      Quotient->clear();

    // The quotient is in Q. Reconstitute the quotient into Quotient's low
    // order words.
    if (lhsWords == 1) {
      uint64_t tmp =
          uint64_t(__Q[0]) | ((uint64_t(__Q[1])) << (APINT_BITS_PER_WORD / 2));
      Quotient->set_VAL(tmp);
    } else {
      assert(!Quotient->isSingleWord() &&
             "Quotient ap_private not large enough");
      for (unsigned i = 0; i < lhsWords; ++i)
        Quotient->set_pVal(
            i, uint64_t(__Q[i * 2]) |
                   ((uint64_t(__Q[i * 2 + 1])) << (APINT_BITS_PER_WORD / 2)));
    }
    Quotient->clearUnusedBits();
  }

  // If the caller wants the remainder
  if (Remainder) {
    // Set up the Remainder value's memory.
    if (Remainder->BitWidth != RHS.BitWidth) {
      if (Remainder->isSingleWord()) Remainder->set_VAL(0);
    } else
      Remainder->clear();

    // The remainder is in R. Reconstitute the remainder into Remainder's low
    // order words.
    if (rhsWords == 1) {
      uint64_t tmp =
          uint64_t(__R[0]) | ((uint64_t(__R[1])) << (APINT_BITS_PER_WORD / 2));
      Remainder->set_VAL(tmp);
    } else {
      assert(!Remainder->isSingleWord() &&
             "Remainder ap_private not large enough");
      for (unsigned i = 0; i < rhsWords; ++i)
        Remainder->set_pVal(
            i, uint64_t(__R[i * 2]) |
                   ((uint64_t(__R[i * 2 + 1])) << (APINT_BITS_PER_WORD / 2)));
    }
    Remainder->clearUnusedBits();
  }

  // Clean up the memory we allocated.
  if (__U != &SPACE[0]) {
    delete[] __U;
    delete[] __V;
    delete[] __Q;
    delete[] __R;
  }
}

template <int _AP_W, bool _AP_S>
void divide(const ap_private<_AP_W, _AP_S>& LHS, uint32_t lhsWords,
            uint64_t RHS, ap_private<_AP_W, _AP_S>* Quotient,
            ap_private<_AP_W, _AP_S>* Remainder) {
  uint32_t rhsWords = 1;
  assert(lhsWords >= rhsWords && "Fractional result");
  enum { APINT_BITS_PER_WORD = 64 };
  // First, compose the values into an array of 32-bit words instead of
  // 64-bit words. This is a necessity of both the "short division" algorithm
  // and the the Knuth "classical algorithm" which requires there to be native
  // operations for +, -, and * on an m bit value with an m*2 bit result. We
  // can't use 64-bit operands here because we don't have native results of
  // 128-bits. Furthremore, casting the 64-bit values to 32-bit values won't
  // work on large-endian machines.
  uint64_t mask = ~0ull >> (sizeof(uint32_t) * 8);
  uint32_t n = 2;
  uint32_t m = (lhsWords * 2) - n;

  // Allocate space for the temporary values we need either on the stack, if
  // it will fit, or on the heap if it won't.
  uint32_t SPACE[128];
  uint32_t* __U = 0;
  uint32_t* __V = 0;
  uint32_t* __Q = 0;
  uint32_t* __R = 0;
  if ((Remainder ? 4 : 3) * n + 2 * m + 1 <= 128) {
    __U = &SPACE[0];
    __V = &SPACE[m + n + 1];
    __Q = &SPACE[(m + n + 1) + n];
    if (Remainder) __R = &SPACE[(m + n + 1) + n + (m + n)];
  } else {
    __U = new uint32_t[m + n + 1];
    __V = new uint32_t[n];
    __Q = new uint32_t[m + n];
    if (Remainder) __R = new uint32_t[n];
  }

  // Initialize the dividend
  memset(__U, 0, (m + n + 1) * sizeof(uint32_t));
  for (unsigned i = 0; i < lhsWords; ++i) {
    uint64_t tmp = LHS.get_pVal(i);
    __U[i * 2] = tmp & mask;
    __U[i * 2 + 1] = (tmp) >> (sizeof(uint32_t) * 8);
  }
  __U[m + n] = 0; // this extra word is for "spill" in the Knuth algorithm.

  // Initialize the divisor
  memset(__V, 0, (n) * sizeof(uint32_t));
  __V[0] = RHS & mask;
  __V[1] = (RHS) >> (sizeof(uint32_t) * 8);

  // initialize the quotient and remainder
  memset(__Q, 0, (m + n) * sizeof(uint32_t));
  if (Remainder) memset(__R, 0, n * sizeof(uint32_t));

  // Now, adjust m and n for the Knuth division. n is the number of words in
  // the divisor. m is the number of words by which the dividend exceeds the
  // divisor (i.e. m+n is the length of the dividend). These sizes must not
  // contain any zero words or the Knuth algorithm fails.
  for (unsigned i = n; i > 0 && __V[i - 1] == 0; i--) {
    n--;
    m++;
  }
  for (unsigned i = m + n; i > 0 && __U[i - 1] == 0; i--) m--;

  // If we're left with only a single word for the divisor, Knuth doesn't work
  // so we implement the short division algorithm here. This is much simpler
  // and faster because we are certain that we can divide a 64-bit quantity
  // by a 32-bit quantity at hardware speed and short division is simply a
  // series of such operations. This is just like doing short division but we
  // are using base 2^32 instead of base 10.
  assert(n != 0 && "Divide by zero?");
  if (n == 1) {
    uint32_t divisor = __V[0];
    uint32_t remainder = 0;
    for (int i = m + n - 1; i >= 0; i--) {
      uint64_t partial_dividend = (uint64_t(remainder)) << 32 | __U[i];
      if (partial_dividend == 0) {
        __Q[i] = 0;
        remainder = 0;
      } else if (partial_dividend < divisor) {
        __Q[i] = 0;
        remainder = partial_dividend;
      } else if (partial_dividend == divisor) {
        __Q[i] = 1;
        remainder = 0;
      } else {
        __Q[i] = partial_dividend / divisor;
        remainder = partial_dividend - (__Q[i] * divisor);
      }
    }
    if (__R) __R[0] = remainder;
  } else {
    // Now we're ready to invoke the Knuth classical divide algorithm. In this
    // case n > 1.
    KnuthDiv(__U, __V, __Q, __R, m, n);
  }

  // If the caller wants the quotient
  if (Quotient) {
    // Set up the Quotient value's memory.
    if (Quotient->BitWidth != LHS.BitWidth) {
      if (Quotient->isSingleWord()) Quotient->set_VAL(0);
    } else
      Quotient->clear();

    // The quotient is in Q. Reconstitute the quotient into Quotient's low
    // order words.
    if (lhsWords == 1) {
      uint64_t tmp =
          uint64_t(__Q[0]) | ((uint64_t(__Q[1])) << (APINT_BITS_PER_WORD / 2));
      Quotient->set_VAL(tmp);
    } else {
      assert(!Quotient->isSingleWord() &&
             "Quotient ap_private not large enough");
      for (unsigned i = 0; i < lhsWords; ++i)
        Quotient->set_pVal(
            i, uint64_t(__Q[i * 2]) |
                   ((uint64_t(__Q[i * 2 + 1])) << (APINT_BITS_PER_WORD / 2)));
    }
    Quotient->clearUnusedBits();
  }

  // If the caller wants the remainder
  if (Remainder) {
    // Set up the Remainder value's memory.
    if (Remainder->BitWidth != 64 /* RHS.BitWidth */) {
      if (Remainder->isSingleWord()) Remainder->set_VAL(0);
    } else
      Remainder->clear();

    // The remainder is in __R. Reconstitute the remainder into Remainder's low
    // order words.
    if (rhsWords == 1) {
      uint64_t tmp =
          uint64_t(__R[0]) | ((uint64_t(__R[1])) << (APINT_BITS_PER_WORD / 2));
      Remainder->set_VAL(tmp);
    } else {
      assert(!Remainder->isSingleWord() &&
             "Remainder ap_private not large enough");
      for (unsigned i = 0; i < rhsWords; ++i)
        Remainder->set_pVal(
            i, uint64_t(__R[i * 2]) |
                   ((uint64_t(__R[i * 2 + 1])) << (APINT_BITS_PER_WORD / 2)));
    }
    Remainder->clearUnusedBits();
  }

  // Clean up the memory we allocated.
  if (__U != &SPACE[0]) {
    delete[] __U;
    delete[] __V;
    delete[] __Q;
    delete[] __R;
  }
}

/// @brief Logical right-shift function.
template <int _AP_W, bool _AP_S, bool _AP_C>
INLINE ap_private<_AP_W, _AP_S, _AP_C> lshr(
    const ap_private<_AP_W, _AP_S, _AP_C>& LHS, uint32_t shiftAmt) {
  return LHS.lshr(shiftAmt);
}

/// Left-shift the ap_private by shiftAmt.
/// @brief Left-shift function.
template <int _AP_W, bool _AP_S, bool _AP_C>
INLINE ap_private<_AP_W, _AP_S, _AP_C> shl(
    const ap_private<_AP_W, _AP_S, _AP_C>& LHS, uint32_t shiftAmt) {
  return LHS.shl(shiftAmt);
}

} // namespace ap_private_ops

#endif // LLVM_SUPPORT_MATHEXTRAS_H

/// This enumeration just provides for internal constants used in this
/// translation unit.
enum {
  MIN_INT_BITS = 1, ///< Minimum number of bits that can be specified
  ///< Note that this must remain synchronized with IntegerType::MIN_INT_BITS
  MAX_INT_BITS = (1 << 23) - 1 ///< Maximum number of bits that can be specified
  ///< Note that this must remain synchronized with IntegerType::MAX_INT_BITS
};

//===----------------------------------------------------------------------===//
//                              ap_private Class
//===----------------------------------------------------------------------===//

/// ap_private - This class represents arbitrary precision constant integral
/// values.
/// It is a functional replacement for common case unsigned integer type like
/// "unsigned", "unsigned long" or "uint64_t", but also allows non-byte-width
/// integer sizes and large integer value types such as 3-bits, 15-bits, or more
/// than 64-bits of precision. ap_private provides a variety of arithmetic
/// operators
/// and methods to manipulate integer values of any bit-width. It supports both
/// the typical integer arithmetic and comparison operations as well as bitwise
/// manipulation.
///
/// The class has several invariants worth noting:
///   * All bit, byte, and word positions are zero-based.
///   * Once the bit width is set, it doesn't change except by the Truncate,
///     SignExtend, or ZeroExtend operations.
///   * All binary operators must be on ap_private instances of the same bit
///   width.
///     Attempting to use these operators on instances with different bit
///     widths will yield an assertion.
///   * The value is stored canonically as an unsigned value. For operations
///     where it makes a difference, there are both signed and unsigned variants
///     of the operation. For example, sdiv and udiv. However, because the bit
///     widths must be the same, operations such as Mul and Add produce the same
///     results regardless of whether the values are interpreted as signed or
///     not.
///   * In general, the class tries to follow the style of computation that LLVM
///     uses in its IR. This simplifies its use for LLVM.
///
/// @brief Class for arbitrary precision integers.

#if defined(_MSC_VER)
#if _MSC_VER < 1400 && !defined(for)
#define for if (0); else for
#endif
typedef unsigned __int64 ap_ulong;
typedef signed __int64 ap_slong;
#else
typedef unsigned long long ap_ulong;
typedef signed long long ap_slong;
#endif
template <int _AP_N8, bool _AP_S>
struct valtype;

template <int _AP_N8>
struct valtype<_AP_N8, false> {
  typedef uint64_t Type;
};

template <int _AP_N8>
struct valtype<_AP_N8, true> {
  typedef int64_t Type;
};

template <>
struct valtype<1, false> {
  typedef unsigned char Type;
};
template <>
struct valtype<2, false> {
  typedef unsigned short Type;
};
template <>
struct valtype<3, false> {
  typedef unsigned int Type;
};
template <>
struct valtype<4, false> {
  typedef unsigned int Type;
};
template <>
struct valtype<1, true> {
  typedef signed char Type;
};
template <>
struct valtype<2, true> {
  typedef short Type;
};
template <>
struct valtype<3, true> {
  typedef int Type;
};
template <>
struct valtype<4, true> {
  typedef int Type;
};

template <bool enable>
struct ap_private_enable_if {};
template <>
struct ap_private_enable_if<true> {
  static const bool isValid = true;
};

// When bitwidth < 64
template <int _AP_W, bool _AP_S>
class ap_private<_AP_W, _AP_S, true> {
  // SFINAE pattern.  Only consider this class when _AP_W <= 64
  const static bool valid = ap_private_enable_if<_AP_W <= 64>::isValid;

#ifdef _MSC_VER
#pragma warning(disable : 4521 4522)
#endif
 public:
  typedef typename valtype<(_AP_W + 7) / 8, _AP_S>::Type ValType;
  typedef ap_private<_AP_W, _AP_S> Type;
  template <int _AP_W2, bool _AP_S2>
  struct RType {
    enum {
      mult_w = _AP_W + _AP_W2,
      mult_s = _AP_S || _AP_S2,
      plus_w =
          AP_MAX(_AP_W + (_AP_S2 && !_AP_S), _AP_W2 + (_AP_S && !_AP_S2)) + 1,
      plus_s = _AP_S || _AP_S2,
      minus_w =
          AP_MAX(_AP_W + (_AP_S2 && !_AP_S), _AP_W2 + (_AP_S && !_AP_S2)) + 1,
      minus_s = true,
      div_w = _AP_W + _AP_S2,
      div_s = _AP_S || _AP_S2,
      mod_w = AP_MIN(_AP_W, _AP_W2 + (!_AP_S2 && _AP_S)),
      mod_s = _AP_S,
      logic_w = AP_MAX(_AP_W + (_AP_S2 && !_AP_S), _AP_W2 + (_AP_S && !_AP_S2)),
      logic_s = _AP_S || _AP_S2
    };
    typedef ap_private<mult_w, mult_s> mult;
    typedef ap_private<plus_w, plus_s> plus;
    typedef ap_private<minus_w, minus_s> minus;
    typedef ap_private<logic_w, logic_s> logic;
    typedef ap_private<div_w, div_s> div;
    typedef ap_private<mod_w, mod_s> mod;
    typedef ap_private<_AP_W, _AP_S> arg1;
    typedef bool reduce;
  };
  enum { APINT_BITS_PER_WORD = sizeof(uint64_t) * 8 };
  enum {
    excess_bits = (_AP_W % APINT_BITS_PER_WORD)
                      ? APINT_BITS_PER_WORD - (_AP_W % APINT_BITS_PER_WORD)
                      : 0
  };
  static const uint64_t mask = ((uint64_t)~0ULL >> (excess_bits));
  static const uint64_t not_mask = ~mask;
  static const uint64_t sign_bit_mask = 1ULL << (APINT_BITS_PER_WORD - 1);
  template <int _AP_W1>
  struct sign_ext_mask {
    static const uint64_t mask = ~0ULL << _AP_W1;
  };
  static const int width = _AP_W;

  enum {
    BitWidth = _AP_W,
    _AP_N = 1,
  };
  ValType VAL; ///< Used to store the <= 64 bits integer value.
#ifdef AP_CANARY
  ValType CANARY;
  void check_canary() { assert(CANARY == (ValType)0xDEADBEEFDEADBEEF); }
  void set_canary() { CANARY = (ValType)0xDEADBEEFDEADBEEF; }
#else
  void check_canary() {}
  void set_canary() {}
#endif

  INLINE ValType& get_VAL(void) { return VAL; }
  INLINE ValType get_VAL(void) const { return VAL; }
  INLINE ValType get_VAL(void) const volatile { return VAL; }
  INLINE void set_VAL(uint64_t value) { VAL = (ValType)value; }
  INLINE ValType& get_pVal(int i) { return VAL; }
  INLINE ValType get_pVal(int i) const { return VAL; }
  INLINE const uint64_t* get_pVal() const {
    assert(0 && "invalid usage");
    return 0;
  }
  INLINE ValType get_pVal(int i) const volatile { return VAL; }
  INLINE uint64_t* get_pVal() const volatile {
    assert(0 && "invalid usage");
    return 0;
  }
  INLINE void set_pVal(int i, uint64_t value) { VAL = (ValType)value; }

  INLINE uint32_t getBitWidth() const { return BitWidth; }

  template <int _AP_W1, bool _AP_S1>
  ap_private<_AP_W, _AP_S>& operator=(const ap_private<_AP_W1, _AP_S1>& RHS) {
    VAL = (ValType)(RHS.get_VAL());
    clearUnusedBits();
    return *this;
  }

  template <int _AP_W1, bool _AP_S1>
  ap_private<_AP_W, _AP_S>& operator=(
      const volatile ap_private<_AP_W1, _AP_S1>& RHS) {
    VAL = (ValType)(RHS.get_VAL()); // TODO check here about ap_private<W,S,false>
    clearUnusedBits();
    return *this;
  }

  void operator=(const ap_private& RHS) volatile {
    // Don't do anything for X = X
    VAL = RHS.get_VAL(); // No need to check because no harm done by copying.
    clearUnusedBits();
  }

  ap_private& operator=(const ap_private& RHS) {
    // Don't do anything for X = X
    VAL = RHS.get_VAL(); // No need to check because no harm done by copying.
    clearUnusedBits();
    return *this;
  }

  void operator=(const volatile ap_private& RHS) volatile {
    // Don't do anything for X = X
    VAL = RHS.get_VAL(); // No need to check because no harm done by copying.
    clearUnusedBits();
  }

  ap_private& operator=(const volatile ap_private& RHS) {
    // Don't do anything for X = X
    VAL = RHS.get_VAL(); // No need to check because no harm done by copying.
    clearUnusedBits();
    return *this;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_private& operator=(const _private_range_ref<_AP_W2, _AP_S2>& op2) {
    *this = ap_private<_AP_W2, false>(op2);
    return *this;
  }

#define ASSIGN_OP_FROM_INT(C_TYPE)               \
  INLINE ap_private& operator=(const C_TYPE v) { \
    set_canary();                                \
    this->VAL = (ValType)v;                      \
    clearUnusedBits();                           \
    check_canary();                              \
    return *this;                                \
  }

ASSIGN_OP_FROM_INT(bool)
ASSIGN_OP_FROM_INT(char)
ASSIGN_OP_FROM_INT(signed char)
ASSIGN_OP_FROM_INT(unsigned char)
ASSIGN_OP_FROM_INT(short)
ASSIGN_OP_FROM_INT(unsigned short)
ASSIGN_OP_FROM_INT(int)
ASSIGN_OP_FROM_INT(unsigned int)
ASSIGN_OP_FROM_INT(long)
ASSIGN_OP_FROM_INT(unsigned long)
ASSIGN_OP_FROM_INT(ap_slong)
ASSIGN_OP_FROM_INT(ap_ulong)
#if 0
ASSIGN_OP_FROM_INT(half)
ASSIGN_OP_FROM_INT(float)
ASSIGN_OP_FROM_INT(double)
#endif
#undef ASSIGN_OP_FROM_INT

  // XXX This is a must to prevent pointer being converted to bool.
  INLINE ap_private& operator=(const char* s) {
    ap_private tmp(s); // XXX direct-initialization, as ctor is explicit.
    operator=(tmp);
    return *this;
  }

 private:
  explicit INLINE ap_private(uint64_t* val) : VAL(val[0]) {
    set_canary();
    clearUnusedBits();
    check_canary();
  }

  INLINE bool isSingleWord() const { return true; }

 public:
  INLINE void fromString(const char* strStart, uint32_t slen, uint8_t radix) {
    bool isNeg = strStart[0] == '-';
    if (isNeg) {
      strStart++;
      slen--;
    }

    if (strStart[0] == '0' && (strStart[1] == 'b' || strStart[1] == 'B')) {
      //if(radix == 0) radix = 2;
      _AP_WARNING(radix != 2, "%s seems to have base %d, but %d given.", strStart, 2, radix);
      strStart += 2;
      slen -=2;
    } else if (strStart[0] == '0' && (strStart[1] == 'o' || strStart[1] == 'O')) {
      //if (radix == 0) radix = 8;
      _AP_WARNING(radix != 8, "%s seems to have base %d, but %d given.", strStart, 8, radix);
      strStart += 2;
      slen -=2;
    } else if (strStart[0] == '0' && (strStart[1] == 'x' || strStart[1] == 'X')) {
      //if (radix == 0) radix = 16;
      _AP_WARNING(radix != 16, "%s seems to have base %d, but %d given.", strStart, 16, radix);
      strStart += 2;
      slen -=2;
    } else if (strStart[0] == '0' && (strStart[1] == 'd' || strStart[1] == 'D')) {
      //if (radix == 0) radix = 10;
      _AP_WARNING(radix != 10, "%s seems to have base %d, but %d given.", strStart, 10, radix);
      strStart += 2;
      slen -=2;
    } else if (radix == 0) {
      //radix = 2; // XXX default value
    }

    // Check our assumptions here
    assert((radix == 10 || radix == 8 || radix == 16 || radix == 2) &&
           "Radix should be 2, 8, 10, or 16!");
    assert(strStart && "String is null?");

    // Clear bits.
    uint64_t tmpVAL = VAL = 0;

    switch (radix) {
      case 2:
        //        sscanf(strStart,"%b",&VAL);
        // tmpVAL = *strStart =='1' ? ~0ULL : 0;
        for (; *strStart; ++strStart) {
          assert((*strStart == '0' || *strStart == '1') &&
                 ("Wrong binary number"));
          tmpVAL <<= 1;
          tmpVAL |= (*strStart - '0');
        }
        break;
      case 8:
#ifdef _MSC_VER
        sscanf_s(strStart, "%llo", &tmpVAL, slen + 1);
#else
#if defined(__x86_64__) && !defined(__MINGW32__) && !defined(__WIN32__)
        sscanf(strStart, "%lo", &tmpVAL);
#else
        sscanf(strStart, "%llo", &tmpVAL);
#endif //__x86_64__
#endif //_MSC_VER
        break;
      case 10:
#ifdef _MSC_VER
        sscanf_s(strStart, "%llu", &tmpVAL, slen + 1);
#else
#if defined(__x86_64__) && !defined(__MINGW32__) && !defined(__WIN32__)
        sscanf(strStart, "%lu", &tmpVAL);
#else
        sscanf(strStart, "%llu", &tmpVAL);
#endif //__x86_64__
#endif //_MSC_VER
        break;
      case 16:
#ifdef _MSC_VER
        sscanf_s(strStart, "%llx", &tmpVAL, slen + 1);
#else
#if defined(__x86_64__) && !defined(__MINGW32__) && !defined(__WIN32__)
        sscanf(strStart, "%lx", &tmpVAL);
#else
        sscanf(strStart, "%llx", &tmpVAL);
#endif //__x86_64__
#endif //_MSC_VER
        break;
      default:
        assert(true && "Unknown radix");
        // error
    }
    VAL = isNeg ? (ValType)(-tmpVAL) : (ValType)(tmpVAL);

    clearUnusedBits();
  }

 private:
  INLINE ap_private(const std::string& val, uint8_t radix = 2) : VAL(0) {
    assert(!val.empty() && "String empty?");
    set_canary();
    fromString(val.c_str(), val.size(), radix);
    check_canary();
  }

  INLINE ap_private(const char strStart[], uint32_t slen, uint8_t radix)
      : VAL(0) {
    set_canary();
    fromString(strStart, slen, radix);
    check_canary();
  }

  INLINE ap_private(uint32_t numWords, const uint64_t bigVal[])
      : VAL(bigVal[0]) {
    set_canary();
    clearUnusedBits();
    check_canary();
  }

 public:
  INLINE ap_private() {
    set_canary();
    clearUnusedBits();
    check_canary();
  }

#define CTOR(TYPE)                              \
  INLINE ap_private(TYPE v) : VAL((ValType)v) { \
    set_canary();                               \
    clearUnusedBits();                          \
    check_canary();                             \
  }
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
#if 0
  CTOR(half)
  CTOR(float)
  CTOR(double)
#endif
#undef CTOR

  template <int _AP_W1, bool _AP_S1, bool _AP_OPT>
  INLINE ap_private(const ap_private<_AP_W1, _AP_S1, _AP_OPT>& that)
      : VAL((ValType)that.get_VAL()) {
    set_canary();
    clearUnusedBits();
    check_canary();
  }

  template <int _AP_W1, bool _AP_S1, bool _AP_OPT>
  INLINE ap_private(const volatile ap_private<_AP_W1, _AP_S1, _AP_OPT>& that)
      : VAL((ValType)that.get_VAL()) {
    set_canary();
    clearUnusedBits();
    check_canary();
  }

  explicit INLINE ap_private(const char* val) {
    set_canary();
    unsigned char radix = 10;
    std::string str = ap_private_ops::parseString(val, radix); // will set radix.
    std::string::size_type pos = str.find('.');
    // trunc all fraction part
    if (pos != std::string::npos) str = str.substr(pos);

    ap_private<_AP_W, _AP_S> ap_private_val(str, radix);
    operator=(ap_private_val);
    check_canary();
  }

  INLINE ap_private(const char* val, signed char rd) {
    set_canary();
    unsigned char radix = rd;
    std::string str = ap_private_ops::parseString(val, radix); // will set radix.
    std::string::size_type pos = str.find('.');
    // trunc all fraction part
    if (pos != std::string::npos) str = str.substr(pos);

    ap_private<_AP_W, _AP_S> ap_private_val(str, radix);
    operator=(ap_private_val);
    check_canary();
  }

  INLINE ~ap_private() { check_canary(); }

  INLINE bool isNegative() const {
    static const uint64_t sign_mask = 1ULL << (_AP_W - 1);
    return _AP_S && (sign_mask & VAL);
  }

  INLINE bool isPositive() const { return !isNegative(); }

  INLINE bool isStrictlyPositive() const { return !isNegative() && VAL != 0; }

  INLINE bool isAllOnesValue() const { return (mask & VAL) == mask; }

  INLINE bool operator==(const ap_private<_AP_W, _AP_S>& RHS) const {
    return VAL == RHS.get_VAL();
  }
  INLINE bool operator==(const ap_private<_AP_W, !_AP_S>& RHS) const {
    return (uint64_t)VAL == (uint64_t)RHS.get_VAL();
  }

  INLINE bool operator==(uint64_t Val) const { return ((uint64_t)VAL == Val); }
  INLINE bool operator!=(uint64_t Val) const { return ((uint64_t)VAL != Val); }
  INLINE bool operator!=(const ap_private<_AP_W, _AP_S>& RHS) const {
    return VAL != RHS.get_VAL();
  }
  INLINE bool operator!=(const ap_private<_AP_W, !_AP_S>& RHS) const {
    return (uint64_t)VAL != (uint64_t)RHS.get_VAL();
  }

  /// postfix increment.
  const ap_private operator++(int) {
    ap_private orig(*this);
    VAL++;
    clearUnusedBits();
    return orig;
  }

  /// prefix increment.
  const ap_private operator++() {
    ++VAL;
    clearUnusedBits();
    return *this;
  }

  /// postfix decrement.
  const ap_private operator--(int) {
    ap_private orig(*this);
    --VAL;
    clearUnusedBits();
    return orig;
  }

  /// prefix decrement.
  const ap_private operator--() {
    --VAL;
    clearUnusedBits();
    return *this;
  }

  /// one's complement.
  INLINE ap_private<_AP_W + !_AP_S, true> operator~() const {
    ap_private<_AP_W + !_AP_S, true> Result(*this);
    Result.flip();
    return Result;
  }

  /// two's complement.
  INLINE typename RType<1, false>::minus operator-() const {
    return ap_private<1, false>(0) - (*this);
  }

  /// logic negation.
  INLINE bool operator!() const { return !VAL; }

  INLINE std::string toString(uint8_t radix, bool wantSigned) const;
  INLINE std::string toStringUnsigned(uint8_t radix = 10) const {
    return toString(radix, false);
  }
  INLINE std::string toStringSigned(uint8_t radix = 10) const {
    return toString(radix, true);
  }
  INLINE void clear() { VAL = 0; }
  INLINE ap_private& clear(uint32_t bitPosition) {
    VAL &= ~(1ULL << (bitPosition));
    clearUnusedBits();
    return *this;
  }

  INLINE ap_private ashr(uint32_t shiftAmt) const {
    if (_AP_S)
      return ap_private((shiftAmt == BitWidth) ? 0
                                               : ((int64_t)VAL) >> (shiftAmt));
    else
      return ap_private((shiftAmt == BitWidth) ? 0
                                               : ((uint64_t)VAL) >> (shiftAmt));
  }

  INLINE ap_private lshr(uint32_t shiftAmt) const {
    return ap_private((shiftAmt == BitWidth)
                          ? ap_private(0)
                          : ap_private((VAL & mask) >> (shiftAmt)));
  }

  INLINE ap_private shl(uint32_t shiftAmt) const
// just for clang compiler
#if defined(__clang__) && !defined(__CLANG_3_1__)
      __attribute__((no_sanitize("undefined")))
#endif
  {
    if (shiftAmt > BitWidth) {
      if (!isNegative())
        return ap_private(0);
      else
        return ap_private(-1);
    }
    if (shiftAmt == BitWidth)
      return ap_private(0);
    else
      return ap_private((VAL) << (shiftAmt));
    // return ap_private((shiftAmt == BitWidth) ? ap_private(0ULL) :
    // ap_private(VAL << shiftAmt));
  }

  INLINE int64_t getSExtValue() const { return VAL; }

  // XXX XXX this function is used in CBE
  INLINE uint64_t getZExtValue() const { return VAL & mask; }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_private(const _private_range_ref<_AP_W2, _AP_S2>& ref) {
    set_canary();
    *this = ref.get();
    check_canary();
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_private(const _private_bit_ref<_AP_W2, _AP_S2>& ref) {
    set_canary();
    *this = ((uint64_t)(bool)ref);
    check_canary();
  }

//  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
//  INLINE ap_private(const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>& ref) {
//    set_canary();
//    *this = ref.get();
//    check_canary();
//  }
//
//  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
//            ap_o_mode _AP_O2, int _AP_N2>
//  INLINE ap_private(
//      const af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& val) {
//    set_canary();
//    *this = ((val.operator ap_private<_AP_W2, false>()));
//    check_canary();
//  }
//
//  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
//            ap_o_mode _AP_O2, int _AP_N2>
//  INLINE ap_private(
//      const af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& val) {
//    set_canary();
//    *this = (uint64_t)(bool)val;
//    check_canary();
//  }

  INLINE void write(const ap_private<_AP_W, _AP_S>& op2) volatile {
    *this = (op2);
  }

  // Explicit conversions to C interger types
  //-----------------------------------------------------------
  INLINE operator ValType() const { return get_VAL(); }

  INLINE int to_uchar() const { return (unsigned char)get_VAL(); }

  INLINE int to_char() const { return (signed char)get_VAL(); }

  INLINE int to_ushort() const { return (unsigned short)get_VAL(); }

  INLINE int to_short() const { return (short)get_VAL(); }

  INLINE int to_int() const {
    //      ap_private<64 /* _AP_W */, _AP_S> res(V);
    return (int)get_VAL();
  }

  INLINE unsigned to_uint() const { return (unsigned)get_VAL(); }

  INLINE long to_long() const { return (long)get_VAL(); }

  INLINE unsigned long to_ulong() const { return (unsigned long)get_VAL(); }

  INLINE ap_slong to_int64() const { return (ap_slong)get_VAL(); }

  INLINE ap_ulong to_uint64() const { return (ap_ulong)get_VAL(); }

  INLINE double to_double() const {
    if (isNegative())
      return roundToDouble(true);
    else
      return roundToDouble(false);
  }

  INLINE unsigned length() const { return _AP_W; }

  INLINE bool isMinValue() const { return VAL == 0; }
  template <int _AP_W1, bool _AP_S1>
  INLINE ap_private& operator&=(const ap_private<_AP_W1, _AP_S1>& RHS) {
    VAL = (ValType)(((uint64_t)VAL) & RHS.get_VAL());
    clearUnusedBits();
    return *this;
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE ap_private& operator|=(const ap_private<_AP_W1, _AP_S1>& RHS) {
    VAL = (ValType)(((uint64_t)VAL) | RHS.get_VAL());
    clearUnusedBits();
    return *this;
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE ap_private& operator^=(const ap_private<_AP_W1, _AP_S1>& RHS) {
    VAL = (ValType)(((uint64_t)VAL) ^ RHS.get_VAL());
    clearUnusedBits();
    return *this;
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE ap_private& operator*=(const ap_private<_AP_W1, _AP_S1>& RHS) {
    VAL = (ValType)(((uint64_t)VAL) * RHS.get_VAL());
    clearUnusedBits();
    return *this;
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE ap_private& operator+=(const ap_private<_AP_W1, _AP_S1>& RHS) {
    VAL = (ValType)(((uint64_t)VAL) + RHS.get_VAL());
    clearUnusedBits();
    return *this;
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE ap_private& operator-=(const ap_private<_AP_W1, _AP_S1>& RHS) {
    VAL = (ValType)(((uint64_t)VAL) - RHS.get_VAL());
    clearUnusedBits();
    return *this;
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE typename RType<_AP_W1, _AP_S1>::logic operator&(
      const ap_private<_AP_W1, _AP_S1>& RHS) const {
    if (RType<_AP_W1, _AP_S1>::logic_w <= 64) {
      typename RType<_AP_W1, _AP_S1>::logic Ret(((uint64_t)VAL) &
                                                RHS.get_VAL());
      return Ret;
    } else {
      typename RType<_AP_W1, _AP_S1>::logic Ret = *this;
      return Ret & RHS;
    }
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE typename RType<_AP_W1, _AP_S1>::logic operator^(
      const ap_private<_AP_W1, _AP_S1>& RHS) const {
    if (RType<_AP_W1, _AP_S1>::logic_w <= 64) {
      typename RType<_AP_W1, _AP_S1>::logic Ret(((uint64_t)VAL) ^
                                                RHS.get_VAL());
      return Ret;
    } else {
      typename RType<_AP_W1, _AP_S1>::logic Ret = *this;
      return Ret ^ RHS;
    }
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE typename RType<_AP_W1, _AP_S1>::logic operator|(
      const ap_private<_AP_W1, _AP_S1>& RHS) const {
    if (RType<_AP_W1, _AP_S1>::logic_w <= 64) {
      typename RType<_AP_W1, _AP_S1>::logic Ret(((uint64_t)VAL) |
                                                RHS.get_VAL());
      return Ret;
    } else {
      typename RType<_AP_W1, _AP_S1>::logic Ret = *this;
      return Ret | RHS;
    }
  }

  INLINE ap_private And(const ap_private& RHS) const {
    return ap_private(VAL & RHS.get_VAL());
  }

  INLINE ap_private Or(const ap_private& RHS) const {
    return ap_private(VAL | RHS.get_VAL());
  }

  INLINE ap_private Xor(const ap_private& RHS) const {
    return ap_private(VAL ^ RHS.get_VAL());
  }
#if 1
  template <int _AP_W1, bool _AP_S1>
  INLINE typename RType<_AP_W1, _AP_S1>::mult operator*(
      const ap_private<_AP_W1, _AP_S1>& RHS) const {
    if (RType<_AP_W1, _AP_S1>::mult_w <= 64) {
      typename RType<_AP_W1, _AP_S1>::mult Result(((uint64_t)VAL) *
                                                  RHS.get_VAL());
      return Result;
    } else {
      typename RType<_AP_W1, _AP_S1>::mult Result(*this);
      Result *= RHS;
      return Result;
    }
  }
#endif
  INLINE ap_private Mul(const ap_private& RHS) const {
    return ap_private(VAL * RHS.get_VAL());
  }

  INLINE ap_private Add(const ap_private& RHS) const {
    return ap_private(VAL + RHS.get_VAL());
  }

  INLINE ap_private Sub(const ap_private& RHS) const {
    return ap_private(VAL - RHS.get_VAL());
  }

  INLINE ap_private& operator&=(uint64_t RHS) {
    VAL &= (ValType)RHS;
    clearUnusedBits();
    return *this;
  }
  INLINE ap_private& operator|=(uint64_t RHS) {
    VAL |= (ValType)RHS;
    clearUnusedBits();
    return *this;
  }
  INLINE ap_private& operator^=(uint64_t RHS) {
    VAL ^= (ValType)RHS;
    clearUnusedBits();
    return *this;
  }
  INLINE ap_private& operator*=(uint64_t RHS) {
    VAL *= (ValType)RHS;
    clearUnusedBits();
    return *this;
  }
  INLINE ap_private& operator+=(uint64_t RHS) {
    VAL += (ValType)RHS;
    clearUnusedBits();
    return *this;
  }
  INLINE ap_private& operator-=(uint64_t RHS) {
    VAL -= (ValType)RHS;
    clearUnusedBits();
    return *this;
  }

  INLINE bool isMinSignedValue() const {
    static const uint64_t min_mask = ~(~0ULL << (_AP_W - 1));
    return BitWidth == 1 ? VAL == 1
                         : (ap_private_ops::isNegative<_AP_W>(*this) &&
                            ((min_mask & VAL) == 0));
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE typename RType<_AP_W1, _AP_S1>::plus operator+(
      const ap_private<_AP_W1, _AP_S1>& RHS) const {
    if (RType<_AP_W1, _AP_S1>::plus_w <= 64)
      return typename RType<_AP_W1, _AP_S1>::plus(
          RType<_AP_W1, _AP_S1>::plus_s
              ? int64_t(((uint64_t)VAL) + RHS.get_VAL())
              : uint64_t(((uint64_t)VAL) + RHS.get_VAL()));
    typename RType<_AP_W1, _AP_S1>::plus Result = RHS;
    Result += VAL;
    return Result;
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE typename RType<_AP_W1, _AP_S1>::minus operator-(
      const ap_private<_AP_W1, _AP_S1>& RHS) const {
    if (RType<_AP_W1, _AP_S1>::minus_w <= 64)
      return typename RType<_AP_W1, _AP_S1>::minus(
          int64_t(((uint64_t)VAL) - RHS.get_VAL()));
    typename RType<_AP_W1, _AP_S1>::minus Result = *this;
    Result -= RHS;
    return Result;
  }

  INLINE uint32_t countPopulation() const {
    return ap_private_ops::CountPopulation_64(VAL);
  }
  INLINE uint32_t countLeadingZeros() const {
    int remainder = BitWidth % 64;
    int excessBits = (64 - remainder) % 64;
    uint32_t Count = ap_private_ops::CountLeadingZeros_64(VAL);
    if (Count) Count -= excessBits;
    return AESL_std::min(Count, (uint32_t)_AP_W);
  }

  /// HiBits - This function returns the high "numBits" bits of this ap_private.
  INLINE ap_private<_AP_W, _AP_S> getHiBits(uint32_t numBits) const {
    ap_private<_AP_W, _AP_S> ret(*this);
    ret = (ret) >> (BitWidth - numBits);
    return ret;
  }

  /// LoBits - This function returns the low "numBits" bits of this ap_private.
  INLINE ap_private<_AP_W, _AP_S> getLoBits(uint32_t numBits) const {
    ap_private<_AP_W, _AP_S> ret(((uint64_t)VAL) << (BitWidth - numBits));
    ret = (ret) >> (BitWidth - numBits);
    return ret;
    // return ap_private(numBits, (VAL << (BitWidth - numBits))>> (BitWidth -
    // numBits));
  }

  INLINE ap_private<_AP_W, _AP_S>& set(uint32_t bitPosition) {
    VAL |= (1ULL << (bitPosition));
    clearUnusedBits();
    return *this; // clearUnusedBits();
  }

  INLINE void set() {
    VAL = (ValType)~0ULL;
    clearUnusedBits();
  }

  template <int _AP_W3>
  INLINE void set(const ap_private<_AP_W3, false>& val) {
    operator=(ap_private<_AP_W3, _AP_S>(val));
  }

  INLINE void set(const ap_private& val) { operator=(val); }

  INLINE void clearUnusedBits(void) volatile
// just for clang compiler
#if defined(__clang__) && !defined(__CLANG_3_1__)
      __attribute__((no_sanitize("undefined")))
#endif
  {
    enum { excess_bits = (_AP_W % 64) ? 64 - _AP_W % 64 : 0 };
    VAL = (ValType)(
        _AP_S
            ? ((((int64_t)VAL) << (excess_bits)) >> (excess_bits))
            : (excess_bits ? (((uint64_t)VAL) << (excess_bits)) >> (excess_bits)
                           : (uint64_t)VAL));
  }

  INLINE void clearUnusedBitsToZero(void) {
    enum { excess_bits = (_AP_W % 64) ? 64 - _AP_W % 64 : 0 };
    static uint64_t mask = ~0ULL >> (excess_bits);
    VAL &= mask;
  }

  INLINE ap_private udiv(const ap_private& RHS) const {
    return ap_private((uint64_t)VAL / RHS.get_VAL());
  }

  /// Signed divide this ap_private by ap_private RHS.
  /// @brief Signed division function for ap_private.
  INLINE ap_private sdiv(const ap_private& RHS) const {
    if (isNegative())
      if (RHS.isNegative())
        return ((uint64_t)(0 - (*this))) / (uint64_t)(0 - RHS);
      else
        return 0 - ((uint64_t)(0 - (*this)) / (uint64_t)(RHS));
    else if (RHS.isNegative())
      return 0 - (this->udiv((ap_private)(0 - RHS)));
    return this->udiv(RHS);
  }

  template <bool _AP_S2>
  INLINE ap_private urem(const ap_private<_AP_W, _AP_S2>& RHS) const {
    assert(RHS.get_VAL() != 0 && "Divide by 0");
    return ap_private(((uint64_t)VAL) % ((uint64_t)RHS.get_VAL()));
  }

  /// Signed remainder operation on ap_private.
  /// @brief Function for signed remainder operation.
  template <bool _AP_S2>
  INLINE ap_private srem(const ap_private<_AP_W, _AP_S2>& RHS) const {
    if (isNegative()) {
      ap_private lhs = 0 - (*this);
      if (RHS.isNegative()) {
        ap_private rhs = 0 - RHS;
        return 0 - (lhs.urem(rhs));
      } else
        return 0 - (lhs.urem(RHS));
    } else if (RHS.isNegative()) {
      ap_private rhs = 0 - RHS;
      return this->urem(rhs);
    }
    return this->urem(RHS);
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE bool eq(const ap_private<_AP_W1, _AP_S1>& RHS) const {
    return (*this) == RHS;
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE bool ne(const ap_private<_AP_W1, _AP_S1>& RHS) const {
    return !((*this) == RHS);
  }

  /// Regards both *this and RHS as unsigned quantities and compares them for
  /// the validity of the less-than relationship.
  /// @returns true if *this < RHS when both are considered unsigned.
  /// @brief Unsigned less than comparison
  template <int _AP_W1, bool _AP_S1>
  INLINE bool ult(const ap_private<_AP_W1, _AP_S1>& RHS) const {
    if (_AP_W1 <= 64) {
      uint64_t lhsZext = ((uint64_t(VAL)) << (64 - _AP_W)) >> (64 - _AP_W);
      uint64_t rhsZext =
          ((uint64_t(RHS.get_VAL())) << (64 - _AP_W1)) >> (64 - _AP_W1);
      return lhsZext < rhsZext;
    } else
      return RHS.uge(*this);
  }

  /// Regards both *this and RHS as signed quantities and compares them for
  /// validity of the less-than relationship.
  /// @returns true if *this < RHS when both are considered signed.
  /// @brief Signed less than comparison
  template <int _AP_W1, bool _AP_S1>
  INLINE bool slt(const ap_private<_AP_W1, _AP_S1>& RHS) const
// just for clang compiler
#if defined(__clang__) && !defined(__CLANG_3_1__)
      __attribute__((no_sanitize("undefined")))
#endif
  {
    if (_AP_W1 <= 64) {
      int64_t lhsSext = ((int64_t(VAL)) << (64 - _AP_W)) >> (64 - _AP_W);
      int64_t rhsSext =
          ((int64_t(RHS.get_VAL())) << (64 - _AP_W1)) >> (64 - _AP_W1);
      return lhsSext < rhsSext;
    } else
      return RHS.sge(*this);
  }

  /// Regards both *this and RHS as unsigned quantities and compares them for
  /// validity of the less-or-equal relationship.
  /// @returns true if *this <= RHS when both are considered unsigned.
  /// @brief Unsigned less or equal comparison
  template <int _AP_W1, bool _AP_S1>
  INLINE bool ule(const ap_private<_AP_W1, _AP_S1>& RHS) const {
    return ult(RHS) || eq(RHS);
  }

  /// Regards both *this and RHS as signed quantities and compares them for
  /// validity of the less-or-equal relationship.
  /// @returns true if *this <= RHS when both are considered signed.
  /// @brief Signed less or equal comparison
  template <int _AP_W1, bool _AP_S1>
  INLINE bool sle(const ap_private<_AP_W1, _AP_S1>& RHS) const {
    return slt(RHS) || eq(RHS);
  }

  /// Regards both *this and RHS as unsigned quantities and compares them for
  /// the validity of the greater-than relationship.
  /// @returns true if *this > RHS when both are considered unsigned.
  /// @brief Unsigned greather than comparison
  template <int _AP_W1, bool _AP_S1>
  INLINE bool ugt(const ap_private<_AP_W1, _AP_S1>& RHS) const {
    return !ult(RHS) && !eq(RHS);
  }

  /// Regards both *this and RHS as signed quantities and compares them for
  /// the validity of the greater-than relationship.
  /// @returns true if *this > RHS when both are considered signed.
  /// @brief Signed greather than comparison
  template <int _AP_W1, bool _AP_S1>
  INLINE bool sgt(const ap_private<_AP_W1, _AP_S1>& RHS) const {
    return !slt(RHS) && !eq(RHS);
  }

  /// Regards both *this and RHS as unsigned quantities and compares them for
  /// validity of the greater-or-equal relationship.
  /// @returns true if *this >= RHS when both are considered unsigned.
  /// @brief Unsigned greater or equal comparison
  template <int _AP_W1, bool _AP_S1>
  INLINE bool uge(const ap_private<_AP_W1, _AP_S1>& RHS) const {
    return !ult(RHS);
  }

  /// Regards both *this and RHS as signed quantities and compares them for
  /// validity of the greater-or-equal relationship.
  /// @returns true if *this >= RHS when both are considered signed.
  /// @brief Signed greather or equal comparison
  template <int _AP_W1, bool _AP_S1>
  INLINE bool sge(const ap_private<_AP_W1, _AP_S1>& RHS) const {
    return !slt(RHS);
  }

  INLINE ap_private abs() const {
    if (isNegative()) return -(*this);
    return *this;
  }

  INLINE ap_private<_AP_W, false> get() const {
    ap_private<_AP_W, false> ret(*this);
    return ret;
  }

  INLINE static uint32_t getBitsNeeded(const char* str, uint32_t slen,
                                       uint8_t radix) {
    return _AP_W;
  }

  INLINE uint32_t getActiveBits() const {
    uint32_t bits = _AP_W - countLeadingZeros();
    return bits ? bits : 1;
  }

  INLINE double roundToDouble(bool isSigned = false) const {
    return isSigned ? double((int64_t)VAL) : double((uint64_t)VAL);
  }

  /*Reverse the contents of ap_private instance. I.e. LSB becomes MSB and vise
   * versa*/
  INLINE ap_private& reverse() {
    for (int i = 0; i < _AP_W / 2; ++i) {
      bool tmp = operator[](i);
      if (operator[](_AP_W - 1 - i))
        set(i);
      else
        clear(i);
      if (tmp)
        set(_AP_W - 1 - i);
      else
        clear(_AP_W - 1 - i);
    }
    clearUnusedBits();
    return *this;
  }

  /*Return true if the value of ap_private instance is zero*/
  INLINE bool iszero() const { return isMinValue(); }

  INLINE bool to_bool() const { return !iszero(); }

  /* x < 0 */
  INLINE bool sign() const {
    if (isNegative()) return true;
    return false;
  }

  /* x[i] = !x[i] */
  INLINE void invert(int i) {
    assert(i >= 0 && "Attempting to read bit with negative index");
    assert(i < _AP_W && "Attempting to read bit beyond MSB");
    flip(i);
  }

  /* x[i] */
  INLINE bool test(int i) const {
    assert(i >= 0 && "Attempting to read bit with negative index");
    assert(i < _AP_W && "Attempting to read bit beyond MSB");
    return operator[](i);
  }

  // This is used for sc_lv and sc_bv, which is implemented by sc_uint
  // Rotate an ap_private object n places to the left
  INLINE void lrotate(int n) {
    assert(n >= 0 && "Attempting to shift negative index");
    assert(n < _AP_W && "Shift value larger than bit width");
    operator=(shl(n) | lshr(_AP_W - n));
  }

  // This is used for sc_lv and sc_bv, which is implemented by sc_uint
  // Rotate an ap_private object n places to the right
  INLINE void rrotate(int n) {
    assert(n >= 0 && "Attempting to shift negative index");
    assert(n < _AP_W && "Shift value larger than bit width");
    operator=(lshr(n) | shl(_AP_W - n));
  }

  // Set the ith bit into v
  INLINE void set(int i, bool v) {
    assert(i >= 0 && "Attempting to write bit with negative index");
    assert(i < _AP_W && "Attempting to write bit beyond MSB");
    v ? set(i) : clear(i);
  }

  // Set the ith bit into v
  INLINE void set_bit(int i, bool v) {
    assert(i >= 0 && "Attempting to write bit with negative index");
    assert(i < _AP_W && "Attempting to write bit beyond MSB");
    v ? set(i) : clear(i);
  }

  // Get the value of ith bit
  INLINE bool get_bit(int i) const {
    assert(i >= 0 && "Attempting to read bit with negative index");
    assert(i < _AP_W && "Attempting to read bit beyond MSB");
    return (((1ULL << i) & VAL) != 0);
  }

  /// Toggle all bits.
  INLINE ap_private& flip() {
    VAL = (ValType)((~0ULL ^ VAL) & mask);
    clearUnusedBits();
    return *this;
  }

  /// Toggles a given bit to its opposite value.
  INLINE ap_private& flip(uint32_t bitPosition) {
    assert(bitPosition < BitWidth && "Out of the bit-width range!");
    set_bit(bitPosition, !get_bit(bitPosition));
    return *this;
  }

  // complements every bit
  INLINE void b_not() { flip(); }

// Binary Arithmetic
//-----------------------------------------------------------
#define OP_BIN_AP(Sym, Rty, Fun)                           \
  template <int _AP_W2, bool _AP_S2>                       \
  INLINE typename RType<_AP_W2, _AP_S2>::Rty operator Sym( \
      const ap_private<_AP_W2, _AP_S2>& op) const {        \
    typename RType<_AP_W2, _AP_S2>::Rty lhs(*this);        \
    typename RType<_AP_W2, _AP_S2>::Rty rhs(op);           \
    return lhs.Fun(rhs);                                   \
  }

/// Bitwise and, or, xor
// OP_BIN_AP(&,logic, And)
// OP_BIN_AP(|,logic, Or)
// OP_BIN_AP(^,logic, Xor)
#undef OP_BIN_AP

  template <int _AP_W2, bool _AP_S2>
  INLINE typename RType<_AP_W2, _AP_S2>::div operator/(
      const ap_private<_AP_W2, _AP_S2>& op) const {
    ap_private<AP_MAX(_AP_W + (_AP_S || _AP_S2), _AP_W2 + (_AP_S || _AP_S2)),
               (_AP_W > _AP_W2 ? _AP_S
                               : (_AP_W2 > _AP_W ? _AP_S2 : _AP_S || _AP_S2))>
        lhs = *this;
    ap_private<AP_MAX(_AP_W + (_AP_S || _AP_S2), _AP_W2 + (_AP_S || _AP_S2)),
               (_AP_W > _AP_W2 ? _AP_S
                               : (_AP_W2 > _AP_W ? _AP_S2 : _AP_S || _AP_S2))>
        rhs = op;
    return typename RType<_AP_W2, _AP_S2>::div(
        (_AP_S || _AP_S2) ? lhs.sdiv(rhs) : lhs.udiv(rhs));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE typename RType<_AP_W2, _AP_S2>::mod operator%(
      const ap_private<_AP_W2, _AP_S2>& op) const {
    ap_private<AP_MAX(_AP_W + (_AP_S || _AP_S2), _AP_W2 + (_AP_S || _AP_S2)),
               (_AP_W > _AP_W2 ? _AP_S
                               : (_AP_W2 > _AP_W ? _AP_S2 : _AP_S || _AP_S2))>
        lhs = *this;
    ap_private<AP_MAX(_AP_W + (_AP_S || _AP_S2), _AP_W2 + (_AP_S || _AP_S2)),
               (_AP_W > _AP_W2 ? _AP_S
                               : (_AP_W2 > _AP_W ? _AP_S2 : _AP_S || _AP_S2))>
        rhs = op;
    typename RType<_AP_W2, _AP_S2>::mod res =
        typename RType<_AP_W2, _AP_S2>::mod(_AP_S ? lhs.srem(rhs)
                                                  : lhs.urem(rhs));
    return res;
  }

#define OP_ASSIGN_AP_2(Sym)                         \
  template <int _AP_W2, bool _AP_S2>                \
  INLINE ap_private<_AP_W, _AP_S>& operator Sym##=( \
      const ap_private<_AP_W2, _AP_S2>& op) {       \
    *this = operator Sym(op);                       \
    return *this;                                   \
  }

  OP_ASSIGN_AP_2(/)
  OP_ASSIGN_AP_2(%)
#undef OP_ASSIGN_AP_2

/// Bitwise assign: and, or, xor
//-------------------------------------------------------------
//    OP_ASSIGN_AP(&)
//    OP_ASSIGN_AP(^)
//    OP_ASSIGN_AP(|)

#define OP_LEFT_SHIFT_CTYPE(TYPE, SIGNED)             \
  INLINE ap_private operator<<(const TYPE op) const { \
    if (op >= _AP_W) return ap_private(0);            \
    if (SIGNED && op < 0) return *this >> (0 - op);   \
    return shl(op);                                   \
  }

  // OP_LEFT_SHIFT_CTYPE(bool, false)
  OP_LEFT_SHIFT_CTYPE(char, CHAR_IS_SIGNED)
  OP_LEFT_SHIFT_CTYPE(signed char, true)
  OP_LEFT_SHIFT_CTYPE(unsigned char, false)
  OP_LEFT_SHIFT_CTYPE(short, true)
  OP_LEFT_SHIFT_CTYPE(unsigned short, false)
  OP_LEFT_SHIFT_CTYPE(int, true)
  OP_LEFT_SHIFT_CTYPE(unsigned int, false)
  OP_LEFT_SHIFT_CTYPE(long, true)
  OP_LEFT_SHIFT_CTYPE(unsigned long, false)
  OP_LEFT_SHIFT_CTYPE(long long, true)
  OP_LEFT_SHIFT_CTYPE(unsigned long long, false)
#if 0
  OP_LEFT_SHIFT_CTYPE(half, false)
  OP_LEFT_SHIFT_CTYPE(float, false)
  OP_LEFT_SHIFT_CTYPE(double, false)
#endif

#undef OP_LEFT_SHIFT_CTYPE

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_private operator<<(const ap_private<_AP_W2, _AP_S2>& op2) const {
    if (_AP_S2 == false) {
      uint32_t sh = op2.to_uint();
      return *this << sh;
    } else {
      int sh = op2.to_int();
      return *this << sh;
    }
  }

#define OP_RIGHT_SHIFT_CTYPE(TYPE, SIGNED)            \
  INLINE ap_private operator>>(const TYPE op) const { \
    if (op >= _AP_W) {                                \
      if (isNegative())                               \
        return ap_private(-1);                        \
      else                                            \
        return ap_private(0);                         \
    }                                                 \
    if ((SIGNED) && op < 0) return *this << (0 - op); \
    if (_AP_S)                                        \
      return ashr(op);                                \
    else                                              \
      return lshr(op);                                \
  }

  // OP_RIGHT_SHIFT_CTYPE(bool, false)
  OP_RIGHT_SHIFT_CTYPE(char, CHAR_IS_SIGNED)
  OP_RIGHT_SHIFT_CTYPE(signed char, true)
  OP_RIGHT_SHIFT_CTYPE(unsigned char, false)
  OP_RIGHT_SHIFT_CTYPE(short, true)
  OP_RIGHT_SHIFT_CTYPE(unsigned short, false)
  OP_RIGHT_SHIFT_CTYPE(int, true)
  OP_RIGHT_SHIFT_CTYPE(unsigned int, false)
  OP_RIGHT_SHIFT_CTYPE(long, true)
  OP_RIGHT_SHIFT_CTYPE(unsigned long, false)
  OP_RIGHT_SHIFT_CTYPE(unsigned long long, false)
  OP_RIGHT_SHIFT_CTYPE(long long, true)
#if 0
  OP_RIGHT_SHIFT_CTYPE(half, false)
  OP_RIGHT_SHIFT_CTYPE(float, false)
  OP_RIGHT_SHIFT_CTYPE(double, false)
#endif

#undef OP_RIGHT_SHIFT_CTYPE

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_private operator>>(const ap_private<_AP_W2, _AP_S2>& op2) const {
    if (_AP_S2 == false) {
      uint32_t sh = op2.to_uint();
      return *this >> sh;
    } else {
      int sh = op2.to_int();
      return *this >> sh;
    }
  }

  /// Shift assign
  //-----------------------------------------------------------------

  //INLINE const ap_private& operator<<=(uint32_t shiftAmt) {
  //  VAL <<= shiftAmt;
  //  clearUnusedBits();
  //  return *this;
  //}

#define OP_ASSIGN_AP(Sym)                                                    \
  template <int _AP_W2, bool _AP_S2>                                         \
  INLINE ap_private& operator Sym##=(int op) {                               \
    *this = operator Sym(op);                                                \
    clearUnusedBits();                                                       \
    return *this;                                                            \
  }                                                                          \
  INLINE ap_private& operator Sym##=(unsigned int op) {                      \
    *this = operator Sym(op);                                                \
    clearUnusedBits();                                                       \
    return *this;                                                            \
  }                                                                          \
  template <int _AP_W2, bool _AP_S2>                                         \
  INLINE ap_private& operator Sym##=(const ap_private<_AP_W2, _AP_S2>& op) { \
    *this = operator Sym(op);                                                \
    clearUnusedBits();                                                       \
    return *this;                                                            \
  }

  OP_ASSIGN_AP(>>)
  OP_ASSIGN_AP(<<)
#undef OP_ASSIGN_AP

  /// Comparisons
  //-----------------------------------------------------------------
  template <int _AP_W1, bool _AP_S1>
  INLINE bool operator==(const ap_private<_AP_W1, _AP_S1>& op) const {
    enum { _AP_MAX_W = AP_MAX(AP_MAX(_AP_W, _AP_W1), 32) };
    ap_private<_AP_MAX_W, false> lhs(*this);
    ap_private<_AP_MAX_W, false> rhs(op);
    if (_AP_MAX_W <= 64) {
      return (uint64_t)lhs.get_VAL() == (uint64_t)rhs.get_VAL();
    } else
      return lhs == rhs;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator!=(const ap_private<_AP_W2, _AP_S2>& op) const {
    return !(*this == op);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator>(const ap_private<_AP_W2, _AP_S2>& op) const {
    enum {
      _AP_MAX_W = AP_MAX(_AP_W + (_AP_S || _AP_S2), _AP_W2 + (_AP_S || _AP_S2))
    };
    ap_private<_AP_MAX_W, _AP_S> lhs(*this);
    ap_private<_AP_MAX_W, _AP_S2> rhs(op);
    // this will follow gcc rule for comparison
    // between different bitwidth and signness
    if (_AP_S == _AP_S2)
      return _AP_S ? lhs.sgt(rhs) : lhs.ugt(rhs);
    else if (_AP_W < 32 && _AP_W2 < 32)
      // different signness but both bitwidth is less than 32
      return lhs.sgt(rhs);
    else
        // different signness but bigger bitwidth
        // is greater or equal to 32
        if (_AP_S)
      if (_AP_W2 >= _AP_W)
        return lhs.ugt(rhs);
      else
        return lhs.sgt(rhs);
    else if (_AP_W >= _AP_W2)
      return lhs.ugt(rhs);
    else
      return lhs.sgt(rhs);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator<=(const ap_private<_AP_W2, _AP_S2>& op) const {
    return !(*this > op);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator<(const ap_private<_AP_W2, _AP_S2>& op) const {
    enum {
      _AP_MAX_W = AP_MAX(_AP_W + (_AP_S || _AP_S2), _AP_W2 + (_AP_S || _AP_S2))
    };
    ap_private<_AP_MAX_W, _AP_S> lhs(*this);
    ap_private<_AP_MAX_W, _AP_S2> rhs(op);
    if (_AP_S == _AP_S2)
      return _AP_S ? lhs.slt(rhs) : lhs.ult(rhs);
    else if (_AP_W < 32 && _AP_W2 < 32)
      return lhs.slt(rhs);
    else if (_AP_S)
      if (_AP_W2 >= _AP_W)
        return lhs.ult(rhs);
      else
        return lhs.slt(rhs);
    else if (_AP_W >= _AP_W2)
      return lhs.ult(rhs);
    else
      return lhs.slt(rhs);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator>=(const ap_private<_AP_W2, _AP_S2>& op) const {
    return !(*this < op);
  }

  /// Bit and Part Select
  //--------------------------------------------------------------
  // FIXME now _private_range_ref refs to _AP_ROOT_TYPE(struct ssdm_int).
  INLINE _private_range_ref<_AP_W, _AP_S> operator()(int Hi, int Lo) {
    return _private_range_ref<_AP_W, _AP_S>(this, Hi, Lo);
  }

  INLINE _private_range_ref<_AP_W, _AP_S> operator()(int Hi, int Lo) const {
    return _private_range_ref<_AP_W, _AP_S>(
        const_cast<ap_private<_AP_W, _AP_S>*>(this), Hi, Lo);
  }

  INLINE _private_range_ref<_AP_W, _AP_S> range(int Hi, int Lo) const {
    return _private_range_ref<_AP_W, _AP_S>(
        (const_cast<ap_private<_AP_W, _AP_S>*>(this)), Hi, Lo);
  }

  INLINE _private_range_ref<_AP_W, _AP_S> range(int Hi, int Lo) {
    return _private_range_ref<_AP_W, _AP_S>(this, Hi, Lo);
  }

  INLINE _private_bit_ref<_AP_W, _AP_S> operator[](int index) {
    return _private_bit_ref<_AP_W, _AP_S>(*this, index);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE _private_bit_ref<_AP_W, _AP_S> operator[](
      const ap_private<_AP_W2, _AP_S2>& index) {
    return _private_bit_ref<_AP_W, _AP_S>(*this, index.to_int());
  }

  INLINE const _private_bit_ref<_AP_W, _AP_S> operator[](int index) const {
    return _private_bit_ref<_AP_W, _AP_S>(
        const_cast<ap_private<_AP_W, _AP_S>&>(*this), index);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE const _private_bit_ref<_AP_W, _AP_S> operator[](
      const ap_private<_AP_W2, _AP_S2>& index) const {
    return _private_bit_ref<_AP_W, _AP_S>(
        const_cast<ap_private<_AP_W, _AP_S>&>(*this), index.to_int());
  }

  INLINE _private_bit_ref<_AP_W, _AP_S> bit(int index) {
    return _private_bit_ref<_AP_W, _AP_S>(*this, index);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE _private_bit_ref<_AP_W, _AP_S> bit(const ap_private<_AP_W2, _AP_S2>& index) {
    return _private_bit_ref<_AP_W, _AP_S>(*this, index.to_int());
  }

  INLINE const _private_bit_ref<_AP_W, _AP_S> bit(int index) const {
    return _private_bit_ref<_AP_W, _AP_S>(
        const_cast<ap_private<_AP_W, _AP_S>&>(*this), index);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE const _private_bit_ref<_AP_W, _AP_S> bit(
      const ap_private<_AP_W2, _AP_S2>& index) const {
    return _private_bit_ref<_AP_W, _AP_S>(
        const_cast<ap_private<_AP_W, _AP_S>&>(*this), index.to_int());
  }

//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2,
//                       ap_private<_AP_W2, _AP_S2> >
//  concat(const ap_private<_AP_W2, _AP_S2>& a2) const {
//    return ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2,
//                         ap_private<_AP_W2, _AP_S2> >(
//        const_cast<ap_private<_AP_W, _AP_S>&>(*this),
//        const_cast<ap_private<_AP_W2, _AP_S2>&>(a2));
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2,
//                       ap_private<_AP_W2, _AP_S2> >
//  concat(ap_private<_AP_W2, _AP_S2>& a2) {
//    return ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2,
//                         ap_private<_AP_W2, _AP_S2> >(*this, a2);
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, ap_private, _AP_W2, ap_private<_AP_W2, _AP_S2> >
//  operator,(const ap_private<_AP_W2, _AP_S2> &a2) const {
//    return ap_concat_ref<_AP_W, ap_private, _AP_W2,
//                         ap_private<_AP_W2, _AP_S2> >(
//        const_cast<ap_private<_AP_W, _AP_S>&>(*this),
//        const_cast<ap_private<_AP_W2, _AP_S2>&>(a2));
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, ap_private, _AP_W2, ap_private<_AP_W2, _AP_S2> >
//  operator,(const ap_private<_AP_W2, _AP_S2> &a2) {
//    return ap_concat_ref<_AP_W, ap_private, _AP_W2,
//                         ap_private<_AP_W2, _AP_S2> >(
//        *this, const_cast<ap_private<_AP_W2, _AP_S2>&>(a2));
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, ap_private, _AP_W2, ap_private<_AP_W2, _AP_S2> >
//  operator,(ap_private<_AP_W2, _AP_S2> &a2) const {
//    return ap_concat_ref<_AP_W, ap_private, _AP_W2,
//                         ap_private<_AP_W2, _AP_S2> >(
//        const_cast<ap_private<_AP_W, _AP_S>&>(*this), a2);
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, ap_private, _AP_W2, ap_private<_AP_W2, _AP_S2> >
//  operator,(ap_private<_AP_W2, _AP_S2> &a2) {
//    return ap_concat_ref<_AP_W, ap_private, _AP_W2,
//                         ap_private<_AP_W2, _AP_S2> >(*this, a2);
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2,
//                       _private_range_ref<_AP_W2, _AP_S2> >
//  operator,(const _private_range_ref<_AP_W2, _AP_S2> &a2) const {
//    return ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2,
//                         _private_range_ref<_AP_W2, _AP_S2> >(
//        const_cast<ap_private<_AP_W, _AP_S>&>(*this),
//        const_cast<_private_range_ref<_AP_W2, _AP_S2>&>(a2));
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2,
//                       _private_range_ref<_AP_W2, _AP_S2> >
//  operator,(_private_range_ref<_AP_W2, _AP_S2> &a2) {
//    return ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2,
//                         _private_range_ref<_AP_W2, _AP_S2> >(*this, a2);
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, 1,
//                       _private_bit_ref<_AP_W2, _AP_S2> >
//  operator,(const _private_bit_ref<_AP_W2, _AP_S2> &a2) const {
//    return ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, 1,
//                         _private_bit_ref<_AP_W2, _AP_S2> >(
//        const_cast<ap_private<_AP_W, _AP_S>&>(*this),
//        const_cast<_private_bit_ref<_AP_W2, _AP_S2>&>(a2));
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, 1,
//                       _private_bit_ref<_AP_W2, _AP_S2> >
//  operator,(_private_bit_ref<_AP_W2, _AP_S2> &a2) {
//    return ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, 1,
//                         _private_bit_ref<_AP_W2, _AP_S2> >(*this, a2);
//  }
//
//  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
//  INLINE ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2 + _AP_W3,
//                       ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> >
//  operator,(const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> &a2) const {
//    return ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2 + _AP_W3,
//                         ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> >(
//        const_cast<ap_private<_AP_W, _AP_S>&>(*this),
//        const_cast<ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>&>(a2));
//  }
//
//  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
//  INLINE ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2 + _AP_W3,
//                       ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> >
//  operator,(ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> &a2) {
//    return ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2 + _AP_W3,
//                         ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> >(*this,
//                                                                         a2);
//  }
//
//  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
//            ap_o_mode _AP_O2, int _AP_N2>
//  INLINE ap_concat_ref<
//      _AP_W, ap_private, _AP_W2,
//      af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >
//  operator,(const af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>
//                &a2) const {
//    return ap_concat_ref<
//        _AP_W, ap_private, _AP_W2,
//        af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >(
//        const_cast<ap_private<_AP_W, _AP_S>&>(*this),
//        const_cast<
//            af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>&>(a2));
//  }
//
//  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
//            ap_o_mode _AP_O2, int _AP_N2>
//  INLINE ap_concat_ref<
//      _AP_W, ap_private, _AP_W2,
//      af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >
//  operator,(af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> &a2) {
//    return ap_concat_ref<
//        _AP_W, ap_private, _AP_W2,
//        af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >(*this,
//                                                                       a2);
//  }
//
//  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
//            ap_o_mode _AP_O2, int _AP_N2>
//  INLINE
//      ap_concat_ref<_AP_W, ap_private, 1,
//                    af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >
//      operator,(const af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>
//                    &a2) const {
//    return ap_concat_ref<
//        _AP_W, ap_private, 1,
//        af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >(
//        const_cast<ap_private<_AP_W, _AP_S>&>(*this),
//        const_cast<af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>&>(
//            a2));
//  }
//
//  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
//            ap_o_mode _AP_O2, int _AP_N2>
//  INLINE
//      ap_concat_ref<_AP_W, ap_private, 1,
//                    af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >
//      operator,(
//          af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> &a2) {
//    return ap_concat_ref<
//        _AP_W, ap_private, 1,
//        af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >(*this, a2);
//  }
//
//  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
//  INLINE ap_private<AP_MAX(_AP_W2 + _AP_W3, _AP_W), _AP_S> operator&(
//      const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>& a2) {
//    return *this & a2.get();
//  }
//
//  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
//  INLINE ap_private<AP_MAX(_AP_W2 + _AP_W3, _AP_W), _AP_S> operator|(
//      const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>& a2) {
//    return *this | a2.get();
//  }
//
//  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
//  INLINE ap_private<AP_MAX(_AP_W2 + _AP_W3, _AP_W), _AP_S> operator^(
//      const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>& a2) {
//    return *this ^ a2.get();
//  }

  // Reduce operation
  //-----------------------------------------------------------
  INLINE bool and_reduce() const { return (VAL & mask) == mask; }

  INLINE bool nand_reduce() const { return (VAL & mask) != mask; }

  INLINE bool or_reduce() const { return (bool)VAL; }

  INLINE bool nor_reduce() const { return VAL == 0; }

  INLINE bool xor_reduce() const {
    unsigned int i = countPopulation();
    return (i % 2) ? true : false;
  }

  INLINE bool xnor_reduce() const {
    unsigned int i = countPopulation();
    return (i % 2) ? false : true;
  }

  INLINE std::string to_string(uint8_t radix = 2, bool sign = false) const {
    return toString(radix, radix == 10 ? _AP_S : sign);
  }
}; // End of class ap_private <_AP_W, _AP_S, true>

template <int _AP_W, bool _AP_S>
std::string ap_private<_AP_W, _AP_S, true>::toString(uint8_t radix,
                                                     bool wantSigned) const {
  assert((radix == 10 || radix == 8 || radix == 16 || radix == 2) &&
         "Radix should be 2, 8, 10, or 16!");
  static const char* digits[] = {"0", "1", "2", "3", "4", "5", "6", "7",
                                 "8", "9", "a", "b", "c", "d", "e", "f"};
  std::string result;
  if (radix != 10) {
    // For the 2, 8 and 16 bit cases, we can just shift instead of divide
    // because the number of bits per digit (1,3 and 4 respectively) divides
    // equaly. We just shift until there value is zero.

    // First, check for a zero value and just short circuit the logic below.
    if (*this == (uint64_t)(0)) {
      // Always generate a radix indicator because fixed-point
      // formats require it.
      switch (radix) {
        case 2:
          result = "0b0";
          break;
        case 8:
          result = "0o0";
          break;
        case 16:
          result = "0x0";
          break;
        default:
          assert("invalid radix" && 0);
      }
    } else {
      ap_private<_AP_W, false, true> tmp(*this);
      size_t insert_at = 0;
      bool leading_zero = true;
      if (wantSigned && isNegative()) {
        // They want to print the signed version and it is a negative value
        // Flip the bits and add one to turn it into the equivalent positive
        // value and put a '-' in the result.
        tmp.flip();
        tmp++;
        result = "-";
        insert_at = 1;
        leading_zero = false;
      }
      switch (radix) {
        case 2:
          result += "0b";
          break;
        case 8:
          result += "0o";
          break;
        case 16:
          result += "0x";
          break;
        default:
          assert("invalid radix" && 0);
      }
      insert_at += 2;

      // Just shift tmp right for each digit width until it becomes zero
      uint32_t shift = (radix == 16 ? 4 : (radix == 8 ? 3 : 1));
      uint64_t mask = radix - 1;
      ap_private<_AP_W, false, true> zero(0);
      unsigned bits = 0;
      bool msb = false;
      while (tmp.ne(zero)) {
        unsigned digit = (unsigned)(tmp.get_VAL() & mask);
        result.insert(insert_at, digits[digit]);
        tmp = tmp.lshr(shift);
        bits++;
        msb = (digit >> (shift - 1)) == 1;
      }
      bits *= shift;
      if (bits < _AP_W && leading_zero && msb)
        result.insert(insert_at, digits[0]);
    }
    return result;
  }

  ap_private<_AP_W, false, true> tmp(*this);
  ap_private<6, false, true> divisor(radix);
  ap_private<_AP_W, _AP_S, true> zero(0);
  size_t insert_at = 0;
  if (wantSigned && isNegative()) {
    // They want to print the signed version and it is a negative value
    // Flip the bits and add one to turn it into the equivalent positive
    // value and put a '-' in the result.
    tmp.flip();
    tmp++;
    result = "-";
    insert_at = 1;
  }
  if (tmp == ap_private<_AP_W, false, true>(0ULL))
    result = "0";
  else
    while (tmp.ne(zero)) {
      ap_private<_AP_W, false, true> APdigit = tmp % divisor;
      ap_private<_AP_W, false, true> tmp2 = tmp / divisor;
      uint32_t digit = (uint32_t)(APdigit.getZExtValue());
      assert(digit < radix && "divide failed");
      result.insert(insert_at, digits[digit]);
      tmp = tmp2;
    }
  return result;

} // End of ap_private<_AP_W, _AP_S, true>::toString()

// bitwidth > 64
template <int _AP_W, bool _AP_S>
class ap_private<_AP_W, _AP_S, false> {
  // SFINAE pattern.  Only consider this class when _AP_W > 64
  const static bool valid = ap_private_enable_if<(_AP_W > 64)>::isValid;

#ifdef _MSC_VER
#pragma warning(disable : 4521 4522)
#endif
 public:
  enum { BitWidth = _AP_W, _AP_N = (_AP_W + 63) / 64 };
  static const int width = _AP_W;

 private:
  /// This constructor is used only internally for speed of construction of
  /// temporaries. It is unsafe for general use so it is not public.

  /* Constructors */
  /// Note that numWords can be smaller or larger than the corresponding bit
  /// width but any extraneous bits will be dropped.
  /// @param numWords the number of words in bigVal
  /// @param bigVal a sequence of words to form the initial value of the
  /// ap_private
  /// @brief Construct an ap_private, initialized as bigVal[].
  INLINE ap_private(uint32_t numWords, const uint64_t bigVal[]) {
    set_canary();
    assert(bigVal && "Null pointer detected!");
    {
      // Get memory, cleared to 0
      memset(pVal, 0, _AP_N * sizeof(uint64_t));

      // Calculate the number of words to copy
      uint32_t words = AESL_std::min<uint32_t>(numWords, _AP_N);
      // Copy the words from bigVal to pVal
      memcpy(pVal, bigVal, words * APINT_WORD_SIZE);
      if (words >= _AP_W) clearUnusedBits();
      // Make sure unused high bits are cleared
    }
    check_canary();
  }

  /// This constructor interprets Val as a string in the given radix. The
  /// interpretation stops when the first charater that is not suitable for the
  /// radix is encountered. Acceptable radix values are 2, 8, 10 and 16. It is
  /// an error for the value implied by the string to require more bits than
  /// numBits.
  /// @param val the string to be interpreted
  /// @param radix the radix of Val to use for the intepretation
  /// @brief Construct an ap_private from a string representation.
  INLINE ap_private(const std::string& val, uint8_t radix = 2) {
    set_canary();
    assert(!val.empty() && "The input string is empty.");
    const char* c_str = val.c_str();
    fromString(c_str, val.size(), radix);
    check_canary();
  }

  /// This constructor interprets the slen characters starting at StrStart as
  /// a string in the given radix. The interpretation stops when the first
  /// character that is not suitable for the radix is encountered. Acceptable
  /// radix values are 2, 8, 10 and 16. It is an error for the value implied by
  /// the string to require more bits than numBits.
  /// @param strStart the start of the string to be interpreted
  /// @param slen the maximum number of characters to interpret
  /// @param radix the radix to use for the conversion
  /// @brief Construct an ap_private from a string representation.
  /// This method does not consider whether it is negative or not.
  INLINE ap_private(const char strStart[], uint32_t slen, uint8_t radix) {
    set_canary();
    fromString(strStart, slen, radix);
    check_canary();
  }

  INLINE void report() {
    _AP_ERROR(_AP_W > MAX_MODE(AP_INT_MAX_W) * 1024,
              "ap_%sint<%d>: Bitwidth exceeds the "
              "default max value %d. Please use macro "
              "AP_INT_MAX_W to set a larger max value.",
              _AP_S ? "" : "u", _AP_W, MAX_MODE(AP_INT_MAX_W) * 1024);
  }
  /// This union is used to store the integer value. When the
  /// integer bit-width <= 64, it uses VAL, otherwise it uses pVal.

  /// This enum is used to hold the constants we needed for ap_private.
  // uint64_t VAL;    ///< Used to store the <= 64 bits integer value.
  uint64_t pVal[_AP_N]; ///< Used to store the >64 bits integer value.
#ifdef AP_CANARY
  uint64_t CANARY;
  INLINE void check_canary() { assert(CANARY == (uint64_t)0xDEADBEEFDEADBEEF); }
  INLINE void set_canary() { CANARY = (uint64_t)0xDEADBEEFDEADBEEF; }
#else
  INLINE void check_canary() {}
  INLINE void set_canary() {}
#endif

 public:
  typedef typename valtype<8, _AP_S>::Type ValType;
  typedef ap_private<_AP_W, _AP_S> Type;
  // FIXME remove friend type?
  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  friend struct ap_fixed_base;
  /// return type of variety of operations
  //----------------------------------------------------------
  template <int _AP_W2, bool _AP_S2>
  struct RType {
    enum {
      mult_w = _AP_W + _AP_W2,
      mult_s = _AP_S || _AP_S2,
      plus_w =
          AP_MAX(_AP_W + (_AP_S2 && !_AP_S), _AP_W2 + (_AP_S && !_AP_S2)) + 1,
      plus_s = _AP_S || _AP_S2,
      minus_w =
          AP_MAX(_AP_W + (_AP_S2 && !_AP_S), _AP_W2 + (_AP_S && !_AP_S2)) + 1,
      minus_s = true,
      div_w = _AP_W + _AP_S2,
      div_s = _AP_S || _AP_S2,
      mod_w = AP_MIN(_AP_W, _AP_W2 + (!_AP_S2 && _AP_S)),
      mod_s = _AP_S,
      logic_w = AP_MAX(_AP_W + (_AP_S2 && !_AP_S), _AP_W2 + (_AP_S && !_AP_S2)),
      logic_s = _AP_S || _AP_S2
    };
    typedef ap_private<mult_w, mult_s> mult;
    typedef ap_private<plus_w, plus_s> plus;
    typedef ap_private<minus_w, minus_s> minus;
    typedef ap_private<logic_w, logic_s> logic;
    typedef ap_private<div_w, div_s> div;
    typedef ap_private<mod_w, mod_s> mod;
    typedef ap_private<_AP_W, _AP_S> arg1;
    typedef bool reduce;
  };

  INLINE uint64_t& get_VAL(void) { return pVal[0]; }
  INLINE uint64_t get_VAL(void) const { return pVal[0]; }
  INLINE uint64_t get_VAL(void) const volatile { return pVal[0]; }
  INLINE void set_VAL(uint64_t value) { pVal[0] = value; }
  INLINE uint64_t& get_pVal(int index) { return pVal[index]; }
  INLINE uint64_t* get_pVal() { return pVal; }
  INLINE const uint64_t* get_pVal() const { return pVal; }
  INLINE uint64_t get_pVal(int index) const { return pVal[index]; }
  INLINE uint64_t* get_pVal() const volatile { return pVal; }
  INLINE uint64_t get_pVal(int index) const volatile { return pVal[index]; }
  INLINE void set_pVal(int i, uint64_t value) { pVal[i] = value; }

  /// This enum is used to hold the constants we needed for ap_private.
  enum {
    APINT_BITS_PER_WORD = sizeof(uint64_t) * 8, ///< Bits in a word
    APINT_WORD_SIZE = sizeof(uint64_t)          ///< Byte size of a word
  };

  enum {
    excess_bits = (_AP_W % APINT_BITS_PER_WORD)
                      ? APINT_BITS_PER_WORD - (_AP_W % APINT_BITS_PER_WORD)
                      : 0
  };
  static const uint64_t mask = ((uint64_t)~0ULL >> (excess_bits));

 public:
  // NOTE changed to explicit to be consistent with ap_private<W,S,true>
  explicit INLINE ap_private(const char* val) {
    set_canary();
    unsigned char radix = 10;
    std::string str = ap_private_ops::parseString(val, radix); // determine radix.
    std::string::size_type pos = str.find('.');
    if (pos != std::string::npos) str = str.substr(pos);
    ap_private ap_private_val(str, radix);
    operator=(ap_private_val);
    report();
    check_canary();
  }

  INLINE ap_private(const char* val, unsigned char rd) {
    set_canary();
    unsigned char radix = rd;
    std::string str = ap_private_ops::parseString(val, radix); // determine radix.
    std::string::size_type pos = str.find('.');
    if (pos != std::string::npos) str = str.substr(pos);
    ap_private ap_private_val(str, radix);
    operator=(ap_private_val);
    report();

    report();
    check_canary();
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_private(const _private_range_ref<_AP_W2, _AP_S2>& ref) {
    set_canary();
    *this = ref.get();
    report();
    check_canary();
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_private(const _private_bit_ref<_AP_W2, _AP_S2>& ref) {
    set_canary();
    *this = ((uint64_t)(bool)ref);
    report();
    check_canary();
  }

//  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
//  INLINE ap_private(const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>& ref) {
//    set_canary();
//    *this = ref.get();
//    report();
//    check_canary();
//  }
//
//  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
//            ap_o_mode _AP_O2, int _AP_N2>
//  INLINE ap_private(
//      const af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& val) {
//    set_canary();
//    *this = ((val.operator ap_private<_AP_W2, false>()));
//    report();
//    check_canary();
//  }
//
//  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
//            ap_o_mode _AP_O2, int _AP_N2>
//  INLINE ap_private(
//      const af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& val) {
//    set_canary();
//    *this = (uint64_t)(bool)val;
//    report();
//    check_canary();
//  }

  /// Simply makes *this a copy of that.
  /// @brief Copy Constructor.
  INLINE ap_private(const ap_private& that) {
      set_canary();
      memcpy(pVal, that.get_pVal(), _AP_N * APINT_WORD_SIZE);
      clearUnusedBits();
      check_canary();
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE ap_private(const ap_private<_AP_W1, _AP_S1, false>& that) {
    set_canary();
    operator=(that);
    check_canary();
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE ap_private(const volatile ap_private<_AP_W1, _AP_S1, false>& that) {
    set_canary();
    operator=(const_cast<const ap_private<_AP_W1, _AP_S1, false>&>(that));
    check_canary();
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE ap_private(const ap_private<_AP_W1, _AP_S1, true>& that) {
    set_canary();
    static const uint64_t that_sign_ext_mask =
        (_AP_W1 == APINT_BITS_PER_WORD)
            ? 0
            : ~0ULL >> (_AP_W1 % APINT_BITS_PER_WORD)
                           << (_AP_W1 % APINT_BITS_PER_WORD);
    if (that.isNegative()) {
      pVal[0] = that.get_VAL() | that_sign_ext_mask;
      memset(pVal + 1, ~0, sizeof(uint64_t) * (_AP_N - 1));
    } else {
      pVal[0] = that.get_VAL();
      memset(pVal + 1, 0, sizeof(uint64_t) * (_AP_N - 1));
    }
    clearUnusedBits();
    check_canary();
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE ap_private(const volatile ap_private<_AP_W1, _AP_S1, true>& that) {
    set_canary();
    operator=(const_cast<const ap_private<_AP_W1, _AP_S1, true>&>(that));
    check_canary();
  }

  /// @brief Destructor.
  // virtual ~ap_private() {}
  INLINE ~ap_private() { check_canary(); }

  /// @name Constructors
  /// @{

  /// Default constructor that creates an uninitialized ap_private.  This is
  /// useful
  ///  for object deserialization (pair this with the static method Read).
  INLINE ap_private() {
    set_canary();
    clearUnusedBits();
    check_canary();
  }

  INLINE ap_private(uint64_t* val, uint32_t bits = _AP_W) { assert(0); }
  INLINE ap_private(const uint64_t* const val, uint32_t bits) { assert(0); }

/// If isSigned is true then val is treated as if it were a signed value
/// (i.e. as an int64_t) and the appropriate sign extension to the bit width
/// will be done. Otherwise, no sign extension occurs (high order bits beyond
/// the range of val are zero filled).
/// @param numBits the bit width of the constructed ap_private
/// @param val the initial value of the ap_private
/// @param isSigned how to treat signedness of val
/// @brief Create a new ap_private of numBits width, initialized as val.
#define CTOR(TYPE, SIGNED)                                  \
  INLINE ap_private(TYPE val, bool isSigned = SIGNED) {     \
    set_canary();                                           \
    pVal[0] = (ValType)val;                                 \
    if (isSigned && int64_t(pVal[0]) < 0) {                 \
      memset(pVal + 1, ~0, sizeof(uint64_t) * (_AP_N - 1)); \
    } else {                                                \
      memset(pVal + 1, 0, sizeof(uint64_t) * (_AP_N - 1));  \
    }                                                       \
    clearUnusedBits();                                      \
    check_canary();                                         \
  }

  CTOR(bool, false)
  CTOR(char, CHAR_IS_SIGNED)
  CTOR(signed char, true)
  CTOR(unsigned char, false)
  CTOR(short, true)
  CTOR(unsigned short, false)
  CTOR(int, true)
  CTOR(unsigned int, false)
  CTOR(long, true)
  CTOR(unsigned long, false)
  CTOR(ap_slong, true)
  CTOR(ap_ulong, false)
#if 0
  CTOR(half, false)
  CTOR(float, false)
  CTOR(double, false)
#endif
#undef CTOR

  /// @returns true if the number of bits <= 64, false otherwise.
  /// @brief Determine if this ap_private just has one word to store value.
  INLINE bool isSingleWord() const { return false; }

  /// @returns the word position for the specified bit position.
  /// @brief Determine which word a bit is in.
  static INLINE uint32_t whichWord(uint32_t bitPosition) {
    //    return bitPosition / APINT_BITS_PER_WORD;
    return (bitPosition) >> 6;
  }

  /// @returns the bit position in a word for the specified bit position
  /// in the ap_private.
  /// @brief Determine which bit in a word a bit is in.
  static INLINE uint32_t whichBit(uint32_t bitPosition) {
    //    return bitPosition % APINT_BITS_PER_WORD;
    return bitPosition & 0x3f;
  }

  /// bit at a specific bit position. This is used to mask the bit in the
  /// corresponding word.
  /// @returns a uint64_t with only bit at "whichBit(bitPosition)" set
  /// @brief Get a single bit mask.
  static INLINE uint64_t maskBit(uint32_t bitPosition) {
    return 1ULL << (whichBit(bitPosition));
  }

  /// @returns the corresponding word for the specified bit position.
  /// @brief Get the word corresponding to a bit position
  INLINE uint64_t getWord(uint32_t bitPosition) const {
    return pVal[whichWord(bitPosition)];
  }

  /// This method is used internally to clear the to "N" bits in the high order
  /// word that are not used by the ap_private. This is needed after the most
  /// significant word is assigned a value to ensure that those bits are
  /// zero'd out.
  /// @brief Clear unused high order bits
  INLINE void clearUnusedBits(void) volatile
// just for clang compiler
#if defined(__clang__) && !defined(__CLANG_3_1__)
      __attribute__((no_sanitize("undefined")))
#endif
  {
    pVal[_AP_N - 1] =
        _AP_S ? ((((int64_t)pVal[_AP_N - 1]) << (excess_bits)) >> excess_bits)
              : (excess_bits
                     ? ((pVal[_AP_N - 1]) << (excess_bits)) >> (excess_bits)
                     : pVal[_AP_N - 1]);
  }

  INLINE void clearUnusedBitsToZero(void) { pVal[_AP_N - 1] &= mask; }

  INLINE void clearUnusedBitsToOne(void) { pVal[_AP_N - 1] |= mask; }

  /// This is used by the constructors that take string arguments.
  /// @brief Convert a char array into an ap_private
  INLINE void fromString(const char* str, uint32_t slen, uint8_t radix) {
    enum { numbits = _AP_W };
    bool isNeg = str[0] == '-';
    if (isNeg) {
      str++;
      slen--;
    }

    if (str[0] == '0' && (str[1] == 'b' || str[1] == 'B')) {
      //if(radix == 0) radix = 2;
      _AP_WARNING(radix != 2, "%s seems to have base %d, but %d given.", str, 2, radix);
      str += 2;
      slen -=2;
    } else if (str[0] == '0' && (str[1] == 'o' || str[1] == 'O')) {
      //if (radix == 0) radix = 8;
      _AP_WARNING(radix != 8, "%s seems to have base %d, but %d given.", str, 8, radix);
      str += 2;
      slen -=2;
    } else if (str[0] == '0' && (str[1] == 'x' || str[1] == 'X')) {
      //if (radix == 0) radix = 16;
      _AP_WARNING(radix != 16, "%s seems to have base %d, but %d given.", str, 16, radix);
      str += 2;
      slen -=2;
    } else if (str[0] == '0' && (str[1] == 'd' || str[1] == 'D')) {
      //if (radix == 0) radix = 10;
      _AP_WARNING(radix != 10, "%s seems to have base %d, but %d given.", str, 10, radix);
      str += 2;
      slen -=2;
    } else if (radix == 0) {
      //radix = 2; // XXX default value
    }

    // Check our assumptions here
    assert((radix == 10 || radix == 8 || radix == 16 || radix == 2) &&
           "Radix should be 2, 8, 10, or 16!");
    assert(str && "String is null?");

    // skip any leading zero
    while (*str == '0' && *(str + 1) != '\0') {
      str++;
      slen--;
    }
    assert((slen <= numbits || radix != 2) && "Insufficient bit width");
    assert(((slen - 1) * 3 <= numbits || radix != 8) &&
           "Insufficient bit width");
    assert(((slen - 1) * 4 <= numbits || radix != 16) &&
           "Insufficient bit width");
    assert((((slen - 1) * 64) / 22 <= numbits || radix != 10) &&
           "Insufficient bit width");

    // clear bits
    memset(pVal, 0, _AP_N * sizeof(uint64_t));

    // Figure out if we can shift instead of multiply
    uint32_t shift = (radix == 16 ? 4 : radix == 8 ? 3 : radix == 2 ? 1 : 0);

    // Set up an ap_private for the digit to add outside the loop so we don't
    // constantly construct/destruct it.
    uint64_t bigVal[_AP_N];
    memset(bigVal, 0, _AP_N * sizeof(uint64_t));
    ap_private<_AP_W, _AP_S> apdigit(getBitWidth(), bigVal);
    ap_private<_AP_W, _AP_S> apradix(radix);

    // Enter digit traversal loop
    for (unsigned i = 0; i < slen; i++) {
      // Get a digit
      uint32_t digit = 0;
      char cdigit = str[i];
      if (radix == 16) {
#define isxdigit(c)                                            \
  (((c) >= '0' && (c) <= '9') || ((c) >= 'a' && (c) <= 'f') || \
   ((c) >= 'A' && (c) <= 'F'))
#define isdigit(c) ((c) >= '0' && (c) <= '9')
        if (!isxdigit(cdigit)) assert(0 && "Invalid hex digit in string");
        if (isdigit(cdigit))
          digit = cdigit - '0';
        else if (cdigit >= 'a')
          digit = cdigit - 'a' + 10;
        else if (cdigit >= 'A')
          digit = cdigit - 'A' + 10;
        else
          assert(0 && "huh? we shouldn't get here");
      } else if (isdigit(cdigit)) {
        digit = cdigit - '0';
      } else if (cdigit != '\0') {
        assert(0 && "Invalid character in digit string");
      }
#undef isxdigit
#undef isdigit
      // Shift or multiply the value by the radix
      if (shift)
        *this <<= shift;
      else
        *this *= apradix;

      // Add in the digit we just interpreted
      apdigit.set_VAL(digit);
      *this += apdigit;
    }
    // If its negative, put it in two's complement form
    if (isNeg) {
      (*this)--;
      this->flip();
    }
    clearUnusedBits();
  }

  INLINE ap_private read() volatile { return *this; }

  INLINE void write(const ap_private& op2) volatile { *this = (op2); }

  INLINE operator ValType() const { return get_VAL(); }

  INLINE int to_uchar() const { return (unsigned char)get_VAL(); }

  INLINE int to_char() const { return (signed char)get_VAL(); }

  INLINE int to_ushort() const { return (unsigned short)get_VAL(); }

  INLINE int to_short() const { return (short)get_VAL(); }

  INLINE int to_int() const { return (int)get_VAL(); }

  INLINE unsigned to_uint() const { return (unsigned)get_VAL(); }

  INLINE long to_long() const { return (long)get_VAL(); }

  INLINE unsigned long to_ulong() const { return (unsigned long)get_VAL(); }

  INLINE ap_slong to_int64() const { return (ap_slong)get_VAL(); }

  INLINE ap_ulong to_uint64() const { return (ap_ulong)get_VAL(); }

  INLINE double to_double() const {
    if (isNegative())
      return roundToDouble(true);
    else
      return roundToDouble(false);
  }

  INLINE unsigned length() const { return _AP_W; }

  /*Reverse the contents of ap_private instance. I.e. LSB becomes MSB and vise
   * versa*/
  INLINE ap_private& reverse() {
    for (int i = 0; i < _AP_W / 2; ++i) {
      bool tmp = operator[](i);
      if (operator[](_AP_W - 1 - i))
        set(i);
      else
        clear(i);
      if (tmp)
        set(_AP_W - 1 - i);
      else
        clear(_AP_W - 1 - i);
    }
    clearUnusedBits();
    return *this;
  }

  /*Return true if the value of ap_private instance is zero*/
  INLINE bool iszero() const { return isMinValue(); }

  INLINE bool to_bool() const { return !iszero(); }

  /* x < 0 */
  INLINE bool sign() const {
    if (isNegative()) return true;
    return false;
  }

  /* x[i] = !x[i] */
  INLINE void invert(int i) {
    assert(i >= 0 && "Attempting to read bit with negative index");
    assert(i < _AP_W && "Attempting to read bit beyond MSB");
    flip(i);
  }

  /* x[i] */
  INLINE bool test(int i) const {
    assert(i >= 0 && "Attempting to read bit with negative index");
    assert(i < _AP_W && "Attempting to read bit beyond MSB");
    return operator[](i);
  }

  // Set the ith bit into v
  INLINE void set(int i, bool v) {
    assert(i >= 0 && "Attempting to write bit with negative index");
    assert(i < _AP_W && "Attempting to write bit beyond MSB");
    v ? set(i) : clear(i);
  }

  // Set the ith bit into v
  INLINE void set_bit(int i, bool v) {
    assert(i >= 0 && "Attempting to write bit with negative index");
    assert(i < _AP_W && "Attempting to write bit beyond MSB");
    v ? set(i) : clear(i);
  }

  // FIXME different argument for different action?
  INLINE ap_private& set(uint32_t bitPosition) {
    pVal[whichWord(bitPosition)] |= maskBit(bitPosition);
    clearUnusedBits();
    return *this;
  }

  INLINE void set() {
    for (int i = 0; i < _AP_N; ++i) pVal[i] = ~0ULL;
    clearUnusedBits();
  }

  // Get the value of ith bit
  INLINE bool get(int i) const {
    assert(i >= 0 && "Attempting to read bit with negative index");
    assert(i < _AP_W && "Attempting to read bit beyond MSB");
    return ((maskBit(i) & (pVal[whichWord(i)])) != 0);
  }

  // Get the value of ith bit
  INLINE bool get_bit(int i) const {
    assert(i >= 0 && "Attempting to read bit with negative index");
    assert(i < _AP_W && "Attempting to read bit beyond MSB");
    return ((maskBit(i) & (pVal[whichWord(i)])) != 0);
  }

  // This is used for sc_lv and sc_bv, which is implemented by sc_uint
  // Rotate an ap_private object n places to the left
  INLINE void lrotate(int n) {
    assert(n >= 0 && "Attempting to shift negative index");
    assert(n < _AP_W && "Shift value larger than bit width");
    operator=(shl(n) | lshr(_AP_W - n));
  }

  // This is used for sc_lv and sc_bv, which is implemented by sc_uint
  // Rotate an ap_private object n places to the right
  INLINE void rrotate(int n) {
    assert(n >= 0 && "Attempting to shift negative index");
    assert(n < _AP_W && "Shift value larger than bit width");
    operator=(lshr(n) | shl(_AP_W - n));
  }

  /// Set the given bit to 0 whose position is given as "bitPosition".
  /// @brief Set a given bit to 0.
  INLINE ap_private& clear(uint32_t bitPosition) {
    pVal[whichWord(bitPosition)] &= ~maskBit(bitPosition);
    clearUnusedBits();
    return *this;
  }

  /// @brief Set every bit to 0.
  INLINE void clear() { memset(pVal, 0, _AP_N * APINT_WORD_SIZE); }

  /// @brief Toggle every bit to its opposite value.
  ap_private& flip() {
    for (int i = 0; i < _AP_N; ++i) pVal[i] ^= ~0ULL;
    clearUnusedBits();
    return *this;
  }

  /// @brief Toggles a given bit to its opposite value.
  INLINE ap_private& flip(uint32_t bitPosition) {
    assert(bitPosition < BitWidth && "Out of the bit-width range!");
    set_bit(bitPosition, !get_bit(bitPosition));
    return *this;
  }

  // complements every bit
  INLINE void b_not() { flip(); }

  INLINE ap_private getLoBits(uint32_t numBits) const {
    return ap_private_ops::lshr(ap_private_ops::shl(*this, _AP_W - numBits),
                                _AP_W - numBits);
  }

  INLINE ap_private getHiBits(uint32_t numBits) const {
    return ap_private_ops::lshr(*this, _AP_W - numBits);
  }

  // Binary Arithmetic
  //-----------------------------------------------------------

//  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
//  INLINE ap_private<AP_MAX(_AP_W2 + _AP_W3, _AP_W), _AP_S> operator&(
//      const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>& a2) {
//    return *this & a2.get();
//  }
//
//  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
//  INLINE ap_private<AP_MAX(_AP_W2 + _AP_W3, _AP_W), _AP_S> operator|(
//      const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>& a2) {
//    return *this | a2.get();
//  }
//
//  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
//  INLINE ap_private<AP_MAX(_AP_W2 + _AP_W3, _AP_W), _AP_S> operator^(
//      const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>& a2) {
//    return *this ^ a2.get();
//  }

/// Arithmetic assign
//-------------------------------------------------------------

#define OP_BIN_LOGIC_ASSIGN_AP(Sym)                                            \
  template <int _AP_W1, bool _AP_S1>                                           \
  INLINE ap_private& operator Sym(const ap_private<_AP_W1, _AP_S1>& RHS) {     \
    const int _AP_N1 = ap_private<_AP_W1, _AP_S1>::_AP_N;                      \
    uint32_t numWords = AESL_std::min((int)_AP_N, _AP_N1);                     \
    uint32_t i;                                                                \
    if (_AP_W != _AP_W1)                                                       \
      fprintf(stderr,                                                          \
              "Warning! Bitsize mismach for ap_[u]int " #Sym " ap_[u]int.\n"); \
    for (i = 0; i < numWords; ++i) pVal[i] Sym RHS.get_pVal(i);                \
    if (_AP_N1 < _AP_N) {                                                      \
      uint64_t ext = RHS.isNegative() ? ~0ULL : 0;                             \
      for (; i < _AP_N; i++) pVal[i] Sym ext;                                  \
    }                                                                          \
    clearUnusedBits();                                                         \
    return *this;                                                              \
  }

  OP_BIN_LOGIC_ASSIGN_AP(&=);
  OP_BIN_LOGIC_ASSIGN_AP(|=);
  OP_BIN_LOGIC_ASSIGN_AP(^=);
#undef OP_BIN_LOGIC_ASSIGN_AP

  /// Adds the RHS APint to this ap_private.
  /// @returns this, after addition of RHS.
  /// @brief Addition assignment operator.
  template <int _AP_W1, bool _AP_S1>
  INLINE ap_private& operator+=(const ap_private<_AP_W1, _AP_S1>& RHS) {
    const int _AP_N1 = ap_private<_AP_W1, _AP_S1>::_AP_N;
    uint64_t RHSpVal[_AP_N1];
    for (int i = 0; i < _AP_N1; ++i) RHSpVal[i] = RHS.get_pVal(i);
    ap_private_ops::add(pVal, pVal, RHSpVal, _AP_N, _AP_N, _AP_N1, _AP_S,
                        _AP_S1);
    clearUnusedBits();
    return *this;
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE ap_private& operator-=(const ap_private<_AP_W1, _AP_S1>& RHS) {
    const int _AP_N1 = ap_private<_AP_W1, _AP_S1>::_AP_N;
    uint64_t RHSpVal[_AP_N1];
    for (int i = 0; i < _AP_N1; ++i) RHSpVal[i] = RHS.get_pVal(i);
    ap_private_ops::sub(pVal, pVal, RHSpVal, _AP_N, _AP_N, _AP_N1, _AP_S,
                        _AP_S1);
    clearUnusedBits();
    return *this;
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE ap_private& operator*=(const ap_private<_AP_W1, _AP_S1>& RHS) {
    // Get some bit facts about LHS and check for zero
    uint32_t lhsBits = getActiveBits();
    uint32_t lhsWords = !lhsBits ? 0 : whichWord(lhsBits - 1) + 1;
    if (!lhsWords) {
      // 0 * X ===> 0
      return *this;
    }

    ap_private dupRHS = RHS;
    // Get some bit facts about RHS and check for zero
    uint32_t rhsBits = dupRHS.getActiveBits();
    uint32_t rhsWords = !rhsBits ? 0 : whichWord(rhsBits - 1) + 1;
    if (!rhsWords) {
      // X * 0 ===> 0
      clear();
      return *this;
    }

    // Allocate space for the result
    uint32_t destWords = rhsWords + lhsWords;
    uint64_t* dest = (uint64_t*)malloc(destWords * sizeof(uint64_t));

    // Perform the long multiply
    ap_private_ops::mul(dest, pVal, lhsWords, dupRHS.get_pVal(), rhsWords,
                        destWords);

    // Copy result back into *this
    clear();
    uint32_t wordsToCopy = destWords >= _AP_N ? _AP_N : destWords;

    memcpy(pVal, dest, wordsToCopy * APINT_WORD_SIZE);

    uint64_t ext = (isNegative() ^ RHS.isNegative()) ? ~0ULL : 0ULL;
    for (int i = wordsToCopy; i < _AP_N; i++) pVal[i] = ext;
    clearUnusedBits();
    // delete dest array and return
    free(dest);
    return *this;
  }

#define OP_ASSIGN_AP(Sym)                                                    \
  template <int _AP_W2, bool _AP_S2>                                         \
  INLINE ap_private& operator Sym##=(const ap_private<_AP_W2, _AP_S2>& op) { \
    *this = operator Sym(op);                                                \
    return *this;                                                            \
  }

  OP_ASSIGN_AP(/)
  OP_ASSIGN_AP(%)
#undef OP_ASSIGN_AP

#define OP_BIN_LOGIC_AP(Sym)                                                  \
  template <int _AP_W1, bool _AP_S1>                                          \
  INLINE typename RType<_AP_W1, _AP_S1>::logic operator Sym(                  \
      const ap_private<_AP_W1, _AP_S1>& RHS) const {                          \
    enum {                                                                    \
      numWords = (RType<_AP_W1, _AP_S1>::logic_w + APINT_BITS_PER_WORD - 1) / \
                 APINT_BITS_PER_WORD                                          \
    };                                                                        \
    typename RType<_AP_W1, _AP_S1>::logic Result;                             \
    uint32_t i;                                                               \
    const int _AP_N1 = ap_private<_AP_W1, _AP_S1>::_AP_N;                     \
    uint32_t min_N = std::min((int)_AP_N, _AP_N1);                            \
    uint32_t max_N = std::max((int)_AP_N, _AP_N1);                            \
    for (i = 0; i < min_N; ++i)                                               \
      Result.set_pVal(i, pVal[i] Sym RHS.get_pVal(i));                        \
    if (numWords > i) {                                                       \
      uint64_t ext = ((_AP_N < _AP_N1 && isNegative()) ||                     \
                      (_AP_N1 < _AP_N && RHS.isNegative()))                   \
                         ? ~0ULL                                              \
                         : 0;                                                 \
      if (_AP_N > _AP_N1)                                                     \
        for (; i < max_N; i++) Result.set_pVal(i, pVal[i] Sym ext);           \
      else                                                                    \
        for (; i < max_N; i++) Result.set_pVal(i, RHS.get_pVal(i) Sym ext);   \
      if (numWords > i) {                                                     \
        uint64_t ext2 = ((_AP_N > _AP_N1 && isNegative()) ||                  \
                         (_AP_N1 > _AP_N && RHS.isNegative()))                \
                            ? ~0ULL                                           \
                            : 0;                                              \
        Result.set_pVal(i, ext Sym ext2);                                     \
      }                                                                       \
    }                                                                         \
    Result.clearUnusedBits();                                                 \
    return Result;                                                            \
  }

  OP_BIN_LOGIC_AP(|);
  OP_BIN_LOGIC_AP(&);
  OP_BIN_LOGIC_AP(^);

#undef OP_BIN_LOGIC_AP

  template <int _AP_W1, bool _AP_S1>
  INLINE typename RType<_AP_W1, _AP_S1>::plus operator+(
      const ap_private<_AP_W1, _AP_S1>& RHS) const {
    typename RType<_AP_W1, _AP_S1>::plus Result, lhs(*this), rhs(RHS);
    const int Result_AP_N = (RType<_AP_W1, _AP_S1>::plus_w + 63) / 64;
    ap_private_ops::add(Result.get_pVal(), lhs.get_pVal(), rhs.get_pVal(),
                        Result_AP_N, Result_AP_N, Result_AP_N, _AP_S, _AP_S1);
    Result.clearUnusedBits();
    return Result;
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE typename RType<_AP_W1, _AP_S1>::minus operator-(
      const ap_private<_AP_W1, _AP_S1>& RHS) const {
    typename RType<_AP_W1, _AP_S1>::minus Result, lhs(*this), rhs(RHS);
    const int Result_AP_N = (RType<_AP_W1, _AP_S1>::minus_w + 63) / 64;
    ap_private_ops::sub(Result.get_pVal(), lhs.get_pVal(), rhs.get_pVal(),
                        Result_AP_N, Result_AP_N, Result_AP_N, _AP_S, _AP_S1);
    Result.clearUnusedBits();
    return Result;
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE typename RType<_AP_W1, _AP_S1>::mult operator*(
      const ap_private<_AP_W1, _AP_S1>& RHS) const {
    typename RType<_AP_W1, _AP_S1>::mult temp = *this;
    temp *= RHS;
    return temp;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE typename RType<_AP_W2, _AP_S2>::div operator/(
      const ap_private<_AP_W2, _AP_S2>& op) const {
    ap_private<AP_MAX(_AP_W + (_AP_S || _AP_S2), _AP_W2 + (_AP_S || _AP_S2)),
               (_AP_W > _AP_W2 ? _AP_S
                               : (_AP_W2 > _AP_W ? _AP_S2 : _AP_S || _AP_S2))>
        lhs = *this;
    ap_private<AP_MAX(_AP_W + (_AP_S || _AP_S2), _AP_W2 + (_AP_S || _AP_S2)),
               (_AP_W > _AP_W2 ? _AP_S
                               : (_AP_W2 > _AP_W ? _AP_S2 : _AP_S || _AP_S2))>
        rhs = op;
    return typename RType<_AP_W2, _AP_S2>::div(
        (_AP_S || _AP_S2) ? lhs.sdiv(rhs) : lhs.udiv(rhs));
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE typename RType<_AP_W2, _AP_S2>::mod operator%(
      const ap_private<_AP_W2, _AP_S2>& op) const {
    ap_private<AP_MAX(_AP_W + (_AP_S || _AP_S2), _AP_W2 + (_AP_S || _AP_S2)),
               (_AP_W > _AP_W2 ? _AP_S
                               : (_AP_W2 > _AP_W ? _AP_S2 : _AP_S || _AP_S2))>
        lhs = *this;
    ap_private<AP_MAX(_AP_W + (_AP_S || _AP_S2), _AP_W2 + (_AP_S || _AP_S2)),
               (_AP_W > _AP_W2 ? _AP_S
                               : (_AP_W2 > _AP_W ? _AP_S2 : _AP_S || _AP_S2))>
        rhs = op;
    typename RType<_AP_W2, _AP_S2>::mod res =
        typename RType<_AP_W2, _AP_S2>::mod(_AP_S ? lhs.srem(rhs)
                                                  : lhs.urem(rhs));
    return res;
  }

#define OP_LEFT_SHIFT_CTYPE(TYPE, SIGNED)             \
  INLINE ap_private operator<<(const TYPE op) const { \
    if (op >= _AP_W) return ap_private(0);            \
    if (SIGNED && op < 0) return *this >> (0 - op);   \
    return shl(op);                                   \
  }

  OP_LEFT_SHIFT_CTYPE(int, true)
  // OP_LEFT_SHIFT_CTYPE(bool, false)
  OP_LEFT_SHIFT_CTYPE(signed char, true)
  OP_LEFT_SHIFT_CTYPE(unsigned char, false)
  OP_LEFT_SHIFT_CTYPE(short, true)
  OP_LEFT_SHIFT_CTYPE(unsigned short, false)
  OP_LEFT_SHIFT_CTYPE(unsigned int, false)
  OP_LEFT_SHIFT_CTYPE(long, true)
  OP_LEFT_SHIFT_CTYPE(unsigned long, false)
  OP_LEFT_SHIFT_CTYPE(unsigned long long, false)
  OP_LEFT_SHIFT_CTYPE(long long, true)
#if 0
  OP_LEFT_SHIFT_CTYPE(half, false)
  OP_LEFT_SHIFT_CTYPE(float, false)
  OP_LEFT_SHIFT_CTYPE(double, false)
#endif
#undef OP_LEFT_SHIFT_CTYPE

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_private operator<<(const ap_private<_AP_W2, _AP_S2>& op2) const {
    if (_AP_S2 == false) {
      uint32_t sh = op2.to_uint();
      return *this << sh;
    } else {
      int sh = op2.to_int();
      return *this << sh;
    }
  }

#define OP_RIGHT_SHIFT_CTYPE(TYPE, SIGNED)            \
  INLINE ap_private operator>>(const TYPE op) const { \
    if (op >= _AP_W) {                                \
      if (isNegative())                               \
        return ap_private(-1);                        \
      else                                            \
        return ap_private(0);                         \
    }                                                 \
    if ((SIGNED) && op < 0) return *this << (0 - op); \
    if (_AP_S)                                        \
      return ashr(op);                                \
    else                                              \
      return lshr(op);                                \
  }

  // OP_RIGHT_SHIFT_CTYPE(bool, false)
  OP_RIGHT_SHIFT_CTYPE(char, CHAR_IS_SIGNED)
  OP_RIGHT_SHIFT_CTYPE(signed char, true)
  OP_RIGHT_SHIFT_CTYPE(unsigned char, false)
  OP_RIGHT_SHIFT_CTYPE(short, true)
  OP_RIGHT_SHIFT_CTYPE(unsigned short, false)
  OP_RIGHT_SHIFT_CTYPE(int, true)
  OP_RIGHT_SHIFT_CTYPE(unsigned int, false)
  OP_RIGHT_SHIFT_CTYPE(long, true)
  OP_RIGHT_SHIFT_CTYPE(unsigned long, false)
  OP_RIGHT_SHIFT_CTYPE(unsigned long long, false)
  OP_RIGHT_SHIFT_CTYPE(long long, true)
#if 0
  OP_RIGHT_SHIFT_CTYPE(half, false)
  OP_RIGHT_SHIFT_CTYPE(float, false)
  OP_RIGHT_SHIFT_CTYPE(double, false)
#endif
#undef OP_RIGHT_SHIFT_CTYPE

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_private operator>>(const ap_private<_AP_W2, _AP_S2>& op2) const {
    if (_AP_S2 == false) {
      uint32_t sh = op2.to_uint();
      return *this >> sh;
    } else {
      int sh = op2.to_int();
      return *this >> sh;
    }
  }

  /// Shift assign
  //------------------------------------------------------------------
  // TODO call clearUnusedBits ?
#define OP_ASSIGN_AP(Sym)                                                    \
  template <int _AP_W2, bool _AP_S2>                                         \
  INLINE ap_private& operator Sym##=(int op) {                               \
    *this = operator Sym(op);                                                \
    return *this;                                                            \
  }                                                                          \
  INLINE ap_private& operator Sym##=(unsigned int op) {                      \
    *this = operator Sym(op);                                                \
    return *this;                                                            \
  }                                                                          \
  template <int _AP_W2, bool _AP_S2>                                         \
  INLINE ap_private& operator Sym##=(const ap_private<_AP_W2, _AP_S2>& op) { \
    *this = operator Sym(op);                                                \
    return *this;                                                            \
  }
  OP_ASSIGN_AP(>>)
  OP_ASSIGN_AP(<<)
#undef OP_ASSIGN_AP

  /// Comparisons
  //-----------------------------------------------------------------
  INLINE bool operator==(const ap_private& RHS) const {
    // Get some facts about the number of bits used in the two operands.
    uint32_t n1 = getActiveBits();
    uint32_t n2 = RHS.getActiveBits();

    // If the number of bits isn't the same, they aren't equal
    if (n1 != n2) return false;

    // If the number of bits fits in a word, we only need to compare the low
    // word.
    if (n1 <= APINT_BITS_PER_WORD) return pVal[0] == RHS.get_pVal(0);

    // Otherwise, compare everything
    for (int i = whichWord(n1 - 1); i >= 0; --i)
      if (pVal[i] != RHS.get_pVal(i)) return false;
    return true;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator==(const ap_private<_AP_W2, _AP_S2>& op) const {
    enum {
      _AP_MAX_W = AP_MAX(_AP_W, _AP_W2),
    };
    ap_private<_AP_MAX_W, false> lhs(*this);
    ap_private<_AP_MAX_W, false> rhs(op);
    return lhs == rhs;
  }

  INLINE bool operator==(uint64_t Val) const {
    uint32_t n = getActiveBits();
    if (n <= APINT_BITS_PER_WORD)
      return pVal[0] == Val;
    else
      return false;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator!=(const ap_private<_AP_W2, _AP_S2>& op) const {
    return !(*this == op);
  }

  template <bool _AP_S1>
  INLINE bool operator!=(const ap_private<_AP_W, _AP_S1>& RHS) const {
    return !((*this) == RHS);
  }

  INLINE bool operator!=(uint64_t Val) const { return !((*this) == Val); }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator<=(const ap_private<_AP_W2, _AP_S2>& op) const {
    return !(*this > op);
  }

  INLINE bool operator<(const ap_private& op) const {
    return _AP_S ? slt(op) : ult(op);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator<(const ap_private<_AP_W2, _AP_S2>& op) const {
    enum {
      _AP_MAX_W = AP_MAX(_AP_W + (_AP_S || _AP_S2), _AP_W2 + (_AP_S || _AP_S2))
    };
    ap_private<_AP_MAX_W, _AP_S> lhs(*this);
    ap_private<_AP_MAX_W, _AP_S2> rhs(op);
    if (_AP_S == _AP_S2)
      return _AP_S ? lhs.slt(rhs) : lhs.ult(rhs);
    else if (_AP_S)
      if (_AP_W2 >= _AP_W)
        return lhs.ult(rhs);
      else
        return lhs.slt(rhs);
    else if (_AP_W >= _AP_W2)
      return lhs.ult(rhs);
    else
      return lhs.slt(rhs);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator>=(const ap_private<_AP_W2, _AP_S2>& op) const {
    return !(*this < op);
  }

  INLINE bool operator>(const ap_private& op) const {
    return _AP_S ? sgt(op) : ugt(op);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator>(const ap_private<_AP_W2, _AP_S2>& op) const {
    enum {
      _AP_MAX_W = AP_MAX(_AP_W + (_AP_S || _AP_S2), _AP_W2 + (_AP_S || _AP_S2))
    };
    ap_private<_AP_MAX_W, _AP_S> lhs(*this);
    ap_private<_AP_MAX_W, _AP_S2> rhs(op);
    if (_AP_S == _AP_S2)
      return _AP_S ? lhs.sgt(rhs) : lhs.ugt(rhs);
    else if (_AP_S)
      if (_AP_W2 >= _AP_W)
        return lhs.ugt(rhs);
      else
        return lhs.sgt(rhs);
    else if (_AP_W >= _AP_W2)
      return lhs.ugt(rhs);
    else
      return lhs.sgt(rhs);
  }

  /// Bit and Part Select
  //--------------------------------------------------------------
  INLINE _private_range_ref<_AP_W, _AP_S> operator()(int Hi, int Lo) {
    return _private_range_ref<_AP_W, _AP_S>(this, Hi, Lo);
  }

  INLINE _private_range_ref<_AP_W, _AP_S> operator()(int Hi, int Lo) const {
    return _private_range_ref<_AP_W, _AP_S>(
        const_cast<ap_private<_AP_W, _AP_S>*>(this), Hi, Lo);
  }

  INLINE _private_range_ref<_AP_W, _AP_S> range(int Hi, int Lo) const {
    return _private_range_ref<_AP_W, _AP_S>(
        (const_cast<ap_private<_AP_W, _AP_S>*>(this)), Hi, Lo);
  }

  INLINE _private_range_ref<_AP_W, _AP_S> range(int Hi, int Lo) {
    return _private_range_ref<_AP_W, _AP_S>(this, Hi, Lo);
  }

  template <int _AP_W2, bool _AP_S2, int _AP_W3, bool _AP_S3>
  INLINE _private_range_ref<_AP_W, _AP_S> range(
      const ap_private<_AP_W2, _AP_S2>& HiIdx,
      const ap_private<_AP_W3, _AP_S3>& LoIdx) {
    int Hi = HiIdx.to_int();
    int Lo = LoIdx.to_int();
    return _private_range_ref<_AP_W, _AP_S>(this, Hi, Lo);
  }

  template <int _AP_W2, bool _AP_S2, int _AP_W3, bool _AP_S3>
  INLINE _private_range_ref<_AP_W, _AP_S> operator()(
      const ap_private<_AP_W2, _AP_S2>& HiIdx,
      const ap_private<_AP_W3, _AP_S3>& LoIdx) {
    int Hi = HiIdx.to_int();
    int Lo = LoIdx.to_int();
    return _private_range_ref<_AP_W, _AP_S>(this, Hi, Lo);
  }

  template <int _AP_W2, bool _AP_S2, int _AP_W3, bool _AP_S3>
  INLINE _private_range_ref<_AP_W, _AP_S> range(
      const ap_private<_AP_W2, _AP_S2>& HiIdx,
      const ap_private<_AP_W3, _AP_S3>& LoIdx) const {
    int Hi = HiIdx.to_int();
    int Lo = LoIdx.to_int();
    return _private_range_ref<_AP_W, _AP_S>(const_cast<ap_private*>(this), Hi, Lo);
  }

  template <int _AP_W2, bool _AP_S2, int _AP_W3, bool _AP_S3>
  INLINE _private_range_ref<_AP_W, _AP_S> operator()(
      const ap_private<_AP_W2, _AP_S2>& HiIdx,
      const ap_private<_AP_W3, _AP_S3>& LoIdx) const {
    int Hi = HiIdx.to_int();
    int Lo = LoIdx.to_int();
    return this->range(Hi, Lo);
  }

  INLINE _private_bit_ref<_AP_W, _AP_S> operator[](int index) {
    return _private_bit_ref<_AP_W, _AP_S>(*this, index);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE _private_bit_ref<_AP_W, _AP_S> operator[](
      const ap_private<_AP_W2, _AP_S2>& index) {
    return _private_bit_ref<_AP_W, _AP_S>(*this, index.to_int());
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE const _private_bit_ref<_AP_W, _AP_S> operator[](
      const ap_private<_AP_W2, _AP_S2>& index) const {
    return _private_bit_ref<_AP_W, _AP_S>(
        const_cast<ap_private<_AP_W, _AP_S>&>(*this), index.to_int());
  }

  INLINE const _private_bit_ref<_AP_W, _AP_S> operator[](int index) const {
    return _private_bit_ref<_AP_W, _AP_S>(
        const_cast<ap_private<_AP_W, _AP_S>&>(*this), index);
  }

  INLINE _private_bit_ref<_AP_W, _AP_S> bit(int index) {
    return _private_bit_ref<_AP_W, _AP_S>(*this, index);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE _private_bit_ref<_AP_W, _AP_S> bit(const ap_private<_AP_W2, _AP_S2>& index) {
    return _private_bit_ref<_AP_W, _AP_S>(*this, index.to_int());
  }

  INLINE const _private_bit_ref<_AP_W, _AP_S> bit(int index) const {
    return _private_bit_ref<_AP_W, _AP_S>(
        const_cast<ap_private<_AP_W, _AP_S>&>(*this), index);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE const _private_bit_ref<_AP_W, _AP_S> bit(
      const ap_private<_AP_W2, _AP_S2>& index) const {
    return _private_bit_ref<_AP_W, _AP_S>(
        const_cast<ap_private<_AP_W, _AP_S>&>(*this), index.to_int());
  }

//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2,
//                       ap_private<_AP_W2, _AP_S2> >
//  concat(ap_private<_AP_W2, _AP_S2>& a2) {
//    return ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2,
//                         ap_private<_AP_W2, _AP_S2> >(*this, a2);
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2,
//                       ap_private<_AP_W2, _AP_S2> >
//  concat(const ap_private<_AP_W2, _AP_S2>& a2) const {
//    return ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2,
//                         ap_private<_AP_W2, _AP_S2> >(
//        const_cast<ap_private<_AP_W, _AP_S>&>(*this),
//        const_cast<ap_private<_AP_W2, _AP_S2>&>(a2));
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, ap_private, _AP_W2, ap_private<_AP_W2, _AP_S2> >
//  operator,(ap_private<_AP_W2, _AP_S2> &a2) {
//    return ap_concat_ref<_AP_W, ap_private, _AP_W2,
//                         ap_private<_AP_W2, _AP_S2> >(*this, a2);
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, ap_private, _AP_W2, ap_private<_AP_W2, _AP_S2> >
//  operator,(ap_private<_AP_W2, _AP_S2> &a2) const {
//    return ap_concat_ref<_AP_W, ap_private, _AP_W2,
//                         ap_private<_AP_W2, _AP_S2> >(
//        const_cast<ap_private<_AP_W, _AP_S>&>(*this), a2);
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, ap_private, _AP_W2, ap_private<_AP_W2, _AP_S2> >
//  operator,(const ap_private<_AP_W2, _AP_S2> &a2) {
//    return ap_concat_ref<_AP_W, ap_private, _AP_W2,
//                         ap_private<_AP_W2, _AP_S2> >(
//        *this, const_cast<ap_private<_AP_W2, _AP_S2>&>(a2));
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, ap_private, _AP_W2, ap_private<_AP_W2, _AP_S2> >
//  operator,(const ap_private<_AP_W2, _AP_S2> &a2) const {
//    return ap_concat_ref<_AP_W, ap_private, _AP_W2,
//                         ap_private<_AP_W2, _AP_S2> >(
//        const_cast<ap_private<_AP_W, _AP_S>&>(*this),
//        const_cast<ap_private<_AP_W2, _AP_S2>&>(a2));
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2,
//                       _private_range_ref<_AP_W2, _AP_S2> >
//  operator,(const _private_range_ref<_AP_W2, _AP_S2> &a2) const {
//    return ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2,
//                         _private_range_ref<_AP_W2, _AP_S2> >(
//        const_cast<ap_private<_AP_W, _AP_S>&>(*this),
//        const_cast<_private_range_ref<_AP_W2, _AP_S2>&>(a2));
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2,
//                       _private_range_ref<_AP_W2, _AP_S2> >
//  operator,(_private_range_ref<_AP_W2, _AP_S2> &a2) {
//    return ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2,
//                         _private_range_ref<_AP_W2, _AP_S2> >(*this, a2);
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, 1,
//                       _private_bit_ref<_AP_W2, _AP_S2> >
//  operator,(const _private_bit_ref<_AP_W2, _AP_S2> &a2) const {
//    return ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, 1,
//                         _private_bit_ref<_AP_W2, _AP_S2> >(
//        const_cast<ap_private<_AP_W, _AP_S>&>(*this),
//        const_cast<_private_bit_ref<_AP_W2, _AP_S2>&>(a2));
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, 1,
//                       _private_bit_ref<_AP_W2, _AP_S2> >
//  operator,(_private_bit_ref<_AP_W2, _AP_S2> &a2) {
//    return ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, 1,
//                         _private_bit_ref<_AP_W2, _AP_S2> >(*this, a2);
//  }
//
//  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
//  INLINE ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2 + _AP_W3,
//                       ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> >
//  operator,(const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> &a2) const {
//    return ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2 + _AP_W3,
//                         ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> >(
//        const_cast<ap_private<_AP_W, _AP_S>&>(*this),
//        const_cast<ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>&>(a2));
//  }
//
//  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
//  INLINE ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2 + _AP_W3,
//                       ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> >
//  operator,(ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> &a2) {
//    return ap_concat_ref<_AP_W, ap_private<_AP_W, _AP_S>, _AP_W2 + _AP_W3,
//                         ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> >(*this,
//                                                                         a2);
//  }
//
//  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
//            ap_o_mode _AP_O2, int _AP_N2>
//  INLINE ap_concat_ref<
//      _AP_W, ap_private, _AP_W2,
//      af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >
//  operator,(const af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>
//                &a2) const {
//    return ap_concat_ref<
//        _AP_W, ap_private, _AP_W2,
//        af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >(
//        const_cast<ap_private<_AP_W, _AP_S>&>(*this),
//        const_cast<
//            af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>&>(a2));
//  }
//
//  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
//            ap_o_mode _AP_O2, int _AP_N2>
//  INLINE ap_concat_ref<
//      _AP_W, ap_private, _AP_W2,
//      af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >
//  operator,(af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> &a2) {
//    return ap_concat_ref<
//        _AP_W, ap_private, _AP_W2,
//        af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >(*this,
//                                                                       a2);
//  }
//
//  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
//            ap_o_mode _AP_O2, int _AP_N2>
//  INLINE
//      ap_concat_ref<_AP_W, ap_private, 1,
//                    af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >
//      operator,(const af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>
//                    &a2) const {
//    return ap_concat_ref<
//        _AP_W, ap_private, 1,
//        af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >(
//        const_cast<ap_private<_AP_W, _AP_S>&>(*this),
//        const_cast<af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>&>(
//            a2));
//  }
//
//  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
//            ap_o_mode _AP_O2, int _AP_N2>
//  INLINE
//      ap_concat_ref<_AP_W, ap_private, 1,
//                    af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >
//      operator,(
//          af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> &a2) {
//    return ap_concat_ref<
//        _AP_W, ap_private, 1,
//        af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >(*this, a2);
//  }

  INLINE ap_private<_AP_W, false> get() const {
    ap_private<_AP_W, false> ret(*this);
    return ret;
  }

  template <int _AP_W3>
  INLINE void set(const ap_private<_AP_W3, false>& val) {
    operator=(ap_private<_AP_W3, _AP_S>(val));
  }

  ///
  /// @name Value Tests
  ///
  /// This tests the high bit of this ap_private to determine if it is set.
  /// @returns true if this ap_private is negative, false otherwise
  /// @brief Determine sign of this ap_private.
  INLINE bool isNegative() const {
    // just for get rid of warnings
    enum { shift = (_AP_W - APINT_BITS_PER_WORD * (_AP_N - 1) - 1) };
    static const uint64_t mask = 1ULL << (shift);
    return _AP_S && (pVal[_AP_N - 1] & mask);
  }

  /// This tests the high bit of the ap_private to determine if it is unset.
  /// @brief Determine if this ap_private Value is positive (not negative).
  INLINE bool isPositive() const { return !isNegative(); }

  /// This tests if the value of this ap_private is strictly positive (> 0).
  /// @returns true if this ap_private is Positive and not zero.
  /// @brief Determine if this ap_private Value is strictly positive.
  INLINE bool isStrictlyPositive() const {
    return isPositive() && (*this) != 0;
  }

  /// This checks to see if the value has all bits of the ap_private are set or
  /// not.
  /// @brief Determine if all bits are set
  INLINE bool isAllOnesValue() const { return countPopulation() == _AP_W; }

  /// This checks to see if the value of this ap_private is the maximum unsigned
  /// value for the ap_private's bit width.
  /// @brief Determine if this is the largest unsigned value.
  INLINE bool isMaxValue() const { return countPopulation() == _AP_W; }

  /// This checks to see if the value of this ap_private is the maximum signed
  /// value for the ap_private's bit width.
  /// @brief Determine if this is the largest signed value.
  INLINE bool isMaxSignedValue() const {
    return !isNegative() && countPopulation() == _AP_W - 1;
  }

  /// This checks to see if the value of this ap_private is the minimum unsigned
  /// value for the ap_private's bit width.
  /// @brief Determine if this is the smallest unsigned value.
  INLINE bool isMinValue() const { return countPopulation() == 0; }

  /// This checks to see if the value of this ap_private is the minimum signed
  /// value for the ap_private's bit width.
  /// @brief Determine if this is the smallest signed value.
  INLINE bool isMinSignedValue() const {
    return isNegative() && countPopulation() == 1;
  }

  /// This function returns a pointer to the internal storage of the ap_private.
  /// This is useful for writing out the ap_private in binary form without any
  /// conversions.
  INLINE const uint64_t* getRawData() const { return &pVal[0]; }

  // Square Root - this method computes and returns the square root of "this".
  // Three mechanisms are used for computation. For small values (<= 5 bits),
  // a table lookup is done. This gets some performance for common cases. For
  // values using less than 52 bits, the value is converted to double and then
  // the libc sqrt function is called. The result is rounded and then converted
  // back to a uint64_t which is then used to construct the result. Finally,
  // the Babylonian method for computing square roots is used.
  INLINE ap_private sqrt() const {
    // Determine the magnitude of the value.
    uint32_t magnitude = getActiveBits();

    // Use a fast table for some small values. This also gets rid of some
    // rounding errors in libc sqrt for small values.
    if (magnitude <= 5) {
      static const uint8_t results[32] = {
          /*     0 */ 0,
          /*  1- 2 */ 1, 1,
          /*  3- 6 */ 2, 2, 2, 2,
          /*  7-12 */ 3, 3, 3, 3, 3, 3,
          /* 13-20 */ 4, 4, 4, 4, 4, 4, 4, 4,
          /* 21-30 */ 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
          /*    31 */ 6};
      return ap_private<_AP_W, _AP_S>(/*BitWidth,*/ results[get_VAL()]);
    }

    // If the magnitude of the value fits in less than 52 bits (the precision of
    // an IEEE double precision floating point value), then we can use the
    // libc sqrt function which will probably use a hardware sqrt computation.
    // This should be faster than the algorithm below.
    if (magnitude < 52) {
#ifdef _MSC_VER
      // Amazingly, VC++ doesn't have round().
      return ap_private<_AP_W, _AP_S>(/*BitWidth,*/
                                      uint64_t(::sqrt(double(get_VAL()))) +
                                      0.5);
#else
      return ap_private<_AP_W, _AP_S>(/*BitWidth,*/
                                      uint64_t(
                                          ::round(::sqrt(double(get_VAL())))));
#endif
    }

    // Okay, all the short cuts are exhausted. We must compute it. The following
    // is a classical Babylonian method for computing the square root. This code
    // was adapted to APINt from a wikipedia article on such computations.
    // See http://www.wikipedia.org/ and go to the page named
    // Calculate_an_integer_square_root.
    uint32_t nbits = BitWidth, i = 4;
    ap_private<_AP_W, _AP_S> testy(16);
    ap_private<_AP_W, _AP_S> x_old(/*BitWidth,*/ 1);
    ap_private<_AP_W, _AP_S> x_new(0);
    ap_private<_AP_W, _AP_S> two(/*BitWidth,*/ 2);

    // Select a good starting value using binary logarithms.
    for (;; i += 2, testy = testy.shl(2))
      if (i >= nbits || this->ule(testy)) {
        x_old = x_old.shl(i / 2);
        break;
      }

    // Use the Babylonian method to arrive at the integer square root:
    for (;;) {
      x_new = (this->udiv(x_old) + x_old).udiv(two);
      if (x_old.ule(x_new)) break;
      x_old = x_new;
    }

    // Make sure we return the closest approximation
    // NOTE: The rounding calculation below is correct. It will produce an
    // off-by-one discrepancy with results from pari/gp. That discrepancy has
    // been
    // determined to be a rounding issue with pari/gp as it begins to use a
    // floating point representation after 192 bits. There are no discrepancies
    // between this algorithm and pari/gp for bit widths < 192 bits.
    ap_private<_AP_W, _AP_S> square(x_old * x_old);
    ap_private<_AP_W, _AP_S> nextSquare((x_old + 1) * (x_old + 1));
    if (this->ult(square))
      return x_old;
    else if (this->ule(nextSquare)) {
      ap_private<_AP_W, _AP_S> midpoint((nextSquare - square).udiv(two));
      ap_private<_AP_W, _AP_S> offset(*this - square);
      if (offset.ult(midpoint))
        return x_old;
      else
        return x_old + 1;
    } else
      assert(0 && "Error in ap_private<_AP_W, _AP_S>::sqrt computation");
    return x_old + 1;
  }

  ///
  /// @Assignment Operators
  ///
  /// @returns *this after assignment of RHS.
  /// @brief Copy assignment operator.
  INLINE ap_private& operator=(const ap_private& RHS) {
    if (this != &RHS) memcpy(pVal, RHS.get_pVal(), _AP_N * APINT_WORD_SIZE);
    clearUnusedBits();
    return *this;
  }
  INLINE ap_private& operator=(const volatile ap_private& RHS) {
    if (this != &RHS)
      for (int i = 0; i < _AP_N; ++i) pVal[i] = RHS.get_pVal(i);
    clearUnusedBits();
    return *this;
  }
  INLINE void operator=(const ap_private& RHS) volatile {
    if (this != &RHS)
      for (int i = 0; i < _AP_N; ++i) pVal[i] = RHS.get_pVal(i);
    clearUnusedBits();
  }
  INLINE void operator=(const volatile ap_private& RHS) volatile {
    if (this != &RHS)
      for (int i = 0; i < _AP_N; ++i) pVal[i] = RHS.get_pVal(i);
    clearUnusedBits();
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE ap_private& operator=(const ap_private<_AP_W1, _AP_S1>& RHS) {
    if (_AP_S1)
      cpSextOrTrunc(RHS);
    else
      cpZextOrTrunc(RHS);
    clearUnusedBits();
    return *this;
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE ap_private& operator=(const volatile ap_private<_AP_W1, _AP_S1>& RHS) {
    if (_AP_S1)
      cpSextOrTrunc(RHS);
    else
      cpZextOrTrunc(RHS);
    clearUnusedBits();
    return *this;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE ap_private& operator=(const _private_range_ref<_AP_W2, _AP_S2>& op2) {
    *this = ap_private<_AP_W2, false>(op2);
    return *this;
  }

#if 0
    template<int _AP_W1, bool _AP_S1>
    INLINE ap_private& operator=(const ap_private<_AP_W1, _AP_S1, true>& RHS) {
        static const uint64_t that_sign_ext_mask = (_AP_W1==APINT_BITS_PER_WORD)?0:~0ULL>>(_AP_W1%APINT_BITS_PER_WORD)<<(_AP_W1%APINT_BITS_PER_WORD);
        if (RHS.isNegative()) {
            pVal[0] = RHS.get_VAL() | that_sign_ext_mask;
            memset(pVal+1,~0, APINT_WORD_SIZE*(_AP_N-1));
        } else {
            pVal[0] = RHS.get_VAL();
            memset(pVal+1, 0, APINT_WORD_SIZE*(_AP_N-1));
        }
        clearUnusedBits();
        return *this;
    }

    template<int _AP_W1, bool _AP_S1>
    INLINE ap_private& operator=(const volatile ap_private<_AP_W1, _AP_S1, true>& RHS) {
        static const uint64_t that_sign_ext_mask = (_AP_W1==APINT_BITS_PER_WORD)?0:~0ULL>>(_AP_W1%APINT_BITS_PER_WORD)<<(_AP_W1%APINT_BITS_PER_WORD);
        if (RHS.isNegative()) {
            pVal[0] = RHS.get_VAL() | that_sign_ext_mask;
            memset(pVal+1,~0, APINT_WORD_SIZE*(_AP_N-1));
        } else {
            pVal[0] = RHS.get_VAL();
            memset(pVal+1, 0, APINT_WORD_SIZE*(_AP_N-1));
        }
        clearUnusedBits();
        return *this;
    }
#endif

/// from all c types.
#define ASSIGN_OP_FROM_INT(C_TYPE, _AP_W2, _AP_S2) \
  INLINE ap_private& operator=(const C_TYPE rhs) { \
    ap_private<(_AP_W2), (_AP_S2)> tmp = rhs;      \
    operator=(tmp);                                \
    return *this;                                  \
  }

  ASSIGN_OP_FROM_INT(bool, 1, false)
  ASSIGN_OP_FROM_INT(char, 8, CHAR_IS_SIGNED)
  ASSIGN_OP_FROM_INT(signed char, 8, true)
  ASSIGN_OP_FROM_INT(unsigned char, 8, false)
  ASSIGN_OP_FROM_INT(short, sizeof(short) * 8, true)
  ASSIGN_OP_FROM_INT(unsigned short, sizeof(unsigned short) * 8, false)
  ASSIGN_OP_FROM_INT(int, sizeof(int) * 8, true)
  ASSIGN_OP_FROM_INT(unsigned int, sizeof(unsigned int) * 8, false)
  ASSIGN_OP_FROM_INT(long, sizeof(long) * 8, true)
  ASSIGN_OP_FROM_INT(unsigned long, sizeof(unsigned long) * 8, false)
  ASSIGN_OP_FROM_INT(ap_slong, sizeof(ap_slong) * 8, true)
  ASSIGN_OP_FROM_INT(ap_ulong, sizeof(ap_ulong) * 8, false)
#undef ASSIGN_OP_FROM_INT

  /// from c string.
  // XXX this is a must, to prevent pointer being converted to bool.
  INLINE ap_private& operator=(const char* s) {
    ap_private tmp(s); // XXX direct initialization, as ctor is explicit.
    operator=(tmp);
    return *this;
  }

  ///
  /// @name Unary Operators
  ///
  /// @returns a new ap_private value representing *this incremented by one
  /// @brief Postfix increment operator.
  INLINE const ap_private operator++(int) {
    ap_private API(*this);
    ++(*this);
    return API;
  }

  /// @returns *this incremented by one
  /// @brief Prefix increment operator.
  INLINE ap_private& operator++() {
    ap_private_ops::add_1(pVal, pVal, _AP_N, 1);
    clearUnusedBits();
    return *this;
  }

  /// @returns a new ap_private representing *this decremented by one.
  /// @brief Postfix decrement operator.
  INLINE const ap_private operator--(int) {
    ap_private API(*this);
    --(*this);
    return API;
  }

  /// @returns *this decremented by one.
  /// @brief Prefix decrement operator.
  INLINE ap_private& operator--() {
    ap_private_ops::sub_1(pVal, _AP_N, 1);
    clearUnusedBits();
    return *this;
  }

  /// Performs a bitwise complement operation on this ap_private.
  /// @returns an ap_private that is the bitwise complement of *this
  /// @brief Unary bitwise complement operator.
  INLINE ap_private<_AP_W + !_AP_S, true> operator~() const {
    ap_private<_AP_W + !_AP_S, true> Result(*this);
    Result.flip();
    return Result;
  }

  /// Negates *this using two's complement logic.
  /// @returns An ap_private value representing the negation of *this.
  /// @brief Unary negation operator
  INLINE typename RType<1, false>::minus operator-() const {
    return ap_private<1, false>(0) - (*this);
  }

  /// Performs logical negation operation on this ap_private.
  /// @returns true if *this is zero, false otherwise.
  /// @brief Logical negation operator.
  INLINE bool operator!() const {
    for (int i = 0; i < _AP_N; ++i)
      if (pVal[i]) return false;
    return true;
  }

  template <bool _AP_S1>
  INLINE ap_private<_AP_W, _AP_S || _AP_S1> And(
      const ap_private<_AP_W, _AP_S1>& RHS) const {
    return this->operator&(RHS);
  }
  template <bool _AP_S1>
  INLINE ap_private Or(const ap_private<_AP_W, _AP_S1>& RHS) const {
    return this->operator|(RHS);
  }
  template <bool _AP_S1>
  INLINE ap_private Xor(const ap_private<_AP_W, _AP_S1>& RHS) const {
    return this->operator^(RHS);
  }

  INLINE ap_private Mul(const ap_private& RHS) const {
    ap_private Result(*this);
    Result *= RHS;
    return Result;
  }

  INLINE ap_private Add(const ap_private& RHS) const {
    ap_private Result(0);
    ap_private_ops::add(Result.get_pVal(), pVal, RHS.get_pVal(), _AP_N, _AP_N,
                        _AP_N, _AP_S, _AP_S);
    Result.clearUnusedBits();
    return Result;
  }

  INLINE ap_private Sub(const ap_private& RHS) const {
    ap_private Result(0);
    ap_private_ops::sub(Result.get_pVal(), pVal, RHS.get_pVal(), _AP_N, _AP_N,
                        _AP_N, _AP_S, _AP_S);
    Result.clearUnusedBits();
    return Result;
  }

  /// Arithmetic right-shift this ap_private by shiftAmt.
  /// @brief Arithmetic right-shift function.
  INLINE ap_private ashr(uint32_t shiftAmt) const {
    assert(shiftAmt <= BitWidth && "Invalid shift amount, too big");
    // Handle a degenerate case
    if (shiftAmt == 0) return ap_private(*this);

    // If all the bits were shifted out, the result is, technically, undefined.
    // We return -1 if it was negative, 0 otherwise. We check this early to
    // avoid
    // issues in the algorithm below.
    if (shiftAmt == BitWidth) {
      if (isNegative())
        return ap_private(-1);
      else
        return ap_private(0);
    }

    // Create some space for the result.
    ap_private Retval(0);
    uint64_t* val = Retval.get_pVal();

    // Compute some values needed by the following shift algorithms
    uint32_t wordShift =
        shiftAmt % APINT_BITS_PER_WORD;               // bits to shift per word
    uint32_t offset = shiftAmt / APINT_BITS_PER_WORD; // word offset for shift
    uint32_t breakWord = _AP_N - 1 - offset;          // last word affected
    uint32_t bitsInWord = whichBit(BitWidth); // how many bits in last word?
    if (bitsInWord == 0) bitsInWord = APINT_BITS_PER_WORD;

    // If we are shifting whole words, just move whole words
    if (wordShift == 0) {
      // Move the words containing significant bits
      for (uint32_t i = 0; i <= breakWord; ++i)
        val[i] = pVal[i + offset]; // move whole word

      // Adjust the top significant word for sign bit fill, if negative
      if (isNegative())
        if (bitsInWord < APINT_BITS_PER_WORD)
          val[breakWord] |= ~0ULL << (bitsInWord); // set high bits
    } else {
      // Shift the low order words
      for (uint32_t i = 0; i < breakWord; ++i) {
        // This combines the shifted corresponding word with the low bits from
        // the next word (shifted into this word's high bits).
        val[i] = ((pVal[i + offset]) >> (wordShift));
        val[i] |= ((pVal[i + offset + 1]) << (APINT_BITS_PER_WORD - wordShift));
      }

      // Shift the break word. In this case there are no bits from the next word
      // to include in this word.
      val[breakWord] = (pVal[breakWord + offset]) >> (wordShift);

      // Deal with sign extenstion in the break word, and possibly the word
      // before
      // it.
      if (isNegative()) {
        if (wordShift > bitsInWord) {
          if (breakWord > 0)
            val[breakWord - 1] |=
                ~0ULL << (APINT_BITS_PER_WORD - (wordShift - bitsInWord));
          val[breakWord] |= ~0ULL;
        } else
          val[breakWord] |= (~0ULL << (bitsInWord - wordShift));
      }
    }

    // Remaining words are 0 or -1, just assign them.
    uint64_t fillValue = (isNegative() ? ~0ULL : 0);
    for (int i = breakWord + 1; i < _AP_N; ++i) val[i] = fillValue;
    Retval.clearUnusedBits();
    return Retval;
  }

  /// Logical right-shift this ap_private by shiftAmt.
  /// @brief Logical right-shift function.
  INLINE ap_private lshr(uint32_t shiftAmt) const {
    // If all the bits were shifted out, the result is 0. This avoids issues
    // with shifting by the size of the integer type, which produces undefined
    // results. We define these "undefined results" to always be 0.
    if (shiftAmt == BitWidth) return ap_private(0);

    // If none of the bits are shifted out, the result is *this. This avoids
    // issues with shifting byt he size of the integer type, which produces
    // undefined results in the code below. This is also an optimization.
    if (shiftAmt == 0) return ap_private(*this);

    // Create some space for the result.
    ap_private Retval(0);
    uint64_t* val = Retval.get_pVal();

    // If we are shifting less than a word, compute the shift with a simple
    // carry
    if (shiftAmt < APINT_BITS_PER_WORD) {
      uint64_t carry = 0;
      for (int i = _AP_N - 1; i >= 0; --i) {
        val[i] = ((pVal[i]) >> (shiftAmt)) | carry;
        carry = (pVal[i]) << (APINT_BITS_PER_WORD - shiftAmt);
      }
      Retval.clearUnusedBits();
      return Retval;
    }

    // Compute some values needed by the remaining shift algorithms
    uint32_t wordShift = shiftAmt % APINT_BITS_PER_WORD;
    uint32_t offset = shiftAmt / APINT_BITS_PER_WORD;

    // If we are shifting whole words, just move whole words
    if (wordShift == 0) {
      for (uint32_t i = 0; i < _AP_N - offset; ++i) val[i] = pVal[i + offset];
      for (uint32_t i = _AP_N - offset; i < _AP_N; i++) val[i] = 0;
      Retval.clearUnusedBits();
      return Retval;
    }

    // Shift the low order words
    uint32_t breakWord = _AP_N - offset - 1;
    for (uint32_t i = 0; i < breakWord; ++i)
      val[i] = ((pVal[i + offset]) >> (wordShift)) |
               ((pVal[i + offset + 1]) << (APINT_BITS_PER_WORD - wordShift));
    // Shift the break word.
    val[breakWord] = (pVal[breakWord + offset]) >> (wordShift);

    // Remaining words are 0
    for (int i = breakWord + 1; i < _AP_N; ++i) val[i] = 0;
    Retval.clearUnusedBits();
    return Retval;
  }

  /// Left-shift this ap_private by shiftAmt.
  /// @brief Left-shift function.
  INLINE ap_private shl(uint32_t shiftAmt) const {
    assert(shiftAmt <= BitWidth && "Invalid shift amount, too big");
    // If all the bits were shifted out, the result is 0. This avoids issues
    // with shifting by the size of the integer type, which produces undefined
    // results. We define these "undefined results" to always be 0.
    if (shiftAmt == BitWidth) return ap_private(0);

    // If none of the bits are shifted out, the result is *this. This avoids a
    // lshr by the words size in the loop below which can produce incorrect
    // results. It also avoids the expensive computation below for a common
    // case.
    if (shiftAmt == 0) return ap_private(*this);

    // Create some space for the result.
    ap_private Retval(0);
    uint64_t* val = Retval.get_pVal();
    // If we are shifting less than a word, do it the easy way
    if (shiftAmt < APINT_BITS_PER_WORD) {
      uint64_t carry = 0;
      for (int i = 0; i < _AP_N; i++) {
        val[i] = ((pVal[i]) << (shiftAmt)) | carry;
        carry = (pVal[i]) >> (APINT_BITS_PER_WORD - shiftAmt);
      }
      Retval.clearUnusedBits();
      return Retval;
    }

    // Compute some values needed by the remaining shift algorithms
    uint32_t wordShift = shiftAmt % APINT_BITS_PER_WORD;
    uint32_t offset = shiftAmt / APINT_BITS_PER_WORD;

    // If we are shifting whole words, just move whole words
    if (wordShift == 0) {
      for (uint32_t i = 0; i < offset; i++) val[i] = 0;
      for (int i = offset; i < _AP_N; i++) val[i] = pVal[i - offset];
      Retval.clearUnusedBits();
      return Retval;
    }

    // Copy whole words from this to Result.
    uint32_t i = _AP_N - 1;
    for (; i > offset; --i)
      val[i] = (pVal[i - offset]) << (wordShift) |
               (pVal[i - offset - 1]) >> (APINT_BITS_PER_WORD - wordShift);
    val[offset] = (pVal[0]) << (wordShift);
    for (i = 0; i < offset; ++i) val[i] = 0;
    Retval.clearUnusedBits();
    return Retval;
  }

  INLINE ap_private rotl(uint32_t rotateAmt) const {
    if (rotateAmt == 0) return ap_private(*this);
    // Don't get too fancy, just use existing shift/or facilities
    ap_private hi(*this);
    ap_private lo(*this);
    hi.shl(rotateAmt);
    lo.lshr(BitWidth - rotateAmt);
    return hi | lo;
  }

  INLINE ap_private rotr(uint32_t rotateAmt) const {
    if (rotateAmt == 0) return ap_private(*this);
    // Don't get too fancy, just use existing shift/or facilities
    ap_private hi(*this);
    ap_private lo(*this);
    lo.lshr(rotateAmt);
    hi.shl(BitWidth - rotateAmt);
    return hi | lo;
  }

  /// Perform an unsigned divide operation on this ap_private by RHS. Both this
  /// and
  /// RHS are treated as unsigned quantities for purposes of this division.
  /// @returns a new ap_private value containing the division result
  /// @brief Unsigned division operation.
  INLINE ap_private udiv(const ap_private& RHS) const {
    // Get some facts about the LHS and RHS number of bits and words
    uint32_t rhsBits = RHS.getActiveBits();
    uint32_t rhsWords = !rhsBits ? 0 : (whichWord(rhsBits - 1) + 1);
    assert(rhsWords && "Divided by zero???");
    uint32_t lhsBits = this->getActiveBits();
    uint32_t lhsWords = !lhsBits ? 0 : (whichWord(lhsBits - 1) + 1);

    // Deal with some degenerate cases
    if (!lhsWords)
      // 0 / X ===> 0
      return ap_private(0);
    else if (lhsWords < rhsWords || this->ult(RHS)) {
      // X / Y ===> 0, iff X < Y
      return ap_private(0);
    } else if (*this == RHS) {
      // X / X ===> 1
      return ap_private(1);
    } else if (lhsWords == 1 && rhsWords == 1) {
      // All high words are zero, just use native divide
      return ap_private(this->pVal[0] / RHS.get_pVal(0));
    }

    // We have to compute it the hard way. Invoke the Knuth divide algorithm.
    ap_private Quotient(0); // to hold result.
    ap_private_ops::divide(*this, lhsWords, RHS, rhsWords, &Quotient,
                           (ap_private*)0);
    return Quotient;
  }

  /// Signed divide this ap_private by ap_private RHS.
  /// @brief Signed division function for ap_private.
  INLINE ap_private sdiv(const ap_private& RHS) const {
    if (isNegative())
      if (RHS.isNegative())
        return (-(*this)).udiv(-RHS);
      else
        return -((-(*this)).udiv(RHS));
    else if (RHS.isNegative())
      return -(this->udiv((ap_private)(-RHS)));
    return this->udiv(RHS);
  }

  /// Perform an unsigned remainder operation on this ap_private with RHS being
  /// the
  /// divisor. Both this and RHS are treated as unsigned quantities for purposes
  /// of this operation. Note that this is a true remainder operation and not
  /// a modulo operation because the sign follows the sign of the dividend
  /// which is *this.
  /// @returns a new ap_private value containing the remainder result
  /// @brief Unsigned remainder operation.
  INLINE ap_private urem(const ap_private& RHS) const {
    // Get some facts about the LHS
    uint32_t lhsBits = getActiveBits();
    uint32_t lhsWords = !lhsBits ? 0 : (whichWord(lhsBits - 1) + 1);

    // Get some facts about the RHS
    uint32_t rhsBits = RHS.getActiveBits();
    uint32_t rhsWords = !rhsBits ? 0 : (whichWord(rhsBits - 1) + 1);
    assert(rhsWords && "Performing remainder operation by zero ???");

    // Check the degenerate cases
    if (lhsWords == 0) {
      // 0 % Y ===> 0
      return ap_private(0);
    } else if (lhsWords < rhsWords || this->ult(RHS)) {
      // X % Y ===> X, iff X < Y
      return *this;
    } else if (*this == RHS) {
      // X % X == 0;
      return ap_private(0);
    } else if (lhsWords == 1) {
      // All high words are zero, just use native remainder
      return ap_private(pVal[0] % RHS.get_pVal(0));
    }

    // We have to compute it the hard way. Invoke the Knuth divide algorithm.
    ap_private Remainder(0);
    ap_private_ops::divide(*this, lhsWords, RHS, rhsWords, (ap_private*)(0),
                           &Remainder);
    return Remainder;
  }

  INLINE ap_private urem(uint64_t RHS) const {
    // Get some facts about the LHS
    uint32_t lhsBits = getActiveBits();
    uint32_t lhsWords = !lhsBits ? 0 : (whichWord(lhsBits - 1) + 1);
    // Get some facts about the RHS
    uint32_t rhsWords = 1; //! rhsBits ? 0 : (ap_private<_AP_W,
                           //! _AP_S>::whichWord(rhsBits - 1) + 1);
    assert(rhsWords && "Performing remainder operation by zero ???");
    // Check the degenerate cases
    if (lhsWords == 0) {
      // 0 % Y ===> 0
      return ap_private(0);
    } else if (lhsWords < rhsWords || this->ult(RHS)) {
      // X % Y ===> X, iff X < Y
      return *this;
    } else if (*this == RHS) {
      // X % X == 0;
      return ap_private(0);
    } else if (lhsWords == 1) {
      // All high words are zero, just use native remainder
      return ap_private(pVal[0] % RHS);
    }

    // We have to compute it the hard way. Invoke the Knuth divide algorithm.
    ap_private Remainder(0);
    divide(*this, lhsWords, RHS, (ap_private*)(0), &Remainder);
    return Remainder;
  }

  /// Signed remainder operation on ap_private.
  /// @brief Function for signed remainder operation.
  INLINE ap_private srem(const ap_private& RHS) const {
    if (isNegative()) {
      ap_private lhs = -(*this);
      if (RHS.isNegative()) {
        ap_private rhs = -RHS;
        return -(lhs.urem(rhs));
      } else
        return -(lhs.urem(RHS));
    } else if (RHS.isNegative()) {
      ap_private rhs = -RHS;
      return this->urem(rhs);
    }
    return this->urem(RHS);
  }

  /// Signed remainder operation on ap_private.
  /// @brief Function for signed remainder operation.
  INLINE ap_private srem(int64_t RHS) const {
    if (isNegative())
      if (RHS < 0)
        return -((-(*this)).urem(-RHS));
      else
        return -((-(*this)).urem(RHS));
    else if (RHS < 0)
      return this->urem(-RHS);
    return this->urem(RHS);
  }

  /// Compares this ap_private with RHS for the validity of the equality
  /// relationship.
  /// @returns true if *this == Val
  /// @brief Equality comparison.
  template <bool _AP_S1>
  INLINE bool eq(const ap_private<_AP_W, _AP_S1>& RHS) const {
    return (*this) == RHS;
  }

  /// Compares this ap_private with RHS for the validity of the inequality
  /// relationship.
  /// @returns true if *this != Val
  /// @brief Inequality comparison
  template <bool _AP_S1>
  INLINE bool ne(const ap_private<_AP_W, _AP_S1>& RHS) const {
    return !((*this) == RHS);
  }

  /// Regards both *this and RHS as unsigned quantities and compares them for
  /// the validity of the less-than relationship.
  /// @returns true if *this < RHS when both are considered unsigned.
  /// @brief Unsigned less than comparison
  template <bool _AP_S1>
  INLINE bool ult(const ap_private<_AP_W, _AP_S1>& RHS) const {
    // Get active bit length of both operands
    uint32_t n1 = getActiveBits();
    uint32_t n2 = RHS.getActiveBits();

    // If magnitude of LHS is less than RHS, return true.
    if (n1 < n2) return true;

    // If magnitude of RHS is greather than LHS, return false.
    if (n2 < n1) return false;

    // If they bot fit in a word, just compare the low order word
    if (n1 <= APINT_BITS_PER_WORD && n2 <= APINT_BITS_PER_WORD)
      return pVal[0] < RHS.get_pVal(0);

    // Otherwise, compare all words
    uint32_t topWord = whichWord(AESL_std::max(n1, n2) - 1);
    for (int i = topWord; i >= 0; --i) {
      if (pVal[i] > RHS.get_pVal(i)) return false;
      if (pVal[i] < RHS.get_pVal(i)) return true;
    }
    return false;
  }

  INLINE bool ult(uint64_t RHS) const {
    // Get active bit length of both operands
    uint32_t n1 = getActiveBits();
    uint32_t n2 =
        64 - ap_private_ops::CountLeadingZeros_64(RHS); // RHS.getActiveBits();

    // If magnitude of LHS is less than RHS, return true.
    if (n1 < n2) return true;

    // If magnitude of RHS is greather than LHS, return false.
    if (n2 < n1) return false;

    // If they bot fit in a word, just compare the low order word
    if (n1 <= APINT_BITS_PER_WORD && n2 <= APINT_BITS_PER_WORD)
      return pVal[0] < RHS;
    assert(0);
  }

  template <bool _AP_S1>
  INLINE bool slt(const ap_private<_AP_W, _AP_S1>& RHS) const {
    ap_private lhs(*this);
    ap_private<_AP_W, _AP_S1> rhs(RHS);
    bool lhsNeg = isNegative();
    bool rhsNeg = rhs.isNegative();
    if (lhsNeg) {
      // Sign bit is set so perform two's complement to make it positive
      lhs.flip();
      lhs++;
    }
    if (rhsNeg) {
      // Sign bit is set so perform two's complement to make it positive
      rhs.flip();
      rhs++;
    }

    // Now we have unsigned values to compare so do the comparison if necessary
    // based on the negativeness of the values.
    if (lhsNeg)
      if (rhsNeg)
        return lhs.ugt(rhs);
      else
        return true;
    else if (rhsNeg)
      return false;
    else
      return lhs.ult(rhs);
  }

  /// Regards both *this and RHS as unsigned quantities and compares them for
  /// validity of the less-or-equal relationship.
  /// @returns true if *this <= RHS when both are considered unsigned.
  /// @brief Unsigned less or equal comparison
  template <bool _AP_S1>
  INLINE bool ule(const ap_private<_AP_W, _AP_S1>& RHS) const {
    return ult(RHS) || eq(RHS);
  }

  /// Regards both *this and RHS as signed quantities and compares them for
  /// validity of the less-or-equal relationship.
  /// @returns true if *this <= RHS when both are considered signed.
  /// @brief Signed less or equal comparison
  template <bool _AP_S1>
  INLINE bool sle(const ap_private<_AP_W, _AP_S1>& RHS) const {
    return slt(RHS) || eq(RHS);
  }

  /// Regards both *this and RHS as unsigned quantities and compares them for
  /// the validity of the greater-than relationship.
  /// @returns true if *this > RHS when both are considered unsigned.
  /// @brief Unsigned greather than comparison
  template <bool _AP_S1>
  INLINE bool ugt(const ap_private<_AP_W, _AP_S1>& RHS) const {
    return !ult(RHS) && !eq(RHS);
  }

  /// Regards both *this and RHS as signed quantities and compares them for
  /// the validity of the greater-than relationship.
  /// @returns true if *this > RHS when both are considered signed.
  /// @brief Signed greather than comparison
  template <bool _AP_S1>
  INLINE bool sgt(const ap_private<_AP_W, _AP_S1>& RHS) const {
    return !slt(RHS) && !eq(RHS);
  }

  /// Regards both *this and RHS as unsigned quantities and compares them for
  /// validity of the greater-or-equal relationship.
  /// @returns true if *this >= RHS when both are considered unsigned.
  /// @brief Unsigned greater or equal comparison
  template <bool _AP_S1>
  INLINE bool uge(const ap_private<_AP_W, _AP_S>& RHS) const {
    return !ult(RHS);
  }

  /// Regards both *this and RHS as signed quantities and compares them for
  /// validity of the greater-or-equal relationship.
  /// @returns true if *this >= RHS when both are considered signed.
  /// @brief Signed greather or equal comparison
  template <bool _AP_S1>
  INLINE bool sge(const ap_private<_AP_W, _AP_S1>& RHS) const {
    return !slt(RHS);
  }

  // Sign extend to a new width.
  template <int _AP_W1, bool _AP_S1>
  INLINE void cpSext(const ap_private<_AP_W1, _AP_S1>& that) {
    assert(_AP_W1 < BitWidth && "Invalid ap_private SignExtend request");
    assert(_AP_W1 <= MAX_INT_BITS && "Too many bits");
    // If the sign bit isn't set, this is the same as zext.
    if (!that.isNegative()) {
      cpZext(that);
      return;
    }

    // The sign bit is set. First, get some facts
    enum { wordBits = _AP_W1 % APINT_BITS_PER_WORD };
    const int _AP_N1 = ap_private<_AP_W1, _AP_S1>::_AP_N;
    // Mask the high order word appropriately
    if (_AP_N1 == _AP_N) {
      enum { newWordBits = _AP_W % APINT_BITS_PER_WORD };
      // The extension is contained to the wordsBefore-1th word.
      static const uint64_t mask = wordBits ? (~0ULL << (wordBits)) : 0ULL;
      for (int i = 0; i < _AP_N; ++i) pVal[i] = that.get_pVal(i);
      pVal[_AP_N - 1] |= mask;
      return;
    }

    enum { newWordBits = _AP_W % APINT_BITS_PER_WORD };
    // The extension is contained to the wordsBefore-1th word.
    static const uint64_t mask = wordBits ? (~0ULL << (wordBits)) : 0ULL;
    int i;
    for (i = 0; i < _AP_N1; ++i) pVal[i] = that.get_pVal(i);
    pVal[i - 1] |= mask;
    for (; i < _AP_N - 1; i++) pVal[i] = ~0ULL;
    pVal[i] = ~0ULL;
    clearUnusedBits();
    return;
  }

  //  Zero extend to a new width.
  template <int _AP_W1, bool _AP_S1>
  INLINE void cpZext(const ap_private<_AP_W1, _AP_S1>& that) {
    assert(_AP_W1 < BitWidth && "Invalid ap_private ZeroExtend request");
    assert(_AP_W1 <= MAX_INT_BITS && "Too many bits");
    const int _AP_N1 = ap_private<_AP_W1, _AP_S1>::_AP_N;
    int i = 0;
    for (; i < _AP_N1; ++i) pVal[i] = that.get_pVal(i);
    for (; i < _AP_N; ++i) pVal[i] = 0;
    clearUnusedBits();
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE void cpZextOrTrunc(const ap_private<_AP_W1, _AP_S1>& that) {
    if (BitWidth > _AP_W1)
      cpZext(that);
    else {
      for (int i = 0; i < _AP_N; ++i) pVal[i] = that.get_pVal(i);
      clearUnusedBits();
    }
  }

  template <int _AP_W1, bool _AP_S1>
  INLINE void cpSextOrTrunc(const ap_private<_AP_W1, _AP_S1>& that) {
    if (BitWidth > _AP_W1)
      cpSext(that);
    else {
      for (int i = 0; i < _AP_N; ++i) pVal[i] = that.get_pVal(i);
      clearUnusedBits();
    }
  }

  /// @}
  /// @name Value Characterization Functions
  /// @{

  /// @returns the total number of bits.
  INLINE uint32_t getBitWidth() const { return BitWidth; }

  /// Here one word's bitwidth equals to that of uint64_t.
  /// @returns the number of words to hold the integer value of this ap_private.
  /// @brief Get the number of words.
  INLINE uint32_t getNumWords() const {
    return (BitWidth + APINT_BITS_PER_WORD - 1) / APINT_BITS_PER_WORD;
  }

  /// This function returns the number of active bits which is defined as the
  /// bit width minus the number of leading zeros. This is used in several
  /// computations to see how "wide" the value is.
  /// @brief Compute the number of active bits in the value
  INLINE uint32_t getActiveBits() const {
    uint32_t bits = BitWidth - countLeadingZeros();
    return bits ? bits : 1;
  }

  /// This method attempts to return the value of this ap_private as a zero
  /// extended
  /// uint64_t. The bitwidth must be <= 64 or the value must fit within a
  /// uint64_t. Otherwise an assertion will result.
  /// @brief Get zero extended value
  INLINE uint64_t getZExtValue() const {
    assert(getActiveBits() <= 64 && "Too many bits for uint64_t");
    return *pVal;
  }

  /// This method attempts to return the value of this ap_private as a sign
  /// extended
  /// int64_t. The bit width must be <= 64 or the value must fit within an
  /// int64_t. Otherwise an assertion will result.
  /// @brief Get sign extended value
  INLINE int64_t getSExtValue() const {
    assert(getActiveBits() <= 64 && "Too many bits for int64_t");
    return int64_t(pVal[0]);
  }

  /// This method determines how many bits are required to hold the ap_private
  /// equivalent of the string given by \p str of length \p slen.
  /// @brief Get bits required for string value.
  INLINE static uint32_t getBitsNeeded(const char* str, uint32_t slen,
                                       uint8_t radix) {
    assert(str != 0 && "Invalid value string");
    assert(slen > 0 && "Invalid string length");

    // Each computation below needs to know if its negative
    uint32_t isNegative = str[0] == '-';
    if (isNegative) {
      slen--;
      str++;
    }
    // For radixes of power-of-two values, the bits required is accurately and
    // easily computed
    if (radix == 2) return slen + isNegative;
    if (radix == 8) return slen * 3 + isNegative;
    if (radix == 16) return slen * 4 + isNegative;

    // Otherwise it must be radix == 10, the hard case
    assert(radix == 10 && "Invalid radix");

    // Convert to the actual binary value.
    // ap_private<_AP_W, _AP_S> tmp(sufficient, str, slen, radix);

    // Compute how many bits are required.
    // return isNegative + tmp.logBase2() + 1;
    return isNegative + slen * 4;
  }

  /// countLeadingZeros - This function is an ap_private version of the
  /// countLeadingZeros_{32,64} functions in MathExtras.h. It counts the number
  /// of zeros from the most significant bit to the first one bit.
  /// @returns BitWidth if the value is zero.
  /// @returns the number of zeros from the most significant bit to the first
  /// one bits.
  INLINE uint32_t countLeadingZeros() const {
    enum {
      msw_bits = (BitWidth % APINT_BITS_PER_WORD)
                     ? (BitWidth % APINT_BITS_PER_WORD)
                     : APINT_BITS_PER_WORD,
      excessBits = APINT_BITS_PER_WORD - msw_bits
    };
    uint32_t Count = ap_private_ops::CountLeadingZeros_64(pVal[_AP_N - 1]);
    if (Count >= excessBits) Count -= excessBits;
    if (!pVal[_AP_N - 1]) {
      for (int i = _AP_N - 1; i; --i) {
        if (!pVal[i - 1])
          Count += APINT_BITS_PER_WORD;
        else {
          Count += ap_private_ops::CountLeadingZeros_64(pVal[i - 1]);
          break;
        }
      }
    }
    return Count;
  }

  /// countLeadingOnes - This function counts the number of contiguous 1 bits
  /// in the high order bits. The count stops when the first 0 bit is reached.
  /// @returns 0 if the high order bit is not set
  /// @returns the number of 1 bits from the most significant to the least
  /// @brief Count the number of leading one bits.
  INLINE uint32_t countLeadingOnes() const {
    if (isSingleWord())
      return countLeadingOnes_64(get_VAL(), APINT_BITS_PER_WORD - BitWidth);

    uint32_t highWordBits = BitWidth % APINT_BITS_PER_WORD;
    uint32_t shift =
        (highWordBits == 0 ? 0 : APINT_BITS_PER_WORD - highWordBits);
    int i = _AP_N - 1;
    uint32_t Count = countLeadingOnes_64(get_pVal(i), shift);
    if (Count == highWordBits) {
      for (i--; i >= 0; --i) {
        if (get_pVal(i) == ~0ULL)
          Count += APINT_BITS_PER_WORD;
        else {
          Count += countLeadingOnes_64(get_pVal(i), 0);
          break;
        }
      }
    }
    return Count;
  }

  /// countTrailingZeros - This function is an ap_private version of the
  /// countTrailingZoers_{32,64} functions in MathExtras.h. It counts
  /// the number of zeros from the least significant bit to the first set bit.
  /// @returns BitWidth if the value is zero.
  /// @returns the number of zeros from the least significant bit to the first
  /// one bit.
  /// @brief Count the number of trailing zero bits.
  INLINE uint32_t countTrailingZeros() const {
    uint32_t Count = 0;
    uint32_t i = 0;
    for (; i < _AP_N && get_pVal(i) == 0; ++i) Count += APINT_BITS_PER_WORD;
    if (i < _AP_N) Count += ap_private_ops::CountTrailingZeros_64(get_pVal(i));
    return AESL_std::min(Count, BitWidth);
  }
  /// countPopulation - This function is an ap_private version of the
  /// countPopulation_{32,64} functions in MathExtras.h. It counts the number
  /// of 1 bits in the ap_private value.
  /// @returns 0 if the value is zero.
  /// @returns the number of set bits.
  /// @brief Count the number of bits set.
  INLINE uint32_t countPopulation() const {
    uint32_t Count = 0;
    for (int i = 0; i < _AP_N - 1; ++i)
      Count += ap_private_ops::CountPopulation_64(pVal[i]);
    Count += ap_private_ops::CountPopulation_64(pVal[_AP_N - 1] & mask);
    return Count;
  }

  /// @}
  /// @name Conversion Functions
  /// @

  /// This is used internally to convert an ap_private to a string.
  /// @brief Converts an ap_private to a std::string
  INLINE std::string toString(uint8_t radix, bool wantSigned) const;

  /// Considers the ap_private to be unsigned and converts it into a string in
  /// the
  /// radix given. The radix can be 2, 8, 10 or 16.
  /// @returns a character interpretation of the ap_private
  /// @brief Convert unsigned ap_private to string representation.
  INLINE std::string toStringUnsigned(uint8_t radix = 10) const {
    return toString(radix, false);
  }

  /// Considers the ap_private to be unsigned and converts it into a string in
  /// the
  /// radix given. The radix can be 2, 8, 10 or 16.
  /// @returns a character interpretation of the ap_private
  /// @brief Convert unsigned ap_private to string representation.
  INLINE std::string toStringSigned(uint8_t radix = 10) const {
    return toString(radix, true);
  }

  /// @brief Converts this ap_private to a double value.
  INLINE double roundToDouble(bool isSigned) const {
    // Handle the simple case where the value is contained in one uint64_t.
    if (isSingleWord() || getActiveBits() <= APINT_BITS_PER_WORD) {
      uint64_t val = pVal[0];
      if (isSigned) {
        int64_t sext = ((int64_t(val)) << (64 - BitWidth)) >> (64 - BitWidth);
        return double(sext);
      } else
        return double(val);
    }

    // Determine if the value is negative.
    bool isNeg = isSigned ? (*this)[BitWidth - 1] : false;

    // Construct the absolute value if we're negative.
    ap_private<_AP_W, _AP_S> Tmp(isNeg ? -(*this) : (*this));

    // Figure out how many bits we're using.
    uint32_t n = Tmp.getActiveBits();

    // The exponent (without bias normalization) is just the number of bits
    // we are using. Note that the sign bit is gone since we constructed the
    // absolute value.
    uint64_t exp = n;

    // Return infinity for exponent overflow
    if (exp > 1023) {
      if (!isSigned || !isNeg)
        return std::numeric_limits<double>::infinity();
      else
        return -std::numeric_limits<double>::infinity();
    }
    exp += 1023; // Increment for 1023 bias

    // Number of bits in mantissa is 52. To obtain the mantissa value, we must
    // extract the high 52 bits from the correct words in pVal.
    uint64_t mantissa;
    unsigned hiWord = whichWord(n - 1);
    if (hiWord == 0) {
      mantissa = Tmp.get_pVal(0);
      if (n > 52)
        (mantissa) >>= (n - 52); // shift down, we want the top 52 bits.
    } else {
      assert(hiWord > 0 && "High word is negative?");
      uint64_t hibits = (Tmp.get_pVal(hiWord))
                        << (52 - n % APINT_BITS_PER_WORD);
      uint64_t lobits =
          (Tmp.get_pVal(hiWord - 1)) >> (11 + n % APINT_BITS_PER_WORD);
      mantissa = hibits | lobits;
    }

    // The leading bit of mantissa is implicit, so get rid of it.
    uint64_t sign = isNeg ? (1ULL << (APINT_BITS_PER_WORD - 1)) : 0;
    union {
      double __D;
      uint64_t __I;
    } __T;
    __T.__I = sign | ((exp) << 52) | mantissa;
    return __T.__D;
  }

  /// @brief Converts this unsigned ap_private to a double value.
  INLINE double roundToDouble() const { return roundToDouble(false); }

  /// @brief Converts this signed ap_private to a double value.
  INLINE double signedRoundToDouble() const { return roundToDouble(true); }

  /// The conversion does not do a translation from integer to double, it just
  /// re-interprets the bits as a double. Note that it is valid to do this on
  /// any bit width. Exactly 64 bits will be translated.
  /// @brief Converts ap_private bits to a double
  INLINE double bitsToDouble() const {
    union {
      uint64_t __I;
      double __D;
    } __T;
    __T.__I = pVal[0];
    return __T.__D;
  }

  /// The conversion does not do a translation from integer to float, it just
  /// re-interprets the bits as a float. Note that it is valid to do this on
  /// any bit width. Exactly 32 bits will be translated.
  /// @brief Converts ap_private bits to a double
  INLINE float bitsToFloat() const {
    union {
      uint32_t __I;
      float __F;
    } __T;
    __T.__I = uint32_t(pVal[0]);
    return __T.__F;
  }

  /// The conversion does not do a translation from double to integer, it just
  /// re-interprets the bits of the double. Note that it is valid to do this on
  /// any bit width but bits from V may get truncated.
  /// @brief Converts a double to ap_private bits.
  INLINE ap_private& doubleToBits(double __V) {
    union {
      uint64_t __I;
      double __D;
    } __T;
    __T.__D = __V;
    pVal[0] = __T.__I;
    return *this;
  }

  /// The conversion does not do a translation from float to integer, it just
  /// re-interprets the bits of the float. Note that it is valid to do this on
  /// any bit width but bits from V may get truncated.
  /// @brief Converts a float to ap_private bits.
  INLINE ap_private& floatToBits(float __V) {
    union {
      uint32_t __I;
      float __F;
    } __T;
    __T.__F = __V;
    pVal[0] = __T.__I;
  }

  // Reduce operation
  //-----------------------------------------------------------
  INLINE bool and_reduce() const { return isMaxValue(); }

  INLINE bool nand_reduce() const { return isMinValue(); }

  INLINE bool or_reduce() const { return (bool)countPopulation(); }

  INLINE bool nor_reduce() const { return countPopulation() == 0; }

  INLINE bool xor_reduce() const {
    unsigned int i = countPopulation();
    return (i % 2) ? true : false;
  }

  INLINE bool xnor_reduce() const {
    unsigned int i = countPopulation();
    return (i % 2) ? false : true;
  }
  INLINE std::string to_string(uint8_t radix = 16, bool sign = false) const {
    return toString(radix, radix == 10 ? _AP_S : sign);
  }
}; // End of class ap_private <_AP_W, _AP_S, false>

namespace ap_private_ops {

enum { APINT_BITS_PER_WORD = 64 };
template <int _AP_W, bool _AP_S>
INLINE bool operator==(uint64_t V1, const ap_private<_AP_W, _AP_S>& V2) {
  return V2 == V1;
}

template <int _AP_W, bool _AP_S>
INLINE bool operator!=(uint64_t V1, const ap_private<_AP_W, _AP_S>& V2) {
  return V2 != V1;
}

template <int _AP_W, bool _AP_S, int index>
INLINE bool get(const ap_private<_AP_W, _AP_S>& a) {
  static const uint64_t mask = 1ULL << (index & 0x3f);
  return ((mask & a.get_pVal((index) >> 6)) != 0);
}

template <int _AP_W, bool _AP_S, int msb_index, int lsb_index>
INLINE void set(ap_private<_AP_W, _AP_S>& a,
                const ap_private<AP_MAX(msb_index, 1), true>& mark1 = 0,
                const ap_private<AP_MAX(lsb_index, 1), true>& mark2 = 0) {
  enum {
    APINT_BITS_PER_WORD = 64,
    lsb_word = lsb_index / APINT_BITS_PER_WORD,
    msb_word = msb_index / APINT_BITS_PER_WORD,
    msb = msb_index % APINT_BITS_PER_WORD,
    lsb = lsb_index % APINT_BITS_PER_WORD
  };
  if (msb_word == lsb_word) {
    const uint64_t mask = ~0ULL >>
                          (lsb) << (APINT_BITS_PER_WORD - msb + lsb - 1) >>
                          (APINT_BITS_PER_WORD - msb - 1);
    // a.set_pVal(msb_word, a.get_pVal(msb_word)  | mask);
    a.get_pVal(msb_word) |= mask;
  } else {
    const uint64_t lsb_mask = ~0ULL >> (lsb) << (lsb);
    const uint64_t msb_mask = ~0ULL << (APINT_BITS_PER_WORD - msb - 1) >>
                              (APINT_BITS_PER_WORD - msb - 1);
    // a.set_pVal(lsb_word, a.get_pVal(lsb_word) | lsb_mask);
    a.get_pVal(lsb_word) |= lsb_mask;
    for (int i = lsb_word + 1; i < msb_word; i++) {
      a.set_pVal(i, ~0ULL);
      // a.get_pVal(i)=0;
    }
    // a.set_pVal(msb_word, a.get_pVal(msb_word) | msb_mask);

    a.get_pVal(msb_word) |= msb_mask;
  }
  a.clearUnusedBits();
}

template <int _AP_W, bool _AP_S, int msb_index, int lsb_index>
INLINE void clear(ap_private<_AP_W, _AP_S>& a,
                  const ap_private<AP_MAX(msb_index, 1), true>& mark1 = 0,
                  const ap_private<AP_MAX(lsb_index, 1), true>& mark2 = 0) {
  enum {
    APINT_BITS_PER_WORD = 64,
    lsb_word = lsb_index / APINT_BITS_PER_WORD,
    msb_word = msb_index / APINT_BITS_PER_WORD,
    msb = msb_index % APINT_BITS_PER_WORD,
    lsb = lsb_index % APINT_BITS_PER_WORD
  };
  if (msb_word == lsb_word) {
    const uint64_t mask =
        ~(~0ULL >> (lsb) << (APINT_BITS_PER_WORD - msb + lsb - 1) >>
          (APINT_BITS_PER_WORD - msb - 1));
    // a.set_pVal(msb_word, a.get_pVal(msb_word) & mask);
    a.get_pVal(msb_word) &= mask;
  } else {
    const uint64_t lsb_mask = ~(~0ULL >> (lsb) << (lsb));
    const uint64_t msb_mask = ~(~0ULL << (APINT_BITS_PER_WORD - msb - 1) >>
                                (APINT_BITS_PER_WORD - msb - 1));
    // a.set_pVal(lsb_word, a.get_pVal(lsb_word) & lsb_mask);
    a.get_pVal(lsb_word) &= lsb_mask;
    for (int i = lsb_word + 1; i < msb_word; i++) {
      // a.set_pVal(i, 0);
      a.get_pVal(i) = 0;
    }
    // a.set_pVal(msb_word, a.get_pVal(msb_word) & msb_mask);
    a.get_pVal(msb_word) &= msb_mask;
  }
  a.clearUnusedBits();
}

template <int _AP_W, bool _AP_S, int index>
INLINE void set(ap_private<_AP_W, _AP_S>& a,
                const ap_private<AP_MAX(index, 1), true>& mark = 0) {
  enum { APINT_BITS_PER_WORD = 64, word = index / APINT_BITS_PER_WORD };
  static const uint64_t mask = 1ULL << (index % APINT_BITS_PER_WORD);
  // a.set_pVal(word, a.get_pVal(word) | mask);
  a.get_pVal(word) |= mask;
  a.clearUnusedBits();
}

template <int _AP_W, bool _AP_S, int index>
INLINE void clear(ap_private<_AP_W, _AP_S>& a,
                  const ap_private<AP_MAX(index, 1), true>& mark = 0) {
  enum { APINT_BITS_PER_WORD = 64, word = index / APINT_BITS_PER_WORD };
  static const uint64_t mask = ~(1ULL << (index % APINT_BITS_PER_WORD));
  // a.set_pVal(word, a.get_pVal(word) & mask);
  a.get_pVal(word) &= mask;
  a.clearUnusedBits();
}

} // End of ap_private_ops namespace

template <int _AP_W, bool _AP_S>
INLINE std::string ap_private<_AP_W, _AP_S, false>::toString(
    uint8_t radix, bool wantSigned) const {
  assert((radix == 10 || radix == 8 || radix == 16 || radix == 2) &&
         "Radix should be 2, 8, 10, or 16!");
  static const char* digits[] = {"0", "1", "2", "3", "4", "5", "6", "7",
                                 "8", "9", "A", "B", "C", "D", "E", "F"};
  std::string result;

  if (radix != 10) {
    // For the 2, 8 and 16 bit cases, we can just shift instead of divide
    // because the number of bits per digit (1,3 and 4 respectively) divides
    // equaly. We just shift until there value is zero.

    // First, check for a zero value and just short circuit the logic below.
    if (*this == (uint64_t)(0))
      result = "0";
    else {
      ap_private<_AP_W, false> tmp(*this);
      size_t insert_at = 0;
      bool leading_zero = true;
      if (wantSigned && isNegative()) {
        // They want to print the signed version and it is a negative value
        // Flip the bits and add one to turn it into the equivalent positive
        // value and put a '-' in the result.
        tmp.flip();
        tmp++;
        tmp.clearUnusedBitsToZero();
        result = "-";
        insert_at = 1;
        leading_zero = false;
      }
      switch (radix) {
        case 2:
          result += "0b";
          break;
        case 8:
          result += "0o";
          break;
        case 16:
          result += "0x";
          break;
        default:
          assert("invalid radix" && 0);
      }
      insert_at += 2;
      // Just shift tmp right for each digit width until it becomes zero
      uint32_t shift = (radix == 16 ? 4 : (radix == 8 ? 3 : 1));
      uint64_t mask = radix - 1;
      ap_private<_AP_W, false> zero(0);
      unsigned bits = 0;
      while (tmp.ne(zero)) {
        uint64_t digit = tmp.get_VAL() & mask;
        result.insert(insert_at, digits[digit]);
        tmp = tmp.lshr(shift);
        ++bits;
      }
      bits *= shift;
      if (bits < _AP_W && leading_zero) result.insert(insert_at, digits[0]);
    }
    return result;
  }

  ap_private<_AP_W, false> tmp(*this);
  ap_private<_AP_W, false> divisor(radix);
  ap_private<_AP_W, false> zero(0);
  size_t insert_at = 0;
  if (wantSigned && isNegative()) {
    // They want to print the signed version and it is a negative value
    // Flip the bits and add one to turn it into the equivalent positive
    // value and put a '-' in the result.
    tmp.flip();
    tmp++;
    tmp.clearUnusedBitsToZero();
    result = "-";
    insert_at = 1;
  }
  if (tmp == ap_private<_AP_W, false>(0))
    result = "0";
  else
    while (tmp.ne(zero)) {
      ap_private<_AP_W, false> APdigit(0);
      ap_private<_AP_W, false> tmp2(0);
      ap_private_ops::divide(tmp, tmp.getNumWords(), divisor,
                             divisor.getNumWords(), &tmp2, &APdigit);
      uint64_t digit = APdigit.getZExtValue();
      assert(digit < radix && "divide failed");
      result.insert(insert_at, digits[digit]);
      tmp = tmp2;
    }

  return result;
} // End of ap_private<_AP_W, _AP_S, false>::toString()

template <int _AP_W, bool _AP_S>
std::ostream &operator<<(std::ostream &os, const ap_private<_AP_W, _AP_S> &x) {
  std::ios_base::fmtflags ff = std::cout.flags();
  if (ff & std::cout.hex) {
    os << x.toString(16, false); // don't print sign
  } else if (ff & std::cout.oct) {
    os << x.toString(8, false); // don't print sign
  } else {
    os << x.toString(10, _AP_S);
  }
  return os;
}

// ------------------------------------------------------------ //
//           XXX moved here from ap_int_sim.h  XXX              //
// ------------------------------------------------------------ //

/// Concatination reference.
/// Proxy class which allows concatination to be used as rvalue(for reading) and
/// lvalue(for writing)
// ----------------------------------------------------------------
// template <int _AP_W1, typename _AP_T1, int _AP_W2, typename _AP_T2>
// struct ap_concat_ref {
//#ifdef _MSC_VER
//#pragma warning(disable : 4521 4522)
//#endif
//  enum {
//    _AP_WR = _AP_W1 + _AP_W2,
//  };
//  _AP_T1& mbv1;
//  _AP_T2& mbv2;
//
//  INLINE ap_concat_ref(const ap_concat_ref<_AP_W1, _AP_T1, _AP_W2, _AP_T2>&
//  ref)
//      : mbv1(ref.mbv1), mbv2(ref.mbv2) {}
//
//  INLINE ap_concat_ref(_AP_T1& bv1, _AP_T2& bv2) : mbv1(bv1), mbv2(bv2) {}
//
//  template <int _AP_W3, bool _AP_S3>
//  INLINE ap_concat_ref& operator=(const ap_private<_AP_W3, _AP_S3>& val) {
//    ap_private<_AP_W1 + _AP_W2, false> vval(val);
//    int W_ref1 = mbv1.length();
//    int W_ref2 = mbv2.length();
//    ap_private<_AP_W1, false> mask1(-1);
//    mask1 >>= _AP_W1 - W_ref1;
//    ap_private<_AP_W2, false> mask2(-1);
//    mask2 >>= _AP_W2 - W_ref2;
//    mbv1.set(ap_private<_AP_W1, false>((vval >> W_ref2) & mask1));
//    mbv2.set(ap_private<_AP_W2, false>(vval & mask2));
//    return *this;
//  }
//
//  INLINE ap_concat_ref& operator=(unsigned long long val) {
//    ap_private<_AP_W1 + _AP_W2, false> tmpVal(val);
//    return operator=(tmpVal);
//  }
//
//  template <int _AP_W3, typename _AP_T3, int _AP_W4, typename _AP_T4>
//  INLINE ap_concat_ref& operator=(
//      const ap_concat_ref<_AP_W3, _AP_T3, _AP_W4, _AP_T4>& val) {
//    ap_private<_AP_W1 + _AP_W2, false> tmpVal(val);
//    return operator=(tmpVal);
//  }
//
//  INLINE ap_concat_ref& operator=(
//      const ap_concat_ref<_AP_W1, _AP_T1, _AP_W2, _AP_T2>& val) {
//    ap_private<_AP_W1 + _AP_W2, false> tmpVal(val);
//    return operator=(tmpVal);
//  }
//
//  template <int _AP_W3, bool _AP_S3>
//  INLINE ap_concat_ref& operator=(const _private_bit_ref<_AP_W3, _AP_S3>&
//  val) {
//    ap_private<_AP_W1 + _AP_W2, false> tmpVal(val);
//    return operator=(tmpVal);
//  }
//
//  template <int _AP_W3, bool _AP_S3>
//  INLINE ap_concat_ref& operator=(const _private_range_ref<_AP_W3, _AP_S3>&
//  val) {
//    ap_private<_AP_W1 + _AP_W2, false> tmpVal(val);
//    return operator=(tmpVal);
//  }
//
//  template <int _AP_W3, int _AP_I3, bool _AP_S3, ap_q_mode _AP_Q3,
//            ap_o_mode _AP_O3, int _AP_N3>
//  INLINE ap_concat_ref& operator=(
//      const af_range_ref<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3, _AP_N3>& val)
//      {
//    return operator=((const ap_private<_AP_W3, false>)(val));
//  }
//
//  template <int _AP_W3, int _AP_I3, bool _AP_S3, ap_q_mode _AP_Q3,
//            ap_o_mode _AP_O3, int _AP_N3>
//  INLINE ap_concat_ref& operator=(
//      const ap_fixed_base<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3, _AP_N3>&
//          val) {
//    return operator=(val.to_ap_private());
//  }
//
//  template <int _AP_W3, int _AP_I3, bool _AP_S3, ap_q_mode _AP_Q3,
//            ap_o_mode _AP_O3, int _AP_N3>
//  INLINE ap_concat_ref& operator=(
//      const af_bit_ref<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3, _AP_N3>& val) {
//    return operator=((unsigned long long)(bool)(val));
//  }
//
//  INLINE operator ap_private<_AP_WR, false>() const { return get(); }
//
//  INLINE operator unsigned long long() const { return get().to_uint64(); }
//
//  template <int _AP_W3, bool _AP_S3>
//  INLINE ap_concat_ref<_AP_WR, ap_concat_ref, _AP_W3,
//                       _private_range_ref<_AP_W3, _AP_S3> >
//  operator,(const _private_range_ref<_AP_W3, _AP_S3> &a2) {
//    return ap_concat_ref<_AP_WR, ap_concat_ref, _AP_W3,
//                         _private_range_ref<_AP_W3, _AP_S3> >(
//        *this, const_cast<_private_range_ref<_AP_W3, _AP_S3>&>(a2));
//  }
//
//  template <int _AP_W3, bool _AP_S3>
//  INLINE
//      ap_concat_ref<_AP_WR, ap_concat_ref, _AP_W3, ap_private<_AP_W3, _AP_S3>
//      >
//      operator,(ap_private<_AP_W3, _AP_S3> &a2) {
//    return ap_concat_ref<_AP_WR, ap_concat_ref, _AP_W3,
//                         ap_private<_AP_W3, _AP_S3> >(*this, a2);
//  }
//
//  template <int _AP_W3, bool _AP_S3>
//  INLINE
//      ap_concat_ref<_AP_WR, ap_concat_ref, _AP_W3, ap_private<_AP_W3, _AP_S3>
//      >
//      operator,(const ap_private<_AP_W3, _AP_S3> &a2) {
//    return ap_concat_ref<_AP_WR, ap_concat_ref, _AP_W3,
//                         ap_private<_AP_W3, _AP_S3> >(
//        *this, const_cast<ap_private<_AP_W3, _AP_S3>&>(a2));
//  }
//
//  template <int _AP_W3, bool _AP_S3>
//  INLINE ap_concat_ref<_AP_WR, ap_concat_ref, 1, _private_bit_ref<_AP_W3,
//  _AP_S3> >
//  operator,(const _private_bit_ref<_AP_W3, _AP_S3> &a2) {
//    return ap_concat_ref<_AP_WR, ap_concat_ref, 1, _private_bit_ref<_AP_W3,
//    _AP_S3> >(
//        *this, const_cast<_private_bit_ref<_AP_W3, _AP_S3>&>(a2));
//  }
//
//  template <int _AP_W3, typename _AP_T3, int _AP_W4, typename _AP_T4>
//  INLINE ap_concat_ref<_AP_WR, ap_concat_ref, _AP_W3 + _AP_W4,
//                       ap_concat_ref<_AP_W3, _AP_T3, _AP_W4, _AP_T4> >
//  operator,(const ap_concat_ref<_AP_W3, _AP_T3, _AP_W4, _AP_T4> &a2) {
//    return ap_concat_ref<_AP_WR, ap_concat_ref, _AP_W3 + _AP_W4,
//                         ap_concat_ref<_AP_W3, _AP_T3, _AP_W4, _AP_T4> >(
//        *this, const_cast<ap_concat_ref<_AP_W3, _AP_T3, _AP_W4,
//        _AP_T4>&>(a2));
//  }
//
//  template <int _AP_W3, int _AP_I3, bool _AP_S3, ap_q_mode _AP_Q3,
//            ap_o_mode _AP_O3, int _AP_N3>
//  INLINE ap_concat_ref<
//      _AP_WR, ap_concat_ref, _AP_W3,
//      af_range_ref<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3, _AP_N3> >
//  operator,(
//      const af_range_ref<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3, _AP_N3> &a2)
//      {
//    return ap_concat_ref<
//        _AP_WR, ap_concat_ref, _AP_W3,
//        af_range_ref<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3, _AP_N3> >(
//        *this,
//        const_cast<
//            af_range_ref<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3,
//            _AP_N3>&>(a2));
//  }
//
//  template <int _AP_W3, int _AP_I3, bool _AP_S3, ap_q_mode _AP_Q3,
//            ap_o_mode _AP_O3, int _AP_N3>
//  INLINE
//      ap_concat_ref<_AP_WR, ap_concat_ref, 1,
//                    af_bit_ref<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3, _AP_N3>
//                    >
//      operator,(const af_bit_ref<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3,
//      _AP_N3>
//                    &a2) {
//    return ap_concat_ref<
//        _AP_WR, ap_concat_ref, 1,
//        af_bit_ref<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3, _AP_N3> >(
//        *this,
//        const_cast<af_bit_ref<_AP_W3, _AP_I3, _AP_S3, _AP_Q3, _AP_O3,
//        _AP_N3>&>(
//            a2));
//  }
//
//  template <int _AP_W3, bool _AP_S3>
//  INLINE ap_private<AP_MAX(_AP_WR, _AP_W3), _AP_S3> operator&(
//      const ap_private<_AP_W3, _AP_S3>& a2) {
//    return get() & a2;
//  }
//
//  template <int _AP_W3, bool _AP_S3>
//  INLINE ap_private<AP_MAX(_AP_WR, _AP_W3), _AP_S3> operator|(
//      const ap_private<_AP_W3, _AP_S3>& a2) {
//    return get() | a2;
//  }
//
//  template <int _AP_W3, bool _AP_S3>
//  INLINE ap_private<AP_MAX(_AP_WR, _AP_W3), _AP_S3> operator^(
//      const ap_private<_AP_W3, _AP_S3>& a2) {
//    return ap_private<AP_MAX(_AP_WR, _AP_W3), _AP_S3>(get() ^ a2);
//  }
//
//  INLINE const ap_private<_AP_WR, false> get() const {
//    ap_private<_AP_W1 + _AP_W2, false> tmpVal =
//        ap_private<_AP_W1 + _AP_W2, false>(mbv1.get());
//    ap_private<_AP_W1 + _AP_W2, false> tmpVal2 =
//        ap_private<_AP_W1 + _AP_W2, false>(mbv2.get());
//    int W_ref2 = mbv2.length();
//    tmpVal <<= W_ref2;
//    tmpVal |= tmpVal2;
//    return tmpVal;
//  }
//
//  INLINE const ap_private<_AP_WR, false> get() {
//    ap_private<_AP_W1 + _AP_W2, false> tmpVal =
//        ap_private<_AP_W1 + _AP_W2, false>(mbv1.get());
//    ap_private<_AP_W1 + _AP_W2, false> tmpVal2 =
//        ap_private<_AP_W1 + _AP_W2, false>(mbv2.get());
//    int W_ref2 = mbv2.length();
//    tmpVal <<= W_ref2;
//    tmpVal |= tmpVal2;
//    return tmpVal;
//  }
//
//  template <int _AP_W3>
//  INLINE void set(const ap_private<_AP_W3, false>& val) {
//    ap_private<_AP_W1 + _AP_W2, false> vval(val);
//    int W_ref1 = mbv1.length();
//    int W_ref2 = mbv2.length();
//    ap_private<_AP_W1, false> mask1(-1);
//    mask1 >>= _AP_W1 - W_ref1;
//    ap_private<_AP_W2, false> mask2(-1);
//    mask2 >>= _AP_W2 - W_ref2;
//    mbv1.set(ap_private<_AP_W1, false>((vval >> W_ref2) & mask1));
//    mbv2.set(ap_private<_AP_W2, false>(vval & mask2));
//  }
//
//  INLINE int length() const { return mbv1.length() + mbv2.length(); }
//
//  INLINE std::string to_string(uint8_t radix = 2) const {
//    return get().to_string(radix);
//  }
//}; // struct ap_concat_ref.

/// Range(slice) reference
/// Proxy class, which allows part selection to be used as rvalue(for reading)
/// and lvalue(for writing)
//------------------------------------------------------------
template <int _AP_W, bool _AP_S>
struct _private_range_ref {
#ifdef _MSC_VER
#pragma warning(disable : 4521 4522)
#endif
  ap_private<_AP_W, _AP_S>& d_bv;
  int l_index;
  int h_index;

 public:
  /// copy ctor.
  INLINE _private_range_ref(const _private_range_ref<_AP_W, _AP_S>& ref)
      : d_bv(ref.d_bv), l_index(ref.l_index), h_index(ref.h_index) {}

  /// direct ctor.
  INLINE _private_range_ref(ap_private<_AP_W, _AP_S>* bv, int h, int l)
      : d_bv(*bv), l_index(l), h_index(h) {
    _AP_WARNING(h < 0 || l < 0,
                "Higher bound (%d) and lower bound (%d) cannot be "
                "negative.",
                h, l);
    _AP_WARNING(h >= _AP_W || l >= _AP_W,
                "Higher bound (%d) or lower bound (%d) out of range (%d).", h, l,
                _AP_W);
  }

  /// compound or assignment.
  template <int _AP_W2, bool _AP_S2>
  INLINE _private_range_ref<_AP_W, _AP_S>& operator|=(
      const _private_range_ref<_AP_W2, _AP_S2>& ref) {
    _AP_WARNING((h_index - l_index) != (ref.h_index - ref.l_index),
                "Bitsize mismach for ap_private<>.range() &= "
                "ap_private<>.range().");
    this->d_bv |= ref.d_bv;
    return *this;
  }

  /// compound or assignment with root type.
  template <int _AP_W2, bool _AP_S2>
  INLINE _private_range_ref<_AP_W, _AP_S>& operator|=(
      const _AP_ROOT_TYPE<_AP_W2, _AP_S2>& ref) {
    _AP_WARNING((h_index - l_index + 1) != _AP_W2,
                "Bitsize mismach for ap_private<>.range() |= _AP_ROOT_TYPE<>.");
    this->d_bv |= ref.V;
    return *this;
  }

  /// compound and assignment.
  template <int _AP_W2, bool _AP_S2>
  INLINE _private_range_ref<_AP_W, _AP_S>& operator&=(
      const _private_range_ref<_AP_W2, _AP_S2>& ref) {
    _AP_WARNING((h_index - l_index) != (ref.h_index - ref.l_index),
                "Bitsize mismach for ap_private<>.range() &= "
                "ap_private<>.range().");
    this->d_bv &= ref.d_bv;
    return *this;
  };

  /// compound and assignment with root type.
  template <int _AP_W2, bool _AP_S2>
  INLINE _private_range_ref<_AP_W, _AP_S>& operator&=(
      const _AP_ROOT_TYPE<_AP_W2, _AP_S2>& ref) {
    _AP_WARNING((h_index - l_index + 1) != _AP_W2,
                "Bitsize mismach for ap_private<>.range() &= _AP_ROOT_TYPE<>.");
    this->d_bv &= ref.V;
    return *this;
  }

  /// compound xor assignment.
  template <int _AP_W2, bool _AP_S2>
  INLINE _private_range_ref<_AP_W, _AP_S>& operator^=(
      const _private_range_ref<_AP_W2, _AP_S2>& ref) {
    _AP_WARNING((h_index - l_index) != (ref.h_index - ref.l_index),
                "Bitsize mismach for ap_private<>.range() ^= "
                "ap_private<>.range().");
    this->d_bv ^= ref.d_bv;
    return *this;
  };

  /// compound xor assignment with root type.
  template <int _AP_W2, bool _AP_S2>
  INLINE _private_range_ref<_AP_W, _AP_S>& operator^=(
      const _AP_ROOT_TYPE<_AP_W2, _AP_S2>& ref) {
    _AP_WARNING((h_index - l_index + 1) != _AP_W2,
                "Bitsize mismach for ap_private<>.range() ^= _AP_ROOT_TYPE<>.");
    this->d_bv ^= ref.V;
    return *this;
  }

  /// @name convertors.
  //  @{
  INLINE operator ap_private<_AP_W, false>() const {
    ap_private<_AP_W, false> val(0);
    if (h_index >= l_index) {
      if (_AP_W > 64) {
        val = d_bv;
        ap_private<_AP_W, false> mask(-1);
        mask >>= _AP_W - (h_index - l_index + 1);
        val >>= l_index;
        val &= mask;
      } else {
        const static uint64_t mask = (~0ULL >> (64 > _AP_W ? (64 - _AP_W) : 0));
        val = (d_bv >> l_index) & (mask >> (_AP_W - (h_index - l_index + 1)));
      }
    } else {
      for (int i = 0, j = l_index; j >= 0 && j >= h_index; j--, i++)
        if ((d_bv)[j]) val.set(i);
    }
    return val;
  }

  INLINE operator unsigned long long() const { return to_uint64(); }
  //  @}

  template <int _AP_W2, bool _AP_S2>
  INLINE _private_range_ref& operator=(const ap_private<_AP_W2, _AP_S2>& val) {
    ap_private<_AP_W, false> vval = ap_private<_AP_W, false>(val);
    if (l_index > h_index) {
      for (int i = 0, j = l_index; j >= 0 && j >= h_index; j--, i++)
        (vval)[i] ? d_bv.set(j) : d_bv.clear(j);
    } else {
      if (_AP_W > 64) {
        ap_private<_AP_W, false> mask(-1);
        if (l_index > 0) {
          mask <<= l_index;
          vval <<= l_index;
        }
        if (h_index < _AP_W - 1) {
          ap_private<_AP_W, false> mask2(-1);
          mask2 >>= _AP_W - h_index - 1;
          mask &= mask2;
          vval &= mask2;
        }
        mask.flip();
        d_bv &= mask;
        d_bv |= vval;
      } else {
        unsigned shift = 64 - _AP_W;
        uint64_t mask = ~0ULL >> (shift);
        if (l_index > 0) {
          vval = mask & vval << l_index;
          mask = mask & mask << l_index;
        }
        if (h_index < _AP_W - 1) {
          uint64_t mask2 = mask;
          mask2 >>= (_AP_W - h_index - 1);
          mask &= mask2;
          vval &= mask2;
        }
        mask = ~mask;
        d_bv &= mask;
        d_bv |= vval;
      }
    }
    return *this;
  } // operator=(const ap_private<>&)

  INLINE _private_range_ref& operator=(unsigned long long val) {
    const ap_private<_AP_W, _AP_S> vval = val;
    return operator=(vval);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE _private_range_ref& operator=(
      const _private_bit_ref<_AP_W2, _AP_S2>& val) {
    return operator=((unsigned long long)(bool)val);
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE _private_range_ref& operator=(
      const _private_range_ref<_AP_W2, _AP_S2>& val) {
    const ap_private<_AP_W, false> tmpVal(val);
    return operator=(tmpVal);
  }

//  template <int _AP_W3, typename _AP_T3, int _AP_W4, typename _AP_T4>
//  INLINE _private_range_ref& operator=(
//      const ap_concat_ref<_AP_W3, _AP_T3, _AP_W4, _AP_T4>& val) {
//    const ap_private<_AP_W, false> tmpVal(val);
//    return operator=(tmpVal);
//  }

  // TODO from ap_int_base, ap_bit_ref and ap_range_ref.

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE _private_range_ref& operator=(
      const ap_fixed_base<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& val) {
    return operator=(val.to_ap_int_base().V);
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE _private_range_ref& operator=(
      const af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& val) {
    return operator=(val.operator ap_int_base<_AP_W2, false>().V);
  }

  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
            ap_o_mode _AP_O2, int _AP_N2>
  INLINE _private_range_ref& operator=(
      const af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>& val) {
    return operator=((unsigned long long)(bool)val);
  }

//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, _private_range_ref, _AP_W2,
//                       _private_range_ref<_AP_W2, _AP_S2> >
//  operator,(const _private_range_ref<_AP_W2, _AP_S2> &a2) {
//    return ap_concat_ref<_AP_W, _private_range_ref, _AP_W2,
//                         _private_range_ref<_AP_W2, _AP_S2> >(
//        *this, const_cast<_private_range_ref<_AP_W2, _AP_S2>&>(a2));
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, _private_range_ref, _AP_W2,
//                       ap_private<_AP_W2, _AP_S2> >
//  operator,(ap_private<_AP_W2, _AP_S2> &a2) {
//    return ap_concat_ref<_AP_W, _private_range_ref, _AP_W2,
//                         ap_private<_AP_W2, _AP_S2> >(*this, a2);
//  }
//
//  INLINE
//  ap_concat_ref<_AP_W, _private_range_ref, _AP_W, ap_private<_AP_W, _AP_S> >
//  operator,(ap_private<_AP_W, _AP_S>& a2) {
//    return ap_concat_ref<_AP_W, _private_range_ref, _AP_W,
//                         ap_private<_AP_W, _AP_S> >(*this, a2);
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<_AP_W, _private_range_ref, 1,
//                       _private_bit_ref<_AP_W2, _AP_S2> >
//  operator,(const _private_bit_ref<_AP_W2, _AP_S2> &a2) {
//    return ap_concat_ref<_AP_W, _private_range_ref, 1,
//                         _private_bit_ref<_AP_W2, _AP_S2> >(
//        *this, const_cast<_private_bit_ref<_AP_W2, _AP_S2>&>(a2));
//  }
//
//  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
//  INLINE ap_concat_ref<_AP_W, _private_range_ref, _AP_W2 + _AP_W3,
//                       ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> >
//  operator,(const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> &a2) {
//    return ap_concat_ref<_AP_W, _private_range_ref, _AP_W2 + _AP_W3,
//                         ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> >(
//        *this, const_cast<ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>&>(a2));
//  }
//
//  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
//            ap_o_mode _AP_O2, int _AP_N2>
//  INLINE ap_concat_ref<
//      _AP_W, _private_range_ref, _AP_W2,
//      af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >
//  operator,(
//      const af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> &a2) {
//    return ap_concat_ref<
//        _AP_W, _private_range_ref, _AP_W2,
//        af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >(
//        *this,
//        const_cast<
//            af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>&>(a2));
//  }
//
//  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
//            ap_o_mode _AP_O2, int _AP_N2>
//  INLINE
//      ap_concat_ref<_AP_W, _private_range_ref, 1,
//                    af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >
//      operator,(const af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>
//                    &a2) {
//    return ap_concat_ref<
//        _AP_W, _private_range_ref, 1,
//        af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >(
//        *this,
//        const_cast<af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2>&>(
//            a2));
//  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator==(const _private_range_ref<_AP_W2, _AP_S2>& op2) {
    ap_private<_AP_W, false> lhs = get();
    ap_private<_AP_W2, false> rhs = op2.get();
    return lhs == rhs;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator!=(const _private_range_ref<_AP_W2, _AP_S2>& op2) {
    ap_private<_AP_W, false> lhs = get();
    ap_private<_AP_W2, false> rhs = op2.get();
    return lhs != rhs;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator>(const _private_range_ref<_AP_W2, _AP_S2>& op2) {
    ap_private<_AP_W, false> lhs = get();
    ap_private<_AP_W2, false> rhs = op2.get();
    return lhs > rhs;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator>=(const _private_range_ref<_AP_W2, _AP_S2>& op2) {
    ap_private<_AP_W, false> lhs = get();
    ap_private<_AP_W2, false> rhs = op2.get();
    return lhs >= rhs;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator<(const _private_range_ref<_AP_W2, _AP_S2>& op2) {
    ap_private<_AP_W, false> lhs = get();
    ap_private<_AP_W2, false> rhs = op2.get();
    return lhs < rhs;
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator<=(const _private_range_ref<_AP_W2, _AP_S2>& op2) {
    ap_private<_AP_W, false> lhs = get();
    ap_private<_AP_W2, false> rhs = op2.get();
    return lhs <= rhs;
  }

  template <int _AP_W2>
  INLINE void set(const ap_private<_AP_W2, false>& val) {
    ap_private<_AP_W, _AP_S> vval = val;
    if (l_index > h_index) {
      for (int i = 0, j = l_index; j >= 0 && j >= h_index; j--, i++)
        (vval)[i] ? d_bv.set(j) : d_bv.clear(j);
    } else {
      if (_AP_W > 64) {
        ap_private<_AP_W, _AP_S> mask(-1);
        if (l_index > 0) {
          ap_private<_AP_W, false> mask1(-1);
          mask1 >>= _AP_W - l_index;
          mask1.flip();
          mask = mask1;
          // vval&=mask1;
          vval <<= l_index;
        }
        if (h_index < _AP_W - 1) {
          ap_private<_AP_W, false> mask2(-1);
          mask2 <<= h_index + 1;
          mask2.flip();
          mask &= mask2;
          vval &= mask2;
        }
        mask.flip();
        d_bv &= mask;
        d_bv |= vval;
      } else {
        uint64_t mask = ~0ULL >> (64 - _AP_W);
        if (l_index > 0) {
          uint64_t mask1 = mask;
          mask1 = mask & (mask1 >> (_AP_W - l_index));
          vval = mask & (vval << l_index);
          mask = ~mask1 & mask;
          // vval&=mask1;
        }
        if (h_index < _AP_W - 1) {
          uint64_t mask2 = ~0ULL >> (64 - _AP_W);
          mask2 = mask & (mask2 << (h_index + 1));
          mask &= ~mask2;
          vval &= ~mask2;
        }
        d_bv &= (~mask & (~0ULL >> (64 - _AP_W)));
        d_bv |= vval;
      }
    }
  }

  INLINE ap_private<_AP_W, false> get() const {
    ap_private<_AP_W, false> val(0);
    if (h_index < l_index) {
      for (int i = 0, j = l_index; j >= 0 && j >= h_index; j--, i++)
        if ((d_bv)[j]) val.set(i);
    } else {
      val = d_bv;
      val >>= l_index;
      if (h_index < _AP_W - 1) {
        if (_AP_W <= 64) {
          const static uint64_t mask =
              (~0ULL >> (64 > _AP_W ? (64 - _AP_W) : 0));
          val &= (mask >> (_AP_W - (h_index - l_index + 1)));
        } else {
          ap_private<_AP_W, false> mask(-1);
          mask >>= _AP_W - (h_index - l_index + 1);
          val &= mask;
        }
      }
    }
    return val;
  }

  INLINE ap_private<_AP_W, false> get() {
    ap_private<_AP_W, false> val(0);
    if (h_index < l_index) {
      for (int i = 0, j = l_index; j >= 0 && j >= h_index; j--, i++)
        if ((d_bv)[j]) val.set(i);
    } else {
      val = d_bv;
      val >>= l_index;
      if (h_index < _AP_W - 1) {
        if (_AP_W <= 64) {
          static const uint64_t mask = ~0ULL >> (64 > _AP_W ? (64 - _AP_W) : 0);
          return val &= ((mask) >> (_AP_W - (h_index - l_index + 1)));
        } else {
          ap_private<_AP_W, false> mask(-1);
          mask >>= _AP_W - (h_index - l_index + 1);
          val &= mask;
        }
      }
    }
    return val;
  }

  INLINE int length() const {
    return h_index >= l_index ? h_index - l_index + 1 : l_index - h_index + 1;
  }

  INLINE int to_int() const {
    ap_private<_AP_W, false> val = get();
    return val.to_int();
  }

  INLINE unsigned int to_uint() const {
    ap_private<_AP_W, false> val = get();
    return val.to_uint();
  }

  INLINE long to_long() const {
    ap_private<_AP_W, false> val = get();
    return val.to_long();
  }

  INLINE unsigned long to_ulong() const {
    ap_private<_AP_W, false> val = get();
    return val.to_ulong();
  }

  INLINE ap_slong to_int64() const {
    ap_private<_AP_W, false> val = get();
    return val.to_int64();
  }

  INLINE ap_ulong to_uint64() const {
    ap_private<_AP_W, false> val = get();
    return val.to_uint64();
  }

  INLINE std::string to_string(uint8_t radix = 2) const {
    return get().to_string(radix);
  }

  INLINE bool and_reduce() {
    bool ret = true;
    bool reverse = l_index > h_index;
    unsigned low = reverse ? h_index : l_index;
    unsigned high = reverse ? l_index : h_index;
    for (unsigned i = low; i != high; ++i) ret &= d_bv[i];
    return ret;
  }

  INLINE bool or_reduce() {
    bool ret = false;
    bool reverse = l_index > h_index;
    unsigned low = reverse ? h_index : l_index;
    unsigned high = reverse ? l_index : h_index;
    for (unsigned i = low; i != high; ++i) ret |= d_bv[i];
    return ret;
  }

  INLINE bool xor_reduce() {
    bool ret = false;
    bool reverse = l_index > h_index;
    unsigned low = reverse ? h_index : l_index;
    unsigned high = reverse ? l_index : h_index;
    for (unsigned i = low; i != high; ++i) ret ^= d_bv[i];
    return ret;
  }
}; // struct _private_range_ref.

/// Bit reference
/// Proxy class, which allows bit selection to be used as rvalue(for reading)
/// and lvalue(for writing)
//--------------------------------------------------------------
template <int _AP_W, bool _AP_S>
struct _private_bit_ref {
#ifdef _MSC_VER
#pragma warning(disable : 4521 4522)
#endif
  ap_private<_AP_W, _AP_S>& d_bv;
  int d_index;

 public:
  // copy ctor.
  INLINE _private_bit_ref(const _private_bit_ref<_AP_W, _AP_S>& ref)
      : d_bv(ref.d_bv), d_index(ref.d_index) {}

  // director ctor.
  INLINE _private_bit_ref(ap_private<_AP_W, _AP_S>& bv, int index = 0)
      : d_bv(bv), d_index(index) {
    _AP_WARNING(d_index < 0, "Index of bit vector  (%d) cannot be negative.\n",
                d_index);
    _AP_WARNING(d_index >= _AP_W,
                "Index of bit vector (%d) out of range (%d).\n", d_index, _AP_W);
  }

  INLINE operator bool() const { return d_bv.get_bit(d_index); }

  INLINE bool to_bool() const { return operator bool(); }

  template <typename T>
  INLINE _private_bit_ref& operator=(const T& val) {
    if (!!val)
      d_bv.set(d_index);
    else
      d_bv.clear(d_index);
    return *this;
  }

//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<1, _private_bit_ref, _AP_W2, ap_private<_AP_W2,
//  _AP_S2> >
//  operator,(ap_private<_AP_W2, _AP_S2> &a2) const {
//    return ap_concat_ref<1, _private_bit_ref, _AP_W2, ap_private<_AP_W2,
//    _AP_S2> >(
//        const_cast<_private_bit_ref<_AP_W, _AP_S>&>(*this), a2);
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<1, _private_bit_ref, _AP_W2,
//  _private_range_ref<_AP_W2,
//  _AP_S2> >
//  operator,(const _private_range_ref<_AP_W2, _AP_S2> &a2) const {
//    return ap_concat_ref<1, _private_bit_ref, _AP_W2,
//    _private_range_ref<_AP_W2,
//    _AP_S2> >(
//        const_cast<_private_bit_ref<_AP_W, _AP_S>&>(*this),
//        const_cast<_private_range_ref<_AP_W2, _AP_S2>&>(a2));
//  }
//
//  template <int _AP_W2, bool _AP_S2>
//  INLINE ap_concat_ref<1, _private_bit_ref, 1, _private_bit_ref<_AP_W2,
//  _AP_S2> > operator,(
//      const _private_bit_ref<_AP_W2, _AP_S2> &a2) const {
//    return ap_concat_ref<1, _private_bit_ref, 1,
//    _private_bit_ref<_AP_W2, _AP_S2> >(
//        const_cast<_private_bit_ref<_AP_W, _AP_S>&>(*this),
//        const_cast<_private_bit_ref<_AP_W2, _AP_S2>&>(a2));
//  }
//
//  INLINE ap_concat_ref<1, _private_bit_ref, 1, _private_bit_ref>
//  operator,(
//      const _private_bit_ref &a2) const {
//    return ap_concat_ref<1, _private_bit_ref, 1, _private_bit_ref>(
//        const_cast<_private_bit_ref<_AP_W, _AP_S>&>(*this),
//        const_cast<_private_bit_ref&>(a2));
//  }
//
//  template <int _AP_W2, typename _AP_T2, int _AP_W3, typename _AP_T3>
//  INLINE ap_concat_ref<1, _private_bit_ref, _AP_W2 + _AP_W3,
//                       ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> >
//  operator,(const ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> &a2) const {
//    return ap_concat_ref<1, _private_bit_ref, _AP_W2 + _AP_W3,
//                         ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3> >(
//        const_cast<_private_bit_ref<_AP_W, _AP_S>&>(*this),
//        const_cast<ap_concat_ref<_AP_W2, _AP_T2, _AP_W3, _AP_T3>&>(a2));
//  }
//
//  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
//            ap_o_mode _AP_O2, int _AP_N2>
//  INLINE ap_concat_ref<
//      1, _private_bit_ref, _AP_W2,
//      af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >
//  operator,(const af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2,
//  _AP_N2>
//                &a2) const {
//    return ap_concat_ref<
//        1, _private_bit_ref, _AP_W2,
//        af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2, _AP_N2> >(
//        const_cast<_private_bit_ref<_AP_W, _AP_S>&>(*this),
//        const_cast<
//            af_range_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2,
//            _AP_N2>&>(a2));
//  }
//
//  template <int _AP_W2, int _AP_I2, bool _AP_S2, ap_q_mode _AP_Q2,
//            ap_o_mode _AP_O2, int _AP_N2>
//  INLINE
//      ap_concat_ref<1, _private_bit_ref, 1,
//                    af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2,
//                    _AP_N2> >
//      operator,(const af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2,
//      _AP_N2>
//                    &a2) const {
//    return ap_concat_ref<1, _private_bit_ref, 1, af_bit_ref<_AP_W2,
//    _AP_I2, _AP_S2,
//                                                      _AP_Q2, _AP_O2,
//                                                      _AP_N2> >(
//        const_cast<_private_bit_ref<_AP_W, _AP_S>&>(*this),
//        const_cast<af_bit_ref<_AP_W2, _AP_I2, _AP_S2, _AP_Q2, _AP_O2,
//        _AP_N2>&>(
//            a2));
//  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator==(const _private_bit_ref<_AP_W2, _AP_S2>& op) const {
    return get() == op.get();
  }

  template <int _AP_W2, bool _AP_S2>
  INLINE bool operator!=(const _private_bit_ref<_AP_W2, _AP_S2>& op) const {
    return get() != op.get();
  }

  INLINE bool get() const { return operator bool(); }

  //  template <int _AP_W3>
  //  INLINE void set(const ap_private<_AP_W3, false>& val) {
  //    operator=(val);
  //  }

  //  INLINE bool operator~() const {
  //    bool bit = (d_bv)[d_index];
  //    return bit ? false : true;
  //  }

  INLINE int length() const { return 1; }

  //  INLINE std::string to_string() const {
  //    bool val = get();
  //    return val ? "1" : "0";
  //  }

}; // struct _private_bit_ref.

// char a[100];
// char* ptr = a;
// ap_int<2> n = 3;
// char* ptr2 = ptr + n*2;
// avoid ambiguous errors
#define OP_BIN_MIX_PTR(BIN_OP)                                           \
  template <typename PTR_TYPE, int _AP_W, bool _AP_S>                    \
  INLINE PTR_TYPE* operator BIN_OP(PTR_TYPE* i_op,                       \
                                   const ap_private<_AP_W, _AP_S>& op) { \
    typename ap_private<_AP_W, _AP_S>::ValType op2 = op;                 \
    return i_op BIN_OP op2;                                              \
  }                                                                      \
  template <typename PTR_TYPE, int _AP_W, bool _AP_S>                    \
  INLINE PTR_TYPE* operator BIN_OP(const ap_private<_AP_W, _AP_S>& op,   \
                                   PTR_TYPE* i_op) {                     \
    typename ap_private<_AP_W, _AP_S>::ValType op2 = op;                 \
    return op2 BIN_OP i_op;                                              \
  }

OP_BIN_MIX_PTR(+)
OP_BIN_MIX_PTR(-)
#undef OP_BIN_MIX_PTR

// float OP ap_int
// when ap_int<wa>'s width > 64, then trunc ap_int<w> to ap_int<64>
#define OP_BIN_MIX_FLOAT(BIN_OP, C_TYPE)                              \
  template <int _AP_W, bool _AP_S>                                    \
  INLINE C_TYPE operator BIN_OP(C_TYPE i_op,                          \
                                const ap_private<_AP_W, _AP_S>& op) { \
    typename ap_private<_AP_W, _AP_S>::ValType op2 = op;              \
    return i_op BIN_OP op2;                                           \
  }                                                                   \
  template <int _AP_W, bool _AP_S>                                    \
  INLINE C_TYPE operator BIN_OP(const ap_private<_AP_W, _AP_S>& op,   \
                                C_TYPE i_op) {                        \
    typename ap_private<_AP_W, _AP_S>::ValType op2 = op;              \
    return op2 BIN_OP i_op;                                           \
  }

#define OPS_MIX_FLOAT(C_TYPE) \
  OP_BIN_MIX_FLOAT(*, C_TYPE) \
  OP_BIN_MIX_FLOAT(/, C_TYPE) \
  OP_BIN_MIX_FLOAT(+, C_TYPE) \
  OP_BIN_MIX_FLOAT(-, C_TYPE)

OPS_MIX_FLOAT(float)
OPS_MIX_FLOAT(double)
#undef OP_BIN_MIX_FLOAT
#undef OPS_MIX_FLOAT

/// Operators mixing Integers with AP_Int
// ----------------------------------------------------------------

// partially specialize template argument _AP_C in order that:
// for _AP_W > 64, we will explicitly convert operand with native data type
// into corresponding ap_private
// for _AP_W <= 64, we will implicitly convert operand with ap_private into
// (unsigned) long long
#define OP_BIN_MIX_INT(BIN_OP, C_TYPE, _AP_WI, _AP_SI, RTYPE)                  \
  template <int _AP_W, bool _AP_S>                                             \
  INLINE                                                                       \
      typename ap_private<_AP_WI, _AP_SI>::template RType<_AP_W, _AP_S>::RTYPE \
      operator BIN_OP(C_TYPE i_op, const ap_private<_AP_W, _AP_S>& op) {       \
    return ap_private<_AP_WI, _AP_SI>(i_op).operator BIN_OP(op);               \
  }                                                                            \
  template <int _AP_W, bool _AP_S>                                             \
  INLINE                                                                       \
      typename ap_private<_AP_W, _AP_S>::template RType<_AP_WI, _AP_SI>::RTYPE \
      operator BIN_OP(const ap_private<_AP_W, _AP_S>& op, C_TYPE i_op) {       \
    return op.operator BIN_OP(ap_private<_AP_WI, _AP_SI>(i_op));               \
  }

#define OP_REL_MIX_INT(REL_OP, C_TYPE, _AP_W2, _AP_S2)                     \
  template <int _AP_W, bool _AP_S>                                         \
  INLINE bool operator REL_OP(const ap_private<_AP_W, _AP_S>& op,          \
                              C_TYPE op2) {                                \
    return op.operator REL_OP(ap_private<_AP_W2, _AP_S2>(op2));            \
  }                                                                        \
  template <int _AP_W, bool _AP_S>                                         \
  INLINE bool operator REL_OP(C_TYPE op2,                                  \
                              const ap_private<_AP_W, _AP_S, false>& op) { \
    return ap_private<_AP_W2, _AP_S2>(op2).operator REL_OP(op);            \
  }

#define OP_ASSIGN_MIX_INT(ASSIGN_OP, C_TYPE, _AP_W2, _AP_S2)       \
  template <int _AP_W, bool _AP_S>                                 \
  INLINE ap_private<_AP_W, _AP_S>& operator ASSIGN_OP(             \
      ap_private<_AP_W, _AP_S>& op, C_TYPE op2) {                  \
    return op.operator ASSIGN_OP(ap_private<_AP_W2, _AP_S2>(op2)); \
  }

#define OP_BIN_SHIFT_INT(BIN_OP, C_TYPE, _AP_WI, _AP_SI, RTYPE)                \
  template <int _AP_W, bool _AP_S>                                             \
  C_TYPE operator BIN_OP(C_TYPE i_op,                                          \
                         const ap_private<_AP_W, _AP_S, false>& op) {          \
    return i_op BIN_OP(op.get_VAL());                                          \
  }                                                                            \
  template <int _AP_W, bool _AP_S>                                             \
  INLINE                                                                       \
      typename ap_private<_AP_W, _AP_S>::template RType<_AP_WI, _AP_SI>::RTYPE \
      operator BIN_OP(const ap_private<_AP_W, _AP_S>& op, C_TYPE i_op) {       \
    return op.operator BIN_OP(i_op);                                           \
  }

#define OP_ASSIGN_RSHIFT_INT(ASSIGN_OP, C_TYPE, _AP_W2, _AP_S2) \
  template <int _AP_W, bool _AP_S>                              \
  INLINE ap_private<_AP_W, _AP_S>& operator ASSIGN_OP(          \
      ap_private<_AP_W, _AP_S>& op, C_TYPE op2) {               \
    op = op.operator>>(op2);                                    \
    return op;                                                  \
  }

#define OP_ASSIGN_LSHIFT_INT(ASSIGN_OP, C_TYPE, _AP_W2, _AP_S2) \
  template <int _AP_W, bool _AP_S>                              \
  INLINE ap_private<_AP_W, _AP_S>& operator ASSIGN_OP(          \
      ap_private<_AP_W, _AP_S>& op, C_TYPE op2) {               \
    op = op.operator<<(op2);                                    \
    return op;                                                  \
  }

#define OPS_MIX_INT(C_TYPE, _AP_W2, _AP_S2)              \
  OP_BIN_MIX_INT(*, C_TYPE, (_AP_W2), (_AP_S2), mult)    \
  OP_BIN_MIX_INT(+, C_TYPE, (_AP_W2), (_AP_S2), plus)    \
  OP_BIN_MIX_INT(-, C_TYPE, (_AP_W2), (_AP_S2), minus)   \
  OP_BIN_MIX_INT(/, C_TYPE, (_AP_W2), (_AP_S2), div)     \
  OP_BIN_MIX_INT(%, C_TYPE, (_AP_W2), (_AP_S2), mod)     \
  OP_BIN_MIX_INT(&, C_TYPE, (_AP_W2), (_AP_S2), logic)   \
  OP_BIN_MIX_INT(|, C_TYPE, (_AP_W2), (_AP_S2), logic)   \
  OP_BIN_MIX_INT (^, C_TYPE, (_AP_W2), (_AP_S2), logic)  \
  OP_BIN_SHIFT_INT(>>, C_TYPE, (_AP_W2), (_AP_S2), arg1) \
  OP_BIN_SHIFT_INT(<<, C_TYPE, (_AP_W2), (_AP_S2), arg1) \
                                                         \
  OP_ASSIGN_MIX_INT(+=, C_TYPE, (_AP_W2), (_AP_S2))      \
  OP_ASSIGN_MIX_INT(-=, C_TYPE, (_AP_W2), (_AP_S2))      \
  OP_ASSIGN_MIX_INT(*=, C_TYPE, (_AP_W2), (_AP_S2))      \
  OP_ASSIGN_MIX_INT(/=, C_TYPE, (_AP_W2), (_AP_S2))      \
  OP_ASSIGN_MIX_INT(%=, C_TYPE, (_AP_W2), (_AP_S2))      \
  OP_ASSIGN_MIX_INT(&=, C_TYPE, (_AP_W2), (_AP_S2))      \
  OP_ASSIGN_MIX_INT(|=, C_TYPE, (_AP_W2), (_AP_S2))      \
  OP_ASSIGN_MIX_INT(^=, C_TYPE, (_AP_W2), (_AP_S2))      \
  OP_ASSIGN_RSHIFT_INT(>>=, C_TYPE, (_AP_W2), (_AP_S2))  \
  OP_ASSIGN_LSHIFT_INT(<<=, C_TYPE, (_AP_W2), (_AP_S2))  \
                                                         \
  OP_REL_MIX_INT(>, C_TYPE, (_AP_W2), (_AP_S2))          \
  OP_REL_MIX_INT(<, C_TYPE, (_AP_W2), (_AP_S2))          \
  OP_REL_MIX_INT(>=, C_TYPE, (_AP_W2), (_AP_S2))         \
  OP_REL_MIX_INT(<=, C_TYPE, (_AP_W2), (_AP_S2))         \
  OP_REL_MIX_INT(==, C_TYPE, (_AP_W2), (_AP_S2))         \
  OP_REL_MIX_INT(!=, C_TYPE, (_AP_W2), (_AP_S2))

OPS_MIX_INT(bool, 1, false)
OPS_MIX_INT(char, 8, CHAR_IS_SIGNED)
OPS_MIX_INT(signed char, 8, true)
OPS_MIX_INT(unsigned char, 8, false)
OPS_MIX_INT(short, sizeof(short) * 8, true)
OPS_MIX_INT(unsigned short, sizeof(unsigned short) * 8, false)
OPS_MIX_INT(int, sizeof(int) * 8, true)
OPS_MIX_INT(unsigned int, sizeof(unsigned int) * 8, false)
OPS_MIX_INT(long, sizeof(long) * 8, true)
OPS_MIX_INT(unsigned long, sizeof(unsigned long) * 8, false)
OPS_MIX_INT(ap_slong, sizeof(ap_slong) * 8, true)
OPS_MIX_INT(ap_ulong, sizeof(ap_ulong) * 8, false)

#undef OP_BIN_MIX_INT
#undef OP_BIN_SHIFT_INT
#undef OP_ASSIGN_MIX_INT
#undef OP_ASSIGN_RSHIFT_INT
#undef OP_ASSIGN_LSHIFT_INT
#undef OP_REL_MIX_INT
#undef OPS_MIX_INT

#define OP_BIN_MIX_RANGE(BIN_OP, RTYPE)                                     \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>               \
  INLINE typename ap_private<_AP_W1, _AP_S1>::template RType<_AP_W2,        \
                                                             _AP_S2>::RTYPE \
  operator BIN_OP(const _private_range_ref<_AP_W1, _AP_S1>& op1,            \
                  const ap_private<_AP_W2, _AP_S2>& op2) {                  \
    return ap_private<_AP_W1, false>(op1).operator BIN_OP(op2);             \
  }                                                                         \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>               \
  INLINE typename ap_private<_AP_W1, _AP_S1>::template RType<_AP_W2,        \
                                                             _AP_S2>::RTYPE \
  operator BIN_OP(const ap_private<_AP_W1, _AP_S1>& op1,                    \
                  const _private_range_ref<_AP_W2, _AP_S2>& op2) {          \
    return op1.operator BIN_OP(ap_private<_AP_W2, false>(op2));             \
  }

#define OP_ASSIGN_MIX_RANGE(ASSIGN_OP)                             \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>      \
  INLINE ap_private<_AP_W1, _AP_S1>& operator ASSIGN_OP(           \
      ap_private<_AP_W1, _AP_S1>& op1,                             \
      const _private_range_ref<_AP_W2, _AP_S2>& op2) {             \
    return op1.operator ASSIGN_OP(ap_private<_AP_W2, false>(op2)); \
  }                                                                \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>      \
  INLINE _private_range_ref<_AP_W1, _AP_S1>& operator ASSIGN_OP(   \
      _private_range_ref<_AP_W1, _AP_S1>& op1,                     \
      ap_private<_AP_W2, _AP_S2>& op2) {                           \
    ap_private<_AP_W1, false> tmp(op1);                            \
    tmp.operator ASSIGN_OP(op2);                                   \
    op1 = tmp;                                                     \
    return op1;                                                    \
  }

#define OP_REL_MIX_RANGE(REL_OP)                                               \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>                  \
  INLINE bool operator REL_OP(const _private_range_ref<_AP_W1, _AP_S1>& op1,   \
                              const ap_private<_AP_W2, _AP_S2>& op2) {         \
    return ap_private<_AP_W1, false>(op1).operator REL_OP(op2);                \
  }                                                                            \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>                  \
  INLINE bool operator REL_OP(const ap_private<_AP_W1, _AP_S1>& op1,           \
                              const _private_range_ref<_AP_W2, _AP_S2>& op2) { \
    return op1.operator REL_OP(op2.operator ap_private<_AP_W2, false>());      \
  }

OP_BIN_MIX_RANGE(+, plus)
OP_BIN_MIX_RANGE(-, minus)
OP_BIN_MIX_RANGE(*, mult)
OP_BIN_MIX_RANGE(/, div)
OP_BIN_MIX_RANGE(%, mod)
OP_BIN_MIX_RANGE(&, logic)
OP_BIN_MIX_RANGE(|, logic)
OP_BIN_MIX_RANGE(^, logic)
OP_BIN_MIX_RANGE(>>, arg1)
OP_BIN_MIX_RANGE(<<, arg1)
#undef OP_BIN_MIX_RANGE

OP_ASSIGN_MIX_RANGE(+=)
OP_ASSIGN_MIX_RANGE(-=)
OP_ASSIGN_MIX_RANGE(*=)
OP_ASSIGN_MIX_RANGE(/=)
OP_ASSIGN_MIX_RANGE(%=)
OP_ASSIGN_MIX_RANGE(&=)
OP_ASSIGN_MIX_RANGE(|=)
OP_ASSIGN_MIX_RANGE(^=)
OP_ASSIGN_MIX_RANGE(>>=)
OP_ASSIGN_MIX_RANGE(<<=)
#undef OP_ASSIGN_MIX_RANGE

OP_REL_MIX_RANGE(>)
OP_REL_MIX_RANGE(<)
OP_REL_MIX_RANGE(>=)
OP_REL_MIX_RANGE(<=)
OP_REL_MIX_RANGE(==)
OP_REL_MIX_RANGE(!=)
#undef OP_REL_MIX_RANGE

#define OP_BIN_MIX_BIT(BIN_OP, RTYPE)                                         \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>                 \
  INLINE typename ap_private<1, false>::template RType<_AP_W2, _AP_S2>::RTYPE \
  operator BIN_OP(const _private_bit_ref<_AP_W1, _AP_S1>& op1,                \
                  const ap_private<_AP_W2, _AP_S2>& op2) {                    \
    return ap_private<1, false>(op1).operator BIN_OP(op2);                    \
  }                                                                           \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>                 \
  INLINE typename ap_private<_AP_W1, _AP_S1>::template RType<1, false>::RTYPE \
  operator BIN_OP(const ap_private<_AP_W1, _AP_S1>& op1,                      \
                  const _private_bit_ref<_AP_W2, _AP_S2>& op2) {              \
    return op1.operator BIN_OP(ap_private<1, false>(op2));                    \
  }

#define OP_ASSIGN_MIX_BIT(ASSIGN_OP)                           \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>  \
  INLINE ap_private<_AP_W1, _AP_S1>& operator ASSIGN_OP(       \
      ap_private<_AP_W1, _AP_S1>& op1,                         \
      _private_bit_ref<_AP_W2, _AP_S2>& op2) {                 \
    return op1.operator ASSIGN_OP(ap_private<1, false>(op2));  \
  }                                                            \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>  \
  INLINE _private_bit_ref<_AP_W1, _AP_S1>& operator ASSIGN_OP( \
      _private_bit_ref<_AP_W1, _AP_S1>& op1,                   \
      ap_private<_AP_W2, _AP_S2>& op2) {                       \
    ap_private<1, false> tmp(op1);                             \
    tmp.operator ASSIGN_OP(op2);                               \
    op1 = tmp;                                                 \
    return op1;                                                \
  }

#define OP_REL_MIX_BIT(REL_OP)                                               \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>                \
  INLINE bool operator REL_OP(const _private_bit_ref<_AP_W1, _AP_S1>& op1,   \
                              const ap_private<_AP_W2, _AP_S2>& op2) {       \
    return ap_private<_AP_W1, false>(op1).operator REL_OP(op2);              \
  }                                                                          \
  template <int _AP_W1, bool _AP_S1, int _AP_W2, bool _AP_S2>                \
  INLINE bool operator REL_OP(const ap_private<_AP_W1, _AP_S1>& op1,         \
                              const _private_bit_ref<_AP_W2, _AP_S2>& op2) { \
    return op1.operator REL_OP(ap_private<1, false>(op2));                   \
  }

OP_ASSIGN_MIX_BIT(+=)
OP_ASSIGN_MIX_BIT(-=)
OP_ASSIGN_MIX_BIT(*=)
OP_ASSIGN_MIX_BIT(/=)
OP_ASSIGN_MIX_BIT(%=)
OP_ASSIGN_MIX_BIT(&=)
OP_ASSIGN_MIX_BIT(|=)
OP_ASSIGN_MIX_BIT(^=)
OP_ASSIGN_MIX_BIT(>>=)
OP_ASSIGN_MIX_BIT(<<=)
#undef OP_ASSIGN_MIX_BIT

OP_BIN_MIX_BIT(+, plus)
OP_BIN_MIX_BIT(-, minus)
OP_BIN_MIX_BIT(*, mult)
OP_BIN_MIX_BIT(/, div)
OP_BIN_MIX_BIT(%, mod)
OP_BIN_MIX_BIT(&, logic)
OP_BIN_MIX_BIT(|, logic)
OP_BIN_MIX_BIT(^, logic)
OP_BIN_MIX_BIT(>>, arg1)
OP_BIN_MIX_BIT(<<, arg1)
#undef OP_BIN_MIX_BIT

OP_REL_MIX_BIT(>)
OP_REL_MIX_BIT(<)
OP_REL_MIX_BIT(<=)
OP_REL_MIX_BIT(>=)
OP_REL_MIX_BIT(==)
OP_REL_MIX_BIT(!=)
#undef OP_REL_MIX_BIT

#define REF_REL_OP_MIX_INT(REL_OP, C_TYPE, _AP_W2, _AP_S2)                  \
  template <int _AP_W, bool _AP_S>                                          \
  INLINE bool operator REL_OP(const _private_range_ref<_AP_W, _AP_S>& op,   \
                              C_TYPE op2) {                                 \
    return (ap_private<_AP_W, false>(op))                                   \
        .                                                                   \
        operator REL_OP(ap_private<_AP_W2, _AP_S2>(op2));                   \
  }                                                                         \
  template <int _AP_W, bool _AP_S>                                          \
  INLINE bool operator REL_OP(C_TYPE op2,                                   \
                              const _private_range_ref<_AP_W, _AP_S>& op) { \
    return ap_private<_AP_W2, _AP_S2>(op2).operator REL_OP(                 \
        ap_private<_AP_W, false>(op));                                      \
  }                                                                         \
  template <int _AP_W, bool _AP_S>                                          \
  INLINE bool operator REL_OP(const _private_bit_ref<_AP_W, _AP_S>& op,     \
                              C_TYPE op2) {                                 \
    return (bool(op))REL_OP op2;                                            \
  }                                                                         \
  template <int _AP_W, bool _AP_S>                                          \
  INLINE bool operator REL_OP(C_TYPE op2,                                   \
                              const _private_bit_ref<_AP_W, _AP_S>& op) {   \
    return op2 REL_OP(bool(op));                                            \
  }

#define REF_REL_MIX_INT(C_TYPE, _AP_W2, _AP_S2)      \
  REF_REL_OP_MIX_INT(>, C_TYPE, (_AP_W2), (_AP_S2))  \
  REF_REL_OP_MIX_INT(<, C_TYPE, (_AP_W2), (_AP_S2))  \
  REF_REL_OP_MIX_INT(>=, C_TYPE, (_AP_W2), (_AP_S2)) \
  REF_REL_OP_MIX_INT(<=, C_TYPE, (_AP_W2), (_AP_S2)) \
  REF_REL_OP_MIX_INT(==, C_TYPE, (_AP_W2), (_AP_S2)) \
  REF_REL_OP_MIX_INT(!=, C_TYPE, (_AP_W2), (_AP_S2))

REF_REL_MIX_INT(bool, 1, false)
REF_REL_MIX_INT(char, 8, CHAR_IS_SIGNED)
REF_REL_MIX_INT(signed char, 8, true)
REF_REL_MIX_INT(unsigned char, 8, false)
REF_REL_MIX_INT(short, sizeof(short) * 8, true)
REF_REL_MIX_INT(unsigned short, sizeof(unsigned short) * 8, false)
REF_REL_MIX_INT(int, sizeof(int) * 8, true)
REF_REL_MIX_INT(unsigned int, sizeof(unsigned int) * 8, false)
REF_REL_MIX_INT(long, sizeof(long) * 8, true)
REF_REL_MIX_INT(unsigned long, sizeof(unsigned long) * 8, false)
REF_REL_MIX_INT(ap_slong, sizeof(ap_slong) * 8, true)
REF_REL_MIX_INT(ap_ulong, sizeof(ap_ulong) * 8, false)
#undef REF_REL_OP_MIX_INT
#undef REF_REL_MIX_INT

#define REF_BIN_OP_MIX_INT(BIN_OP, RTYPE, C_TYPE, _AP_W2, _AP_S2)              \
  template <int _AP_W, bool _AP_S>                                             \
  INLINE                                                                       \
      typename ap_private<_AP_W, false>::template RType<_AP_W2, _AP_S2>::RTYPE \
      operator BIN_OP(const _private_range_ref<_AP_W, _AP_S>& op,              \
                      C_TYPE op2) {                                            \
    return (ap_private<_AP_W, false>(op))                                      \
        .                                                                      \
        operator BIN_OP(ap_private<_AP_W2, _AP_S2>(op2));                      \
  }                                                                            \
  template <int _AP_W, bool _AP_S>                                             \
  INLINE                                                                       \
      typename ap_private<_AP_W2, _AP_S2>::template RType<_AP_W, false>::RTYPE \
      operator BIN_OP(C_TYPE op2,                                              \
                      const _private_range_ref<_AP_W, _AP_S>& op) {            \
    return ap_private<_AP_W2, _AP_S2>(op2).operator BIN_OP(                    \
        ap_private<_AP_W, false>(op));                                         \
  }

#define REF_BIN_MIX_INT(C_TYPE, _AP_W2, _AP_S2)            \
  REF_BIN_OP_MIX_INT(+, plus, C_TYPE, (_AP_W2), (_AP_S2))  \
  REF_BIN_OP_MIX_INT(-, minus, C_TYPE, (_AP_W2), (_AP_S2)) \
  REF_BIN_OP_MIX_INT(*, mult, C_TYPE, (_AP_W2), (_AP_S2))  \
  REF_BIN_OP_MIX_INT(/, div, C_TYPE, (_AP_W2), (_AP_S2))   \
  REF_BIN_OP_MIX_INT(%, mod, C_TYPE, (_AP_W2), (_AP_S2))   \
  REF_BIN_OP_MIX_INT(&, logic, C_TYPE, (_AP_W2), (_AP_S2)) \
  REF_BIN_OP_MIX_INT(|, logic, C_TYPE, (_AP_W2), (_AP_S2)) \
  REF_BIN_OP_MIX_INT(^, logic, C_TYPE, (_AP_W2), (_AP_S2)) \
  REF_BIN_OP_MIX_INT(>>, arg1, C_TYPE, (_AP_W2), (_AP_S2)) \
  REF_BIN_OP_MIX_INT(<<, arg1, C_TYPE, (_AP_W2), (_AP_S2))

REF_BIN_MIX_INT(bool, 1, false)
REF_BIN_MIX_INT(char, 8, CHAR_IS_SIGNED)
REF_BIN_MIX_INT(signed char, 8, true)
REF_BIN_MIX_INT(unsigned char, 8, false)
REF_BIN_MIX_INT(short, sizeof(short) * 8, true)
REF_BIN_MIX_INT(unsigned short, sizeof(unsigned short) * 8, false)
REF_BIN_MIX_INT(int, sizeof(int) * 8, true)
REF_BIN_MIX_INT(unsigned int, sizeof(unsigned int) * 8, false)
REF_BIN_MIX_INT(long, sizeof(long) * 8, true)
REF_BIN_MIX_INT(unsigned long, sizeof(unsigned long) * 8, false)
REF_BIN_MIX_INT(ap_slong, sizeof(ap_slong) * 8, true)
REF_BIN_MIX_INT(ap_ulong, sizeof(ap_ulong) * 8, false)
#undef REF_BIN_OP_MIX_INT
#undef REF_BIN_MIX_INT

#define REF_BIN_OP(BIN_OP, RTYPE)                                             \
  template <int _AP_W, bool _AP_S, int _AP_W2, bool _AP_S2>                   \
  INLINE                                                                      \
      typename ap_private<_AP_W, false>::template RType<_AP_W2, false>::RTYPE \
      operator BIN_OP(const _private_range_ref<_AP_W, _AP_S>& lhs,            \
                      const _private_range_ref<_AP_W2, _AP_S2>& rhs) {        \
    return ap_private<_AP_W, false>(lhs).operator BIN_OP(                     \
        ap_private<_AP_W2, false>(rhs));                                      \
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
#undef REF_BIN_OP

//************************************************************************
//  Implement
//      ap_private<M+N> = ap_concat_ref<M> OP ap_concat_ref<N>
//  for operators  +, -, *, /, %, >>, <<, &, |, ^
//  Without these operators the operands are converted to int64 and
//  larger results lose informations (higher order bits).
//
//                       operand OP
//                      /          |
//              left-concat        right-concat
//                /     |           /         |
//         <LW1,LT1>  <LW2,LT2>   <RW1,RT1>   <RW2,RT2>
//
//      _AP_LW1, _AP_LT1 (width and type of left-concat's left side)
//      _AP_LW2, _AP_LT2 (width and type of left-concat's right side)
//  Similarly for RHS of operand OP: _AP_RW1, AP_RW2, _AP_RT1, _AP_RT2
//
//  In Verilog 2001 result of concatenation is always unsigned even
//  when both sides are signed.
//************************************************************************

#endif // ifndef __AP_PRIVATE_H__

// -*- cpp -*-
