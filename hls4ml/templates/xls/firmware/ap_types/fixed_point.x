// Copyright 2025 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// A fixed point number type and operations on it.

import std;
import apfloat;

// A fixed point number represented in the type as a number of bits and binary point offset, and at
// runtime by a significand (some bits). To convert this to a Real value, treat significand as an
// integer and multiply by 2^(BINARY_EXPONENT).
//
// Documentation below uses the term 'representable' to mean the bits that could be 1 or 0 in a
// fixed point number. Bits that are always 0 are not considered representable (i.e., the least
// significant integer bits that are always zero for a value with a positive binary exponent that is
// larger than the width, or the most significant fractional bits that are always zero for a value
// with a negative binary exponent and a width that is smaller than the magnitude of the binary
// exponent).
//
// Examples:
// 0.75 would be represented using minimal bits as FixedPoint2<2, -2> { significand: 0b11 }
// 0.75 would be represented using 2 extra bits as FixedPoint2<4, -2> { significand: 0b0011 }
// (1/16 + 1/64) would be represented using minimal bits as FixedPoint2<3, -6> { significand: 0b101
// }
// 20 would be represented using minimal bits as FixedPoint2<3, 2> { significand: 0b101 }
//
// TODO when https://github.com/google/xls/issues/1841 is resolved, undo the workaround
// that changed BINARY_EXPONENT:s32 to (EXPONENT_IS_NEGATIVE:u32, BINARY_UEXPONENT: u32).
//
// BINARY_UEXPONENT means unsigned exponent. It is the magnitude of the binary exponent.
//
// TODO when https://github.com/google/xls/issues/1848 is resolved, delete the two unused
// fields
//
// TODO when https://github.com/google/xls/issues/1861 is resolved, make the type
// sign-parametric (i.e. xN[sign][NUM_BITS])
pub struct FixedPoint<NUM_BITS: u32, EXPONENT_IS_NEGATIVE: u32, BINARY_UEXPONENT: u32> {
    significand: sN[NUM_BITS],  // concatenation of integer and fraction bits
    // TODO delete when https://github.com/google/xls/issues/1848 is resolved
    unused_eis: uN[EXPONENT_IS_NEGATIVE],
    // TODO delete when https://github.com/google/xls/issues/1848 is resolved
    unused_exp: uN[BINARY_UEXPONENT],
}

// Creates a fixed point number with the given significand. The two unused fields are set to 0.
// Exists because it's annoying to set the unused fields to 0 manually.
//
// TODO delete when https://github.com/google/xls/issues/1848 is resolved: we won't
// need this helper to set the two dummy fields to 0
pub fn make_fixed_point_with_zeros<NUM_BITS: u32, EXPONENT_IS_NEGATIVE: u32, BINARY_UEXPONENT: u32>
    (significand: sN[NUM_BITS]) -> FixedPoint<NUM_BITS, EXPONENT_IS_NEGATIVE, BINARY_UEXPONENT> {
    FixedPoint<NUM_BITS, EXPONENT_IS_NEGATIVE, BINARY_UEXPONENT> {
        significand,
        unused_eis: uN[EXPONENT_IS_NEGATIVE]:0,
        unused_exp: uN[BINARY_UEXPONENT]:0,
    }
}

// Converts from sign & magnitude to two's complement.
//
// TODO delete when https://github.com/google/xls/issues/1848 is resolved:
// we won't need to convert between two's complement and sign & magnitude representations.
pub fn binary_exponent(EXPONENT_IS_NEGATIVE: u32, BINARY_UEXPONENT: u32) -> s32 {
    if EXPONENT_IS_NEGATIVE > u32:0 { -BINARY_UEXPONENT as s32 } else { BINARY_UEXPONENT as s32 }
}

// Converts from two's complement to sign of sign & magnitude representation.
//
// TODO delete when https://github.com/google/xls/issues/1848 is resolved:
// we won't need to convert between two's complement and sign & magnitude representations.
pub fn is_negative(binary_exponent: s32) -> u32 {
    if binary_exponent < s32:0 { u32:1 } else { u32:0 }
}

// Converts from two's complement to magnitude of sign & magnitude representation.
//
// TODO delete when https://github.com/google/xls/issues/1848 is resolved:
// we won't need to convert between two's complement and sign & magnitude representations.
pub fn binary_uexponent(binary_exponent: s32) -> u32 {
    if binary_exponent < s32:0 { (-binary_exponent) as u32 } else { binary_exponent as u32 }
}

// Creates a FixedPoint of with appropriate sign and magnitude representation, given the signed
// binary exponent. This is a convenience function to avoid having to determine the sign and
// magnitude.
//
// Note that BINARY_EXPONENT is located first so that you can specify it and elide the
// other type parameters, as they are inferrable.
// E.g. make_fixed_point<s32:-2>(s6:31) = 31 * 2^-2 = 7.75
//
// TODO change when https://github.com/google/xls/issues/1848 is resolved:
// we won't need to convert between two's complement and sign & magnitude representations.
pub fn make_fixed_point
    <BINARY_EXPONENT: s32, NUM_BITS: u32,
     EXPONENT_IS_NEGATIVE: u32 = {is_negative(BINARY_EXPONENT)},
     BINARY_UEXPONENT: u32 = {binary_uexponent(BINARY_EXPONENT)}>
    (significand: sN[NUM_BITS]) -> FixedPoint<NUM_BITS, EXPONENT_IS_NEGATIVE, BINARY_UEXPONENT> {
    make_fixed_point_with_zeros<NUM_BITS, EXPONENT_IS_NEGATIVE, BINARY_UEXPONENT>(significand)
}

// Returns a FixedPoint equivalent to the given integer.
pub fn from_integer<NUM_BITS: u32>
    (significand: sN[NUM_BITS]) -> FixedPoint<NUM_BITS, u32:0, u32:0> {
    make_fixed_point_with_zeros<NUM_BITS, u32:0, u32:0>(significand)
}

// Returns the number of integer bits representable by a fixed point number with these parameters.
// Note the third example, where the two least significant integer bits, which must always be zero,
// are not counted.
//
// This does not examine the bits set in a particular value.
//
// Example:
// num_nonzero_integer_bits(4, -8) == 0
// num_nonzero_integer_bits(4, -1) == 3
// num_nonzero_integer_bits(4, 6) == 4
pub fn num_nonzero_integer_bits(NUM_BITS: u32, BINARY_EXPONENT: s32) -> u32 {
    if BINARY_EXPONENT < s32:0 {
        if std::abs(BINARY_EXPONENT) as s33 >= NUM_BITS as s33 {
            u32:0
        } else {
            (NUM_BITS as s33 + BINARY_EXPONENT as s33) as u32
        }
    } else {
        NUM_BITS
    }
}

// Returns the number of fractional bits representable by a fixed point number with these
// parameters. Note the first example, where the four most significant fractional bits, which must
// always be zero, are not counted.
//
// This does not examine the bits set in a particular value.
//
// Example:
// num_nonzero_fractional_bits(4, -8) == 4
// num_nonzero_fractional_bits(4, -1) == 1
// num_nonzero_fractional_bits(4, 6) == 0
pub fn num_nonzero_fractional_bits(NUM_BITS: u32, BINARY_EXPONENT: s32) -> u32 {
    NUM_BITS - num_nonzero_integer_bits(NUM_BITS, BINARY_EXPONENT)
}

// Returns the bits of a fixed point number's fractional part. These bits are _not_ shifted or
// normalized in any sense. E.g. it would be wrong to add the raw fractional parts of two different
// fixed point numbers without first aligning their binary points.
pub fn fractional_bits_raw
    <NB: u32, BE: s32, F: u32 = {num_nonzero_fractional_bits(NB, BE)},
     EXPONENT_IS_NEGATIVE: u32 = {is_negative(BE)}>
    (a: FixedPoint<NB, EXPONENT_IS_NEGATIVE, BE>) -> uN[F] {
    a.significand[0+:uN[F]]
}

// Returns the bits of a fixed point number's integer part. These bits are _not_ shifted or
// normalized in any sense. Less-significant bits that are always zero are not included. E.g. it
// would be wrong to add the raw integer parts of two different fixed point numbers without first
// aligning their binary points.
pub fn integer_bits_raw
    <NB: u32, BE: s32, I: u32 = {num_nonzero_integer_bits(NB, BE)},
     EXPONENT_IS_NEGATIVE: u32 = {is_negative(BE)}>
    (a: FixedPoint<NB, EXPONENT_IS_NEGATIVE, BE>) -> uN[I] {
    let F = num_nonzero_fractional_bits(NB, BE);
    a.significand[F+:uN[I]]
}

// Multiplies two unsigned fixed point numbers.
//
// The number of bits in the result is the sum of the number of bits in the inputs.
pub fn mul
    <NB_A: u32, EN_A: u32, BU_A: u32, NB_B: u32, EN_B: u32, BU_B: u32,
     EXP_SUM: s32 = {binary_exponent(EN_A, BU_A) + binary_exponent(EN_B, BU_B)},
     NB_R: u32 = {NB_A + NB_B}, EN_R: u32 = {is_negative(EXP_SUM)},
     BU_R: u32 = {binary_uexponent(EXP_SUM)}>
    (a: FixedPoint<NB_A, EN_A, BU_A>, b: FixedPoint<NB_B, EN_B, BU_B>)
    -> FixedPoint<NB_R, EN_R, BU_R> {
    make_fixed_point<EXP_SUM>(std::smul(a.significand, b.significand))
}

// Returns the position of the most significant bit, where 0 is the bit just left of the binary
// point.
//
// E.g. consider a value like x.xxxb, which corresponds to NB=4 BE=-3.
// most_significant_bit_position(4,-3) is 0
fn most_significant_bit_position(NB: u32, BE: s32) -> s33 { NB as s33 + BE as s33 - s33:1 }

// Returns the position of the least significant bit, where 0 is the bit just left of the binary
// point.
//
// E.g. consider a value like xxxx.b, which corresponds to NB=4 BE=0.
// least_significant_bit_position(4,0) is 0
fn least_significant_bit_position(NB: u32, BE: s32) -> s32 { BE }

// Returns the number of representable bits where two fixed point numbers overlap.
//
// These examples use x to indicate a representable bit:
// num_bits_overlapping(2,-1, 2,-1) -> x.x and x.x overlap = 2
// num_bits_overlapping(2, -1, 3, -2) -> x.x and x.xx overlap = 2
// num_bits_overlapping(4, 0, 2, -1)  -> xxxx and x.x overlap = 1
// num_bits_overlapping(4, 1, 1, 0)  -> xxxx0 and x overlap = 0
// num_bits_overlapping(4, 0, 2, -2)  -> xxxx and .xx overlap = 0
// num_bits_overlapping(4, 0, 2, -3)  -> xxxx and .0xx overlap = 0
pub fn num_bits_overlapping(NB_A: u32, BE_A: s32, NB_B: u32, BE_B: s32) -> u32 {
    let msb_a = most_significant_bit_position(NB_A, BE_A);
    let msb_b = most_significant_bit_position(NB_B, BE_B);
    let lsb_a = least_significant_bit_position(NB_A, BE_A) as s33;
    let lsb_b = least_significant_bit_position(NB_B, BE_B) as s33;
    let overlap = std::min(msb_a, msb_b) - std::max(lsb_a, lsb_b) + s33:1;
    std::max(overlap, s33:0) as u32
}

// Returns the total width of two fixed point numbers when their binary points are aligned and the
// representable bits are unioned. Includes the bits that would always be zero if these values were
// aligned and then ANDed or ORed.
pub fn aligned_width(NB_A: u32, BE_A: s32, NB_B: u32, BE_B: s32) -> u32 {
    assert!(NB_A > u32:0, "0_width_will_yield_nonsensical_results");
    assert!(NB_B > u32:0, "0_width_will_yield_nonsensical_results");

    let msb_a = most_significant_bit_position(NB_A, BE_A);
    let msb_b = most_significant_bit_position(NB_B, BE_B);
    let lsb_a = least_significant_bit_position(NB_A, BE_A);
    let lsb_b = least_significant_bit_position(NB_B, BE_B);
    let msb = std::max(msb_a, msb_b);
    let lsb = std::min(lsb_a, lsb_b) as s33;
    let num_bits = msb - lsb + s33:1;
    num_bits as u32
}

// Adds two fixed point numbers.
//
// Note: when there is no overlap of aligned inputs, then there is no chance of carry out and result
// width is not increased by 1
pub fn add
    <NB_A: u32, EN_A: u32, BU_A: u32, NB_B: u32, EN_B: u32, BU_B: u32,
     BE_A: s32 = {binary_exponent(EN_A, BU_A)}, BE_B: s32 = {binary_exponent(EN_B, BU_B)},
     NB_R:
     u32 = {
         aligned_width(NB_A, BE_A, NB_B, BE_B) +
         if num_bits_overlapping(NB_A, BE_A, NB_B, BE_B) == u32:0 { u32:0 } else { u32:1 }},
     BE_R: s32 = {std::min(BE_A, BE_B)}, EN_R: u32 = {is_negative(BE_R)},
     BU_R: u32 = {binary_uexponent(BE_R)}>
    (lhs: FixedPoint<NB_A, EN_A, BU_A>, rhs: FixedPoint<NB_B, EN_B, BU_B>)
    -> FixedPoint<NB_R, EN_R, BU_R> {
    // Widen before left shifting to avoid overflow
    let aligned_lhs = (lhs.significand as sN[NB_R]) << (BE_A - BE_R) as u32;
    let aligned_rhs = (rhs.significand as sN[NB_R]) << (BE_B - BE_R) as u32;

    make_fixed_point<BE_R>(aligned_lhs + aligned_rhs)
}

// Subtracts two unsigned fixed point numbers, returns lhs - rhs
pub fn sub
    <NB_A: u32, EN_A: u32, BU_A: u32, NB_B: u32, EN_B: u32, BU_B: u32,
     BE_A: s32 = {binary_exponent(EN_A, BU_A)}, BE_B: s32 = {binary_exponent(EN_B, BU_B)},
     NB_R:
     u32 = {
         aligned_width(NB_A, BE_A, NB_B, BE_B) +
         if num_bits_overlapping(NB_A, BE_A, NB_B, BE_B) == u32:0 { u32:0 } else { u32:1 }},
     BE_R: s32 = {std::min(BE_A, BE_B)}, EN_R: u32 = {is_negative(BE_R)},
     BU_R: u32 = {binary_uexponent(BE_R)}>
    (lhs: FixedPoint<NB_A, EN_A, BU_A>, rhs: FixedPoint<NB_B, EN_B, BU_B>)
    -> FixedPoint<NB_R, EN_R, BU_R> {
    // Widen before left shifting to avoid overflow
    let aligned_lhs = (lhs.significand as sN[NB_R]) << (BE_A - BE_R) as u32;
    let aligned_rhs = (rhs.significand as sN[NB_R]) << (BE_B - BE_R) as u32;

    make_fixed_point<BE_R>(aligned_lhs - aligned_rhs)
}

// Returns the binary exponent after truncating or rounding a fixed point number to a smaller width.
fn binary_exponent_after_truncation
    (num_bits_result: u32, num_bits_a: u32, binary_exponent_a: s32) -> s32 {
    assert!(
        num_bits_a >= num_bits_result, "truncation_cannot_increase_the_number_of_bits_in_the_result");
    let bits_reduced_by = num_bits_a - num_bits_result;
    (binary_exponent_a as s33 + bits_reduced_by as s33) as s32
}

// Truncates a fixed point number to a smaller width, preserving the most significant bits. The
// first type parameter, NB_R, is the number of bits in the result.
pub fn truncate
    <NB_R: u32, NB_A: u32, EN_A: u32, BU_A: u32, NUM_BITS_TRUNCATED: u32 = {NB_A - NB_R},
     BE_R: s32 = {binary_exponent_after_truncation(NB_R, NB_A, binary_exponent(EN_A, BU_A))},
     EN_R: u32 = {is_negative(BE_R)}, BU_R: u32 = {binary_uexponent(BE_R)}>
    (a: FixedPoint<NB_A, EN_A, BU_A>) -> FixedPoint<NB_R, EN_R, BU_R> {
    // Shift the significand to preserve the most significant bits
    let truncated_data = a.significand >> NUM_BITS_TRUNCATED;

    make_fixed_point<BE_R>(truncated_data as sN[NB_R])
}

// Round to nearest, ties to even: rounds a fixed point number to fewer bits, preserving the
// most significant bits. The first type parameter is the number of bits that are rounded away.
// E.g. round_ne_bits_discarded<u32:3> would reduce the NUM_BITS of the argument by 3.
//
// WARNING: this function does not handle overflow (the result should have 1 more significant
// bit to handle overflow - consider what happens when rounding up and the retained bits are
// already at maximum).
//
// The type of rounding is Round To Nearest, ties to Even (RTNE).
// Imagine the binary point is just left of the discarded bits, such that they have a value in
// [0.0, 1) E.g. they are .xxxxb
// If the discarded bits > half, round up (e.g. .1001b)
// If the discarded bits < half, round down (e.g. .0111b)
// If the discarded bits == half, we have to consider the least significant retained bit:
//  * if it is odd, round up    (e.g. 01.1000b -> 10.b)
//  * if it is even, round down (e.g. 00.1000b -> 00.b)
//
// TODO create a version of this that is wider to accept overflow?
//
// The IEEE 754 standard denotes “round to nearest, ties to even” with the abbreviation RNE. We
// keep "round" in the name to avoid excessive brevity.
pub fn round_ne_bits_discarded
    <NUM_BITS_ROUNDED: u32, NB_A: u32, EN_A: u32, BU_A: u32, NB_R: u32 = {NB_A - NUM_BITS_ROUNDED},
     BE_R: s32 = {binary_exponent_after_truncation(NB_R, NB_A, binary_exponent(EN_A, BU_A))},
     EN_R: u32 = {is_negative(BE_R)}, BU_R: u32 = {binary_uexponent(BE_R)}>
    (a: FixedPoint<NB_A, EN_A, BU_A>) -> FixedPoint<NB_R, EN_R, BU_R> {
    if NUM_BITS_ROUNDED == u32:0 {
        // no rounding needed, but we have to make DSLX happy about unifying the types
        // (otherwise we'd just return `a`)
        make_fixed_point<BE_R>(a.significand as sN[NB_R])
    } else {
        // keeps the least significant retained bit
        let lsb_bit_mask = uN[NB_A]:1 << NUM_BITS_ROUNDED;

        // the index of the bit that is equal to half of the result's ULP
        let halfway_idx = NUM_BITS_ROUNDED as uN[NB_A] - uN[NB_A]:1;

        // keeps the half-ULP bit
        let halfway_bit_mask = uN[NB_A]:1 << halfway_idx;

        // keeps the discarded bits
        let discarded_mask = std::mask_bits<NUM_BITS_ROUNDED>() as uN[NB_A];

        let unsigned_significand = a.significand as uN[NB_A];
        let discarded_bits = discarded_mask & unsigned_significand;

        let discarded_bits_gt_half = discarded_bits > halfway_bit_mask;
        let discarded_bits_equal_half = discarded_bits == halfway_bit_mask;

        let retained_is_odd = (unsigned_significand & lsb_bit_mask) == lsb_bit_mask;

        // do we round up because discarded bits are 0.5 and the retained bits are odd? (if we don't
        // round up, then the result will be odd)
        let round_up_to_even = discarded_bits_equal_half && retained_is_odd;

        let round_up = discarded_bits_gt_half || round_up_to_even;

        let retained = (a.significand >> NUM_BITS_ROUNDED) as sN[NB_R];
        let raw_significand = if round_up { retained + sN[NB_R]:1 } else { retained };
        make_fixed_point<BE_R>(raw_significand)
    }
}

// Round to nearest, ties to even: rounds a fixed point number to fewer bits, preserving the
// most significant bits. The first type parameter is the number of bits in the result.
// E.g. round_ne_target_width<u32:20> rounds to 20 bits.
//
// WARNING: this function does not handle overflow (the result should have 1 more significant
// bit to handle overflow - consider what happens when rounding up and the retained bits are
// already at maximum).
pub fn round_ne_target_width
    <NB_R: u32, NB_A: u32, EN_A: u32, BU_A: u32, NUM_BITS_ROUNDED: u32 = {NB_A - NB_R},
     BE_R: s32 = {binary_exponent_after_truncation(NB_R, NB_A, binary_exponent(EN_A, BU_A))},
     EN_R: u32 = {is_negative(BE_R)}, BU_R: u32 = {binary_uexponent(BE_R)}>
    (a: FixedPoint<NB_A, EN_A, BU_A>) -> FixedPoint<NB_R, EN_R, BU_R> {
    // NUM_BITS_ROUNDED must be non-negative
    const_assert!(NB_A >= NB_R);
    round_ne_bits_discarded<NUM_BITS_ROUNDED>(a)
}

// Round to nearest, ties to even: rounds a fixed point number to fewer bits, preserving the
// most significant bits. The first type parameter is the (signed) binary exponent of the result.
// E.g. round_ne_target_exponent<s32:-20> rounds to a binary exponent of -20 (assuming a's
// binary exponent <= -20).
//
// WARNING: this function does not handle overflow (the result should have 1 more significant
// bit to handle overflow - consider what happens when rounding up and the retained bits are
// already at maximum).
pub fn round_ne_target_exponent
    <BE_R: s32, NB_A: u32, EN_A: u32, BU_A: u32, BE_A: s32 = {binary_exponent(EN_A, BU_A)},
     NUM_BITS_ROUNDED: u32 = {(BE_R - BE_A) as u32}, NB_R: u32 = {NB_A - NUM_BITS_ROUNDED},
     EN_R: u32 = {is_negative(BE_R)}, BU_R: u32 = {binary_uexponent(BE_R)}>
    (a: FixedPoint<NB_A, EN_A, BU_A>) -> FixedPoint<NB_R, EN_R, BU_R> {
    // rounding cannot decrease the binary exponent
    const_assert!(BE_R >= BE_A);
    round_ne_target_width<NB_R>(a)
}

// Discards the given number of most significant bits of this fixed point number (thereby
// reducing the width). The first type parameter, NUM_DISCARDED, is the number of bits
// discarded.
//
// WARNING: will overflow if the result is too small to hold the input!
//
// Currently only supports discarding bits from the integer part of the number. This means the
// binary exponent can't change. This could be relaxed with a little bit of work.
pub fn narrow_by
    <NUM_DISCARDED: u32, NB_A: u32, EN_A: u32, BU_A: u32, NB_R: u32 = {NB_A - NUM_DISCARDED}>
    (a: FixedPoint<NB_A, EN_A, BU_A>) -> FixedPoint<NB_R, EN_A, BU_A> {
    assert!(NUM_DISCARDED <= NB_A, "narrow_by_cant_yet_discard_fractional_bits");
    make_fixed_point_with_zeros<NB_R, EN_A, BU_A>(a.significand as sN[NB_R])
}

// Returns a FixedPoint that uses a common num bits and binary exponent.
//
// The intended usage is so that fixed point constants can be specified in their most reduced form
// (i.e. fewest number of bits used) by the generating program, and then all co-normalized so that
// they have the same type in DSLX.
//
// Assumes that EXPONENT_IS_NEGATIVE of `x` matches the result's EXPONENT_IS_NEGATIVE.
//
// When COMMON_BINARY_UEXPONENT > BINARY_UEXPONENT, the significand is shifted right, and there is
// potential information loss, so this branch is currently a `fail!`.
//
// WARNING:Does not check that the result's bitwidth is wide enough to hold `x.significand` shifted
// appropriately.

pub fn to_common_type
    <COMMON_NUM_BITS: u32, COMMON_BINARY_UEXPONENT: u32, NUM_BITS: u32, EXPONENT_IS_NEGATIVE: u32,
     BINARY_UEXPONENT: u32>
    (x: FixedPoint<NUM_BITS, EXPONENT_IS_NEGATIVE, BINARY_UEXPONENT>)
    -> FixedPoint<COMMON_NUM_BITS, EXPONENT_IS_NEGATIVE, COMMON_BINARY_UEXPONENT> {
    let x_exp = binary_exponent(EXPONENT_IS_NEGATIVE, BINARY_UEXPONENT);
    let result_exp = binary_exponent(EXPONENT_IS_NEGATIVE, COMMON_BINARY_UEXPONENT);
    let significand = if result_exp > x_exp {
        // If the exponent is increasing, then the significand needs to decrease.
        // let expr = (x.significand as sN[COMMON_NUM_BITS]) >> (result_exp - x_exp) as u32;
        // fail!("you_are_losing_information_is_this_really_what_you_want", expr)
        // BUGFIX+ENABLE: Andrei
        let expr = (x.significand >> (result_exp - x_exp) as u32) as sN[COMMON_NUM_BITS];
        expr
    } else {
        // If the exponent is decreasing, then the significand needs to increase.
        (x.significand as sN[COMMON_NUM_BITS]) << (x_exp - result_exp) as u32
    };
    make_fixed_point<result_exp>(significand)
}

// Round to nearest, ties to even (aka roundTiesToEven).
// if truncated bits > halfway bit: round up.
// if truncated bits < halfway bit: round down.
// if truncated bits == halfway bit and lsb bit is odd: round up.
// if truncated bits == halfway bit and lsb bit is even: round down.
//
// TODO this is apfloat's rne, because apfloat's is not public. Make apfloat's rne public?
// Consolidate with apfloat' implementation.
fn rne<FRACTION_SZ: u32, LSB_INDEX_SZ: u32 = {std::clog2(FRACTION_SZ)}>
    (fraction: uN[FRACTION_SZ], lsb_idx: uN[LSB_INDEX_SZ]) -> bool {
    let lsb_bit_mask = uN[FRACTION_SZ]:1 << lsb_idx;
    let halfway_idx = lsb_idx as uN[FRACTION_SZ] - uN[FRACTION_SZ]:1;
    let halfway_bit_mask = uN[FRACTION_SZ]:1 << halfway_idx;
    let trunc_mask = (uN[FRACTION_SZ]:1 << lsb_idx) - uN[FRACTION_SZ]:1;
    let trunc_bits = trunc_mask & fraction;
    let trunc_bits_gt_half = trunc_bits > halfway_bit_mask;
    let trunc_bits_are_halfway = trunc_bits == halfway_bit_mask;
    let to_fraction_is_odd = (fraction & lsb_bit_mask) == lsb_bit_mask;
    let round_to_even = trunc_bits_are_halfway && to_fraction_is_odd;
    let round_up = trunc_bits_gt_half || round_to_even;
    round_up
}

pub enum SubnormalOutputs : u1 {
    Produced = 0,
    FlushToZero = 1,
}

// Converts the fixed point number to a floating point number using round to nearest, ties to
// even as the rounding mode.
pub fn convert_to_float_using_round_ties_to_even
    <SUBNORMAL_OUTPUTS: SubnormalOutputs, EXP_SZ: u32, FRACTION_SZ: u32, NUM_BITS: u32,
     EXPONENT_IS_NEGATIVE: u32, BINARY_UEXPONENT: u32>
    (src: FixedPoint<NUM_BITS, EXPONENT_IS_NEGATIVE, BINARY_UEXPONENT>)
    -> apfloat::APFloat<EXP_SZ, FRACTION_SZ> {
    let magnitude = std::abs(src.significand as sN[NUM_BITS + u32:1]) as uN[NUM_BITS];
    let leading_zeroes = clz(magnitude);
    let num_trailing_nonzeros = NUM_BITS - leading_zeroes as u32;

    // A note on terminology: the significand is the 1.ffff where the f's are the fractional
    // bits.
    const SIGNIFICAND_WIDTH = FRACTION_SZ + u32:1;
    const PRE_NORMALIZE_WIDTH = std::max(SIGNIFICAND_WIDTH, NUM_BITS);
    let unnormalized_significand = magnitude as uN[PRE_NORMALIZE_WIDTH];

    // Form the normalized significand: 1.xxxx...xxxx
    // When NUM_BITS < SIGNIFICAND_WIDTH we need to shift left to normalize the significand.
    // When NUM_BITS = SIGNIFICAND_WIDTH AND num_trailing_nonzeros < SIGNIFICAND_WIDTH we need
    // to shift left to normalize the significand.
    // When NUM_BITS > SIGNIFICAND_WIDTH we may need to left shift, do nothing, or round. It
    // depends on compare(num_trailing_nonzeros, SIGNIFICAND_WIDTH)

    const NUM_BITS_COMPARED_SIGNIFICAND_WIDTH = std::compare(NUM_BITS, SIGNIFICAND_WIDTH);
    let (normalized_significand, increment_exponent) = match NUM_BITS_COMPARED_SIGNIFICAND_WIDTH {
        std::Ordering::Less =>  // we need to shift left to normalize the significand
            (unnormalized_significand << (SIGNIFICAND_WIDTH - num_trailing_nonzeros), u1:0),
        std::Ordering::Equal => (
            unnormalized_significand << (SIGNIFICAND_WIDTH - num_trailing_nonzeros), u1:0
        ),
        std::Ordering::Greater => {
            match std::compare(num_trailing_nonzeros, SIGNIFICAND_WIDTH) {
                std::Ordering::Less => (
                    unnormalized_significand << (SIGNIFICAND_WIDTH - num_trailing_nonzeros), u1:0
                ),
                std::Ordering::Equal => (unnormalized_significand, u1:0),
                std::Ordering::Greater => {
                    let num_bits_to_round_off = (num_trailing_nonzeros - SIGNIFICAND_WIDTH) as
                                                uN[std::clog2(PRE_NORMALIZE_WIDTH)];
                    let right_aligned = unnormalized_significand >> num_bits_to_round_off;
                    let round_up = rne(unnormalized_significand, num_bits_to_round_off);
                    let rounded = if round_up {
                        let rounded_up = right_aligned + uN[PRE_NORMALIZE_WIDTH]:1;
                        let significand_overflow =
                            (rounded_up as uN[SIGNIFICAND_WIDTH]) == uN[SIGNIFICAND_WIDTH]:0;
                        (rounded_up, significand_overflow)
                    } else {
                        let significand_overflow = false;
                        (right_aligned, significand_overflow)
                    };
                    rounded
                },
            }
            },
    };

    // We now discard the leading 1 in the normalized significand (however, when
    // significand_overflow (see above), the leading 1 is actually one bit to the left, but we
    // want fraction to be 0, so the logic works out).
    let fraction = normalized_significand as uN[FRACTION_SZ];

    const BINARY_EXPONENT_OF_X = binary_exponent(EXPONENT_IS_NEGATIVE, BINARY_UEXPONENT);
    let exponent =
        BINARY_EXPONENT_OF_X + num_trailing_nonzeros as s32 + increment_exponent as s32 - s32:1;

    const MAX_NORMAL_EXP = apfloat::max_normal_exp<EXP_SZ>();
    let exponent_overflows = exponent > MAX_NORMAL_EXP as s32;

    // When implementing SubnormalOutputs::Produced, handle case where exponent_underflows but
    // the shifted significand is not zero
    const_assert!(SUBNORMAL_OUTPUTS == SubnormalOutputs::FlushToZero);

    const MIN_NORMAL_EXP = apfloat::min_normal_exp<EXP_SZ>();
    let exponent_underflows = exponent < MIN_NORMAL_EXP as s32;

    let biased_exponent = apfloat::bias(exponent as sN[EXP_SZ]);

    let is_negative = src.significand < sN[NUM_BITS]:0;
    let is_zero = magnitude == uN[NUM_BITS]:0;

    match (exponent_overflows, exponent_underflows || is_zero) {
        (true, _) => apfloat::inf<EXP_SZ, FRACTION_SZ>(is_negative),
        (_, true) => apfloat::zero<EXP_SZ, FRACTION_SZ>(is_negative),
        (false, false) => apfloat::APFloat<EXP_SZ, FRACTION_SZ> {
            sign: is_negative,
            bexp: biased_exponent,
            fraction,
        },
    }
}

// Converts a FixedPoint to its underlying bits; i.e. the significand.
//
// Note: discards the signedness, hence 'u' in the name.
pub fn to_ubits<NB: u32, EN: u32, BU: u32>(x: FixedPoint<NB, EN, BU>) -> uN[NB] {
    x.significand as uN[NB]
}

#[test]
fn test_most_significant_bit_position() {
    // Test case 1: Standard positive exponents
    assert_eq(most_significant_bit_position(u32:4, s32:3), s33:6);

    // Test case 2: Zero exponent
    assert_eq(most_significant_bit_position(u32:4, s32:0), s33:3);  // xxxx.b

    // Test case 3: Negative exponent
    assert_eq(most_significant_bit_position(u32:4, s32:-4), s33:-1);
    assert_eq(most_significant_bit_position(u32:4, s32:-3), s33:0);  // x.xxxb
    assert_eq(most_significant_bit_position(u32:4, s32:-2), s33:1);

    // Test case 4: Maximum u32 value
    assert_eq(most_significant_bit_position(u32:4294967295, s32:0), s33:4294967294);
    assert_eq(most_significant_bit_position(u32:4294967294, s32:1), s33:4294967294);

    // Test case 5: Minimum s32 exponent
    assert_eq(most_significant_bit_position(u32:4294967295, s32:-2147483648), s33:2147483646);
}

#[test]
fn test_least_significant_bit_position() {
    // Test case 1: Standard positive exponents
    assert_eq(least_significant_bit_position(u32:1, s32:1), s32:1);  // x0.b
    assert_eq(least_significant_bit_position(u32:1, s32:2), s32:2);

    // Test case 2: Zero exponent
    assert_eq(least_significant_bit_position(u32:4, s32:0), s32:0);

    // Test case 3: Negative exponent
    assert_eq(least_significant_bit_position(u32:1, s32:-1), s32:-1);

    // Test case 4: Maximum u32 value
    assert_eq(least_significant_bit_position(u32:4294967295, s32:0), s32:0);

    // Test case 5: Minimum s32 exponent
    assert_eq(least_significant_bit_position(u32:1, s32:-2147483648), s32:-2147483648);
}

#[test]
fn test_num_bits_overlapping() {
    assert_eq(num_bits_overlapping(u32:0, s32:0, u32:0, s32:0), u32:0);
    assert_eq(num_bits_overlapping(u32:1, s32:0, u32:1, s32:0), u32:1);
    assert_eq(num_bits_overlapping(u32:1, s32:-1, u32:1, s32:-1), u32:1);

    // Test identical widths and binary exponents
    assert_eq(num_bits_overlapping(u32:5, s32:0, u32:5, s32:0), u32:5);

    // Different binary exponents, same widths
    assert_eq(num_bits_overlapping(u32:5, s32:0, u32:5, s32:1), u32:4);
    assert_eq(num_bits_overlapping(u32:5, s32:1, u32:5, s32:0), u32:4);
    assert_eq(num_bits_overlapping(u32:5, s32:1, u32:5, s32:-1), u32:3);

    // Different widths, same binary exponent
    assert_eq(num_bits_overlapping(u32:5, s32:0, u32:6, s32:0), u32:5);

    // Different widths and binary exponents
    assert_eq(num_bits_overlapping(u32:5, s32:0, u32:6, s32:1), u32:4);

    // Neighboring, excactly zero overlap
    assert_eq(num_bits_overlapping(u32:4, s32:0, u32:2, s32:-2), u32:0);
    assert_eq(num_bits_overlapping(u32:2, s32:-2, u32:4, s32:0), u32:0);
    assert_eq(num_bits_overlapping(u32:32, s32:31, u32:31, s32:0), u32:0);

    // Gap of 1
    assert_eq(num_bits_overlapping(u32:4, s32:0, u32:2, s32:-3), u32:0);
    assert_eq(num_bits_overlapping(u32:2, s32:-3, u32:4, s32:0), u32:0);

    // partial overlap
    assert_eq(num_bits_overlapping(u32:4, s32:0, u32:2, s32:-1), u32:1);
    assert_eq(num_bits_overlapping(u32:2, s32:-1, u32:3, s32:-2), u32:2);

    // big gap
    assert_eq(num_bits_overlapping(u32:32, s32:-31, u32:32, s32:31), u32:0);
}

#[test]
fn test_aligned_width() {
    // Test minimum NB and BE
    assert_eq(aligned_width(u32:1, s32:0, u32:1, s32:0), u32:1);

    // Test identical NB and BE
    assert_eq(aligned_width(u32:8, s32:0, u32:8, s32:0), u32:8);

    // Test different NB values
    assert_eq(aligned_width(u32:16, s32:0, u32:8, s32:0), u32:16);

    // Test different BE values
    assert_eq(aligned_width(u32:8, s32:2, u32:8, s32:-2), u32:12);

    // There is a gap, so no need to increase width (i.e. no need to account for carry out)
    assert_eq(aligned_width(u32:1, s32:1, u32:1, s32:0), u32:2);
    assert_eq(aligned_width(u32:8, s32:16, u32:8, s32:0), u32:24);

    // Test negative BE values
    assert_eq(aligned_width(u32:8, s32:-8, u32:8, s32:0), u32:16);

    // Test + and - BE values
    assert_eq(aligned_width(u32:31, s32:16, u32:37, s32:-15), u32:62);
}

#[test]
fn test_from_integer() { assert_eq(from_integer(s3:0b111), make_fixed_point<s32:0>(s3:0b111)); }

#[test]
fn test_mul_zero_zero() {
    let a = make_fixed_point<s32:-4>(s5:0);
    let b = make_fixed_point<s32:-4>(s5:0);
    let result = mul(a, b);
    assert_eq(result, make_fixed_point<s32:-8>(s10:0));
}

#[test]
fn test_mul_zero_nonzero() {
    let a = make_fixed_point<s32:-4>(s4:0);
    let b = make_fixed_point<s32:-4>(s6:5);
    let result = mul(a, b);
    assert_eq(result, make_fixed_point<s32:-8>(s10:0));
}

#[test]
fn test_mul_exponent_zero() {
    // 5 * 2^0 = 5 and 3 * 2^0 = 3 => product = 15 => raw = 15 when exponent is 0
    let a = make_fixed_point<s32:0>(s5:5);
    let b = make_fixed_point<s32:0>(s5:3);
    let result = mul(a, b);
    assert_eq(result, make_fixed_point<s32:0>(s10:15));
}

#[test]
fn test_mul_max_data_bits() {
    // 15/16 * 1/16 = 15/256 => raw = 15 for 8 bits => exponent is -8
    let a = make_fixed_point<s32:-4>(s5:15);
    let b = make_fixed_point<s32:-4>(s5:1);
    let result = mul(a, b);
    assert_eq(result, make_fixed_point<s32:-8>(s10:15));
}

#[test]
fn test_mul_half_half() {
    // 1/2 * 1/2 = 1/4 => significand = 1
    let a = make_fixed_point<s32:-1>(s2:1);
    let b = make_fixed_point<s32:-1>(s2:1);
    let result = mul(a, b);
    assert_eq(result, make_fixed_point<s32:-2>(s4:0b01));
}

#[test]
fn test_mul_one_one() {
    // 1/16 * 1/16 = 1/256 => significand = 1
    let a = make_fixed_point<s32:-4>(s5:1);
    let b = make_fixed_point<s32:-4>(s5:1);
    let result = mul(a, b);
    assert_eq(result, make_fixed_point<s32:-8>(s10:1));
}

#[test]
fn test_mul_one_two() {
    // 1/16 * 2/16 = 2/256 => significand = 2
    let a = make_fixed_point<s32:-4>(s5:1);
    let b = make_fixed_point<s32:-4>(s5:2);
    let result = mul(a, b);
    assert_eq(result, make_fixed_point<s32:-8>(s10:2));
}

#[test]
fn test_mul_max_max() {
    // 15/16 * 15/16 = 225/256 => significand = 225
    let a = make_fixed_point<s32:-4>(s5:15);
    let b = make_fixed_point<s32:-4>(s5:15);
    let result = mul(a, b);
    assert_eq(result, make_fixed_point<s32:-8>(s10:225));
}

#[test]
fn test_mul_large_positive_exponent() {
    // 3 * 2^5 = 96 and 2 * 2^5 = 64 => 96 * 64 = 6144 => raw = 6 when exponent is 10
    let a = make_fixed_point<s32:5>(s5:3);
    let b = make_fixed_point<s32:5>(s5:2);
    let result = mul(a, b);
    assert_eq(result, make_fixed_point<s32:10>(s10:6));
}

#[test]
fn test_mul_more_negative_exponent() {
    // 3 * 2^-6 = 3/64 and 8 * 2^-6 = 1/8 => product = 3/512 => raw = 24 when exponent is -12
    let a = make_fixed_point<s32:-6>(s5:3);
    let b = make_fixed_point<s32:-6>(s5:8);
    let result = mul(a, b);
    assert_eq(result, make_fixed_point<s32:-12>(s10:24));
}

#[test]
fn test_mul_int_fractional() {
    // 7 * 2^2 = 7 and 3 * 2^-5 = 3/16 => product = 21/8 => raw = 21 when exponent is -3
    let a = make_fixed_point<s32:2>(s5:7);
    let b = make_fixed_point<s32:-5>(s5:3);
    let result = mul(a, b);
    assert_eq(result, make_fixed_point<s32:-3>(s10:21));
}

#[test]
fn test_mul_min_exponent() {
    // 1 * 2^-8 = 1/256 and 1 * 2^-8 = 1/256 => product = 1/65536 => raw = 1 when exponent is -16
    let a = make_fixed_point<s32:-8>(s5:1);
    let b = make_fixed_point<s32:-8>(s5:1);
    let result = mul(a, b);
    assert_eq(result, make_fixed_point<s32:-16>(s10:1));
}

#[test]
fn test_mul_large_positive_exponents() {
    // a: 63 * 2^8 = 16128
    // b: 31 * 2^7 = 3968
    // product: 16128 * 3968 = 1953 * 2^15 = 63,995,904
    let a = make_fixed_point<s32:8>(s7:63);
    let b = make_fixed_point<s32:7>(s6:31);
    let result = mul(a, b);
    assert_eq(result, make_fixed_point<s32:15>(s13:1953));
}

#[test]
fn test_mul_different_exponents() {
    // 7 * 2^-2 = 7/4 and 4 * 2^3 = 32 => product = 56 => raw = 28 when exponent is 1
    let a = make_fixed_point<s32:-2>(s5:7);
    let b = make_fixed_point<s32:3>(s4:4);
    let result = mul(a, b);
    assert_eq(result, make_fixed_point<s32:1>(s9:28));
}

#[test]
fn test_uadd_zero_zero() {
    let a = make_fixed_point<s32:0>(s5:0);
    let b = make_fixed_point<s32:0>(s5:0);
    let result = add(a, b);
    assert_eq(result, make_fixed_point<s32:0>(s6:0));

    let a = make_fixed_point<s32:1>(s5:0);
    let b = make_fixed_point<s32:1>(s5:0);
    let result = add(a, b);
    assert_eq(result, make_fixed_point<s32:1>(s6:0));

    let a = make_fixed_point<s32:-2>(s5:0);
    let b = make_fixed_point<s32:-2>(s5:0);
    let result = add(a, b);
    assert_eq(result, make_fixed_point<s32:-2>(s6:0));
}

#[test]
fn test_uadd_zero_five() {
    let a = make_fixed_point<s32:0>(s2:0b0);
    let b = make_fixed_point<s32:0>(s4:0b101);
    let result = add(a, b);
    assert_eq(result, make_fixed_point<s32:0>(s5:0b101));
}

#[test]
fn test_uadd_1_5() {
    let a = make_fixed_point<s32:0>(s2:0b1);  // 1
    let b = make_fixed_point<s32:0>(s4:0b101);  // 5
    let result = add(a, b);
    assert_eq(result, make_fixed_point<s32:0>(s5:0b110));  // 6
}

#[test]
fn test_uadd_1_5_exp1() {
    let a = make_fixed_point<s32:1>(s2:0b1);  // 1*2^1 = 2
    let b = make_fixed_point<s32:1>(s4:0b101);  // 5*2^1 = 10
    let result = add(a, b);
    assert_eq(result, make_fixed_point<s32:1>(s5:0b110));  // 6*2^1 = 12
}

#[test]
fn test_uadd_carry_out() {
    let a = make_fixed_point<s32:0>(s5:0b1111);  // 15
    let b = make_fixed_point<s32:0>(s5:0b0001);  // 1
    let result = add(a, b);
    assert_eq(result, make_fixed_point<s32:0>(s6:0b10000));  // 16
}

#[test]
fn test_uadd_different_exps() {
    let a = make_fixed_point<s32:1>(s3:0b01);  // 1*2^1 = 2
    let b = make_fixed_point<s32:2>(s3:0b01);  // 1*2^2 = 4
    let result = add(a, b);
    assert_eq(result, make_fixed_point<s32:1>(s5:0b11));  // 3*2^1 = 6

    let a = make_fixed_point<s32:1>(s3:0b11);  // 3*2^1 = 6
    let b = make_fixed_point<s32:2>(s3:0b11);  // 3*2^2 = 12
    let result = add(a, b);
    assert_eq(result, make_fixed_point<s32:1>(s5:0b1001));  // 9*2^1 = 18
}

#[test]
fn test_uadd_2_7_exp2() {
    let a = make_fixed_point<s32:2>(s2:0b1);  // 2*2^2 = 8
    let b = make_fixed_point<s32:2>(s4:0b111);  // 7*2^2 = 28
    let result = add(a, b);
    assert_eq(result, make_fixed_point<s32:2>(s5:0b1000));  // 8*2^2 = 32
}

#[test]
fn test_uadd_4_3_exp1() {
    let a = make_fixed_point<s32:1>(s4:0b100);  // 4*2^1 = 8
    let b = make_fixed_point<s32:1>(s3:0b11);  // 3*2^1 = 6
    let result = add(a, b);
    assert_eq(result, make_fixed_point<s32:1>(s5:0b111));  // 7*2^1 = 14
}

#[test]
fn test_uadd_1_1_exp3_partial_overlap() {
    let a = make_fixed_point<s32:3>(s2:0b1);  // 1*2^3 = 8
    let b = make_fixed_point<s32:0>(s5:0b1111);  // 15*2^0 = 15
    let result = add(a, b);
    assert_eq(result, make_fixed_point<s32:0>(s6:0b10111));  // 23*2^0 = 23
}

#[test]
fn test_uadd_2_4_exp0_non_overlap() {
    let a = make_fixed_point<s32:0>(s3:0b01);  // 1*2^0 = 1
    let b = make_fixed_point<s32:2>(s5:0b1000);  // 8*2^2 = 32
    let result = add(a, b);
    // Bits don't overlap after alignment so there is no carry out
    assert_eq(result, make_fixed_point<s32:0>(s8:0b100001));  // 33*2^0 = 33
}

#[test]
fn test_uadd_wide_exp2() {
    let a = make_fixed_point<s32:0>(s5:0b1111);  // 15*2^0 = 15
    let b = make_fixed_point<s32:1>(s4:0b111);  // 7*2^1 = 14
    let result = add(a, b);
    // Fully overlapping bits
    assert_eq(result, make_fixed_point<s32:0>(s6:0b11101));  // 29*2^0 = 29
}

#[test]
fn test_uadd_neg_neg_exp2() {
    let a = make_fixed_point<s32:-2>(s3:0b01);  // 1*2^-2 = 0.25
    let b = make_fixed_point<s32:-2>(s3:0b10);  // 2*2^-2 = 0.5
    let result = add(a, b);
    assert_eq(result, make_fixed_point<s32:-2>(s4:0b11));  // 0.75
}

#[test]
fn test_uadd_neg_pos_exp1() {
    let a = make_fixed_point<s32:-1>(s5:0b111);  // 7*2^-1 = 3.5
    let b = make_fixed_point<s32:1>(s4:0b111);  // 7*2^1 = 14
    let result = add(a, b);
    assert_eq(result, make_fixed_point<s32:-1>(s7:0b100011));  // 35*2^-1 = 17.5
}

#[test]
fn test_uadd_pos_neg_exp0() {
    let a = make_fixed_point<s32:0>(s5:0b1001);  // 9*2^0 = 9
    let b = make_fixed_point<s32:-3>(s4:0b101);  // 5*2^-3 = 0.625
    let result = add(a, b);
    // no overlap after alignment; no carry out
    assert_eq(result, make_fixed_point<s32:-3>(s9:0b1001101));  // 77*2^-3=  9.6255
}

// ++++ sub tests ++++
#[test]
fn test_sub_zero_zero_exp0() {
    // 0 * 2^0 = 0
    // 0 * 2^0 = 0
    // Expected: 0
    let a = make_fixed_point<s32:0>(s2:0b0);
    let b = make_fixed_point<s32:0>(s2:0b0);
    let result = sub(a, b);
    assert_eq(result, make_fixed_point<s32:0>(s3:0b0));
}

#[test]
fn test_sub_3_1_exp0() {
    // 3 * 2^0 = 3
    // 1 * 2^0 = 1
    // Expected: 2 * 2^0 = 2
    let a = make_fixed_point<s32:0>(s3:0b11);
    let b = make_fixed_point<s32:0>(s3:0b01);
    let result = sub(a, b);
    assert_eq(result, make_fixed_point<s32:0>(s4:0b10));
}

#[test]
fn test_sub_6_2_exp1() {
    // 6 * 2^1 = 12
    // 2 * 2^1 = 4
    // Expected: 8 * 2^1 = 16
    let a = make_fixed_point<s32:1>(s4:0b110);
    let b = make_fixed_point<s32:1>(s3:0b10);
    let result = sub(a, b);
    assert_eq(result, make_fixed_point<s32:1>(s5:0b100));
}

#[test]
fn test_sub_8_3_exp_neg1() {
    // 8 * 2^1 = 16
    // 3 * 2^-1 = 1.5
    // Expected: 14.5 => (29 * 2^-1) in binary = u4:0b0101
    // 1000.00
    //-0000.11
    let a = make_fixed_point<s32:1>(s5:0b1000);
    let b = make_fixed_point<s32:-1>(s5:0b0011);
    let result = sub(a, b);
    assert_eq(result, make_fixed_point<s32:-1>(s8:0b11101));
}

#[test]
fn test_sub_lhs_has_smaller_exponent() {
    // 172.75
    // 21 * 2^3 = 168
    // Expected: 14.5 => (29 * 2^-1) in binary = u4:0b0101
    // 1000.00
    //-0000.11
    let a = make_fixed_point<s32:-2>(s20:0b1010110011);
    let b = make_fixed_point<s32:3>(s6:0b10101);
    let result = sub(a, b);
    assert_eq(result, make_fixed_point<s32:-2>(s21:0b10011));
}

#[test]
fn test_sub_negative_result() {
    // 1 * 2^0 = 1
    // 3 * 2^0 = 3
    let a = make_fixed_point<s32:0>(s3:1);
    let b = make_fixed_point<s32:0>(s3:3);
    let result = sub(a, b);
    assert_eq(result, make_fixed_point<s32:0>(s4:-2));  // -2 * 2^0 = -2
}

#[test]
fn test_sub_negative_result_fractional_only() {
    // 0.25 - 0.75 = -0.5
    // -0.5 = -4 * 2^-3
    let a = make_fixed_point<s32:-2>(s6:1);
    let b = make_fixed_point<s32:-3>(s6:6);
    let result = sub(a, b);
    assert_eq(result, make_fixed_point<s32:-3>(s8:-4));
}

#[test]
fn test_sub_negative_result_lhs_neg_exponent() {
    // 12 * 2^-2 = 3
    // 4 * 2^0 = 4
    // 3 - 4 = -1 = -4 * 2^-2
    let a = make_fixed_point<s32:-2>(s6:12);
    let b = make_fixed_point<s32:0>(s4:4);
    let result = sub(a, b);
    assert_eq(result, make_fixed_point<s32:-2>(s7:-4));
}

#[test]
fn test_sub_negative_result_rhs_neg_exponent() {
    // 2.0 - 2.75 = -0.75
    // 2.75 => 11 * 2^-2
    let a = make_fixed_point<s32:0>(s3:2);
    let b = make_fixed_point<s32:-2>(s6:11);
    let result = sub(a, b);
    assert_eq(result, make_fixed_point<s32:-2>(s7:-3));
}

#[test]
fn test_sub_negative_result_both_neg_exponent() {
    // 5.5 - 6.0 = -0.5
    // 5.5 => 22 * 2^-2
    // 6.0 => 24 * 2^-2
    let a = make_fixed_point<s32:-2>(s6:22);
    let b = make_fixed_point<s32:-2>(s6:24);
    let result = sub(a, b);
    assert_eq(result, make_fixed_point<s32:-2>(s7:-2));
}

#[test]
fn test_sub_result_neg_pos_exponent() {
    // 3 * 2^-5 = 3/32
    // 11 * 2^4 = 176
    // 3/32 - 176 = -(175 + 29/32)
    // ... = -5629/32
    let a = make_fixed_point<s32:-5>(s3:3);
    let b = make_fixed_point<s32:4>(s6:11);
    let result = sub(a, b);
    assert_eq(result, make_fixed_point<s32:-5>(s15:-5629));
}

#[test]
fn test_sub_result_pos_neg_exponent() {
    // 2 * 2^3 = 16
    // 11 * 2^-4 = 0.6875
    // 16 - 0.6875 = 15.3125
    // 15.3125 = 245 * 2^-4
    let a = make_fixed_point<s32:3>(s3:2);
    let b = make_fixed_point<s32:-4>(s6:11);
    let result = sub(a, b);
    assert_eq(result, make_fixed_point<s32:-4>(s10:245));
}

#[test]
fn test_add_overflow() {
    // Max s4 value 0b0111 = 7
    let a = make_fixed_point<s32:0>(s4:7);

    // 7 + 7 = 14, overflow an s4 number
    let result = add(a, a);

    // Expected result: 7 + 7 = 14 (0b01110)
    assert_eq(result, make_fixed_point<s32:0>(s5:14));
}

#[test]
fn test_sub_overflow() {
    // Max s4 value 0b0111 = 7
    // Min s4 value 0b1000 = -8
    let a = make_fixed_point<s32:0>(s4:7);
    let b = make_fixed_point<s32:0>(s4:-8);

    // 7 - (-8) = 15, overflow the s4 number
    let result = sub(a, b);

    // Expected result: 7 - (-8) = 15 (0b01111)
    assert_eq(result, make_fixed_point<s32:0>(s5:15));
}

#[test]
fn test_add_no_overlap() {
    // 7 = 0b0111
    let a = make_fixed_point<s32:0>(s4:7);

    // 3 = 0b0011
    let b = make_fixed_point<s32:-4>(s4:3);

    // No overlap
    let result = add(a, b);

    // Expected result with no overlap
    //   a      = 0b0111_0000
    //   b      = 0b0000_0011
    // +
    //   result = 0b0111_0011
    assert_eq(result, make_fixed_point<s32:-4>(s8:0b0111_0011));
}

#[test]
fn test_sub_no_overlap() {
    // 7 = 0b0111
    let a = make_fixed_point<s32:0>(s4:7);

    // 3 = 0b0011
    let b = make_fixed_point<s32:-4>(s4:3);

    // No overlap
    let result = sub(a, b);

    // Expected result with no overlap
    //   a      = 0b0111_0000
    //   b      = 0b0000_0011
    // -
    //   result = 0b0110_1101
    assert_eq(result, make_fixed_point<s32:-4>(s8:0b0110_1101));
}

#[test]
fn test_binary_exponent_after_truncation_combined() {
    // Test no truncation
    let result = binary_exponent_after_truncation(u32:8, u32:8, s32:2);
    assert_eq(result, s32:2);

    // Test almost all truncated
    let result = binary_exponent_after_truncation(u32:1, u32:8, s32:2);
    assert_eq(result, s32:9);

    // Test fractional truncated
    let result = binary_exponent_after_truncation(u32:6, u32:8, s32:-2);
    assert_eq(result, s32:0);

    // Test integer and fractional truncated
    let result = binary_exponent_after_truncation(u32:4, u32:9, s32:-3);
    assert_eq(result, s32:2);

    // Test negative exponent
    let result = binary_exponent_after_truncation(u32:4, u32:8, s32:-1);
    assert_eq(result, s32:3);

    // Test zero result bits
    let result = binary_exponent_after_truncation(u32:0, u32:8, s32:1);
    assert_eq(result, s32:9);
}

#[test]
fn test_truncate() {
    // Test no truncation
    assert_eq(
        truncate<u32:9>(make_fixed_point<s32:2>(s9:0b10101010)),
        make_fixed_point<s32:2>(s9:0b10101010));
    assert_eq(
        truncate<u32:9>(make_fixed_point<s32:2>(s9:0b01010101)),
        make_fixed_point<s32:2>(s9:0b01010101));

    // Truncate by 1 bit
    assert_eq(
        truncate<u32:8>(make_fixed_point<s32:0>(s9:0b10101010)),
        make_fixed_point<s32:1>(s8:0b1010101));
    assert_eq(
        truncate<u32:8>(make_fixed_point<s32:0>(s9:0b01010101)),
        make_fixed_point<s32:1>(s8:0b0101010));

    // Truncate by 2 bits
    assert_eq(
        truncate<u32:7>(make_fixed_point<s32:0>(s9:0b01011111)),
        make_fixed_point<s32:2>(s7:0b010111));
    assert_eq(
        truncate<u32:7>(make_fixed_point<s32:0>(s9:0b10100000)),
        make_fixed_point<s32:2>(s7:0b101000));

    // Truncate by almost all bits
    assert_eq(
        truncate<u32:2>(make_fixed_point<s32:0>(s9:0b10101010)), make_fixed_point<s32:7>(s2:0b1));
    assert_eq(
        truncate<u32:2>(make_fixed_point<s32:0>(s9:0b01111111)), make_fixed_point<s32:7>(s2:0b0));

    // Truncate, input is 0
    assert_eq(
        truncate<u32:6>(make_fixed_point<s32:3>(s9:0b00000000)), make_fixed_point<s32:6>(s6:0b00000));
    assert_eq(truncate<u32:5>(make_fixed_point<s32:3>(s31:0b0)), make_fixed_point<s32:29>(s5:0b0));

    // Truncate an all-fractional number. exponent will reduce in magnitude
    assert_eq(
        truncate<u32:7>(make_fixed_point<s32:-10>(s13:0b101101101101)),
        make_fixed_point<s32:-4>(s7:0b101101));

    // Truncate resulting in zero
    assert_eq(
        truncate<u32:3>(make_fixed_point<s32:1>(s7:0b001111)), make_fixed_point<s32:5>(s3:0b00));
}

#[test]
fn test_round_ne_target_width() {
    // Test no rounding
    assert_eq(
        round_ne_target_width<u32:9>(make_fixed_point<s32:2>(s9:0b10101010)),
        make_fixed_point<s32:2>(s9:0b10101010));
    assert_eq(
        round_ne_target_width<u32:9>(make_fixed_point<s32:2>(s9:0b01010101)),
        make_fixed_point<s32:2>(s9:0b01010101));

    // We want to test these cases:
    // If the discarded bits > half, round up (e.g. .1001b)
    // If the discarded bits < half, round down (e.g. .0111b)
    // If the discarded bits == half, we have to consider the least significant retained bit:
    //  * if it is odd, round up    (e.g. 01.1000b -> 10.b)
    //  * if it is even, round down (e.g. 00.1000b -> 00.b)

    // the discarded bits > half, round up (e.g. .1001b)
    assert_eq(
        round_ne_target_width<u32:2>(make_fixed_point<s32:0>(s5:0b10101)),
        make_fixed_point<s32:3>(s2:0b11));
    assert_eq(
        round_ne_target_width<u32:2>(make_fixed_point<s32:0>(s5:0b10110)),
        make_fixed_point<s32:3>(s2:0b11));
    assert_eq(
        round_ne_target_width<u32:2>(make_fixed_point<s32:0>(s5:0b10111)),
        make_fixed_point<s32:3>(s2:0b11));

    // If the discarded bits < half, round down (e.g. .0111b)
    assert_eq(
        round_ne_target_width<u32:2>(make_fixed_point<s32:0>(s5:0b10000)),
        make_fixed_point<s32:3>(s2:0b10));
    assert_eq(
        round_ne_target_width<u32:2>(make_fixed_point<s32:0>(s5:0b10001)),
        make_fixed_point<s32:3>(s2:0b10));
    assert_eq(
        round_ne_target_width<u32:2>(make_fixed_point<s32:0>(s5:0b10010)),
        make_fixed_point<s32:3>(s2:0b10));
    assert_eq(
        round_ne_target_width<u32:2>(make_fixed_point<s32:0>(s5:0b10011)),
        make_fixed_point<s32:3>(s2:0b10));

    // If the discarded bits == half, we have to consider the least significant retained bit:
    //  * if it is odd, round up    (e.g. 01.1000b -> 10.b)
    assert_eq(
        round_ne_target_width<u32:2>(make_fixed_point<s32:0>(s5:0b01100)),
        make_fixed_point<s32:3>(s2:0b10));

    // If the discarded bits == half, we have to consider the least significant retained bit:
    //  * if it is even, round down (e.g. 00.1000b -> 00.b)
    assert_eq(
        round_ne_target_width<u32:2>(make_fixed_point<s32:0>(s5:0b10100)),
        make_fixed_point<s32:3>(s2:0b10));

    // round up and overflow
    assert_eq(
        round_ne_target_width<u32:2>(make_fixed_point<s32:0>(s5:0b11111)),
        make_fixed_point<s32:3>(s2:0b00));
}

#[test]
fn test_round_ne_target_exponent() {
    // Check that the type arithmetic is correct
    assert_eq(
        round_ne_target_exponent<s32:3>(make_fixed_point<s32:0>(s10:0)),
        make_fixed_point<s32:3>(s7:0));
    assert_eq(
        round_ne_target_exponent<s32:-2>(make_fixed_point<s32:-4>(s10:0)),
        make_fixed_point<s32:-2>(s8:0));

    // We're not going to do comprehensive unit testing because the function is just a
    // wrapper around round_ne_target_width. We adapt a few of round_ne_target_width's unit tests:

    // If the discarded bits == half, we have to consider the least significant retained bit:
    //  * if it is odd, round up    (e.g. 01.1000b -> 10.b)
    assert_eq(
        round_ne_target_exponent<s32:3>(make_fixed_point<s32:0>(s5:0b01100)),
        make_fixed_point<s32:3>(s2:0b10));

    // If the discarded bits == half, we have to consider the least significant retained bit:
    //  * if it is even, round down (e.g. 00.1000b -> 00.b)
    assert_eq(
        round_ne_target_exponent<s32:3>(make_fixed_point<s32:0>(s5:0b10100)),
        make_fixed_point<s32:3>(s2:0b10));
}

#[test]
fn test_narrow_by() {
    // Test no rounding
    let x = make_fixed_point<s32:2>(s9:0b10101010);
    assert_eq(narrow_by<u32:0>(x), x);
    let x = make_fixed_point<s32:-3>(s9:0b10101010);
    assert_eq(narrow_by<u32:0>(x), x);

    // Test no overflow case
    // posiitve input
    assert_eq(
        narrow_by<u32:1>(make_fixed_point<s32:-2>(s9:0b011111111)),
        make_fixed_point<s32:-2>(s8:0b11111111));
    // negative input
    assert_eq(
        narrow_by<u32:1>(make_fixed_point<s32:-2>(s9:0b111111111)),
        make_fixed_point<s32:-2>(s8:0b11111111));

    // Test overflow occurs but is not detected
    // positive input
    assert_eq(
        narrow_by<u32:2>(make_fixed_point<s32:-2>(s9:0b011111111)),
        make_fixed_point<s32:-2>(s7:0b1111111));
    // negative input
    assert_eq(
        narrow_by<u32:3>(make_fixed_point<s32:-2>(s9:0b100000000)), make_fixed_point<s32:-2>(s6:0));

    // can discard all integer bits
    assert_eq(
        narrow_by<u32:3>(make_fixed_point<s32:-1>(s4:0b1111)), make_fixed_point<s32:-1>(s1:0b1));
}

#[test]
fn test_to_common_numbits_and_exponent() {
    // exponent decrease by 1. numbits increase by 1
    assert_eq(
        to_common_type<u32:11, u32:4>(make_fixed_point<s32:-3>(s10:375)),
        make_fixed_point<s32:-4>(s11:750));

    // exponent decrease by 2. numbits unchanged.
    assert_eq(
        to_common_type<u32:12, u32:5>(make_fixed_point<s32:-3>(s12:253)),
        make_fixed_point<s32:-5>(s12:1012));

    // exponent decrease by 3. numbits increases by 3. negative significand. If casting before
    // shifting is not done, the shift will overflow.
    assert_eq(
        to_common_type<u32:10, u32:15>(make_fixed_point<s32:-12>(s7:-63)),
        make_fixed_point<s32:-15>(s10:-504));
}

import float32;

#[test]
fn test_convert_to_float_using_round_ties_to_even() {
    type F32 = float32::F32;
    type ExpBits = sN[float32::F32_EXP_SZ];
    type FractionBits = uN[float32::F32_FRACTION_SZ];

    // ↓↓↓↓ fxp is zero with varying {exponents, widths} ↓↓↓↓
    let fxp = make_fixed_point<s32:0>(s2:0);
    let expected = float32::zero(false);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);

    let fxp = make_fixed_point<s32:0>(s8:0);
    let expected = float32::zero(false);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);

    let fxp = make_fixed_point<s32:-3>(s33:0);
    let expected = float32::zero(false);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);

    let fxp = make_fixed_point<s32:4>(s17:0);
    let expected = float32::zero(false);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);

    // ↓↓↓↓ fxp is most-negative representable value; does magnitude =
    // std::abs(src.significand)
    // work? ↓↓↓↓
    let fxp = make_fixed_point<s32:0>(s3:-4);
    let expected = float32::from_int32(s32:-4);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);

    // ↓↓↓↓ fxp is ∞ with varying {exponents, widths} ↓↓↓↓
    // testing that 1*2^127 is finite while numbers at least 2x larger are ∞
    let fxp = make_fixed_point<s32:127>(s2:1);
    assert_eq(
        apfloat::tag(
            convert_to_float_using_round_ties_to_even<
                SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
                fxp)), apfloat::APFloatTag::NORMAL);
    let fxp = make_fixed_point<s32:126>(s3:1);
    assert_eq(
        apfloat::tag(
            convert_to_float_using_round_ties_to_even<
                SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
                fxp)), apfloat::APFloatTag::NORMAL);
    let fxp = make_fixed_point<s32:126>(s3:2);
    assert_eq(
        apfloat::tag(
            convert_to_float_using_round_ties_to_even<
                SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
                fxp)), apfloat::APFloatTag::NORMAL);
    let fxp = make_fixed_point<s32:126>(s3:3);
    assert_eq(
        apfloat::tag(
            convert_to_float_using_round_ties_to_even<
                SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
                fxp)), apfloat::APFloatTag::NORMAL);
    let fxp = make_fixed_point<s32:127>(s10:1);
    assert_eq(
        apfloat::tag(
            convert_to_float_using_round_ties_to_even<
                SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
                fxp)), apfloat::APFloatTag::NORMAL);
    // ±∞ are produced
    let fxp = make_fixed_point<s32:128>(s2:1);
    let expected = float32::inf(false);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    let fxp = make_fixed_point<s32:128>(s2:-1);
    let expected = float32::inf(true);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    // exp is smaller but the fixed point significand 2x larger
    let fxp = make_fixed_point<s32:127>(s3:2);
    let expected = float32::inf(false);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    let fxp = make_fixed_point<s32:127>(s3:-2);
    let expected = float32::inf(true);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);

    let fxp = make_fixed_point<s32:126>(s4:4);
    let expected = float32::inf(false);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);

    // ↓↓↓↓ subnormals are flushed to zero ↓↓↓↓
    // 2^-126 is normal, 2^-127 is subnormal and is flushed to zero
    let fxp = make_fixed_point<s32:-126>(s32:1);
    assert_eq(
        apfloat::tag(
            convert_to_float_using_round_ties_to_even<
                SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
                fxp)), apfloat::APFloatTag::NORMAL);
    let fxp = make_fixed_point<s32:-127>(s32:1);
    let expected = float32::zero(false);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    let fxp = make_fixed_point<s32:-127>(s32:-1);
    let expected = float32::zero(true);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    // twice as big is normal
    let fxp = make_fixed_point<s32:-127>(s32:2);
    assert_eq(
        apfloat::tag(
            convert_to_float_using_round_ties_to_even<
                SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
                fxp)), apfloat::APFloatTag::NORMAL);
    let fxp = make_fixed_point<s32:-127>(s32:-2);
    assert_eq(
        apfloat::tag(
            convert_to_float_using_round_ties_to_even<
                SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
                fxp)), apfloat::APFloatTag::NORMAL);
    // reduce exponent by 1 and these are subnormal
    let fxp = make_fixed_point<s32:-128>(s32:2);
    let expected = float32::zero(false);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    let fxp = make_fixed_point<s32:-128>(s32:-2);
    let expected = float32::zero(true);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    // 3*2^-128 is subnormal
    let fxp = make_fixed_point<s32:-128>(s32:3);
    let expected = float32::zero(false);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    let fxp = make_fixed_point<s32:-128>(s32:-3);
    let expected = float32::zero(true);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    // 4*2^-128 is normal
    let fxp = make_fixed_point<s32:-128>(s32:4);
    assert_eq(
        apfloat::tag(
            convert_to_float_using_round_ties_to_even<
                SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
                fxp)), apfloat::APFloatTag::NORMAL);
    let fxp = make_fixed_point<s32:-128>(s32:-4);
    assert_eq(
        apfloat::tag(
            convert_to_float_using_round_ties_to_even<
                SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
                fxp)), apfloat::APFloatTag::NORMAL);

    // ↓↓↓↓ normalized values ↓↓↓↓
    // ↓↓↓↓ integers created via fxp with non-negative binary exponent ↓↓↓↓
    let fxp = make_fixed_point<s32:0>(s32:1);
    let expected = float32::from_int32(s32:1);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    let fxp = make_fixed_point<s32:0>(s32:2);
    let expected = float32::from_int32(s32:2);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    let fxp = make_fixed_point<s32:1>(s32:1);
    let expected = float32::from_int32(s32:2);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    let fxp = make_fixed_point<s32:30>(s32:1);
    let expected = float32::from_int32(s32:1073741824);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    let fxp = make_fixed_point<s32:29>(s32:2);
    let expected = float32::from_int32(s32:1073741824);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);

    // ↓↓↓↓ integers created via fxp with negative binary exponent ↓↓↓↓
    let fxp = make_fixed_point<s32:-1>(s32:2);
    let expected = float32::from_int32(s32:1);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    let fxp = make_fixed_point<s32:-2>(s32:4);
    let expected = float32::from_int32(s32:1);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    let fxp = make_fixed_point<s32:-2>(s32:16);
    let expected = float32::from_int32(s32:4);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    let fxp = make_fixed_point<s32:-2>(s32:12);
    let expected = float32::from_int32(s32:3);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    let fxp = make_fixed_point<s32:-2>(s32:20);
    let expected = float32::from_int32(s32:5);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);

    // ↓↓↓↓ integers approach the exactly representable threshold ↓↓↓↓
    let fxp = make_fixed_point<s32:0>(s32:0b100000000000000000000000);
    let expected = float32::from_int32(s32:8388608);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    let fxp = make_fixed_point<s32:0>(s32:0b111111111111111111111110);
    let expected = float32::from_int32(s32:16777214);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    let fxp = make_fixed_point<s32:0>(s32:0b111111111111111111111111);
    let expected = float32::from_int32(s32:16777215);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);

    // ↓↓↓↓ integers that are not exactly representable ↓↓↓↓
    // smallest int not exactly representable. rounds down to even
    let fxp = make_fixed_point<s32:0>(s32:16777217);
    let expected = float32::from_int32(s32:16777216);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    // rounds up to even
    let fxp = make_fixed_point<s32:0>(s32:16777219);
    let expected = float32::from_int32(s32:16777220);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);

    // negative fxp binary exponent
    // smallest int not exactly representable. rounds down to even
    let fxp = make_fixed_point<s32:-1>(s32:33554434);
    let expected = float32::from_int32(s32:16777216);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    // rounds up to even
    let fxp = make_fixed_point<s32:-1>(s32:33554438);
    let expected = float32::from_int32(s32:16777220);
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);

    // ↓↓↓↓ a wide value that must be rounded, overflows when rounding, increases the
    // exponent ↓↓↓↓
    // We start with an exactly representable value (i.e. up to contiguous 24 set bits). Then we
    // add 1 to it (producing 25 set bits) and observe that rounding (during conversion)
    // increases the result's f32's exponent
    let fxp = make_fixed_point<s32:0>(s32:0b1111111111111111111111110);
    let expected = F32 {
        sign: u1:0,
        bexp: float32::bias(s8:24),
        fraction: FractionBits:0b11111111111111111111111,
    };
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    let fxp = add(make_fixed_point<s32:0>(s32:1), fxp);
    let expected = F32 { sign: u1:0, bexp: float32::bias(s8:25), fraction: FractionBits:0b0 };
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    // Let's do it again with a value that has a negative binary exponent
    let fxp = make_fixed_point<s32:-24>(s32:0b1111111111111111111111110);
    let expected = F32 {
        sign: u1:0,
        bexp: float32::bias(s8:0),
        fraction: FractionBits:0b11111111111111111111111,
    };
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
    let fxp = make_fixed_point<s32:-24>(s32:0b1111111111111111111111111);
    let expected = F32 { sign: u1:0, bexp: float32::bias(s8:1), fraction: FractionBits:0b0 };
    assert_eq(
        convert_to_float_using_round_ties_to_even<
            SubnormalOutputs::FlushToZero, float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(
            fxp), expected);
}
