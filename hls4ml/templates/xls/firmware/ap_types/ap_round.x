// The code below is an adaptation of std.round.x (https://github.com/google/xls/blob/dd8e4023ffd993e542bb8a397aa60a0f29583cf7/xls/dslx/stdlib/round.x)
// that uses its own RoundingMode which matches hls4ml.model.types.RoundingMode (or ap_fixed quantization modes).

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


// Implements rounding for all rounding modes defined by hls4ml.model.types.RoundingMode (or ap_fixed type).
//
// It handles unsigned, signed (two's complement), and sign-and-magnitude values.
// Versions with compile-time 'num bits rounded' argument are provided, truncating the rounded-away
// bits.
// Versions with runtime 'num bits rounded' argument are provided, returning the full-width rounded
// result, with the rounded-away bits zeroed.
//
// Note: XLS prunes unused specializations. If callers pass compile-time constants for
// `num_bits_rounded` or restrict `RoundingMode` via an adapter, the optimizer keeps only the
// cases that remain reachable. (E.g. wrap this API with your own enum of three rounding modes
// and convert to `RoundingMode`, then the others will fold away.)
//
// Let's say you only want 3 rounding modes. Your public API should be a new enum with only
// those 3 rounding modes, and a function that takes that enum, and translates it to the
// RoundingMode enum of this library. XLS optimizer will remove the unused rounding modes from
// the optimized code.

import std;
import round as std_round;

// Rounding modes defined by hls4ml.model.types.RoundingMode (or ap_fixed type).
// NB: do not confuse with std_round::RoundingMode!
// Note that the last five (RND, RND_ZERO, RND_INF, RND_MIN_INF, RND_CONV) always round to the nearest value, and when two potential
// results are equally close, a tie-breaking rule is applied. The first two (TRN, TRN_ZERO)
// first establish a direction on the Extended Real number line, and then round to the nearest
// value in that direction. If there is a closer value in the opposite direction, it is never
// returned.
//
// On naming: those that round to the nearest value begin with "RND", and those that always round
// in the same direction begin with "TRN".
type RoundingModeIntegerType = u3;
pub enum RoundingMode: RoundingModeIntegerType {
    // Directed truncation toward -inf.
    // equivalent to std_round::RoundingMode::RTN
    TRN         = 1,
    // Directed truncation toward 0.
    // equivalent to std_round::RoundingMode::RTZ
    TRN_ZERO    = 2,
    // Round to nearest, ties toward +inf.
    RND         = 3,
    // Round to nearest, ties toward 0.
    RND_ZERO    = 4,
    // Round to nearest, ties away from 0.
    // equivalent to std_round::RoundingMode::RNA
    RND_INF     = 5,
    // Round to nearest, ties toward -inf.
    RND_MIN_INF = 6,
    // Round to nearest, ties to even.
    // equivalent to std_round::RoundingMode::RNE
    RND_CONV    = 7
}

// Indicates a positive (more precisely: non-negative) or negative number.
pub type Sign = std_round::Sign;

// Conversion to/from std_round::RoundingMode

const TO_STD_MODE = [
    (RoundingMode::TRN, std_round::RoundingMode::RTN),
    (RoundingMode::TRN_ZERO, std_round::RoundingMode::RTZ),
    (RoundingMode::RND_INF, std_round::RoundingMode::RNA),
    (RoundingMode::RND_CONV, std_round::RoundingMode::RNE),
];

pub fn try_convert_to_std(rm: RoundingMode) -> (bool, std_round::RoundingMode) {
    let res = (false, std_round::RoundingMode::RTN);
    for ((from, to), res) in TO_STD_MODE {
        if (rm == from) {
            (true, to)
        } else {
            res
        }
    }(res)
}

pub fn try_convert_from_std(rm: std_round::RoundingMode) -> (bool, RoundingMode) {
    let res = (false, RoundingMode::TRN);
    for ((to, from), res) in TO_STD_MODE {
        if (rm == from) {
            (true, to)
        } else {
            res
        }
    }(res)
}

pub fn convert_to_std(rm: RoundingMode) -> std_round::RoundingMode {
    let (ok, res) = try_convert_to_std(rm);
    assert!(ok, "ap_round_convert_to_std_failed");
    res
}

pub fn convert_from_std(rm: std_round::RoundingMode) -> RoundingMode {
    let (ok, res) = try_convert_from_std(rm);
    assert!(ok, "ap_round_convert_from_std_failed");
    res
}

// Rounds off the `num_bits_rounded` least significant bits. Returns (overflow, rounded result).
//
// Works for:
// - unsigned integers
//   - `sign` must be `NonNegative` (otherwise you have the sign-and-magnitude case, see below)
// - signed (two's complement) integers
//   - `sign` is ignored. The most significant bit of `unrounded` is used to determine the sign.
// - sign and magnitude values
//   - `sign` must be `Negative` when the represented value is a negative number, otherwise
//     `sign` must be `NonNegative`
//   - rounding may produce a zero magnitude from a negative input; callers must decide whether
//     to keep or flip the sign in that case
//
// The `num_bits_rounded` lsbs of the rounded result will always be 0.
//
// Users should interpret `unrounded` as a fixed-point quantity with num_bits_rounded fractional
// bits, being rounded to an integer. For unsigned inputs the corresponding Real value is
// unrounded / 2^num_bits_rounded, and the rounding modes apply directly to that Real
// number. This viewpoint also explains the RND_CONV tie case when every retained bit is discarded:
// the surviving integer portion is zero (an even value), so ties resolve towards zero.
//
// Overflow is 1 when the Real rounded result isn't a representable result (because the increase
// in magnitude requires a wider result type). Some non-exhaustive examples of when that can
// occur:
//  * RND_CONV(3.5) = 4 -> overflow when round(RND_CONV, 2, NonNegative, u4:0b11_10)
//  * TRN(-1.0625) = -2 -> overflow when round(TRN, 4, Negative, u5:0b1_0001)
//  * TRN(-0.03125) = -1 -> overflow when round(TRN, 5, NonNegative, s5:0b11111)
// The rounded result is 0 when overflow is 1.
//
// When num_bits_rounded > N, all source bits are treated as fractional. The rounded integer is 0
// unless the rounding mode requires +/-1, in which case overflow is signaled and 0 is returned.
//
// As mentioned above, during a tie, RND_CONV looks at the least significant retained bit to
// determine round up or down. When there are no retained bits (i.e. num_bits_rounded >= N),
// round down is chosen. E.g.
// round(RND_CONV, 4 bits, unsigned, u5:0b1_1000) -> rounds up (retained msb is 1)
// round(RND_CONV, 4 bits, unsigned, u4:0b1000) -> rounds down (no retained bits)
pub fn round<S: bool, N: u32, W_NBR: u32 = {std::clog2(N + u32:1)}>
    (rounding_mode: RoundingMode, num_bits_rounded: uN[W_NBR], sign: Sign, unrounded: xN[S][N])
    -> (u1, xN[S][N]) {
    type NumBitsRoundedT = uN[W_NBR];

    // Works even when N is zero.
    type SafeWord = uN[std::max(N, u32:1)];

    // Wide enough to represent overflow.
    type ExtendedWord = uN[N + u32:1];

    // Compute sign bit while avoiding issues when N is zero.
    let unrounded_u = unrounded as uN[N];
    let sign_shift = std::usub_or_zero(N, u32:1) as NumBitsRoundedT;
    let unrounded_sign_bit = std::lsb((unrounded_u as SafeWord) >> sign_shift);

    // determine sign when unrounded is two's complement
    let sign = if S {
        if unrounded_sign_bit == u1:1 { Sign::Negative } else { Sign::NonNegative }
    } else {
        sign
    };

    if N == u32:0 {
        (u1:0, xN[S][N]:0)
    } else if num_bits_rounded == NumBitsRoundedT:0 {
        (u1:0, unrounded)
    } else if num_bits_rounded as u32 > N {
        let is_zero = unrounded_u == uN[N]:0;
        let is_strictly_negative = !is_zero && sign == Sign::Negative;
        let overflow = match rounding_mode {
            RoundingMode::TRN => if is_strictly_negative { u1:1 } else { u1:0 },
            _ => u1:0,
        };
        (overflow, xN[S][N]:0)
    } else {
        let negative_twos_complement = S && sign == Sign::Negative;

        // The bits rounded away; these bits are always 0 in the result.
        let rounded_bits = std::keep_lsbs(unrounded_u, num_bits_rounded);
        let rounded_bits_safe = rounded_bits as SafeWord;

        // The bits that will be returned, before any rounding adjustment.
        let retained_bits = std::clear_lsbs(unrounded_u, num_bits_rounded);

        // Note: zero retained bits means retained_bits_are_odd is false.
        let retained_bits_are_odd = std::lsb(retained_bits >> num_bits_rounded);

        let rounded_bits_are_nonzero = rounded_bits_safe != SafeWord:0;

        // This is the value of 0.5 when num_bits_rounded is interpreted as a negative binary
        // exponent (and by implication, `unrounded` is a binary fixed point value). We are
        // rounding the fixed point value to a nearby integer. This is 0.5 in this fixed point
        // format.
        let half_value = (SafeWord:1) << (num_bits_rounded as SafeWord - SafeWord:1);

        // as we defined half above, we use a similar definition of one
        let one = (ExtendedWord:1) << num_bits_rounded;
        let zero = ExtendedWord:0;

        // Beware rounded_gt_half when unrounded is two's complement and negative; it's
        // misleading.
        let rounded_gt_half = rounded_bits_safe > half_value;
        let rounded_eq_half = rounded_bits_safe == half_value;

        let adjustment = match rounding_mode {
            RoundingMode::RND_CONV => {
                // round to nearest, ties to even
                // when |rounded_bits| > |half| or (|rounded_bits| == |half| and the retained bits
                // are odd)
                // the adjustment is:
                // unsigned -> 1
                // sign & magnitude, positive value -> 1
                // sign & magnitude, negative value -> 1
                // two's complement, positive value -> 1
                // two's complement, negative value is more complex, see below
                let tie_to_even = rounded_eq_half && retained_bits_are_odd;
                if negative_twos_complement {
                    // recall that rounded > 0.5 means the (negative two's complement) value is
                    // closer to 0 than half. E.g. -4 + 0.75 = -3.25 is closer to 0 than -3.5 is.
                    let closer_to_zero_than_half_is = rounded_gt_half;
                    if closer_to_zero_than_half_is || tie_to_even {
                        // RND_CONV(-3.25) -> -3, retained=-4, thus adjustment=1
                        // RND_CONV(-2.5) -> -2, retained=-3, thus adjustment=+1
                        one
                    } else {
                        // case: further from 0 than half is (e.g. -4 + 0.25 = -3.75 which is
                        // further from 0 than -3.5 is) OR rounded=0.5 and retained bits are even.
                        // RND_CONV(-3.75) -> -4, retained=-4, thus adjustment=0
                        // RND_CONV(-3.5) -> -4, retained=-4, thus adjustment=0
                        zero
                    }
                } else {
                    if rounded_gt_half || tie_to_even { one } else { zero }
                }
            },
            RoundingMode::RND_INF => {
                // round to nearest, ties away from zero
                // when |rounded_bits| >= |half| the adjustment is:
                // unsigned -> 1
                // sign & magnitude, positive value -> 1
                // sign & magnitude, negative value -> 1
                // two's complement, positive value -> 1
                // two's complement, negative value -> 0 (because truncation is toward -∞)
                //
                // you'll notice that RND_CONV and RND_INF are the same w.r.t. the adjustment, and only
                // differ in the case of a tie (they agree when |rounded_bits| > |half|)
                if negative_twos_complement {
                    // recall that rounded > 0.5 means the (negative two's complement) value is
                    // closer to 0 than half. E.g. -4 + 0.75 = -3.25 is closer to 0 than -3.5 is.
                    let closer_to_zero_than_half_is = rounded_gt_half;
                    if closer_to_zero_than_half_is {
                        // RND_INF(-3.25) -> -3, retained=-4, thus adjustment=1
                        one
                    } else {
                        // RND_INF(-3.5) -> -4, retained=-4, thus adjustment=0
                        // RND_INF(-3.75) -> -4, retained=-4, thus adjustment=0
                        zero
                    }
                } else {
                    // unsigned or sign-magnitude and positive two's-complement
                    if rounded_gt_half || rounded_eq_half { one } else { zero }
                }
            },
            RoundingMode::RND => {
                // round to nearest, ties toward positive infinity
                if rounded_gt_half {
                    one
                } else if rounded_eq_half {
                    match (S, sign) {
                        // sign & magnitude, negative value
                        (false, Sign::Negative) => zero,
                        _ => one,
                    }
                } else {
                    zero
                }
            },
            RoundingMode::RND_MIN_INF => {
                // round to nearest, ties toward negative infinity
                if rounded_gt_half {
                    one
                } else if rounded_eq_half {
                    match (S, sign) {
                        // sign & magnitude, negative value
                        (false, Sign::Negative) => one,
                        _ => zero,
                    }
                } else {
                    zero
                }
            },
            RoundingMode::RND_ZERO => {
                // round to nearest, ties toward zero
                if rounded_gt_half {
                    one
                } else if rounded_eq_half {
                    if negative_twos_complement { one } else { zero }
                } else {
                    zero
                }
            },
            RoundingMode::TRN_ZERO => {
                // round toward zero
                // when rounded_bits != zero, the adjustment is:
                // unsigned -> 0
                // sign & magnitude, positive value -> 0
                // sign & magnitude, negative value -> 0
                // two's complement, positive value -> 0
                // two's complement, negative value -> 1
                if negative_twos_complement && rounded_bits_are_nonzero { one } else { zero }
            },
            RoundingMode::TRN => {
                // round toward negative infinity
                // when rounded_bits != zero, the adjustment is:
                // unsigned -> 0
                // sign & magnitude, positive value -> 0
                // sign & magnitude, negative value -> 1
                // two's complement, positive value -> 0
                // two's complement, negative value -> 0 (because truncation is toward -∞)
                if rounded_bits_are_nonzero {
                    match (S, sign) {
                        (false, Sign::Negative) => one,
                        _ => zero,
                    }
                } else {
                    zero
                }
            },
        };

        let sum = retained_bits as ExtendedWord + adjustment;
        let (carry, rounded_u) = std::split_msbs<u32:1>(sum);

        let rounded_sign_bit = std::lsb((rounded_u as SafeWord) >> sign_shift);
        let sign_changed = S && rounded_sign_bit != unrounded_sign_bit;
        let adjustment_is_one = adjustment == one;
        let rounding_all_bits = (num_bits_rounded as u32) == N;
        let overflow = if !S {
            // Unsigned or sign-and-magnitude: any carry-out indicates overflow.
            carry
        } else {
            match (sign, adjustment_is_one) {
                // Positive argument, no adjustment - never overflows.
                (Sign::NonNegative, false) => false,
                (Sign::NonNegative, true) => {
                    // Positive argument, adjustment of +1.
                    //   - If we rounded away every bit, rely on the carry-out.
                    //   - If we kept at least one integer bit, check for sign change.
                    if rounding_all_bits { carry } else { sign_changed }
                },
                // Negative argument with +1 adjustment never overflows (-1 -> 0 etc.).
                (Sign::Negative, true) => false,
                (Sign::Negative, false) => {
                    // Negative argument with no adjustment.
                    // When every bit is rounded away the result is 0 (TRN_ZERO/RND/RND_ZERO/RND_CONV)
                    // or -1 (TRN/RND_MIN_INF/RND_INF). The latter is not representable because no integer bits
                    // remain.
                    if rounding_all_bits &&
                    (rounding_mode == RoundingMode::TRN ||
                     rounding_mode == RoundingMode::RND_MIN_INF ||
                     rounding_mode == RoundingMode::RND_INF) {
                        true
                    } else {
                        false
                    }
                },
            }
        };

        let rounded_u = if sign_changed {
            // handles cases like:
            // argument is two's complement, positive, and rounding away all bits
            // RND_CONV(0.9375) = 1 -> overflow
            // round(RoundingMode::RND_CONV, 4, NonNegative, s5:0b0_1111))
            // without this correction, result would be s5:0b1_0000, i.e. -1
            uN[N]:0
        } else {
            rounded_u
        };
        (overflow, rounded_u as xN[S][N])
    }
}

// Rounds an unsigned integer:
//  - rounds a runtime-specified number (`num_bits_rounded`) of least significant bits,
//  - returns the full-width rounded result with the least significant `num_bits_rounded` bits
//    zeroed.
// Returns (overflow, rounded result).
pub fn round_u<N: u32, W_NBR: u32 = {std::clog2(N + u32:1)}>
    (rounding_mode: RoundingMode, num_bits_rounded: uN[W_NBR], unrounded: uN[N]) -> (u1, uN[N]) {
    round(rounding_mode, num_bits_rounded, Sign::NonNegative, unrounded)
}

// Rounds an unsigned integer:
// - rounds a compile-time-constant (`num_bits_rounded`) number of least significant bits,
// - returns only the most significant bits (i.e., the rounded result), discarding the rounded-off
//   bits.
// Returns (overflow, rounded result).
pub fn round_trunc_u<NumBitsRounded: u32, N: u32, R: u32 = {N - NumBitsRounded}>
    (rounding_mode: RoundingMode, unrounded: uN[N]) -> (u1, uN[R]) {
    const_assert!(NumBitsRounded <= N);
    const W_NBR: u32 = std::clog2(N + u32:1);

    let (overflow, rounded) = round_u(rounding_mode, NumBitsRounded as uN[W_NBR], unrounded);
    let (rounded_msbs, _) = std::split_msbs<R>(rounded);
    (overflow, rounded_msbs)
}

// Rounds an unsigned integer:
// - such that after rounding it is `AtMost` (a compile-time constant) bits wide,
// - returns only the most significant bits (i.e., the rounded result), discarding the rounded-off
//   bits.
// Returns (overflow, rounded result).
pub fn round_trunc_to_u<AtMost: u32, N: u32, R: u32 = {std::min(AtMost, N)}>
    (rounding_mode: RoundingMode, unrounded: uN[N]) -> (u1, uN[R]) {
    const NUM_BITS_ROUNDED: u32 = std::usub_or_zero(N, R);

    if NUM_BITS_ROUNDED == u32:0 {
        // This no-op cast is required by the type checker. When this branch is not taken, this
        // cast op unifies the types of the branches.
        let unrounded = unrounded as uN[R];
        (u1:0, unrounded)
    } else {
        round_trunc_u<NUM_BITS_ROUNDED>(rounding_mode, unrounded)
    }
}

// Rounds a signed integer:
//  - rounds a runtime-specified number (`num_bits_rounded`) of least significant bits,
//  - returns the full-width rounded result with the least significant `num_bits_rounded` bits
//    zeroed.
// Returns (overflow, rounded result).
pub fn round_s<N: u32, W_NBR: u32 = {std::clog2(N + u32:1)}>
    (rounding_mode: RoundingMode, num_bits_rounded: uN[W_NBR], unrounded: sN[N]) -> (u1, sN[N]) {
    round(rounding_mode, num_bits_rounded, Sign::NonNegative, unrounded)
}

// Rounds a signed integer:
// - rounds a compile-time-constant (`num_bits_rounded`) number of least significant bits,
// - returns only the most significant bits (i.e., the rounded result), discarding the rounded-off
//   bits.
// Returns (overflow, rounded result).
pub fn round_trunc_s<num_bits_rounded: u32, N: u32, R: u32 = {N - num_bits_rounded}>
    (rounding_mode: RoundingMode, unrounded: sN[N]) -> (u1, sN[R]) {
    const_assert!(num_bits_rounded <= N);
    const W_NBR: u32 = std::clog2(N + u32:1);

    if R == u32:0 {
        let (overflow, _) = round_s(rounding_mode, num_bits_rounded as uN[W_NBR], unrounded);
        (overflow, zero!<sN[R]>())
    } else {
        let (overflow, rounded) = round_s(rounding_mode, num_bits_rounded as uN[W_NBR], unrounded);
        let (rounded_msbs, _) = std::split_msbs<R>(rounded as uN[N]);
        (overflow, rounded_msbs as sN[R])
    }
}

// Rounds a signed integer:
// - such that after rounding it is `AtMost` (a compile-time constant) bits wide,
// - returns only the most significant bits (i.e., the rounded result), discarding the rounded-off
//   bits.
// Returns (overflow, rounded result).
pub fn round_trunc_to_s<AtMost: u32, N: u32, R: u32 = {std::min(AtMost, N)}>
    (rounding_mode: RoundingMode, unrounded: sN[N]) -> (u1, sN[R]) {
    const NUM_BITS_ROUNDED: u32 = std::usub_or_zero(N, R);

    if NUM_BITS_ROUNDED == u32:0 {
        let unrounded = unrounded as sN[R];
        (u1:0, unrounded)
    } else {
        round_trunc_s<NUM_BITS_ROUNDED>(rounding_mode, unrounded)
    }
}

// Rounds a sign-and-magnitude integer:
//  - rounds a runtime-specified number (`num_bits_rounded`) of least significant bits,
//  - returns the full-width rounded result with the least significant `num_bits_rounded` bits
//    zeroed.
// Returns (overflow, rounded result).
pub fn round_sm<N: u32, W_NBR: u32 = {std::clog2(N + u32:1)}>
    (rounding_mode: RoundingMode, num_bits_rounded: uN[W_NBR], sign: Sign, magnitude: uN[N])
    -> (u1, uN[N]) {
    round(rounding_mode, num_bits_rounded, sign, magnitude)
}

// Rounds a sign-and-magnitude integer:
// - rounds a compile-time-constant (`num_bits_rounded`) number of least significant bits,
// - returns only the most significant bits (i.e., the rounded result), discarding the rounded-off
//   bits.
// Returns (overflow, rounded result).
pub fn round_trunc_sm<num_bits_rounded: u32, N: u32, R: u32 = {N - num_bits_rounded}>
    (rounding_mode: RoundingMode, sign: Sign, magnitude: uN[N]) -> (u1, uN[R]) {
    const_assert!(num_bits_rounded <= N);
    const W_NBR: u32 = std::clog2(N + u32:1);

    let (overflow, rounded) =
        round_sm(rounding_mode, num_bits_rounded as uN[W_NBR], sign, magnitude);
    let (rounded_msbs, _) = std::split_msbs<R>(rounded);
    (overflow, rounded_msbs)
}

// Rounds a sign-and-magnitude integer:
// - such that after rounding it is `AtMost` (a compile-time constant) bits wide,
// - returns only the most significant bits (i.e., the rounded result), discarding the rounded-off
//   bits.
// Returns (overflow, rounded result).
pub fn round_trunc_to_sm<AtMost: u32, N: u32, R: u32 = {std::min(AtMost, N)}>
    (rounding_mode: RoundingMode, sign: Sign, magnitude: uN[N]) -> (u1, uN[R]) {
    const NUM_BITS_ROUNDED: u32 = std::usub_or_zero(N, R);

    if NUM_BITS_ROUNDED == u32:0 {
        let magnitude = magnitude as uN[R];
        (u1:0, magnitude)
    } else {
        round_trunc_sm<NUM_BITS_ROUNDED>(rounding_mode, sign, magnitude)
    }
}

#[test]
fn test_rounding_modes_unsigned() {
    assert_eq(round_u(RoundingMode::TRN, u3:2, u4:2), (u1:0, u4:0));
    assert_eq(round_u(RoundingMode::TRN_ZERO, u3:2, u4:2), (u1:0, u4:0));
    assert_eq(round_u(RoundingMode::RND, u3:2, u4:2), (u1:0, u4:4));
    assert_eq(round_u(RoundingMode::RND_ZERO, u3:2, u4:2), (u1:0, u4:0));
    assert_eq(round_u(RoundingMode::RND_INF, u3:2, u4:2), (u1:0, u4:4));
    assert_eq(round_u(RoundingMode::RND_MIN_INF, u3:2, u4:2), (u1:0, u4:0));
    assert_eq(round_u(RoundingMode::RND_CONV, u3:2, u4:2), (u1:0, u4:0));
}

#[test]
fn test_rounding_modes_signed() {
    assert_eq(round_s(RoundingMode::TRN, u3:2, s4:-2), (u1:0, s4:-4));
    assert_eq(round_s(RoundingMode::TRN_ZERO, u3:2, s4:-2), (u1:0, s4:0));
    assert_eq(round_s(RoundingMode::RND, u3:2, s4:-2), (u1:0, s4:0));
    assert_eq(round_s(RoundingMode::RND_ZERO, u3:2, s4:-2), (u1:0, s4:0));
    assert_eq(round_s(RoundingMode::RND_INF, u3:2, s4:-2), (u1:0, s4:-4));
    assert_eq(round_s(RoundingMode::RND_MIN_INF, u3:2, s4:-2), (u1:0, s4:-4));
    assert_eq(round_s(RoundingMode::RND_CONV, u3:2, s4:-2), (u1:0, s4:0));
}

const ROUNDING_MODES = [
    RoundingMode::TRN,
    RoundingMode::TRN_ZERO,
    RoundingMode::RND,
    RoundingMode::RND_ZERO,
    RoundingMode::RND_INF,
    RoundingMode::RND_MIN_INF,
    RoundingMode::RND_CONV,
];

// Round 2 bits of a 5-bit signed integer => no overflow.
fn check_all_rounding_modes(value: s5, expected: s5[7]) {
    let overflow = u1:0;
    for (i, _) in u32:0..array_size(ROUNDING_MODES) {
        assert_eq(round_s(ROUNDING_MODES[i], u3:2, value), (overflow, expected[i]));
    }(());
}

#[test]
fn test_all_rounding_modes() {
    check_all_rounding_modes(s5:1, [s5:0, 0, 0, 0, 0, 0, 0]);
    check_all_rounding_modes(s5:2, [s5:0, 0, 4, 0, 4, 0, 0]);
    check_all_rounding_modes(s5:3, [s5:0, 0, 4, 4, 4, 4, 4]);
    check_all_rounding_modes(s5:4, [s5:4, 4, 4, 4, 4, 4, 4]);
    check_all_rounding_modes(s5:5, [s5:4, 4, 4, 4, 4, 4, 4]);
    check_all_rounding_modes(s5:6, [s5:4, 4, 8, 4, 8, 4, 8]);
    check_all_rounding_modes(s5:-1, [s5:-4, 0, 0, 0, 0, 0, 0]);
    check_all_rounding_modes(s5:-2, [s5:-4, 0, 0, 0, -4, -4, 0]);
    check_all_rounding_modes(s5:-3, [s5:-4, 0, -4, -4, -4, -4, -4]);
    check_all_rounding_modes(s5:-4, [s5:-4, -4, -4, -4, -4, -4, -4]);
    check_all_rounding_modes(s5:-5, [s5:-8, -4, -4, -4, -4, -4, -4]);
    check_all_rounding_modes(s5:-6, [s5:-8, -4, -4, -4, -8, -8, -8]);
    check_all_rounding_modes(s5:-7, [s5:-8, -4, -8, -8, -8, -8, -8]);
}

// Check that std::round produces the same results as our version.
#[test]
fn test_stdlib_equivalent_rounding_modes() {
    for (value, _) in s5:-7..8 {
        for(num_bits_rounded, _) in u3:0..7 {
            for ((rounding, std_rounding), _) in TO_STD_MODE {
                assert_eq(
                    round_s(rounding, num_bits_rounded, value),
                    std_round::round_s(std_rounding, num_bits_rounded, value)
                );
                let u_value = value as u5;
                assert_eq(
                    round_u(rounding, num_bits_rounded, u_value),
                    std_round::round_u(std_rounding, num_bits_rounded, u_value)
                );
            }(());
        }(());
    }(());
}

fn check_unsigned_overflow(value: u5, num_bits_rounded: u3, expected: (u1, u5)[7]) {
    for (i, _) in u32:0..array_size(ROUNDING_MODES) {
        assert_eq(round_u(ROUNDING_MODES[i], num_bits_rounded, value), expected[i]);
    }(());
}

fn check_signed_overflow(value: s5, num_bits_rounded: u3, expected: (u1, s5)[7]) {
    for (i, _) in u32:0..array_size(ROUNDING_MODES) {
        assert_eq(round_s(ROUNDING_MODES[i], num_bits_rounded, value), expected[i]);
    }(());
}

#[test]
fn test_rounding_overflow() {
    // u5:31 is 7.75; rounding to an integer overflows when it reaches 8.
    check_unsigned_overflow(u5:31, u3:2,
                            [(u1:0, u5:28), (u1:0, u5:28), (u1:1, u5:0),
                             (u1:1, u5:0), (u1:1, u5:0), (u1:1, u5:0), (u1:1, u5:0)]);

    // s5:15 is 3.75; rounding to 4 changes the sign bit and overflows.
    check_signed_overflow(s5:15, u3:2,
                          [(u1:0, s5:12), (u1:0, s5:12), (u1:1, s5:0),
                           (u1:1, s5:0), (u1:1, s5:0), (u1:1, s5:0), (u1:1, s5:0)]);
}

#[test]
fn test_rounding_more_fractional_bits_overflow() {
    // With more discarded bits than source bits, nearest modes remain zero.
    check_unsigned_overflow(u5:1, u3:6,
                            [(u1:0, u5:0), (u1:0, u5:0), (u1:0, u5:0),
                             (u1:0, u5:0), (u1:0, u5:0), (u1:0, u5:0), (u1:0, u5:0)]);
    check_signed_overflow(s5:-1, u3:6,
                          [(u1:1, s5:0), (u1:0, s5:0), (u1:0, s5:0),
                           (u1:0, s5:0), (u1:0, s5:0), (u1:0, s5:0), (u1:0, s5:0)]);
    assert_eq(round_sm(RoundingMode::TRN, u3:6, Sign::Negative, u5:1), (u1:1, u5:0));
    assert_eq(round_sm(RoundingMode::RND, u3:6, Sign::Negative, u5:1), (u1:0, u5:0));
    assert_eq(round_sm(RoundingMode::RND, u3:6, Sign::NonNegative, u5:1), (u1:0, u5:0));
}
