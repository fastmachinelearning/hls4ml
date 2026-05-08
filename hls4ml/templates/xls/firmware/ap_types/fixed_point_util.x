// Collection of utility functions for fixed_point::FixedPoint<NUM_BITS, BINARY_EXPONENT>.
// Here we use abbreviations NB -> NUM_BITS, BE -> BINARY_EXPONENT.
// fixed_point::FixedPoint<NB, BE>{significand: sN[NB]} represents a real number (significand * 2^BE)

import std;
import fixed_point;
import round;

type FixedPoint = fixed_point::FixedPoint;
type Sign = round::Sign;

// All modes from hls4ml.model.types.RoundingMode
// NB: do not confuse with round.RoundingMode!
// TODO: not all modes are currently supported, see convert_rounding_mode()
type RoundingModeIntegerType = u3;
pub enum RoundingMode: RoundingModeIntegerType {
    // Trunacte toward -inf
    TRN         = 1,
    // Truncate towards 0
    TRN_ZERO    = 2,
    // Round towards +inf
    RND         = 3,
    // Round towards 0
    RND_ZERO    = 4,
    // Round towards +-inf
    RND_INF     = 5,
    // Round towards -inf
    RND_MIN_INF = 6,
    // Round towards nearest even
    RND_CONV    = 7
}

// Same oveflow modes as in ac_fixed type and in hls4ml
type OverflowModeIntegerType = u2;
pub enum OverflowMode: OverflowModeIntegerType {
    // Drop bits to the left of MSB
    WRAP      = 0,
    // Saturate to [MIN, MAX]
    SAT       = 1,
    // Set to 0 on overflow
    SAT_ZERO  = 2,
    // Saturate to [-MAX, MAX]
    SAT_SYM   = 3
}

// === Non-public functions copied from stdlib/fixed_point.x ===

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
    let NB = msb - lsb + s33:1;
    NB as u32
}

// === Create FixedPoint constants ===

pub fn one<NB: u32, BE: s32>() -> FixedPoint<NB, BE> {
    // If BE > 0, 1 is below quantization limit
    const_assert!(BE <= s32:0);
    let SHIFT = std::abs(BE) as u32;
    const_assert!(SHIFT <= NB);
    let x = sN[NB]:1 << SHIFT;
    fixed_point::make_fixed_point<BE>(x)
}

pub fn max_value<NB: u32, BE: s32>() -> FixedPoint<NB, BE> {
    fixed_point::make_fixed_point<BE>(std::signed_max_value<NB>())
}

pub fn min_value<NB: u32, BE: s32>() -> FixedPoint<NB, BE> {
    fixed_point::make_fixed_point<BE>(std::signed_min_value<NB>())
}

// === Create FixedPoint arrays numbers from arrays of significands sN[NB] ===


pub fn make_fixed_points_1d
    <BE: s32, NB: u32, DIM: u32>
    (significands: sN[NB][DIM])
    -> FixedPoint<NB, BE>[DIM] {
    map(significands, fixed_point::make_fixed_point<BE>)
}

pub fn make_fixed_points_2d
    <BE: s32, NB: u32, DIM_0: u32, DIM_1: u32>
    (significands: sN[NB][DIM_1][DIM_0])
    -> FixedPoint<NB, BE>[DIM_1][DIM_0] {
    map(significands, make_fixed_points_1d<BE>)
}

pub fn make_fixed_points_3d
    <BE: s32, NB: u32, DIM_0: u32, DIM_1: u32, DIM_2: u32>
    (significands: sN[NB][DIM_2][DIM_1][DIM_0])
    -> FixedPoint<NB, BE>[DIM_2][DIM_1][DIM_0] {
    map(significands, make_fixed_points_2d<BE>)
}

pub fn make_fixed_points_4d
    <BE: s32, NB: u32, DIM_0: u32, DIM_1: u32, DIM_2: u32, DIM_3: u32>
    (significands: sN[NB][DIM_3][DIM_2][DIM_1][DIM_0])
    -> FixedPoint<NB, BE>[DIM_3][DIM_2][DIM_1][DIM_0] {        
    map(significands, make_fixed_points_3d<BE>)
}

pub fn const_array_1d
    <DIM: u32, NB: u32, BE: s32>
    (value: FixedPoint<NB, BE>)
    -> FixedPoint<NB, BE>[DIM] {
    FixedPoint<NB, BE>[DIM]:[value, ...]
}

pub fn const_array_2d
    <DIM_0: u32, DIM_1: u32, NB: u32, BE: s32>
    (value: FixedPoint<NB, BE>)
    -> FixedPoint<NB, BE>[DIM_1][DIM_0] {
    FixedPoint<NB, BE>[DIM_1][DIM_0]:[const_array_1d<DIM_1>(value), ...]
}

pub fn const_array_3d
    <DIM_0: u32, DIM_1: u32, DIM_2: u32, NB: u32, BE: s32>
    (value: FixedPoint<NB, BE>)
    -> FixedPoint<NB, BE>[DIM_2][DIM_1][DIM_0] {
    FixedPoint<NB, BE>[DIM_2][DIM_1][DIM_0]:[const_array_2d<DIM_1, DIM_2>(value), ...]
}

pub fn const_array_4d
    <DIM_0: u32, DIM_1: u32, DIM_2: u32, DIM_3: u32, NB: u32, BE: s32>
    (value: FixedPoint<NB, BE>)
    -> FixedPoint<NB, BE>[DIM_3][DIM_2][DIM_1][DIM_0] {
    FixedPoint<NB, BE>[DIM_3][DIM_2][DIM_1][DIM_0]:[const_array_3d<DIM_0, DIM_1, DIM_2>(value), ...]
}


// === Compare ===

pub enum Compare: s2 {
    LESS = -1,
    EQUAL = 0,
    GREATER = 1
}

pub fn compare<
    NB_A: u32, BE_A: s32,
    NB_B: u32, BE_B: s32
>(
    a: FixedPoint<NB_A, BE_A>,
    b: FixedPoint<NB_B, BE_B>
) -> Compare {
    let diff = fixed_point::sub(a, b).significand;
    if (diff == 0)
        { Compare::EQUAL }
    else if (std::msb(diff) == u1:1)
        { Compare::LESS }
    else
        { Compare::GREATER }
}

pub fn greater<
    NB_A: u32, BE_A: s32,
    NB_B: u32, BE_B: s32
>(
    a: FixedPoint<NB_A, BE_A>,
    b: FixedPoint<NB_B, BE_B>
) -> bool {
    compare(a, b) as s2 == Compare::GREATER as s2
}

pub fn greater_or_equal<
    NB_A: u32, BE_A: s32,
    NB_B: u32, BE_B: s32
>(
    a: FixedPoint<NB_A, BE_A>,
    b: FixedPoint<NB_B, BE_B>
) -> bool {
    compare(a, b) as s2 >= Compare::EQUAL as s2
}

pub fn less<
    NB_A: u32, BE_A: s32,
    NB_B: u32, BE_B: s32
>(
    a: FixedPoint<NB_A, BE_A>,
    b: FixedPoint<NB_B, BE_B>
) -> bool {
    compare(a, b) as s2 == Compare::LESS as s2
}

pub fn less_or_equal<
    NB_A: u32, BE_A: s32,
    NB_B: u32, BE_B: s32
>(
    a: FixedPoint<NB_A, BE_A>,
    b: FixedPoint<NB_B, BE_B>
) -> bool {
    compare(a, b) as s2 <= Compare::EQUAL as s2
}

pub fn equal<
    NB_A: u32, BE_A: s32,
    NB_B: u32, BE_B: s32
>(
    a: FixedPoint<NB_A, BE_A>,
    b: FixedPoint<NB_B, BE_B>
) -> bool {
    compare(a, b) as s2 == Compare::EQUAL as s2
}

fn check_compare_impl<
    NB_A: u32, BE_A: s32,
    NB_B: u32, BE_B: s32
>(
    a: FixedPoint<NB_A, BE_A>,
    b: FixedPoint<NB_B, BE_B>,
    expected_compare_result: Compare
) {
    let compare_result = compare(a, b);
    assert_eq(compare_result as s2, expected_compare_result as s2);
    
    match expected_compare_result {
        Compare::LESS => {
            assert_eq(less(a,b), true);
            assert_eq(less_or_equal(a,b), true);
            assert_eq(equal(a,b), false);
            assert_eq(greater_or_equal(a,b), false);
            assert_eq(greater(a,b), false);
        },
        Compare::EQUAL => {
            assert_eq(less(a,b), false);
            assert_eq(less_or_equal(a,b), true);
            assert_eq(equal(a,b), true);
            assert_eq(greater_or_equal(a,b), true);
            assert_eq(greater(a,b), false);
        },
        Compare::GREATER => {
                assert_eq(less(a,b), false);
                assert_eq(less_or_equal(a,b), false);
                assert_eq(equal(a,b), false);
                assert_eq(greater_or_equal(a,b), true);
                assert_eq(greater(a,b), true);
        }
    };
}

fn check_compare<
    NB_A: u32, BE_A: s32,
    NB_B: u32, BE_B: s32
>(
    a: FixedPoint<NB_A, BE_A>,
    b: FixedPoint<NB_B, BE_B>,
    expected_compare_result: Compare
) {
    check_compare_impl(a, b, expected_compare_result);
    check_compare_impl(b, a, match expected_compare_result {
        Compare::LESS => Compare::GREATER,
        Compare::EQUAL => Compare::EQUAL,
        Compare::GREATER => Compare::LESS
    });
}

#[test]
fn test_compare() {
    let minus_one = fixed_point::from_integer(s3:-1);
    let zero = fixed_point::from_integer(s3:0);
    let one = fixed_point::from_integer(s3:1);
    let two = fixed_point::from_integer(s3:2);

    let minus_one_big = fixed_point::make_fixed_point<-8>(s16:-256);
    let zero_big = fixed_point::make_fixed_point<-4>(s8:0);
    let one_big = fixed_point::make_fixed_point<-5>(s12:32);    
    let two_big = fixed_point::make_fixed_point<-1>(s12:4);    

    let values = [minus_one, zero, one, two];
    // Cannot make it an array because of different types
    let values_big = (minus_one_big, zero_big, one_big, two_big);

    check_compare(minus_one, minus_one_big, Compare::EQUAL);
    check_compare(minus_one, minus_one_big, Compare::EQUAL);

    for (i, _) in u32:0..4 {
        for (j, _) in u32:0..4 {
            let expected_result = if (i < j) {
                Compare::LESS
            } else if (i == j) {
                Compare::EQUAL
            } else {
                Compare::GREATER
            };
            let a = values[i];
            // values_big[i] or values_big.i does not compile,
            // so we iterate manually
            match j {
                u32:0 => check_compare(a, values_big.0, expected_result),
                u32:1 => check_compare(a, values_big.1, expected_result),
                u32:2 => check_compare(a, values_big.2, expected_result),
                u32:3 => check_compare(a, values_big.3, expected_result),
                _     => fail!("index_out_of_bounds", ())
            }
        }(())
    }(())
}


// === Transpose ===

pub fn transpose
<NB: u32, BE: s32, DIM_0: u32, DIM_1: u32>
(x: FixedPoint<NB, BE>[DIM_1][DIM_0])
-> FixedPoint<NB, BE>[DIM_0][DIM_1] {
    let res = zero!<FixedPoint<NB, BE>[DIM_0][DIM_1]>();
    for (i, res) in 0..DIM_0 {
        for (j, res) in 0..DIM_1 {
            update(res, (j,i), x[i][j])
        }(res)
    }(res)
}

#[test]
fn test_transpose() {
    let x = make_fixed_points_2d<0>([[s16:1, 2, 3], [s16:4, 5, 6]]);
    let x_t = make_fixed_points_2d<0>([[s16:1, 4], [s16:2, 5], [s16:3, 6]]);
    assert_eq(x_t, transpose(x));
    assert_eq(x, transpose(x_t));
}

// Reshape to and from 1D arrays with C-style (row-major) ordering.

pub fn flatten_2d<
    NB: u32, BE: s32,
    DIM_0: u32, DIM_1: u32,
    DIM: u32 = {DIM_0 * DIM_1}
>
(x: FixedPoint<NB, BE>[DIM_1][DIM_0])
-> FixedPoint<NB, BE>[DIM] {
    let res = zero!<FixedPoint<NB, BE>[DIM]>();
    for (i, res) in 0..DIM_0 {
        for (j, res) in 0..DIM_1 {
            update(res, i * DIM_1 + j, x[i][j])
        }(res)
    }(res)
}

pub fn flatten_3d<
    NB: u32, BE: s32,
    DIM_0: u32, DIM_1: u32, DIM_2: u32,
    DIM: u32 = {DIM_0 * DIM_1 * DIM_2}
>(x: FixedPoint<NB, BE>[DIM_2][DIM_1][DIM_0]) 
-> FixedPoint<NB, BE>[DIM] {
    flatten_2d(map(x, flatten_2d))
}

pub fn flatten_4d<
    NB: u32, BE: s32,
    DIM_0: u32, DIM_1: u32, DIM_2: u32, DIM_3: u32,
    DIM: u32 = {DIM_0 * DIM_1 * DIM_2 * DIM_3}
>(x: FixedPoint<NB, BE>[DIM_3][DIM_2][DIM_1][DIM_0])
-> FixedPoint<NB, BE>[DIM] {
    flatten_2d(map(x, flatten_3d))
}

pub fn reshape_to_2d<
    DIM_0: u32, DIM_1: u32,
    NB: u32, BE: s32,
    DIM: u32 = {DIM_0 * DIM_1}>
(x: FixedPoint<NB, BE>[DIM])
-> FixedPoint<NB, BE>[DIM_1][DIM_0] {
    let res = zero!<FixedPoint<NB, BE>[DIM_1][DIM_0]>();
    for (i, res) in 0..DIM_0 {
        for (j, res) in 0..DIM_1 {
            update(res, (i, j), x[i * DIM_1 + j])
        }(res)
    }(res)
}

pub fn reshape_to_3d<
    DIM_0: u32, DIM_1: u32, DIM_2: u32,
    NB: u32, BE: s32,
    DIM: u32 = {DIM_0 * DIM_1 * DIM_2}>
(x: FixedPoint<NB, BE>[DIM])
-> FixedPoint<NB, BE>[DIM_2][DIM_1][DIM_0] {
    let x_2d = reshape_to_2d<DIM_0, {DIM_1 * DIM_2}>(x);
    map(x_2d, reshape_to_2d<DIM_1, DIM_2>)
}

pub fn reshape_to_4d<
    DIM_0: u32, DIM_1: u32, DIM_2: u32, DIM_3: u32,
    NB: u32, BE: s32,
    DIM: u32 = {DIM_0 * DIM_1 * DIM_2 * DIM_3}>
(x: FixedPoint<NB, BE>[DIM])
-> FixedPoint<NB, BE>[DIM_3][DIM_2][DIM_1][DIM_0] {
    let x_2d = reshape_to_2d<DIM_0, {DIM_1 * DIM_2 * DIM_3}>(x);
    map(x_2d, reshape_to_3d<DIM_1, DIM_2, DIM_3>)
}

#[test]
fn test_reshape_2d() {
    let x_flat = make_fixed_points_1d<0>([s16:1, 2, 3, 4, 5, 6]);
    let x = make_fixed_points_2d<0>([[s16:1, 2, 3], [s16:4, 5, 6]]);
    assert_eq(x, reshape_to_2d<2,3>(x_flat));
    assert_eq(x_flat, flatten_2d(x));
}

#[test]
fn test_reshape_3d() {
    let x_flat = make_fixed_points_1d<0>([s16:1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let x = make_fixed_points_3d<0>([[[s16:1, 2], [s16:3, 4], [s16:5, 6]], [[s16:7, 8], [s16:9, 10], [s16:11, 12]]]);
    assert_eq(x, reshape_to_3d<2,3,2>(x_flat));
    assert_eq(x_flat, flatten_3d(x));
}

#[test]
fn test_reshape_4d() {
    let x = make_fixed_points_4d<0>([[[[s16:1, 2], [s16:3, 4], [s16:5, 6]]], [[[s16:7, 8], [s16:9, 10], [s16:11, 12]]]]);
    let x_flat = make_fixed_points_1d<0>([s16:1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    assert_eq(x, reshape_to_4d<2,1,3,2>(x_flat));
    assert_eq(x_flat, flatten_4d(x));
}

// === Convert FixedPoint array to array of significands sN[NB] ===

pub fn to_significand
    <NB: u32, BE: s32>
    (x: FixedPoint<NB, BE>) 
    -> sN[NB] {
    x.significand
}

pub fn to_significand_1d
    <NB: u32, BE: s32, DIM_0: u32>
    (x: FixedPoint<NB, BE>[DIM_0])
    -> sN[NB][DIM_0] {
    map(x, to_significand)
}
pub fn to_significand_2d
    <NB: u32, BE: s32, DIM_0: u32, DIM_1: u32>
    (x: FixedPoint<NB, BE>[DIM_1][DIM_0])
    -> sN[NB][DIM_1][DIM_0] {
    map(x, to_significand_1d)
}
pub fn to_significand_3d
    <NB: u32, BE: s32, DIM_0: u32, DIM_1: u32, DIM_2: u32>
    (x: FixedPoint<NB, BE>[DIM_2][DIM_1][DIM_0])
    -> sN[NB][DIM_2][DIM_1][DIM_0] {
    map(x, to_significand_2d)
}
pub fn to_significand_4d
    <NB: u32, BE: s32, DIM_0: u32, DIM_1: u32, DIM_2: u32, DIM_3: u32>
    (x: FixedPoint<NB, BE>[DIM_3][DIM_2][DIM_1][DIM_0])
    -> sN[NB][DIM_3][DIM_2][DIM_1][DIM_0] {
    map(x, to_significand_3d)
}

// === Change width and exponent ===

fn overflow_truncated<OVERFLOW: OverflowMode, N: u32>(
    // result of truncate_msbs(x) or truncate_lsbs(x)
    truncated: sN[N],
    // Sign of the result (need to pass it because is could be lost during truncation)
    sign: Sign,
    // Did overflow happen during truncation?
    had_overflow: bool
    ) -> sN[N] {
   
    assert!(N != 0, "illegal_zero_width");
    // TODO: this fails due to eager instantiation for N=0
    // let MAX = std::signed_max_value<N>();
    // let MIN = std::signed_max_value<N>();
    let MAX = (std::signed_max_value<{N+2}>() >> 2) as sN[N];
    let MIN = (std::signed_min_value<{N+2}>() >> 2) as sN[N];

    let has_overflow = match OVERFLOW {
        OverflowMode::SAT_SYM => had_overflow || (truncated == MIN),
        _ => had_overflow
    };

    if has_overflow {
        match OVERFLOW {
            OverflowMode::WRAP => {
                truncated
            },
            OverflowMode::SAT => {
                match sign {
                    Sign::NonNegative => MAX,
                    Sign::Negative    => MIN
                }
            },
            OverflowMode::SAT_ZERO => {
                sN[N]:0
            },
            OverflowMode::SAT_SYM => {
                match sign {
                    Sign::NonNegative => MAX,
                    Sign::Negative    => -MAX
                }
            }
        }
    }
    else {
        truncated
    }
}

// Drop (NB_IN - NB_OUT) MSBs and handle overflow
fn truncate_msbs<NB_OUT: u32, OVERFLOW: OverflowMode, NB_IN: u32>
    (x: sN[NB_IN]) -> sN[NB_OUT] {

    // TODO const_assert! fails due to eager instantiation.
    // const_assert!(NB_IN > NB_OUT);
    // let NB_OVERFLOW = NB_IN - NB_OUT;
    assert!(NB_IN > NB_OUT, "truncate_msbs_nothing_to_truncate");
    let NB_OVERFLOW = std::usub_or_zero(NB_IN, NB_OUT);
    
    // TODO: this causes const_assert! in split_lsbs.
    // So we have to introduce NB_SPLIT
    // let (msbs, lsbs) = std::split_lsbs<NB_OUT>(std::to_unsigned(x));
    let NB_SPLIT = std::min(NB_IN, NB_OUT);
    let (_, lsbs) = std::split_lsbs<NB_SPLIT>(std::to_unsigned(x));
    let truncated = std::to_signed(lsbs) as sN[NB_OUT];
    
    // TODO this fails due to eager instantiation for NB_IN = 0 
    // let sign:Sign = std::msb(x) as Sign;
    let sign:Sign = std::msb((x as sN[NB_IN + 1]) << 1) as Sign;
    
    // TODO this fails due to eager instantiation for NB_IN = 0 
    // let NB_SIGN_EXT = NB_OVERFLOW + 1;
    let NB_SIGN_EXT = std::min(NB_OVERFLOW + 1, NB_IN);
    // If there is no overflow, overflow_bits and are either 000..0 or 111..1
    let sign_ext = match sign {
        Sign::NonNegative => zero!<uN[NB_SIGN_EXT]>(),
        Sign::Negative => all_ones!<uN[NB_SIGN_EXT]>()
    };
    // Take all truncated bits and the sign bit
    let (msbs, _) = std::split_msbs<NB_SIGN_EXT>(std::to_unsigned(x));
    
    // NB: overflow also happens when truncated == MIN for OverflowMode::SAT_SYM
    // We handle this inside overflow_truncated()
    let had_overflow = (msbs != sign_ext);
    overflow_truncated<OVERFLOW>(truncated as sN[NB_OUT], sign, had_overflow)
}

fn convert_rounding_mode<rm: RoundingMode>() -> round::RoundingMode {
    match rm {
        RoundingMode::TRN         => round::RoundingMode::RTN,
        RoundingMode::TRN_ZERO    => round::RoundingMode::RTZ,
        // RoundingMode::RND         => TODO,
        // RoundingMode::RND_ZERO    => TODO,
        RoundingMode::RND_INF     => round::RoundingMode::RNA,
        // RoundingMode::RND_MIN_INF => TODO,
        RoundingMode::RND_CONV    => round::RoundingMode::RNE,
        _ => {
            assert_fmt!(false, "unsupported_RoundingMode_{}", (rm as RoundingModeIntegerType));
            round::RoundingMode::RTN
        }
    }
}

// round::round_trunc_s, but with our RoundingMode
fn round_trunc_s<NUM_BITS_ROUNDED: u32, ROUNDING: RoundingMode, N: u32, R: u32 = {N - NUM_BITS_ROUNDED}>
    (unrounded: sN[N]) -> (u1, sN[R]) {
    round::round_trunc_s<NUM_BITS_ROUNDED>(convert_rounding_mode<ROUNDING>(), unrounded)
}

// Drop (NB_IN - NB_OUT) LSBs using RoundingMode, 
// and handle possible overflow (e.g. rounding MAX up) according to OverflowMode.
fn truncate_lsbs<NB_OUT: u32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode, NB_IN: u32>
    (x: sN[NB_IN]) -> sN[NB_OUT] {

    // TODO const_assert! fails due to eager instantiation
    // const_assert!(NB_IN > NB_OUT);
    // let NUM_BITS_ROUNDED = NB_IN - NB_OUT;
    assert!(NB_IN > NB_OUT, "truncate_lsbs_nothing_to_truncate");
    let NUM_BITS_ROUNDED = std::usub_or_zero(NB_IN, NB_OUT);

    let (had_overflow, truncated) = round_trunc_s<NUM_BITS_ROUNDED, ROUNDING>(x);
    let sign = std::msb(x) as Sign;
    overflow_truncated<OVERFLOW>(truncated as sN[NB_OUT], sign, had_overflow)
}

// FixedPoint<NB, BE> ~ ac_fixed<NB, NB + BE> 
// ~ significand * 2^BE
// 0b00111.001 ~ FixedPoint<8,-3>
pub fn resize<
    NB_OUT: u32, BE_OUT: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    NB_IN: u32, BE_IN: s32>
    (x: FixedPoint<NB_IN, BE_IN>)
    -> FixedPoint<NB_OUT, BE_OUT>{

    let SHIFT: s32 = BE_IN - BE_OUT;

    let NB_ALIGNED = if (SHIFT >= s32:0) {
        NB_IN + std::to_unsigned(SHIFT)
    }
    else {
        std::usub_or_zero(NB_IN, std::to_unsigned(-SHIFT))
    };

    // Align exponent
    let aligned : sN[NB_ALIGNED] =
        if (SHIFT >= s32:0) {
            (x.significand as sN[NB_ALIGNED]) << std::to_unsigned(SHIFT)
        } else if (NB_ALIGNED == 0) {
            // TODO: move this case inside truncate_lsbs?
            zero!<sN[NB_ALIGNED]>()
        } else {
            truncate_lsbs<NB_ALIGNED, ROUNDING, OVERFLOW>(x.significand)
        };
    
    // Resize width
    let resized = if (NB_OUT < NB_ALIGNED) {
        truncate_msbs<NB_OUT, OVERFLOW>(aligned)
    } else if (NB_OUT == NB_ALIGNED){
        // Here overflow_truncated() will change the result on in SAT_SYM mode, if aligned == MIN.
        let sign = std::msb(aligned as sN[NB_OUT]) as Sign;
        let had_overflow = false;
        overflow_truncated<OVERFLOW>(aligned as sN[NB_OUT], sign, had_overflow)
    } else {
        aligned as sN[NB_OUT]
    };

    FixedPoint<NB_OUT, BE_OUT>{ significand: resized }
}

pub fn resize_1d<
    NB_OUT: u32, BE_OUT: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    NB_IN: u32, BE_IN: s32,
    DIM: u32
    >
(x: FixedPoint<NB_IN, BE_IN>[DIM])
-> FixedPoint<NB_OUT, BE_OUT>[DIM] {
    map(x, resize<NB_OUT, BE_OUT, ROUNDING, OVERFLOW>)
}

pub fn resize_2d<
    NB_OUT: u32, BE_OUT: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    NB_IN: u32, BE_IN: s32,
    DIM_0: u32, DIM_1: u32
    >
(x: FixedPoint<NB_IN, BE_IN>[DIM_1][DIM_0])
-> FixedPoint<NB_OUT, BE_OUT>[DIM_1][DIM_0] {
    map(x, resize_1d<NB_OUT, BE_OUT, ROUNDING, OVERFLOW>)
}

pub fn resize_3d<
    NB_OUT: u32, BE_OUT: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    NB_IN: u32, BE_IN: s32,
    DIM_0: u32, DIM_1: u32, DIM_2: u32
    >
(x: FixedPoint<NB_IN, BE_IN>[DIM_2][DIM_1][DIM_0])
-> FixedPoint<NB_OUT, BE_OUT>[DIM_2][DIM_1][DIM_0] {
    map(x, resize_2d<NB_OUT, BE_OUT, ROUNDING, OVERFLOW>)
}

pub fn resize_4d<
    NB_OUT: u32, BE_OUT: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    NB_IN: u32, BE_IN: s32,
    DIM_0: u32, DIM_1: u32, DIM_2: u32, DIM_3: u32
    >
(x: FixedPoint<NB_IN, BE_IN>[DIM_3][DIM_2][DIM_1][DIM_0])
-> FixedPoint<NB_OUT, BE_OUT>[DIM_3][DIM_2][DIM_1][DIM_0] {
    map(x, resize_3d<NB_OUT, BE_OUT, ROUNDING, OVERFLOW>)
}


fn resize_test_case<
    ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    NB_IN: u32, BE_IN: s32,
    NB_OUT: u32, BE_OUT: s32>
    (input: FixedPoint<NB_IN, BE_IN>, expected_output: FixedPoint<NB_OUT, BE_OUT>) {

    let output = resize<NB_OUT, BE_OUT, ROUNDING, OVERFLOW>(input);
    assert_eq(output, expected_output);
}

#[test]
fn test_resize() {
    let R = RoundingMode::TRN;
    let O = OverflowMode::WRAP;
    resize_test_case<R, O>(
        fixed_point::make_fixed_point<0>(s2:1),
        fixed_point::make_fixed_point<-2>(s4:1 << 2)
    );
    resize_test_case<R, O>(
        fixed_point::make_fixed_point<0>(s2:-1),
        fixed_point::make_fixed_point<-2>(s4:-1 << 2)
    );
}

#[test]
fn test_resize_more() {
    let R = RoundingMode::TRN;
    let O = OverflowMode::WRAP;

    // widen width only (sign extension)
    resize_test_case<R,O>(
        fixed_point::make_fixed_point<0>(s2:1),
        fixed_point::make_fixed_point<0>(s4:1)
    );

    resize_test_case<R,O>(
        fixed_point::make_fixed_point<0>(s2:-1),
        fixed_point::make_fixed_point<0>(s4:-1)
    );

    // exponent decrease (SHIFT > 0) → left shift
    resize_test_case<R,O>(
        fixed_point::make_fixed_point<0>(s3:1),
        fixed_point::make_fixed_point<-2>(s5:1 << 2)
    );

    resize_test_case<R,O>(
        fixed_point::make_fixed_point<0>(s3:-2),
        fixed_point::make_fixed_point<-2>(s5:-2 << 2)
    );

    // exponent increase (SHIFT < 0) → truncate LSBs
    resize_test_case<R,O>(
        fixed_point::make_fixed_point<-2>(s4:0b0110), // 1.5
        fixed_point::make_fixed_point<0>(s2:1)
    );

    resize_test_case<R,O>(
        fixed_point::make_fixed_point<-2>(s4:0b1010), // -1.5
        fixed_point::make_fixed_point<0>(s2:-2)
    );

    // full LSB truncation (NB_ALIGNED = 0)
    resize_test_case<R,O>(
        fixed_point::make_fixed_point<-1>(s3:3),
        fixed_point::make_fixed_point<3>(s4:0)
    );

    resize_test_case<R,O>(
        fixed_point::make_fixed_point<-1>(s3:-3),
        fixed_point::make_fixed_point<3>(s4:0)
    );

    // MSB truncation (wrap)
    resize_test_case<R,O>(
        fixed_point::make_fixed_point<0>(s5:0b10110),
        fixed_point::make_fixed_point<0>(s3:0b110)
    );

    resize_test_case<R,O>(
        fixed_point::make_fixed_point<0>(s5:-7),
        fixed_point::make_fixed_point<0>(s3:1)
    );
}

fn resize_overflow_test_case<
    OVERFLOW: OverflowMode,
    NB_IN: u32,
    NB_OUT: u32
>(
    x: sN[NB_IN],
    expected: sN[NB_OUT]
) {
    resize_test_case<RoundingMode::TRN, OVERFLOW>(
        fixed_point::make_fixed_point<0>(x),
        fixed_point::make_fixed_point<0>(expected)
    );
}

#[test]
fn test_resize_overflow_modes() {
    // WRAP
    resize_overflow_test_case<OverflowMode::WRAP>(s5:15, s3:-1);
    resize_overflow_test_case<OverflowMode::WRAP>(s5:8, s3:0);
    // SAT
    resize_overflow_test_case<OverflowMode::SAT>(s5:15, s4:7);
    resize_overflow_test_case<OverflowMode::SAT>(s5:15, s3:3);
    resize_overflow_test_case<OverflowMode::SAT>(s5:-16, s4:-8);
    resize_overflow_test_case<OverflowMode::SAT>(s5:-16, s3:-4);
    // SAT_ZERO
    resize_overflow_test_case<OverflowMode::SAT_ZERO>(s5:15,s3:0);
    resize_overflow_test_case<OverflowMode::SAT_ZERO>(s5:-15,s3:0);
    resize_overflow_test_case<OverflowMode::SAT_ZERO>(s5:-9,s3:0);
    // SAT_SYM
    resize_overflow_test_case<OverflowMode::SAT_SYM>(s5:-16, s3:-3);
    resize_overflow_test_case<OverflowMode::SAT_SYM>(s5:-16, s5:-15);
    resize_overflow_test_case<OverflowMode::SAT_SYM>(s5:15, s5:15);
}


// === Queries ===


pub fn max<NB: u32, BE: s32>
    (x: FixedPoint<NB, BE>, y: FixedPoint<NB, BE>) -> FixedPoint<NB, BE> {
    fixed_point::make_fixed_point<BE>(std::max(x.significand, y.significand))
}

pub fn max_1d
    <NB: u32, BE: s32, DIM: u32>
    (xs: FixedPoint<NB, BE>[DIM])
    -> FixedPoint<NB, BE> {
    // We could do 1..DIM, but compilation fails for empty range
    let max_significand = for (i, acc) in 0..DIM {
        std::max(acc, xs[i].significand)
    }(xs[0].significand);
    fixed_point::make_fixed_point<BE>(max_significand)
}


// === Clip ===

pub fn clip<NB: u32, BE: s32>(
    x: FixedPoint<NB, BE>,
    min_value: FixedPoint<NB, BE>,
    max_value: FixedPoint<NB, BE>
    ) -> FixedPoint<NB, BE> {
    
    if (fixed_point::sub(x, min_value).significand < 0)
        { min_value }
    else if (fixed_point::sub(x, max_value).significand > 0)
        { max_value }
    else
        { x }
}

pub fn clip_resize<
    NB_OUT: u32, BE_OUT: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    NB_IN: u32, BE_IN: s32,
    NB_MIN: u32, BE_MIN: s32,
    NB_MAX: u32, BE_MAX: s32>(
        x: FixedPoint<NB_IN, BE_IN>,
        min_value: FixedPoint<NB_MIN, BE_MIN>,
        max_value: FixedPoint<NB_MAX, BE_MAX>
    ) -> FixedPoint<NB_OUT, BE_OUT> {
    
    if (fixed_point::sub(x, min_value).significand < 0)
        { resize<NB_OUT, BE_OUT, ROUNDING, OVERFLOW>(min_value) }
    else if (fixed_point::sub(x, max_value).significand > 0)
        { resize<NB_OUT, BE_OUT, ROUNDING, OVERFLOW>(max_value) }
    else
        { resize<NB_OUT, BE_OUT, ROUNDING, OVERFLOW>(x) }
}

// === Arithmetic operations ===

// Compute -x
// Adds one extra bit to avoid overflow when x = -2^(NB-1)
pub fn negate<
    NB_IN: u32, BE_IN: s32,
    NB_OUT: u32 = {NB_IN + 1}, BE_OUT: s32 = {BE_IN}
>
(x: FixedPoint<NB_IN, BE_IN>)
-> FixedPoint<NB_OUT, BE_OUT> {
    let xx = x.significand as sN[NB_OUT];
    FixedPoint<NB_OUT, BE_OUT>{ significand: -xx }
}

// Negate without adding extra bit
pub fn negate_with_overflow<
    OVERFLOW: OverflowMode,
    NB: u32, BE: s32
>
(x: FixedPoint<NB, BE>)
-> FixedPoint<NB, BE> {
    let minus_x = negate(x);
    let significand = truncate_msbs<NB, OVERFLOW>(minus_x.significand);
    fixed_point::make_fixed_point<BE>(significand)
}

fn negate_test_case<NB: u32, BE: s32, OVERFLOW: OverflowMode>() {
    let NB_OUT = NB + 1;

    let MIN = std::signed_min_value<NB>();
    let MAX = std::signed_max_value<NB>();

    let ROUNDING = RoundingMode::TRN;
    for (i, _) in MIN..MAX {
        let x = fixed_point::make_fixed_point<BE>(i);
        let expected = fixed_point::make_fixed_point<BE>(-(i as sN[NB_OUT]));
        let expected_with_overflow = resize<NB, BE, ROUNDING, OVERFLOW>(expected);
        assert_eq(expected, negate(x));
        assert_eq(expected_with_overflow, negate_with_overflow<OVERFLOW>(x));
    }(());
}

#[test]
fn test_negate() {
    negate_test_case<3, 0, OverflowMode::WRAP>();
    negate_test_case<3, 0, OverflowMode::SAT>();
    negate_test_case<3, 0, OverflowMode::SAT_ZERO>();
    negate_test_case<3, 0, OverflowMode::SAT_SYM>();
}


// Performs an add assuming that the rhs is already wide enough to not overflow.
// WARNING: rhs must be wide enough to avoid any overflow
pub fn add_already_widened
    <NB_A: u32, BE_A: s32, NB_B: u32, BE_B: s32>
    (fxd_a: FixedPoint<NB_A, BE_A>, fxd_b: FixedPoint<NB_B, BE_B>)
    -> FixedPoint<NB_B, BE_B> {
    // Widen before left shifting to avoid overflow
    let aligned_lhs = (fxd_a.significand as sN[NB_B]) << (BE_A - BE_B) as u32;
    // TODO: I think this is also always the same in the dot product use case. Fraction bits stay
    // the same
    let aligned_rhs = fxd_b.significand;

    fixed_point::make_fixed_point<BE_B>(aligned_lhs + aligned_rhs)
}

// Performs an subtraction assuming that the rhs is already wide enough to not overflow.
// WARNING: rhs must be wide enough to avoid any overflow
pub fn sub_already_widened
    <NB_A: u32, BE_A: u32, NB_B: u32, BE_B: u32>
    (fxd_a: FixedPoint<NB_A, BE_A>, fxd_b: FixedPoint<NB_B, BE_B>)
    -> FixedPoint<NB_B, BE_B> {
    // Widen before left shifting to avoid overflow
    let aligned_lhs = (fxd_a.significand as sN[NB_B]) << (BE_A - BE_B) as u32;
    let aligned_rhs = fxd_b.significand;

    fixed_point::make_fixed_point<BE_B>(aligned_lhs - aligned_rhs)
}

// Performs an fused-multiply-add assuming that the rhs is already wide enough to not overflow.
// WARNING: the add rhs must be wide enough to avoid any overflow
pub fn fmadd_already_widened
    <NB_A: u32, BE_A: s32, NB_B: u32, BE_B: s32, NB_C: u32, BE_C: s32,
     NB_MUL: u32 = {NB_A + NB_B}, BE_MUL: s32 = {BE_A + BE_B}>
    (fxd_a: FixedPoint<NB_A, BE_A>,
    fxd_b: FixedPoint<NB_B, BE_B>,
    fxd_c: FixedPoint<NB_C, BE_C>)
    -> FixedPoint<NB_C, BE_C> {
    let prod = fixed_point::mul<NB_A, BE_A, NB_B, BE_B>(fxd_a, fxd_b);
    add_already_widened<NB_MUL, BE_MUL, NB_C, BE_C>(prod, fxd_c)
}

// Performs a dot product on 2 vectors. To implement this, the final widened result is
// computed before. An accumulator is instantiated with this final size and the fmadd operation
// is reimplemented in such a way as to not widen the output when summing in the accumulator.
//
// TYPE EXPLANATIONS:
// number bits: a multiplication assumes to always double the number of bits. 
//      Since our vectors must be of the same type
//      (each elem. within each vector follow the same fixed point representation) 
//      we know the size of all elem. wise multiplications.
//      We can also guarantee that all elements will have overlapping positions 
//      (again because elems. within vectors have the same type). This means that we must
//      widen by one bit for each element of the vector minus one. Minus one because we performs VEC_SZ - 1 adds.
// binary exponent: The binary exponent will never change with additions since
//      all elem-wise multiplication will result in the same exponent.
// exp is negative: inferred from 'binary exponent'
// unsigned exp:    inferred from 'binary exponent'
// WARNINGS: 
// 1. made aligned_width() and num_bits_overlapping() public in a copy of the fixed_point_lib module.
// to write the type inference
// 2. We use ''already_widened'' functions.
pub fn dot_prod
    <NB_X: u32, BE_X: s32,
    NB_Y: u32, BE_Y: s32,
    VEC_SZ: u32,
    // Precision inference MUL
    NB_MUL: u32 = {NB_X + NB_Y},
    BE_MUL: s32 = {BE_X + BE_Y},
    // Precision Inference DOT PROD
    NB_DOT_PROD: u32 = {NB_MUL + std::clog2(VEC_SZ)},
    BE_DOT_PROD: s32 = {BE_MUL}>
    (x: FixedPoint<NB_X, BE_X>[VEC_SZ], 
     y: FixedPoint<NB_Y, BE_Y>[VEC_SZ])
    -> FixedPoint<NB_DOT_PROD, BE_DOT_PROD> {

    for (i, acc) in 0..VEC_SZ {
        fmadd_already_widened(x[i], y[i], acc)
    }(zero!<FixedPoint<NB_DOT_PROD, BE_DOT_PROD>>())
}

// TODO
// #[test]
// fn fadd_test() {
//     let a = sN[u32:16]:1024; // 1.0
//     let b = sN[u32:16]:1024; // 1.0
//     let c = sN[u32:16]:1024; // 1.0

//     let result = fmadd<u32:16, u32:1, u32:10, u32:16, u32:1, u32:10, u32:16, u32:1, u32:10>(a, b, c);
//     // Solve: x * 2^(-20) = 2 (x must fit in 33 bits)
//     let expected = sN[u32:33]:2097152; // 2.0
//     assert_eq(expected, result);
// }


type FP = FixedPoint<16, -10>;

#[test]
fn dot_prod_test() {
    // [1.5, 1.5]
    let x = make_fixed_points_1d<-10>(sN[16][2]:[1536, ...]); 
    // [2.25, 2.25]
    let y = make_fixed_points_1d<-10>(sN[16][2]:[2304, ...]);
    // 6.75
    let expected = fixed_point::make_fixed_point<-20>(sN[33]:7077888); 
    assert_eq(expected, dot_prod(x, y));

    // [1.0, 1.0, 1.0]
    let x = make_fixed_points_1d<-10>(sN[16][3]:[1024, ...]); 
    // [1.0, 1.0, 1.0]
    let y = make_fixed_points_1d<-10>(sN[16][3]:[1024, ...]);
    // 3.0
    let expected = fixed_point::make_fixed_point<-20>(sN[34]:3145728); 
    assert_eq(expected, dot_prod(x, y));
}
