
import std;

import ap_types.fixed_point_lib;

// ================================================================
// ----------------------- Fixed Point Lib ------------------------

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
    <COMMON_NUM_BITS: u32, COMMON_BINARY_UEXPONENT: u32, 
    NUM_BITS: u32, EXPONENT_IS_NEGATIVE: u32, BINARY_UEXPONENT: u32>
    (x: sN[NUM_BITS])
    -> sN[COMMON_NUM_BITS] {

    let x_exp = fixed_point_lib::binary_exponent(EXPONENT_IS_NEGATIVE, BINARY_UEXPONENT);
    let result_exp = fixed_point_lib::binary_exponent(EXPONENT_IS_NEGATIVE, COMMON_BINARY_UEXPONENT);
    let significand = if result_exp > x_exp {
        // If the exponent is increasing, then the significand needs to decrease.
        // let expr = (x.significand as sN[COMMON_NUM_BITS]) >> (result_exp - x_exp) as u32;
        // fail!("you_are_losing_information_is_this_really_what_you_want", expr)
        // BUGFIX+ENABLE: Andrei
        let expr = (x >> (result_exp - x_exp) as u32) as sN[COMMON_NUM_BITS];
        expr
    } else {
        // If the exponent is decreasing, then the significand needs to increase.
        (x as sN[COMMON_NUM_BITS]) << (x_exp - result_exp) as u32
    };
    significand
}

pub fn mul
    <NB_A: u32, EN_A: u32, BU_A: u32, 
    NB_B: u32, EN_B: u32, BU_B: u32,
    EXP_SUM: s32 = {fixed_point_lib::binary_exponent(EN_A, BU_A) + fixed_point_lib::binary_exponent(EN_B, BU_B)},
    NB_R: u32 = {NB_A + NB_B}, 
    EN_R: u32 = {fixed_point_lib::is_negative(EXP_SUM)},
    BU_R: u32 = {fixed_point_lib::binary_uexponent(EXP_SUM)}>
    (fxd_a: sN[NB_A], 
    fxd_b: sN[NB_B])
    -> sN[NB_R] {

    std::smul(fxd_a, fxd_b)
}

pub fn add
    <NB_A: u32, EN_A: u32, BU_A: u32, 
    NB_B: u32, EN_B: u32, BU_B: u32,
    BE_A: s32 = {fixed_point_lib::binary_exponent(EN_A, BU_A)}, 
    BE_B: s32 = {fixed_point_lib::binary_exponent(EN_B, BU_B)},
    NB_R:
    u32 = {
        fixed_point_lib::aligned_width(NB_A, BE_A, NB_B, BE_B) +
        if fixed_point_lib::num_bits_overlapping(NB_A, BE_A, NB_B, BE_B) == u32:0 { u32:0 } else { u32:1 }},
    BE_R: s32 = {std::min(BE_A, BE_B)}, 
    EN_R: u32 = {fixed_point_lib::is_negative(BE_R)},
    BU_R: u32 = {fixed_point_lib::binary_uexponent(BE_R)}>
    (fxd_a: sN[NB_A], 
    fxd_b: sN[NB_B])
    -> sN[NB_R] {
    // Widen before left shifting to avoid overflow
    let aligned_lhs = (fxd_a as sN[NB_R]) << (BE_A - BE_R) as u32;
    let aligned_rhs = (fxd_b as sN[NB_R]) << (BE_B - BE_R) as u32;

    aligned_lhs + aligned_rhs
}

// Subtracts two unsigned fixed point numbers, returns lhs - rhs
pub fn sub
    <NB_A: u32, EN_A: u32, BU_A: u32, 
    NB_B: u32, EN_B: u32, BU_B: u32,
    BE_A: s32 = {fixed_point_lib::binary_exponent(EN_A, BU_A)}, 
    BE_B: s32 = {fixed_point_lib::binary_exponent(EN_B, BU_B)},
    NB_R:
    u32 = {
        fixed_point_lib::aligned_width(NB_A, BE_A, NB_B, BE_B) +
        if fixed_point_lib::num_bits_overlapping(NB_A, BE_A, NB_B, BE_B) == u32:0 { u32:0 } else { u32:1 }},
    BE_R: s32 = {std::min(BE_A, BE_B)}, EN_R: u32 = {fixed_point_lib::is_negative(BE_R)},
    BU_R: u32 = {fixed_point_lib::binary_uexponent(BE_R)}>
    (fxd_a: sN[NB_A], 
    fxd_b: sN[NB_B])
    -> sN[NB_R] {
    // Widen before left shifting to avoid overflow
    let aligned_lhs = (fxd_a as sN[NB_R]) << (BE_A - BE_R) as u32;
    let aligned_rhs = (fxd_b as sN[NB_R]) << (BE_B - BE_R) as u32;

    aligned_lhs - aligned_rhs
}


// Fused-multiply-add. To infer the final precision, we chain the precision calculation as a multiplication
// followed by an add.
pub fn fmadd
    <NB_A: u32, EN_A: u32, BU_A: u32, 
    NB_B: u32, EN_B: u32, BU_B: u32,
    NB_C: u32, EN_C: u32, BU_C: u32,
    BE_A: s32 = {fixed_point_lib::binary_exponent(EN_A, BU_A)}, // binary exp A
    BE_B: s32 = {fixed_point_lib::binary_exponent(EN_B, BU_B)}, // binary exp B
    BE_C: s32 = {fixed_point_lib::binary_exponent(EN_C, BU_C)}, // binary exp C
    // Precision inference MUL
    BE_MUL: s32 = {BE_A + BE_B},                           // binary exp MUL
    NB_MUL: u32 = {NB_A + NB_B},                           // number bits MUL
    EN_MUL: u32 = {fixed_point_lib::is_negative(BE_MUL)},      // exp is negative MUL
    BU_MUL: u32 = {fixed_point_lib::binary_uexponent(BE_MUL)}, // unsigned exp MUL
    // Precision Inference ADD
    NB_SUM: u32 = {                                 // number bits ADD
        fixed_point_lib::aligned_width(NB_MUL, BE_MUL, NB_C, BE_C) +
        if fixed_point_lib::num_bits_overlapping(NB_MUL, BE_MUL, NB_C, BE_C) == u32:0 { u32:0 } else { u32:1 }
        },
    BE_SUM: s32 = {std::min(BE_MUL, BE_C)},                      // binary exp ADD
    EN_SUM: u32 = {fixed_point_lib::is_negative(BE_SUM)},            // exp is negative ADD
    BU_SUM: u32 = {fixed_point_lib::binary_uexponent(BE_SUM)}>       // unsigned exp ADD
    (fxd_a: sN[NB_A], 
    fxd_b: sN[NB_B], 
    fxd_c: sN[NB_C])
    -> sN[NB_SUM] {

    let prod = mul<NB_A, EN_A, BU_A, NB_B, EN_B, BU_B>(fxd_a, fxd_b);
    add<NB_MUL, EN_MUL, BU_MUL, NB_C, EN_C, BU_C>(prod, fxd_c)
}

// Performs an add assuming that the rhs is already wide enough to not overflow.
// WARNING: rhs must be wide enough to avoid any overflow
pub fn add_already_widened
    <NB_A: u32, EN_A: u32, BU_A: u32, 
    NB_B: u32, EN_B: u32, BU_B: u32,
    BE_A: s32 = {fixed_point_lib::binary_exponent(EN_A, BU_A)}, 
    BE_B: s32 = {fixed_point_lib::binary_exponent(EN_B, BU_B)}>
    (fxd_a: sN[NB_A], fxd_b: sN[NB_B])
    -> sN[NB_B] {
    // Widen before left shifting to avoid overflow
    let aligned_lhs = (fxd_a as sN[NB_B]) << (BE_A - BE_B) as u32; // TODO: I think this is also always the same in the dot product use case. Fraction bits stay the same
    let aligned_rhs = fxd_b;

    aligned_lhs + aligned_rhs
}

// Performs an subtraction assuming that the rhs is already wide enough to not overflow.
// WARNING: rhs must be wide enough to avoid any overflow
pub fn sub_already_widened
    <NB_A: u32, EN_A: u32, BU_A: u32, 
    NB_B: u32, EN_B: u32, BU_B: u32,
    BE_A: s32 = {fixed_point_lib::binary_exponent(EN_A, BU_A)}, 
    BE_B: s32 = {fixed_point_lib::binary_exponent(EN_B, BU_B)}>
    (fxd_a: sN[NB_A], fxd_b: sN[NB_B])
    -> sN[NB_B] {
    // Widen before left shifting to avoid overflow
    let aligned_lhs = (fxd_a as sN[NB_B]) << (BE_A - BE_B) as u32;
    let aligned_rhs = fxd_b;

    aligned_lhs - aligned_rhs
}

// Performs an fused-multiply-add assuming that the rhs is already wide enough to not overflow.
// WARNING: the add rhs must be wide enough to avoid any overflow
pub fn fmadd_already_widened
    <NB_A: u32, EN_A: u32, BU_A: u32, 
    NB_B: u32, EN_B: u32, BU_B: u32,
    NB_C: u32, EN_C: u32, BU_C: u32,
    BE_A: s32 = {fixed_point_lib::binary_exponent(EN_A, BU_A)}, // binary exp A
    BE_B: s32 = {fixed_point_lib::binary_exponent(EN_B, BU_B)}, // binary exp B
    // Precision inference MUL
    BE_MUL: s32 = {BE_A + BE_B},                           // binary exp MUL
    NB_MUL: u32 = {NB_A + NB_B},                           // number bits MUL
    EN_MUL: u32 = {fixed_point_lib::is_negative(BE_MUL)},      // exp is negative MUL
    BU_MUL: u32 = {fixed_point_lib::binary_uexponent(BE_MUL)}> // unsigned exp MUL>
    (fxd_a: sN[NB_A], 
    fxd_b: sN[NB_B], 
    fxd_c: sN[NB_C])
    -> sN[NB_C] {

    let prod = mul<NB_A, EN_A, BU_A, NB_B, EN_B, BU_B>(fxd_a, fxd_b);
    add_already_widened<NB_MUL, EN_MUL, BU_MUL, NB_C, EN_C, BU_C>(prod, fxd_c)
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
    <NB_X: u32, EN_X: u32, BU_X: u32, 
    NB_Y: u32, EN_Y: u32, BU_Y: u32,
    VEC_SZ: u32,
    BE_X: s32 = {fixed_point_lib::binary_exponent(EN_X, BU_X)}, // binary exp X
    BE_Y: s32 = {fixed_point_lib::binary_exponent(EN_Y, BU_Y)}, // binary exp Y
    // Precision inference MUL
    BE_MUL: s32 = {BE_X + BE_Y},                           // binary exp MUL
    NB_MUL: u32 = {NB_X + NB_Y},                           // number bits MUL
    EN_MUL: u32 = {fixed_point_lib::is_negative(BE_MUL)},      // exp is negative MUL
    BU_MUL: u32 = {fixed_point_lib::binary_uexponent(BE_MUL)}, // unsigned exp MUL
    // Precision Inference DOT PROD
    NB_DOT_PROD: u32 = {NB_MUL + std::clog2(VEC_SZ)},                 // number bits DOT PROD
    BE_DOT_PROD: s32 = {BE_MUL},                                     // binary exp DOT PROD
    EN_DOT_PROD: u32 = {fixed_point_lib::is_negative(BE_DOT_PROD)},      // exp is negative DOT PROD
    BU_DOT_PROD: u32 = {fixed_point_lib::binary_uexponent(BE_DOT_PROD)}> // unsigned exp DOT PROD
    (x: sN[NB_X][VEC_SZ], 
     y: sN[NB_Y][VEC_SZ])
    -> sN[NB_DOT_PROD] {

    for (i, acc): (u32, sN[NB_DOT_PROD]) in u32:0..VEC_SZ {
        fmadd_already_widened<NB_X, EN_X, BU_X, NB_Y, EN_Y, BU_Y, NB_DOT_PROD, EN_DOT_PROD, BU_DOT_PROD>(x[i], y[i], acc)
    }(sN[NB_DOT_PROD]:0)
}


#[test]
fn fadd_test() {
    let a = sN[u32:16]:1024; // 1.0
    let b = sN[u32:16]:1024; // 1.0
    let c = sN[u32:16]:1024; // 1.0

    let result = fmadd<u32:16, u32:1, u32:10, u32:16, u32:1, u32:10, u32:16, u32:1, u32:10>(a, b, c);
    // Solve: x * 2^(-20) = 2 (x must fit in 33 bits)
    let expected = sN[u32:33]:2097152; // 2.0
    assert_eq(expected, result);
}

#[test]
fn dot_prod_test() {
    // [1.5, 1.5]
    let x = sN[u32:16][2]:[sN[u32:16]:1536, ...]; 
    // [2.25, 2.25]
    let y = sN[u32:16][2]:[sN[u32:16]:2304, ...];
    // 6.75
    let expected = sN[u32:33]:7077888; 
    assert_eq(expected, dot_prod<u32:16, u32:1, u32:10, u32:16, u32:1, u32:10>(x, y));

    // [1.0, 1.0, 1.0]
    let x = sN[u32:16][3]:[sN[u32:16]:1024, ...]; 
    // [1.0, 1.0, 1.0]
    let y = sN[u32:16][3]:[sN[u32:16]:1024, ...];
    // 3.0
    let expected = sN[u32:34]:3145728; 
    assert_eq(expected, dot_prod<u32:16, u32:1, u32:10, u32:16, u32:1, u32:10>(x, y));
}