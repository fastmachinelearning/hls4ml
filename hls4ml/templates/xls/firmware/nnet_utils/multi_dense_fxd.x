import std;

import ap_types.fixed_point;
type FixedPoint = fixed_point::FixedPoint;

// DEFAULT: (16, 6) => <NUM_BITS: 16, EXPONENT_IS_NEGATIVE: 1, BINARY_UEXPONENT: 10 >
// E.g. make_fixed_point<s32:-2>(s6:31) = 31 * 2^-2 = 7.75

const NB_COMMON = u32:16;
const EN_COMMON = u32:1;
const BU_COMMON = u32:10;
const BE_COMMON = s32:-10;

pub const FXP_6_75_NEG = sN[NB_COMMON]:-6912;
pub const FXP_4_0_NEG  = sN[NB_COMMON]:-4096;
pub const FXP_3_0_NEG  = sN[NB_COMMON]:-3072;
pub const FXP_0_0      = sN[NB_COMMON]:0;
pub const FXP_0_5      = sN[NB_COMMON]:512;
pub const FXP_1_0      = sN[NB_COMMON]:1024;
pub const FXP_1_5      = sN[NB_COMMON]:1536;
pub const FXP_2_0      = sN[NB_COMMON]:2048;
pub const FXP_2_25     = sN[NB_COMMON]:2304;
pub const FXP_4_5      = sN[NB_COMMON]:4608;
pub const FXP_5_5      = sN[NB_COMMON]:5632;
pub const FXP_6_75     = sN[NB_COMMON]:6912;
pub const FXP_12_0     = sN[NB_COMMON]:12288;
pub const FXP_13_5     = sN[NB_COMMON]:13824;

pub type CommonFxdPoint = FixedPoint<NB_COMMON, EN_COMMON, BU_COMMON>;

// let w0 = fixed_point::mul(x[0], y[0]);
// let w1 = fixed_point::mul(x[1], y[1]);
// let w2 = fixed_point::add(w0, w1);
// let w3 = fixed_point::to_common_type<u32:16, u32:10>(w2);


// =============================================================================
// ----------------------- Required Fixed Point Changes ------------------------

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
fn to_common_type
    <COMMON_NUM_BITS: u32, COMMON_BINARY_UEXPONENT: u32, 
    NUM_BITS: u32, EXPONENT_IS_NEGATIVE: u32, BINARY_UEXPONENT: u32>
    (x: sN[NUM_BITS])
    -> sN[COMMON_NUM_BITS] {

    let x_exp = fixed_point::binary_exponent(EXPONENT_IS_NEGATIVE, BINARY_UEXPONENT);
    let result_exp = fixed_point::binary_exponent(EXPONENT_IS_NEGATIVE, COMMON_BINARY_UEXPONENT);
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
    EXP_SUM: s32 = {fixed_point::binary_exponent(EN_A, BU_A) + fixed_point::binary_exponent(EN_B, BU_B)},
    NB_R: u32 = {NB_A + NB_B}, 
    EN_R: u32 = {fixed_point::is_negative(EXP_SUM)},
    BU_R: u32 = {fixed_point::binary_uexponent(EXP_SUM)}>
    (fxd_a: sN[NB_A], 
    fxd_b: sN[NB_B])
    -> sN[NB_R] {

    std::smul(fxd_a, fxd_b)
}

pub fn add
    <NB_A: u32, EN_A: u32, BU_A: u32, 
    NB_B: u32, EN_B: u32, BU_B: u32,
    BE_A: s32 = {fixed_point::binary_exponent(EN_A, BU_A)}, 
    BE_B: s32 = {fixed_point::binary_exponent(EN_B, BU_B)},
    NB_R:
    u32 = {
        fixed_point::aligned_width(NB_A, BE_A, NB_B, BE_B) +
        if fixed_point::num_bits_overlapping(NB_A, BE_A, NB_B, BE_B) == u32:0 { u32:0 } else { u32:1 }},
    BE_R: s32 = {std::min(BE_A, BE_B)}, 
    EN_R: u32 = {fixed_point::is_negative(BE_R)},
    BU_R: u32 = {fixed_point::binary_uexponent(BE_R)}>
    (fxd_a: sN[NB_A], 
    fxd_b: sN[NB_B])
    -> sN[NB_R] {
    // Widen before left shifting to avoid overflow
    let aligned_lhs = (fxd_a as sN[NB_R]) << (BE_A - BE_R) as u32;
    let aligned_rhs = (fxd_b as sN[NB_R]) << (BE_B - BE_R) as u32;

    aligned_lhs + aligned_rhs
}


// Fused-multiply-add. To infer the final precision, we chain the precision calculation as a multiplication
// followed by an add.
fn fmadd
    <NB_A: u32, EN_A: u32, BU_A: u32, 
    NB_B: u32, EN_B: u32, BU_B: u32,
    NB_C: u32, EN_C: u32, BU_C: u32,
    BE_A: s32 = {fixed_point::binary_exponent(EN_A, BU_A)}, // binary exp A
    BE_B: s32 = {fixed_point::binary_exponent(EN_B, BU_B)}, // binary exp B
    BE_C: s32 = {fixed_point::binary_exponent(EN_C, BU_C)}, // binary exp C
    // Precision inference MUL
    BE_MUL: s32 = {BE_A + BE_B},                           // binary exp MUL
    NB_MUL: u32 = {NB_A + NB_B},                           // number bits MUL
    EN_MUL: u32 = {fixed_point::is_negative(BE_MUL)},      // exp is negative MUL
    BU_MUL: u32 = {fixed_point::binary_uexponent(BE_MUL)}, // unsigned exp MUL
    // Precision Inference ADD
    NB_SUM: u32 = {                                 // number bits ADD
        fixed_point::aligned_width(NB_MUL, BE_MUL, NB_C, BE_C) +
        if fixed_point::num_bits_overlapping(NB_MUL, BE_MUL, NB_C, BE_C) == u32:0 { u32:0 } else { u32:1 }
        },
    BE_SUM: s32 = {std::min(BE_MUL, BE_C)},                      // binary exp ADD
    EN_SUM: u32 = {fixed_point::is_negative(BE_SUM)},            // exp is negative ADD
    BU_SUM: u32 = {fixed_point::binary_uexponent(BE_SUM)}>       // unsigned exp ADD
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
    BE_A: s32 = {fixed_point::binary_exponent(EN_A, BU_A)}, 
    BE_B: s32 = {fixed_point::binary_exponent(EN_B, BU_B)}>
    (fxd_a: sN[NB_A], fxd_b: sN[NB_B])
    -> sN[NB_B] {
    // Widen before left shifting to avoid overflow
    let aligned_lhs = (fxd_a as sN[NB_B]) << (BE_A - BE_B) as u32; // TODO: I think this is also always the same in the dot product use case. Fraction bits stay the same
    let aligned_rhs = fxd_b;

    aligned_lhs + aligned_rhs
}

// Performs an fused-multiply-add assuming that the rhs is already wide enough to not overflow.
// WARNING: the add rhs must be wide enough to avoid any overflow
fn fmadd_already_widened
    <NB_A: u32, EN_A: u32, BU_A: u32, 
    NB_B: u32, EN_B: u32, BU_B: u32,
    NB_C: u32, EN_C: u32, BU_C: u32,
    BE_A: s32 = {fixed_point::binary_exponent(EN_A, BU_A)}, // binary exp A
    BE_B: s32 = {fixed_point::binary_exponent(EN_B, BU_B)}, // binary exp B
    // Precision inference MUL
    BE_MUL: s32 = {BE_A + BE_B},                           // binary exp MUL
    NB_MUL: u32 = {NB_A + NB_B},                           // number bits MUL
    EN_MUL: u32 = {fixed_point::is_negative(BE_MUL)},      // exp is negative MUL
    BU_MUL: u32 = {fixed_point::binary_uexponent(BE_MUL)}> // unsigned exp MUL>
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
// 1. made aligned_width() and num_bits_overlapping() public in a copy of the fixed_point module.
// to write the type inference
// 2. We use ''already_widened'' functions.
fn dot_prod
    <NB_X: u32, EN_X: u32, BU_X: u32, 
    NB_Y: u32, EN_Y: u32, BU_Y: u32,
    VEC_SZ: u32,
    BE_X: s32 = {fixed_point::binary_exponent(EN_X, BU_X)}, // binary exp X
    BE_Y: s32 = {fixed_point::binary_exponent(EN_Y, BU_Y)}, // binary exp Y
    // Precision inference MUL
    BE_MUL: s32 = {BE_X + BE_Y},                           // binary exp MUL
    NB_MUL: u32 = {NB_X + NB_Y},                           // number bits MUL
    EN_MUL: u32 = {fixed_point::is_negative(BE_MUL)},      // exp is negative MUL
    BU_MUL: u32 = {fixed_point::binary_uexponent(BE_MUL)}, // unsigned exp MUL
    // Precision Inference DOT PROD
    NB_DOT_PROD: u32 = {NB_MUL + std::clog2(VEC_SZ)},                 // number bits DOT PROD
    BE_DOT_PROD: s32 = {BE_MUL},                                     // binary exp DOT PROD
    EN_DOT_PROD: u32 = {fixed_point::is_negative(BE_DOT_PROD)},      // exp is negative DOT PROD
    BU_DOT_PROD: u32 = {fixed_point::binary_uexponent(BE_DOT_PROD)}> // unsigned exp DOT PROD
    (x: sN[NB_X][VEC_SZ], 
     y: sN[NB_Y][VEC_SZ])
    -> sN[NB_DOT_PROD] {

    for (i, acc): (u32, sN[NB_DOT_PROD]) in u32:0..VEC_SZ {
        fmadd_already_widened<NB_X, EN_X, BU_X, NB_Y, EN_Y, BU_Y, NB_DOT_PROD, EN_DOT_PROD, BU_DOT_PROD>(x[i], y[i], acc)
    }(sN[NB_DOT_PROD]:0)
}

// // ================================================================
// // ----------------------- NN Implementation ----------------------

pub fn relu
    <NB: u32, EN: u32, BU: u32, 
    BE: s32 = {fixed_point::binary_exponent(EN, BU)}>
    (fxd_x: sN[NB]) -> sN[NB] {
    
    if (fxd_x > sN[NB]:0) 
        { fxd_x } 
    else 
        { sN[NB]:0 }
} 

pub fn argmax
    <NB: u32, EN: u32, BU: u32, VEC_SZ: u32,
    BE: s32 = {fixed_point::binary_exponent(EN, BU)},
    SHIFT_LIMIT: u32 = {NB - u32:1}>
    (y: sN[NB][VEC_SZ]) -> sN[NB][VEC_SZ] {
    
    let max_significand = for (i, acc): (u32, sN[NB]) in u32:0..VEC_SZ {
        std::max(y[i], acc)
    }((s32:-1 << SHIFT_LIMIT) as sN[NB]);

    for (i, z): (u32, sN[NB][VEC_SZ]) in u32:0..VEC_SZ {
        if y[i] == max_significand { 
            update(z, i, (u32:1<<BU) as sN[NB]) 
        } else { 
            update(z, i, sN[NB]:0) 
        }
    }(y)
} 

// Wx = y
// When called must specify the fixed point precision that is in the output. 
// This allows the truncation to be done correctly.
pub fn dense
    <NB_IN: u32, EN_IN: u32, BU_IN: u32, 
    NB_OUT: u32, EN_OUT: u32, BU_OUT: u32, 
    COLS: u32, ROWS: u32,
    BE_OUT: s32 = {fixed_point::binary_exponent(EN_OUT, BU_OUT)}, //new

    BE_IN: s32 = {fixed_point::binary_exponent(EN_IN, BU_IN)}, // binary exp X
    // Precision inference MUL
    BE_MUL: s32 = {BE_IN + BE_IN},                           // binary exp MUL
    NB_MUL: u32 = {NB_IN + NB_IN},                           // number bits MUL
    EN_MUL: u32 = {fixed_point::is_negative(BE_MUL)},        // exp is negative MUL
    BU_MUL: u32 = {fixed_point::binary_uexponent(BE_MUL)},   // unsigned exp MUL
    // Precision Inference DOT PROD
    NB_DOT_PROD: u32 = {NB_MUL + std::clog2(ROWS)},                  // number bits DOT PROD
    BE_DOT_PROD: s32 = {BE_MUL},                                     // binary exp DOT PROD
    EN_DOT_PROD: u32 = {fixed_point::is_negative(BE_DOT_PROD)},      // exp is negative DOT PROD
    BU_DOT_PROD: u32 = {fixed_point::binary_uexponent(BE_DOT_PROD)},
    // Precision Inference BIAS
    NB_BIAS: u32 = {
        fixed_point::aligned_width(NB_DOT_PROD, BE_DOT_PROD, NB_IN, BE_IN) +
        if fixed_point::num_bits_overlapping(NB_DOT_PROD, BE_DOT_PROD, NB_IN, BE_IN) == u32:0 { u32:0 } else { u32:1 }},
    BE_BIAS: s32 = {std::min(BE_DOT_PROD, BE_IN)},              // TODO: this is always DOT_PROD 
    EN_BIAS: u32 = {fixed_point::is_negative(BE_BIAS)},
    BU_BIAS: u32 = {fixed_point::binary_uexponent(BE_BIAS)}>
    (x: sN[NB_IN][ROWS], 
    W: sN[NB_IN][ROWS][COLS], 
    bias: sN[NB_IN][COLS]) 
    -> sN[NB_OUT][COLS] {

    for (i, z): (u32, sN[NB_OUT][COLS]) in u32:0..COLS {
        let vec_prod  = dot_prod<NB_IN, EN_IN, BU_IN, NB_IN, EN_IN, BU_IN>(x, W[i]);
        let with_bias = add<NB_DOT_PROD, EN_DOT_PROD, BU_DOT_PROD, NB_IN, EN_IN, BU_IN>(vec_prod, bias[i]);
        let with_bias_common = to_common_type<NB_OUT, BU_OUT, NB_BIAS, EN_BIAS, BU_BIAS>(with_bias);
        update(z, i, with_bias_common)
    }(sN[NB_OUT][COLS]:[sN[NB_OUT]:0, ...]) 
}
// Wx = y
// When called must specify the fixed point precision that is in the output. 
// This allows the truncation to be done correctly.
// TODO: remove inference from called functions, only infer at layer level? (Issue due to bottom up approach when writing the lib)
pub fn dense_relu
    <NB_IN: u32, EN_IN: u32, BU_IN: u32, 
    NB_OUT: u32, EN_OUT: u32, BU_OUT: u32, 
    COLS: u32, ROWS: u32,
    BE_OUT: s32 = {fixed_point::binary_exponent(EN_OUT, BU_OUT)}, //new

    BE_IN: s32 = {fixed_point::binary_exponent(EN_IN, BU_IN)}, // binary exp X
    // Precision inference MUL
    BE_MUL: s32 = {BE_IN + BE_IN},                           // binary exp MUL
    NB_MUL: u32 = {NB_IN + NB_IN},                           // number bits MUL
    EN_MUL: u32 = {fixed_point::is_negative(BE_MUL)},        // exp is negative MUL
    BU_MUL: u32 = {fixed_point::binary_uexponent(BE_MUL)},   // unsigned exp MUL
    // Precision Inference DOT PROD
    NB_DOT_PROD: u32 = {NB_MUL + std::clog2(ROWS)},                  // number bits DOT PROD
    BE_DOT_PROD: s32 = {BE_MUL},                                     // binary exp DOT PROD
    EN_DOT_PROD: u32 = {fixed_point::is_negative(BE_DOT_PROD)},      // exp is negative DOT PROD
    BU_DOT_PROD: u32 = {fixed_point::binary_uexponent(BE_DOT_PROD)},
    // Precision Inference BIAS
    NB_BIAS: u32 = {
        fixed_point::aligned_width(NB_DOT_PROD, BE_DOT_PROD, NB_IN, BE_IN) +
        if fixed_point::num_bits_overlapping(NB_DOT_PROD, BE_DOT_PROD, NB_IN, BE_IN) == u32:0 { u32:0 } else { u32:1 }},
    BE_BIAS: s32 = {std::min(BE_DOT_PROD, BE_IN)},              // TODO: this is always DOT_PROD 
    EN_BIAS: u32 = {fixed_point::is_negative(BE_BIAS)},
    BU_BIAS: u32 = {fixed_point::binary_uexponent(BE_BIAS)}>
    (x: sN[NB_IN][ROWS], 
    W: sN[NB_IN][ROWS][COLS], 
    bias: sN[NB_IN][COLS]) 
    -> sN[NB_OUT][COLS] {

    for (i, z): (u32, sN[NB_OUT][COLS]) in u32:0..COLS {
        let vec_prod  = dot_prod<NB_IN, EN_IN, BU_IN, NB_IN, EN_IN, BU_IN>(x, W[i]);
        let with_bias = add<NB_DOT_PROD, EN_DOT_PROD, BU_DOT_PROD, NB_IN, EN_IN, BU_IN>(vec_prod, bias[i]);
        let with_bias_common = to_common_type<NB_OUT, BU_OUT, NB_BIAS, EN_BIAS, BU_BIAS>(with_bias);
        let with_relu = relu<NB_OUT, EN_OUT, BU_OUT>(with_bias_common);
        update(z, i, with_relu)
    }(sN[NB_OUT][COLS]:[sN[NB_OUT]:0, ...]) 
}


pub fn multi_dense_fxd
    <INPUT_D1: u32, INPUT_D2: u32, 
    IN_L1: u32 = {INPUT_D2}, OUT_L1: u32,
    IN_L2: u32 = {OUT_L1},   OUT_L2: u32>
    (x: sN[NB_COMMON][INPUT_D2][INPUT_D1], 
    w1: sN[NB_COMMON][IN_L1][OUT_L1], 
    b1: sN[NB_COMMON][OUT_L1],
    w2: sN[NB_COMMON][IN_L2][OUT_L2], 
    b2: sN[NB_COMMON][OUT_L2])
    -> sN[NB_COMMON][OUT_L2][INPUT_D1] {

    // ---------------- Layer 1 -----------------
    let z1 = for (batch_idx, layer1): (u32, sN[NB_COMMON][OUT_L1][INPUT_D1]) in u32:0..INPUT_D1 {
        update(
            layer1, 
            batch_idx, 
            dense_relu<u32:16, u32:1, u32:10, u32:16, u32:1, u32:10>(x[batch_idx], w1, b1)
        )
    }(sN[NB_COMMON][OUT_L1][INPUT_D1]:[sN[NB_COMMON][OUT_L1]:[FXP_0_0, ...], ...]); // init matrix w/ zeros

    // ---------------- Layer 2 -----------------
    let z2 = for (batch_idx, layer2): (u32, sN[NB_COMMON][OUT_L2][INPUT_D1]) in u32:0..INPUT_D1 {
        update(
            layer2, 
            batch_idx, 
            dense_relu<u32:16, u32:1, u32:10, u32:16, u32:1, u32:10>(z1[batch_idx], w2, b2)
        )
    }(sN[NB_COMMON][OUT_L2][INPUT_D1]:[sN[NB_COMMON][OUT_L2]:[FXP_0_0, ...], ...]); // init matrix w/ zeros

    // ------------ Output -------------------
    z2
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

#[test]
fn argmax_test() {
    let x = sN[NB_COMMON][2]:[
        sN[NB_COMMON]:1536, 
        sN[NB_COMMON]:1024
    ]; 
    let expected = sN[NB_COMMON][2]:[
        sN[NB_COMMON]:1024, 
        sN[NB_COMMON]:0
    ];  
    assert_eq(expected, argmax<NB_COMMON, u32:1, u32:10>(x));

    let x = sN[NB_COMMON][4]:[
        sN[NB_COMMON]:-1536, 
        sN[NB_COMMON]:-1024,
        sN[NB_COMMON]:0,
        sN[NB_COMMON]:-1024
    ]; 
    let expected = sN[NB_COMMON][4]:[
        sN[NB_COMMON]:0, 
        sN[NB_COMMON]:0,
        sN[NB_COMMON]:1024,
        sN[NB_COMMON]:0,
    ];  
    assert_eq(expected, argmax<NB_COMMON, u32:1, u32:10>(x));

    let x = sN[NB_COMMON][4]:[
        sN[NB_COMMON]:-1536, 
        sN[NB_COMMON]:-1024,
        sN[NB_COMMON]:-512,
        sN[NB_COMMON]:-1024
    ]; 
    let expected = sN[NB_COMMON][4]:[
        sN[NB_COMMON]:0, 
        sN[NB_COMMON]:0,
        sN[NB_COMMON]:1024,
        sN[NB_COMMON]:0,
    ];  
    assert_eq(expected, argmax<NB_COMMON, u32:1, u32:10>(x));
}

#[test]
fn dense_relu_test_pos() { 
    let x  = sN[NB_COMMON][2]:[FXP_1_5, FXP_1_5];
    let w1 = sN[NB_COMMON][2][2]:[[FXP_1_5, FXP_1_5],
                                   [FXP_1_5, FXP_1_5]];
    let b1 = sN[NB_COMMON][2]:[FXP_0_0, FXP_0_0];
    let expected = sN[NB_COMMON][2]:[FXP_4_5, FXP_4_5];
    assert_eq(expected, dense_relu<NB_COMMON, u32:1, u32:10, NB_COMMON, u32:1, u32:10>(x, w1, b1));
}

#[test]
fn dense_relu_test_neg() { 
    let x  = sN[NB_COMMON][2]:[FXP_1_5, FXP_1_5];
    let w1 = sN[NB_COMMON][2][2]:[[FXP_1_5, FXP_1_5],
                                   [FXP_1_5, FXP_1_5]];
    let b1 = sN[NB_COMMON][2]:[FXP_6_75_NEG, FXP_0_0];
    let expected = sN[NB_COMMON][2]:[FXP_0_0, FXP_4_5];
    assert_eq(expected, dense_relu<NB_COMMON, u32:1, u32:10, NB_COMMON, u32:1, u32:10>(x, w1, b1));
}

#[test]
fn multi_dense_test_no_bias() {
    let x = sN[NB_COMMON][2][2]:[[FXP_1_5, FXP_1_5],
                                  [FXP_1_5, FXP_1_5]];
    let w1 = sN[NB_COMMON][2][2]:[[FXP_1_5, FXP_1_5],
                                   [FXP_1_5, FXP_1_5]];
    let b1 = sN[NB_COMMON][2]:[FXP_0_0, FXP_0_0];
    let w2 = sN[NB_COMMON][2][2]:[[FXP_1_5, FXP_1_5],
                                   [FXP_1_5, FXP_1_5]];
    let b2 = sN[NB_COMMON][2]:[FXP_0_0, FXP_0_0];
    let expected = sN[NB_COMMON][2][2]:[[FXP_13_5, FXP_13_5],
                                         [FXP_13_5, FXP_13_5]];
    let result = multi_dense_fxd(x, w1, b1, w2, b2);
    assert_eq(expected, result);
}
