import std;

import ap_types.fixed_point_fix;
import ap_types.fixed_point_lib;

import nnet_utils.activations;

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



// Wx = y
// When called must specify the fixed point precision that is in the output. 
// This allows the truncation to be done correctly.
pub fn dense
    <NB_IN: u32, EN_IN: u32, BU_IN: u32, 
    NB_OUT: u32, EN_OUT: u32, BU_OUT: u32, 
    COLS: u32, ROWS: u32,
    BE_OUT: s32 = {fixed_point_lib::binary_exponent(EN_OUT, BU_OUT)}, //new

    BE_IN: s32 = {fixed_point_lib::binary_exponent(EN_IN, BU_IN)}, // binary exp X
    // Precision inference MUL
    BE_MUL: s32 = {BE_IN + BE_IN},                           // binary exp MUL
    NB_MUL: u32 = {NB_IN + NB_IN},                           // number bits MUL
    EN_MUL: u32 = {fixed_point_lib::is_negative(BE_MUL)},        // exp is negative MUL
    BU_MUL: u32 = {fixed_point_lib::binary_uexponent(BE_MUL)},   // unsigned exp MUL
    // Precision Inference DOT PROD
    NB_DOT_PROD: u32 = {NB_MUL + std::clog2(ROWS)},                  // number bits DOT PROD
    BE_DOT_PROD: s32 = {BE_MUL},                                     // binary exp DOT PROD
    EN_DOT_PROD: u32 = {fixed_point_lib::is_negative(BE_DOT_PROD)},      // exp is negative DOT PROD
    BU_DOT_PROD: u32 = {fixed_point_lib::binary_uexponent(BE_DOT_PROD)},
    // Precision Inference BIAS
    NB_BIAS: u32 = {
        fixed_point_lib::aligned_width(NB_DOT_PROD, BE_DOT_PROD, NB_IN, BE_IN) +
        if fixed_point_lib::num_bits_overlapping(NB_DOT_PROD, BE_DOT_PROD, NB_IN, BE_IN) == u32:0 { u32:0 } else { u32:1 }},
    BE_BIAS: s32 = {std::min(BE_DOT_PROD, BE_IN)},         
    EN_BIAS: u32 = {fixed_point_lib::is_negative(BE_BIAS)},
    BU_BIAS: u32 = {fixed_point_lib::binary_uexponent(BE_BIAS)}>
    (x: sN[NB_IN][ROWS], 
    W: sN[NB_IN][ROWS][COLS], 
    bias: sN[NB_IN][COLS]) 
    -> sN[NB_OUT][COLS] {

    for (i, z): (u32, sN[NB_OUT][COLS]) in u32:0..COLS {
        let vec_prod  = fixed_point_fix::dot_prod<NB_IN, EN_IN, BU_IN, NB_IN, EN_IN, BU_IN>(x, W[i]);
        let with_bias = fixed_point_fix::add<NB_DOT_PROD, EN_DOT_PROD, BU_DOT_PROD, NB_IN, EN_IN, BU_IN>(vec_prod, bias[i]);
        let with_bias_common = fixed_point_fix::to_common_type<NB_OUT, BU_OUT, NB_BIAS, EN_BIAS, BU_BIAS>(with_bias);
        update(z, i, with_bias_common)
    }(sN[NB_OUT][COLS]:[sN[NB_OUT]:0, ...]) 
}
// Wx = y
// When called must specify the fixed point precision that is in the output. 
// This allows the truncation to be done correctly.
pub fn dense_relu
    <NB_IN: u32, EN_IN: u32, BU_IN: u32, 
    NB_OUT: u32, EN_OUT: u32, BU_OUT: u32, 
    COLS: u32, ROWS: u32,
    BE_OUT: s32 = {fixed_point_lib::binary_exponent(EN_OUT, BU_OUT)}, //new

    BE_IN: s32 = {fixed_point_lib::binary_exponent(EN_IN, BU_IN)}, // binary exp X
    // Precision inference MUL
    BE_MUL: s32 = {BE_IN + BE_IN},                           // binary exp MUL
    NB_MUL: u32 = {NB_IN + NB_IN},                           // number bits MUL
    EN_MUL: u32 = {fixed_point_lib::is_negative(BE_MUL)},        // exp is negative MUL
    BU_MUL: u32 = {fixed_point_lib::binary_uexponent(BE_MUL)},   // unsigned exp MUL
    // Precision Inference DOT PROD
    NB_DOT_PROD: u32 = {NB_MUL + std::clog2(ROWS)},                  // number bits DOT PROD
    BE_DOT_PROD: s32 = {BE_MUL},                                     // binary exp DOT PROD
    EN_DOT_PROD: u32 = {fixed_point_lib::is_negative(BE_DOT_PROD)},      // exp is negative DOT PROD
    BU_DOT_PROD: u32 = {fixed_point_lib::binary_uexponent(BE_DOT_PROD)},
    // Precision Inference BIAS
    NB_BIAS: u32 = {
        fixed_point_lib::aligned_width(NB_DOT_PROD, BE_DOT_PROD, NB_IN, BE_IN) +
        if fixed_point_lib::num_bits_overlapping(NB_DOT_PROD, BE_DOT_PROD, NB_IN, BE_IN) == u32:0 { u32:0 } else { u32:1 }},
    BE_BIAS: s32 = {std::min(BE_DOT_PROD, BE_IN)},         
    EN_BIAS: u32 = {fixed_point_lib::is_negative(BE_BIAS)},
    BU_BIAS: u32 = {fixed_point_lib::binary_uexponent(BE_BIAS)}>
    (x: sN[NB_IN][ROWS], 
    W: sN[NB_IN][ROWS][COLS], 
    bias: sN[NB_IN][COLS]) 
    -> sN[NB_OUT][COLS] {

    for (i, z): (u32, sN[NB_OUT][COLS]) in u32:0..COLS {
        let vec_prod  = fixed_point_fix::dot_prod<NB_IN, EN_IN, BU_IN, NB_IN, EN_IN, BU_IN>(x, W[i]);
        let with_bias = fixed_point_fix::add<NB_DOT_PROD, EN_DOT_PROD, BU_DOT_PROD, NB_IN, EN_IN, BU_IN>(vec_prod, bias[i]);
        let with_bias_common = fixed_point_fix::to_common_type<NB_OUT, BU_OUT, NB_BIAS, EN_BIAS, BU_BIAS>(with_bias);
        let with_relu = activations::relu_1elem<NB_OUT>(with_bias_common);
        update(z, i, with_relu)
    }(sN[NB_OUT][COLS]:[sN[NB_OUT]:0, ...]) 
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

fn integration_nn
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
fn integration_test() {
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
    let result = integration_nn(x, w1, b1, w2, b2);
    assert_eq(expected, result);
}
