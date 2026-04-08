import std;
import fixed_point;
import ap_types.fixed_point_util;
import nnet_utils.activations;

type FixedPoint = fixed_point::FixedPoint;
type RoundingMode = fixed_point_util::RoundingMode;
type OverflowMode = fixed_point_util::OverflowMode;

// y = Wx + b 
// When called must specify the fixed point precision that is in the output.
// This allows the truncation to be done correctly.
pub fn dense
    <NB_OUT: u32, BE_OUT: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    NB_IN: u32, BE_IN: s32,
    COLS: u32, ROWS: u32>
    (x: FixedPoint<NB_IN, BE_IN>[ROWS],
    w: FixedPoint<NB_IN, BE_IN>[ROWS][COLS],
    bias: FixedPoint<NB_IN, BE_IN>[COLS])
    -> FixedPoint<NB_OUT, BE_OUT>[COLS] {

    for (i, z) in u32:0..COLS {
        let vec_prod  = fixed_point_util::dot_prod(x, w[i]);
        let with_bias = fixed_point::add(vec_prod, bias[i]);
        let with_bias_out = fixed_point_util::resize<NB_OUT, BE_OUT, ROUNDING, OVERFLOW>(with_bias);
        update(z, i, with_bias_out)
    }(zero!<FixedPoint<NB_OUT, BE_OUT>[COLS]>())
}

// TODO: used only for tests
// y = relu(Wx + b)
// When called must specify the fixed point precision that is in the output.
// This allows the truncation to be done correctly.
pub fn dense_relu
    <NB_OUT: u32, BE_OUT: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    NB_IN: u32, BE_IN: s32,
    COLS: u32, ROWS: u32>
    (x: FixedPoint<NB_IN, BE_IN>[ROWS],
    w: FixedPoint<NB_IN, BE_IN>[ROWS][COLS],
    bias: FixedPoint<NB_IN, BE_IN>[COLS])
    -> FixedPoint<NB_OUT, BE_OUT>[COLS] {

    let y = dense<NB_OUT, BE_OUT, ROUNDING, OVERFLOW>(x, w, bias);
    activations::relu<NB_OUT, BE_OUT, ROUNDING, OVERFLOW>(y)
}

// Testing

const NB_COMMON = u32:16;
const BE_COMMON = s32:-10;
const ROUNDING_COMMON = RoundingMode::TRN;
const OVERFLOW_COMMON = OverflowMode::WRAP;

type FP = FixedPoint<NB_COMMON, BE_COMMON>;

fn make_fixed(x:sN[NB_COMMON]) -> FP {
    fixed_point::make_fixed_point<BE_COMMON>(x)
}

const FXP_6_75_NEG = make_fixed(-6912);
const FXP_4_0_NEG  = make_fixed(-4096);
const FXP_3_0_NEG  = make_fixed(-3072);
const FXP_0_0      = make_fixed(0);
const FXP_0_5      = make_fixed(512);
const FXP_1_0      = make_fixed(1024);
const FXP_1_5      = make_fixed(1536);
const FXP_2_0      = make_fixed(2048);
const FXP_2_25     = make_fixed(2304);
const FXP_4_5      = make_fixed(4608);
const FXP_5_5      = make_fixed(5632);
const FXP_6_75     = make_fixed(6912);
const FXP_12_0     = make_fixed(12288);
const FXP_13_5     = make_fixed(13824);


#[test]
fn dense_relu_test_pos() {
    let x  = [FXP_1_5, FXP_1_5];
    let w1 = [
        [FXP_1_5, FXP_1_5],
        [FXP_1_5, FXP_1_5]
    ];
    let b1 = [FXP_0_0, FXP_0_0];
    let expected = [FXP_4_5, FXP_4_5];
    assert_eq(expected, dense_relu<NB_COMMON, -10, ROUNDING_COMMON, OVERFLOW_COMMON>(x, w1, b1));
}

#[test]
fn dense_relu_test_neg() {
    let x  = [FXP_1_5, FXP_1_5];
    let w1 = [
        [FXP_1_5, FXP_1_5],
        [FXP_1_5, FXP_1_5]
    ];
    let b1 = [FXP_6_75_NEG, FXP_0_0];
    let expected = [FXP_0_0, FXP_4_5];
    assert_eq(expected, dense_relu<NB_COMMON, -10, ROUNDING_COMMON, OVERFLOW_COMMON>(x, w1, b1));
}

fn integration_nn
    <INPUT_D1: u32, INPUT_D2: u32,
    IN_L1: u32 = {INPUT_D2}, OUT_L1: u32,
    IN_L2: u32 = {OUT_L1},   OUT_L2: u32>
    (x: FP[INPUT_D2][INPUT_D1],
    w1: FP[IN_L1][OUT_L1],
    b1: FP[OUT_L1],
    w2: FP[IN_L2][OUT_L2],
    b2: FP[OUT_L2])
    -> FP[OUT_L2][INPUT_D1] {

    // ---------------- Layer 1 -----------------
    let z1 = for (batch_idx, layer1) in 0..INPUT_D1 {
        update(
            layer1,
            batch_idx,
            dense_relu<NB_COMMON, BE_COMMON, ROUNDING_COMMON, OVERFLOW_COMMON>(x[batch_idx], w1, b1)
        )
    }(zero!<FP[OUT_L1][INPUT_D1]>()); // init matrix w/ zeros

    // ---------------- Layer 2 -----------------
    let z2 = for (batch_idx, layer2) in 0..INPUT_D1 {
        update(
            layer2,
            batch_idx,
            dense_relu<NB_COMMON, BE_COMMON, ROUNDING_COMMON, OVERFLOW_COMMON>(z1[batch_idx], w2, b2)
        )
    }(zero!<FP[OUT_L2][INPUT_D1]>()); // init matrix w/ zeros

    // ------------ Output -------------------
    z2
}

#[test]
fn integration_test() {
    let x = [
        [FXP_1_5, FXP_1_5],
        [FXP_1_5, FXP_1_5]
    ];
    let w1 = [
        [FXP_1_5, FXP_1_5],
        [FXP_1_5, FXP_1_5]
    ];
    let b1 = [FXP_0_0, FXP_0_0];
    let w2 = [
        [FXP_1_5, FXP_1_5],
        [FXP_1_5, FXP_1_5]
    ];
    let b2 = [FXP_0_0, FXP_0_0];
    let expected = [
        [FXP_13_5, FXP_13_5],
        [FXP_13_5, FXP_13_5]
    ];
    let result = integration_nn(x, w1, b1, w2, b2);
    assert_eq(expected, result);
}
