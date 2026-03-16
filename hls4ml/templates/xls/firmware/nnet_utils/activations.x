import std;
import fixed_point;

import ap_types.fixed_point_util;
import nnet_utils.lookup_table;

type FixedPoint = fixed_point::FixedPoint;
type RoundingMode = fixed_point_util::RoundingMode;
type OverflowMode = fixed_point_util::OverflowMode;
type LookupTable = lookup_table::LookupTable;


// =========================================================================
// --------------------------------- ReLU ----------------------------------

pub fn relu_1elem
    <NB: u32, BE: s32>
    (fxd_x: FixedPoint<NB, BE>) -> FixedPoint<NB, BE> {
    
    if (fxd_x.significand > 0) 
        { fxd_x } 
    else 
        { zero!<FixedPoint<NB, BE>>() }
} 

pub fn relu
    <NB: u32, BE: s32, DIM: u32>
    (y: FixedPoint<NB, BE>[DIM]) -> FixedPoint<NB, BE>[DIM] {

    map(y, relu_1elem<NB>)
} 

#[test]
fn relu_test() {
    let x = fixed_point_util::make_fixed_points_1d<-10>(sN[16][2]:[
        1536, 1024
    ]); 
    let expected = fixed_point_util::make_fixed_points_1d<-10>(sN[16][2]:[
        1536, 1024
    ]);
    assert_eq(expected, relu(x));

    let x = fixed_point_util::make_fixed_points_1d<-10>(sN[16][4]:[
        -1536, -1024, 0, -1024
    ]); 
    let expected = fixed_point_util::make_fixed_points_1d<-10>(sN[16][4]:[
        0,...
    ]);  
    assert_eq(expected, relu(x));

    let x = fixed_point_util::make_fixed_points_1d<-10>(sN[16][4]:[
        -1536, -1024, 1024, -1024
    ]); 
    let expected = fixed_point_util::make_fixed_points_1d<-10>(sN[16][4]:[
        0, 0, 1024, 0
    ]);
    assert_eq(expected, relu<u32:16>(x));
}

// =========================================================================
// ------------------------------- Argmax ---------------------------------

pub fn argmax
    <NB_OUT: u32, BE_OUT: s32,
    NB_IN: u32, BE_IN: s32,
    DIM: u32>
    (y: FixedPoint<NB_IN, BE_IN>[DIM])
    -> FixedPoint<NB_OUT, BE_OUT>[DIM] {
        let y_max = fixed_point_util::max_1d(y);
        let one = fixed_point_util::resize<
            NB_OUT, BE_OUT,
            RoundingMode::TRN,
            OverflowMode::WRAP
        >(fixed_point::from_integer(s2:1));
        for (i, z) in 0..DIM {
            if y[i] == y_max { 
                update(z, i, one)
            }
            else { 
                z
            }
        }(zero!<FixedPoint<NB_OUT, BE_OUT>[DIM]>())
}

#[test]
fn argmax_test() {
    let x = fixed_point_util::make_fixed_points_1d<-10>(sN[16][2]:[
        1536, 
        1024
    ]); 
    let expected = fixed_point_util::make_fixed_points_1d<-10>(sN[18][2]:[
        1024, 
        0
    ]);
    assert_eq(expected, argmax<18, -10>(x));

    let x = fixed_point_util::make_fixed_points_1d<-10>(sN[16][4]:[
        -1536, 
        -1024,
        0,
        -1024
    ]); 
    let expected = fixed_point_util::make_fixed_points_1d<-10>(sN[18][4]:[
        0, 
        0,
        1024,
        0,
    ]);
    assert_eq(expected, argmax<18, -10>(x));

    let x = fixed_point_util::make_fixed_points_1d<-10>(sN[16][4]:[
        -1536, 
        -1024,
        -512,
        -1024
    ]);
    let expected = fixed_point_util::make_fixed_points_1d<-10>(sN[18][4]:[
        0, 
        0,
        1024,
        0,
    ]);  
    assert_eq(expected, argmax<18, -10>(x));
}

// =========================================================================
// ------------------------------ Softmax ----------------------------------

pub fn softmax_latency
    <NB_OUT: u32, BE_OUT: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    NB_IN: u32, BE_IN: s32, 
    NB_EXP: u32, BE_EXP: s32, SIZE_EXP: u32, LOG2_STEP_EXP: s32,
    NB_INV: u32, BE_INV: s32, SIZE_INV: u32, LOG2_STEP_INV: s32,
    DIM: u32,
    NB_SUM_EXP: u32 = {NB_EXP + std::clog2(DIM)},  
    BE_SUM_EXP: s32 = {BE_EXP}>(
        y: FixedPoint<NB_IN, BE_IN>[DIM],
        exp_lut: LookupTable<NB_IN, BE_IN, NB_EXP, BE_EXP, SIZE_EXP, LOG2_STEP_EXP>,
        inv_lut: LookupTable<NB_INV, BE_INV, NB_INV, BE_INV, SIZE_INV, LOG2_STEP_INV>,
    ) -> FixedPoint<NB_OUT, BE_OUT>[DIM] {
    
    // Compute exp() with Lookup Tables
    let exp = lookup_table::eval_1d(exp_lut, y);

    // Sum all exponents
    let sum_exp = for (i, acc) in 0..DIM {
        fixed_point_util::add_already_widened(exp[i], acc)
    }(zero!<FixedPoint<NB_SUM_EXP, BE_SUM_EXP>>());
    let sum_exp = fixed_point_util::resize<NB_INV, BE_INV, ROUNDING, OVERFLOW>(sum_exp);
    let inv_sum_exp = lookup_table::eval(inv_lut, sum_exp);

    // Compute softmax
    let softmax_result = for (i, inv_vec) in 0..DIM {
        update(inv_vec, i, fixed_point_util::resize<NB_OUT, BE_OUT, ROUNDING, OVERFLOW>(
            fixed_point::mul(exp[i], inv_sum_exp)
        ))
    }(zero!<FixedPoint<NB_OUT, BE_OUT>[DIM]>());

    softmax_result
} 

// softmax(x) = exp(x[i]) / sum(exp(x[k])
// Stable implementation:
// softmax(x) = exp(-(x_max-x[i])) / sum_k(exp(-(x_max-x[k])))
pub fn softmax_stable
    <NB_OUT: u32, BE_OUT: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    NB_IN: u32, BE_IN: s32, 
    NB_EXP: u32, BE_EXP: s32, SIZE_EXP: u32, LOG2_STEP_EXP: s32,
    NB_INV: u32, BE_INV: s32, SIZE_INV: u32, LOG2_STEP_INV: s32,
    DIM: u32,
    // x_max - x_i
    NB_DIFF: u32 = {NB_IN + 1}, BE_DIFF: s32 = {BE_IN},
    // sum(exp(-(x_max-x_i)
    NB_SUM_EXP: u32 = {NB_EXP + std::clog2(DIM)},  
    BE_SUM_EXP: s32 = {BE_EXP}>(
        x: FixedPoint<NB_IN, BE_IN>[DIM],
        // f(x) = exp(-x)
        exp_neg_lut: LookupTable<NB_DIFF, BE_DIFF, NB_EXP, BE_EXP, SIZE_EXP, LOG2_STEP_EXP>,
        // f(x) = 1/x
        inv_lut: LookupTable<NB_INV, BE_INV, NB_INV, BE_INV, SIZE_INV, LOG2_STEP_INV>,
    ) -> FixedPoint<NB_OUT, BE_OUT>[DIM] {

    let x_max = fixed_point_util::max_1d(x);

    // exp(-(x_max-x_i))
    let exp = for (i, acc) in 0..DIM {
        let d_xmax_xi = fixed_point::sub(x_max, x[i]);
        let exp_dx = lookup_table::eval(exp_neg_lut, d_xmax_xi);
        update(acc, i, exp_dx)
    }(zero!<FixedPoint<NB_EXP, BE_EXP>[DIM]>());

    // Sum all exponents
    let sum_exp = for (i, acc) in 0..DIM {
        fixed_point_util::add_already_widened(exp[i], acc)
    }(zero!<FixedPoint<NB_SUM_EXP, BE_SUM_EXP>>());
    // Truncate.
    let sum_exp = fixed_point_util::resize<NB_INV, BE_INV, ROUNDING, OVERFLOW>(sum_exp);
    // 1 / sum(exp)
    let inv_sum_exp = lookup_table::eval(inv_lut, sum_exp);

    // softmax
    let softmax_result = for (i, acc) in 0..DIM {
        update(acc, i, fixed_point_util::resize<NB_OUT, BE_OUT, ROUNDING, OVERFLOW>(
            fixed_point::mul(exp[i], inv_sum_exp)
        ))
    }(zero!<FixedPoint<NB_OUT, BE_OUT>[DIM]>());

    softmax_result
}

// ------------- TODO Tests should be generated depending on the table precision/size

// #[test]
// fn softmax_latency_test() {
//     let x = sN[16][4]:[
//         sN[16]:1024,
//         sN[16]:1024,
//         sN[16]:1024,
//         sN[16]:1024
//     ];
//     let expected = sN[16][4]:[
//         sN[16]:258,  // Ideal 256
//         sN[16]:258,
//         sN[16]:258,
//         sN[16]:258
//     ];
//     assert_eq(expected, softmax_latency        
//         <u32:16, u32:1, u32:10, 
//         u32:16, u32:1, u32:10, 
//         u32:18, u32:1, u32:10, 
//         u32:18, u32:1, u32:10,
//         u32:1024>(x));

//     let x = sN[16][4]:[
//         sN[16]:2048,
//         sN[16]:2048,
//         sN[16]:2048,
//         sN[16]:2048
//     ];
//     let expected = sN[16][4]:[
//         sN[16]:258,  // Ideal 256
//         sN[16]:258,
//         sN[16]:258,
//         sN[16]:258
//     ];
//     assert_eq(expected, softmax_latency        
//         <u32:16, u32:1, u32:10, 
//         u32:16, u32:1, u32:10, 
//         u32:18, u32:1, u32:10, 
//         u32:18, u32:1, u32:10,
//         u32:1024>(x));
// }

// #[test]
// fn softmax_stable_test() {
//     let x = fixed_point_util::make_fixed_points<-10>(sN[16][4]:[
//         1024,
//         1024,
//         1024,
//         1024
//     ]);
//     let expected = fixed_point_util::make_fixed_points<-10>(sN[16][4]:[
//         256,  // Ideal 256
//         256,
//         256,
//         256
//     ]);
//     assert_eq(expected, softmax_stable<16,1,10>(x, EXP_TABLE, INV_TABLE));

//     let x = fixed_point_util::make_fixed_points<-10>(sN[16][4]:[
//         4096,
//         4096,
//         4096,
//         4096
//     ]);
//     let expected = fixed_point_util::make_fixed_points<-10>(sN[16][4]:[
//         256,  // Ideal 256
//         256,
//         256,
//         256
//     ]);
//     assert_eq(expected, softmax_stable<16,1,10>(x, EXP_TABLE, INV_TABLE));
// }