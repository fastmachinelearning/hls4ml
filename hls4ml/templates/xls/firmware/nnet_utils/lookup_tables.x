
import std;
import ap_types.fixed_point_fix;
import ap_types.fixed_point_lib;


// hls-fpga-machine-learning insert exponent table


// hls-fpga-machine-learning insert inversion table



pub fn idx_from_real_val
    <TABLE_SZ: u32, NB: u32,
    N: u32 = {std::clog2(TABLE_SZ)},
    LOW_END: u32 = {if NB > N { NB - N } else { u32:0 }}> // NB-N but it the generated table influences this factor as well
    (x: sN[NB]) -> uN[N] {

    let unsgined_x = x as uN[NB];
    //let idx = (unsgined_x >> LOW_END) & ((uN[NB]:1 << N) - uN[NB]:1);
    let idx = (unsgined_x >> LOW_END);
    idx as uN[N]
}

#[test]
fn idx_from_real_val_test() {
    let x = sN[16]:256; 
    let expected = uN[10]:4;  
    assert_eq(expected, idx_from_real_val<u32:1024, u32:16>(x));

    let x = sN[16]:1024; 
    let expected = uN[10]:16;  
    assert_eq(expected, idx_from_real_val<u32:1024, u32:16>(x));

    let x = sN[18]:1024; 
    let expected = uN[10]:4;  
    assert_eq(expected, idx_from_real_val<u32:1024, u32:18>(x));
}


// =========================================================================
// ------------------------------ Softmax ----------------------------------

pub fn softmax_latency
    <NB_IN: u32, EN_IN: u32, BU_IN: u32, 
    NB_OUT: u32, EN_OUT: u32, BU_OUT: u32, 
    NB_TABLE_EXP: u32, EN_TABLE_EXP: u32, BU_TABLE_EXP: u32,
    NB_TABLE_INV: u32, EN_TABLE_INV: u32, BU_TABLE_INV: u32,
    TABLE_SZ: u32, 
    VEC_SZ: u32,
    // EXP Accum
    BE_TABLE_EXP: s32 = {fixed_point_lib::binary_exponent(EN_TABLE_EXP, BU_TABLE_EXP)}, 
    NB_ACCUM: u32 = {NB_TABLE_EXP + std::clog2(VEC_SZ)},  
    BE_ACCUM: s32 = {BE_TABLE_EXP},
    EN_ACCUM: u32 = {fixed_point_lib::is_negative(BE_ACCUM)},      // exp is negative ACCUM
    BU_ACCUM: u32 = {fixed_point_lib::binary_uexponent(BE_ACCUM)}, // unsigned exp ACCUM
    // INV Multiplication 
    BE_TABLE_INV: s32 = {fixed_point_lib::binary_exponent(EN_TABLE_INV, BU_TABLE_INV)}, 
    EXP_SUM: s32 = {BE_TABLE_INV + BE_TABLE_INV},
    NB_MUL: u32 = {NB_TABLE_INV + NB_TABLE_INV}, 
    EN_MUL: u32 = {fixed_point_lib::is_negative(EXP_SUM)},
    BU_MUL: u32 = {fixed_point_lib::binary_uexponent(EXP_SUM)}>
    (y: sN[NB_IN][VEC_SZ]) -> sN[NB_OUT][VEC_SZ] {
    
    // Compute exp() with Lookup Tables
    let exp_result = for (i, exp_vec): (u32, sN[NB_TABLE_EXP][VEC_SZ]) in u32:0..VEC_SZ {
        let exp_table_idx = idx_from_real_val<TABLE_SZ, NB_IN>(y[i]);
        update(exp_vec, i, EXP_TABLE[exp_table_idx]) 
    }(sN[NB_TABLE_EXP][VEC_SZ]:[sN[NB_TABLE_EXP]:0, ...]);

    // Sum all exponents
    let sum = for (i, acc): (u32, sN[NB_ACCUM]) in u32:0..VEC_SZ {
        fixed_point_fix::add_already_widened<NB_TABLE_EXP, EN_TABLE_EXP, BU_TABLE_EXP, NB_ACCUM, EN_ACCUM, BU_ACCUM>(exp_result[i], acc)
    }(sN[NB_ACCUM]:0);
    let truncate = fixed_point_fix::to_common_type<NB_TABLE_INV, BU_TABLE_INV, NB_ACCUM, EN_ACCUM, BU_ACCUM>(sum);
    let inv_exp_sum = INV_TABLE[idx_from_real_val<TABLE_SZ>(truncate)];

    // Compute softmax
    let softmax_result = for (i, inv_vec): (u32, sN[NB_OUT][VEC_SZ]) in u32:0..VEC_SZ {
        update(inv_vec, i, fixed_point_fix::to_common_type<NB_OUT, BU_OUT, NB_MUL, EN_MUL, BU_MUL>(
            fixed_point_fix::mul<NB_TABLE_INV, EN_TABLE_INV, BU_TABLE_INV, NB_TABLE_INV, EN_TABLE_INV, BU_TABLE_INV>
            (exp_result[i], inv_exp_sum)
        )) 
    }(sN[NB_OUT][VEC_SZ]:[sN[NB_OUT]:0, ...]);

    softmax_result
} 

pub fn softmax_stable
    <NB_IN: u32, EN_IN: u32, BU_IN: u32, 
    NB_OUT: u32, EN_OUT: u32, BU_OUT: u32, 
    NB_TABLE_EXP: u32, EN_TABLE_EXP: u32, BU_TABLE_EXP: u32,
    NB_TABLE_INV: u32, EN_TABLE_INV: u32, BU_TABLE_INV: u32,
    TABLE_SZ: u32, 
    VEC_SZ: u32,
    SHIFT_LIMIT: u32 = {NB_IN - u32:1},
    // EXP Accum
    BE_TABLE_EXP: s32 = {fixed_point_lib::binary_exponent(EN_TABLE_EXP, BU_TABLE_EXP)}, 
    NB_ACCUM: u32 = {NB_TABLE_EXP + std::clog2(VEC_SZ)},  
    BE_ACCUM: s32 = {BE_TABLE_EXP},
    EN_ACCUM: u32 = {fixed_point_lib::is_negative(BE_ACCUM)},      // exp is negative ACCUM
    BU_ACCUM: u32 = {fixed_point_lib::binary_uexponent(BE_ACCUM)}, // unsigned exp ACCUM
    // INV Multiplication 
    BE_TABLE_INV: s32 = {fixed_point_lib::binary_exponent(EN_TABLE_INV, BU_TABLE_INV)}, 
    EXP_SUM: s32 = {BE_TABLE_INV + BE_TABLE_INV},
    NB_MUL: u32 = {NB_TABLE_INV + NB_TABLE_INV}, 
    EN_MUL: u32 = {fixed_point_lib::is_negative(EXP_SUM)},
    BU_MUL: u32 = {fixed_point_lib::binary_uexponent(EXP_SUM)}>
    (y: sN[NB_IN][VEC_SZ]) -> sN[NB_OUT][VEC_SZ] {

    // Find max element
    let y_max = for (i, acc): (u32, sN[NB_IN]) in u32:0..VEC_SZ {
        std::max(y[i], acc)
    }((s32:-1 << SHIFT_LIMIT) as sN[NB_IN]);

    // Compute difference 
    let d_yi_ymax = for (i, z): (u32, sN[NB_IN][VEC_SZ]) in u32:0..VEC_SZ {
        update(z, i, fixed_point_fix::sub_already_widened<NB_IN, EN_IN, BU_IN, NB_IN, EN_IN, BU_IN>(y_max, y[i]) ) 
    }(sN[NB_IN][VEC_SZ]:[sN[NB_IN]:0, ...]);

    // Compute exp() with Lookup Tables
    let exp_result = for (i, exp_vec): (u32, sN[NB_TABLE_EXP][VEC_SZ]) in u32:0..VEC_SZ {
        let exp_table_idx = idx_from_real_val<TABLE_SZ, NB_IN>(d_yi_ymax[i]);
        update(exp_vec, i, EXP_TABLE[exp_table_idx]) 
    }(sN[NB_TABLE_EXP][VEC_SZ]:[sN[NB_TABLE_EXP]:0, ...]);

    // Sum all exponents
    let sum = for (i, acc): (u32, sN[NB_ACCUM]) in u32:0..VEC_SZ {
        fixed_point_fix::add_already_widened<NB_TABLE_EXP, EN_TABLE_EXP, BU_TABLE_EXP, NB_ACCUM, EN_ACCUM, BU_ACCUM>(exp_result[i], acc)
    }(sN[NB_ACCUM]:0);
    let truncate = fixed_point_fix::to_common_type<NB_TABLE_INV, BU_TABLE_INV, NB_ACCUM, EN_ACCUM, BU_ACCUM>(sum);
    let inv_exp_sum = INV_TABLE[idx_from_real_val<TABLE_SZ>(truncate)];

    // Compute softmax
    let softmax_result = for (i, inv_vec): (u32, sN[NB_OUT][VEC_SZ]) in u32:0..VEC_SZ {
        update(inv_vec, i, fixed_point_fix::to_common_type<NB_OUT, BU_OUT, NB_MUL, EN_MUL, BU_MUL>(
            fixed_point_fix::mul<NB_TABLE_INV, EN_TABLE_INV, BU_TABLE_INV, NB_TABLE_INV, EN_TABLE_INV, BU_TABLE_INV>
            (exp_result[i], inv_exp_sum)
        )) 
    }(sN[NB_OUT][VEC_SZ]:[sN[NB_OUT]:0, ...]);

    softmax_result
}

// ------------- Tests should be generated depending on the table precision/size

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
//     let x = sN[16][4]:[
//         sN[16]:1024,
//         sN[16]:1024,
//         sN[16]:1024,
//         sN[16]:1024
//     ];
//     let expected = sN[16][4]:[
//         sN[16]:256,  // Ideal 256
//         sN[16]:256,
//         sN[16]:256,
//         sN[16]:256
//     ];
//     assert_eq(expected, softmax_stable
//         <u32:16, u32:1, u32:10, 
//         u32:16, u32:1, u32:10, 
//         u32:18, u32:1, u32:10, 
//         u32:18, u32:1, u32:10,
//         u32:1024>(x));

//     let x = sN[16][4]:[
//         sN[16]:4096,
//         sN[16]:4096,
//         sN[16]:4096,
//         sN[16]:4096
//     ];
//     let expected = sN[16][4]:[
//         sN[16]:256,  // Ideal 256
//         sN[16]:256,
//         sN[16]:256,
//         sN[16]:256
//     ];
//     assert_eq(expected, softmax_stable        
//         <u32:16, u32:1, u32:10, 
//         u32:16, u32:1, u32:10, 
//         u32:18, u32:1, u32:10, 
//         u32:18, u32:1, u32:10,
//         u32:1024>(x));
// }


