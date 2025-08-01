import std;

import ap_types.fixed_point_fix;
import ap_types.fixed_point_lib;
import ap_types.lookup_tables;


// =========================================================================
// --------------------------------- ReLU ----------------------------------

pub fn relu_1elem
    <NB: u32>
    (fxd_x: sN[NB]) -> sN[NB] {
    
    if (fxd_x > sN[NB]:0) 
        { fxd_x } 
    else 
        { sN[NB]:0 }
} 

pub fn relu
    <NB: u32, VEC_SZ: u32>
    (y: sN[NB][VEC_SZ]) -> sN[NB][VEC_SZ] {
    
    for (i, z): (u32, sN[NB][VEC_SZ]) in u32:0..VEC_SZ {
        let with_relu = relu_1elem<NB>(y[i]);
        update(z, i, with_relu)
    }(y) 
} 

#[test]
fn relu_test() {
    let x = sN[16][2]:[
        sN[16]:1536, 
        sN[16]:1024
    ]; 
    let expected = sN[16][2]:[
        sN[16]:1536, 
        sN[16]:1024
    ];  
    assert_eq(expected, relu<u32:16>(x));

    let x = sN[16][4]:[
        sN[16]:-1536, 
        sN[16]:-1024,
        sN[16]:0,
        sN[16]:-1024
    ]; 
    let expected = sN[16][4]:[
        sN[16]:0, 
        sN[16]:0,
        sN[16]:0,
        sN[16]:0,
    ];  
    assert_eq(expected, relu<u32:16>(x));

    let x = sN[16][4]:[
        sN[16]:-1536, 
        sN[16]:-1024,
        sN[16]:1024,
        sN[16]:-1024
    ]; 
    let expected = sN[16][4]:[
        sN[16]:0, 
        sN[16]:0,
        sN[16]:1024,
        sN[16]:0,
    ];  
    assert_eq(expected, relu<u32:16>(x));
}

// =========================================================================
// ------------------------------- Softmax ---------------------------------

fn get_exp
    <NB_IN: u32, EN_IN: u32, BU_IN: u32, 
    NB_OUT: u32, EN_OUT: u32, BU_OUT: u32, 
    TABLE_SZ: u32, 
    VEC_SZ: u32>
    (y: sN[NB_IN][VEC_SZ]) -> sN[NB_OUT][VEC_SZ] {
    
    // Compute exp() with Lookup Tables
    let exp_result = for (i, exp_vec): (u32, sN[NB_OUT][VEC_SZ]) in u32:0..VEC_SZ {
        let exp_table_idx = lookup_tables::idx_from_real_val<TABLE_SZ, NB_IN>(y[i]);
        update(exp_vec, i, lookup_tables::EXP_TABLE[exp_table_idx]) 
    }(sN[NB_OUT][VEC_SZ]:[sN[NB_OUT]:0, ...]);

    exp_result
} 

#[test]
fn get_exp_test() {
    let x = sN[16][4]:[
        sN[16]:1024,
        sN[16]:1024,
        sN[16]:1024,
        sN[16]:1024
    ];
    let expected = sN[18][4]:[
        sN[18]:2784,
        sN[18]:2784,
        sN[18]:2784,
        sN[18]:2784
    ];
    assert_eq(expected, get_exp<u32:16, u32:1, u32:10, u32:18, u32:1, u32:10, u32:1024>(x));
}

fn get_accum
    <NB_IN: u32, EN_IN: u32, BU_IN: u32, 
    NB_OUT: u32, EN_OUT: u32, BU_OUT: u32, 
    TABLE_SZ: u32, 
    VEC_SZ: u32,
    // EXP Accum
    BE_OUT: s32 = {fixed_point_lib::binary_exponent(EN_OUT, BU_OUT)}, 
    NB_ACCUM: u32 = {NB_OUT + std::clog2(VEC_SZ)},  
    BE_ACCUM: s32 = {BE_OUT},
    EN_ACCUM: u32 = {fixed_point_lib::is_negative(BE_ACCUM)},      // exp is negative ACCUM
    BU_ACCUM: u32 = {fixed_point_lib::binary_uexponent(BE_ACCUM)}> // unsigned exp ACCUM
    (y: sN[NB_IN][VEC_SZ]) -> sN[NB_OUT] {
    
    // Compute exp() with Lookup Tables
    let exp_result = for (i, exp_vec): (u32, sN[NB_OUT][VEC_SZ]) in u32:0..VEC_SZ {
        let exp_table_idx = lookup_tables::idx_from_real_val<TABLE_SZ, NB_IN>(y[i]);
        update(exp_vec, i, lookup_tables::EXP_TABLE[exp_table_idx]) 
    }(sN[NB_OUT][VEC_SZ]:[sN[NB_OUT]:0, ...]);

    let sum = for (i, acc): (u32, sN[NB_ACCUM]) in u32:0..VEC_SZ {
        fixed_point_fix::add_already_widened<NB_OUT, EN_OUT, BU_OUT, NB_ACCUM, EN_ACCUM, BU_ACCUM>(exp_result[i], acc)
    }(sN[NB_ACCUM]:0);
    let truncate = fixed_point_fix::to_common_type<NB_OUT, BU_OUT, NB_ACCUM, EN_ACCUM, BU_ACCUM>(sum);
    let inv_exp_sum = lookup_tables::INV_TABLE[lookup_tables::idx_from_real_val<TABLE_SZ>(truncate)];

    inv_exp_sum
} 

#[test]
fn get_accum_test() {
    let x = sN[16][4]:[
        sN[16]:1024,
        sN[16]:1024,
        sN[16]:1024,
        sN[16]:1024
    ];
    let expected = sN[18]:95; // ideal 95
    assert_eq(expected, get_accum<u32:16, u32:1, u32:10, u32:18, u32:1, u32:10, u32:1024>(x));
}

pub fn softmax_latency
    <NB_IN: u32, EN_IN: u32, BU_IN: u32, 
    NB_OUT: u32, EN_OUT: u32, BU_OUT: u32, 
    TABLE_SZ: u32, 
    VEC_SZ: u32,
    // EXP Accum
    BE_OUT: s32 = {fixed_point_lib::binary_exponent(EN_OUT, BU_OUT)}, 
    NB_ACCUM: u32 = {NB_OUT + std::clog2(VEC_SZ)},  
    BE_ACCUM: s32 = {BE_OUT},
    EN_ACCUM: u32 = {fixed_point_lib::is_negative(BE_ACCUM)},      // exp is negative ACCUM
    BU_ACCUM: u32 = {fixed_point_lib::binary_uexponent(BE_ACCUM)}, // unsigned exp ACCUM
    // INV Multiplication 
    EXP_SUM: s32 = {BE_OUT + BE_OUT},
    NB_MUL: u32 = {NB_OUT + NB_OUT}, 
    EN_MUL: u32 = {fixed_point_lib::is_negative(EXP_SUM)},
    BU_MUL: u32 = {fixed_point_lib::binary_uexponent(EXP_SUM)}>
    (y: sN[NB_IN][VEC_SZ]) -> sN[NB_OUT][VEC_SZ] {
    
    // Compute exp() with Lookup Tables
    let exp_result = for (i, exp_vec): (u32, sN[NB_OUT][VEC_SZ]) in u32:0..VEC_SZ {
        let exp_table_idx = lookup_tables::idx_from_real_val<TABLE_SZ, NB_IN>(y[i]);
        update(exp_vec, i, lookup_tables::EXP_TABLE[exp_table_idx]) 
    }(sN[NB_OUT][VEC_SZ]:[sN[NB_OUT]:0, ...]);

    let sum = for (i, acc): (u32, sN[NB_ACCUM]) in u32:0..VEC_SZ {
        fixed_point_fix::add_already_widened<NB_OUT, EN_OUT, BU_OUT, NB_ACCUM, EN_ACCUM, BU_ACCUM>(exp_result[i], acc)
    }(sN[NB_ACCUM]:0);
    let truncate = fixed_point_fix::to_common_type<NB_OUT, BU_OUT, NB_ACCUM, EN_ACCUM, BU_ACCUM>(sum);
    let inv_exp_sum = lookup_tables::INV_TABLE[lookup_tables::idx_from_real_val<TABLE_SZ>(truncate)];

    let inv_result = for (i, inv_vec): (u32, sN[NB_OUT][VEC_SZ]) in u32:0..VEC_SZ {
        update(inv_vec, i, fixed_point_fix::to_common_type<NB_OUT, BU_OUT, NB_MUL, EN_MUL, BU_MUL>(fixed_point_fix::mul<NB_OUT, EN_OUT, BU_OUT, NB_OUT, EN_OUT, BU_OUT>(exp_result[i], inv_exp_sum))) 
    }(exp_result);

    inv_result
} 

pub fn argmax
    <NB_IN: u32, EN_IN:u32, BU_IN:u32,
    NB_OUT: u32, EN_OUT: u32, BU_OUT: u32, VEC_SZ: u32,
    SHIFT_LIMIT: u32 = {NB_IN - u32:1}>
    (y: sN[NB_IN][VEC_SZ]) -> sN[NB_OUT][VEC_SZ] {
    
    let max_significand = for (i, acc): (u32, sN[NB_IN]) in u32:0..VEC_SZ {
        std::max(y[i], acc)
    }((s32:-1 << SHIFT_LIMIT) as sN[NB_IN]);

    for (i, z): (u32, sN[NB_OUT][VEC_SZ]) in u32:0..VEC_SZ {
        if y[i] == max_significand { 
            update(z, i, (u32:1<<BU_OUT) as sN[NB_OUT]) 
        } else {
            z
        }
    }(sN[NB_OUT][VEC_SZ]:[sN[NB_OUT]:0, ...])
} 

#[test]
fn softmax_latency_test() {
    let x = sN[16][4]:[
        sN[16]:1024,
        sN[16]:1024,
        sN[16]:1024,
        sN[16]:1024
    ];
    let expected = sN[18][4]:[
        sN[18]:258,  // Ideal 256
        sN[18]:258,
        sN[18]:258,
        sN[18]:258
    ];
    assert_eq(expected, softmax_latency<u32:16, u32:1, u32:10, u32:18, u32:1, u32:10, u32:1024>(x));
}

#[test]
fn argmax_test() {
    let x = sN[16][2]:[
        sN[16]:1536, 
        sN[16]:1024
    ]; 
    let expected = sN[18][2]:[
        sN[18]:1024, 
        sN[18]:0
    ];  
    assert_eq(expected, argmax<u32:16, u32:1, u32:10, u32:18, u32:1, u32:10>(x));

    let x = sN[16][4]:[
        sN[16]:-1536, 
        sN[16]:-1024,
        sN[16]:0,
        sN[16]:-1024
    ]; 
    let expected = sN[18][4]:[
        sN[18]:0, 
        sN[18]:0,
        sN[18]:1024,
        sN[18]:0,
    ];  
    assert_eq(expected, argmax<u32:16, u32:1, u32:10, u32:18, u32:1, u32:10>(x));

    let x = sN[16][4]:[
        sN[16]:-1536, 
        sN[16]:-1024,
        sN[16]:-512,
        sN[16]:-1024
    ]; 
    let expected = sN[18][4]:[
        sN[18]:0, 
        sN[18]:0,
        sN[18]:1024,
        sN[18]:0,
    ];  
    assert_eq(expected, argmax<u32:16, u32:1, u32:10, u32:18, u32:1, u32:10>(x));
}
