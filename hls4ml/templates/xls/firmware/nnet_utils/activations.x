import std;

import ap_types.fixed_point_fix;
import ap_types.fixed_point_lib;


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
// ------------------------------- Argmax ---------------------------------

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
