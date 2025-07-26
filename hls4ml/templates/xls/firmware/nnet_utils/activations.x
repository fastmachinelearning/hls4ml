import std;

import ap_types.fixed_point_fix;
import ap_types.fixed_point_lib;

const NB_COMMON = u32:16;
const EN_COMMON = u32:1;
const BU_COMMON = u32:10;
const BE_COMMON = s32:-10;


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

pub fn argmax
    <NB: u32, EN: u32, BU: u32, VEC_SZ: u32,
    BE: s32 = {fixed_point_lib::binary_exponent(EN, BU)},
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
fn relu_test() {
    let x = sN[NB_COMMON][2]:[
        sN[NB_COMMON]:1536, 
        sN[NB_COMMON]:1024
    ]; 
    let expected = sN[NB_COMMON][2]:[
        sN[NB_COMMON]:1536, 
        sN[NB_COMMON]:1024
    ];  
    assert_eq(expected, relu<NB_COMMON>(x));

    let x = sN[NB_COMMON][4]:[
        sN[NB_COMMON]:-1536, 
        sN[NB_COMMON]:-1024,
        sN[NB_COMMON]:0,
        sN[NB_COMMON]:-1024
    ]; 
    let expected = sN[NB_COMMON][4]:[
        sN[NB_COMMON]:0, 
        sN[NB_COMMON]:0,
        sN[NB_COMMON]:0,
        sN[NB_COMMON]:0,
    ];  
    assert_eq(expected, relu<NB_COMMON>(x));

    let x = sN[NB_COMMON][4]:[
        sN[NB_COMMON]:-1536, 
        sN[NB_COMMON]:-1024,
        sN[NB_COMMON]:1024,
        sN[NB_COMMON]:-1024
    ]; 
    let expected = sN[NB_COMMON][4]:[
        sN[NB_COMMON]:0, 
        sN[NB_COMMON]:0,
        sN[NB_COMMON]:1024,
        sN[NB_COMMON]:0,
    ];  
    assert_eq(expected, relu<NB_COMMON>(x));
}