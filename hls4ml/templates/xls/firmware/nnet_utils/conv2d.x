import std;

import ap_types.fixed_point_fix;
import ap_types.fixed_point_lib;

import nnet_utils.activations;


pub fn conv2d_latency
    <NB_IN: u32, EN_IN: u32, BU_IN: u32, 
    NB_OUT: u32, EN_OUT: u32, BU_OUT: u32,
    // Input Image
    IN_HEIGHT: u32, IN_WIDTH: u32, IN_CHANNELS: u32,
    // Kernel Dims
    KERN_HEIGHT: u32, KERN_WIDTH: u32, OUT_FILTERS: u32,
    // Output Image
    OUT_HEIGHT: u32 = {IN_HEIGHT - KERN_HEIGHT + u32:1}, OUT_WIDTH: u32 = {IN_HEIGHT - KERN_HEIGHT + u32:1},
    BE_OUT: s32 = {fixed_point_lib::binary_exponent(EN_OUT, BU_OUT)}, 
    BE_IN: s32 = {fixed_point_lib::binary_exponent(EN_IN, BU_IN)},      // binary exp X
    // Precision inference MUL
    BE_MUL: s32 = {BE_IN + BE_IN},                               // binary exp MUL
    NB_MUL: u32 = {NB_IN + NB_IN},                               // number bits MUL
    EN_MUL: u32 = {fixed_point_lib::is_negative(BE_MUL)},        // exp is negative MUL
    BU_MUL: u32 = {fixed_point_lib::binary_uexponent(BE_MUL)},   // unsigned exp MUL
    // Precision Inference CONV
    NB_CONV: u32 = {NB_MUL + std::clog2(KERN_HEIGHT*KERN_WIDTH*IN_CHANNELS)},    // number bits CONV
    BE_CONV: s32 = {BE_MUL},                                                     // binary exp CONV
    EN_CONV: u32 = {fixed_point_lib::is_negative(BE_CONV)},                      // exp is negative CONV
    BU_CONV: u32 = {fixed_point_lib::binary_uexponent(BE_CONV)},
    // Precision Inference BIAS
    NB_BIAS: u32 = {
        fixed_point_lib::aligned_width(NB_CONV, BE_CONV, NB_IN, BE_IN) +
        if fixed_point_lib::num_bits_overlapping(NB_CONV, BE_CONV, NB_IN, BE_IN) == u32:0 { u32:0 } else { u32:1 }},
    BE_BIAS: s32 = {std::min(BE_CONV, BE_IN)},         
    EN_BIAS: u32 = {fixed_point_lib::is_negative(BE_BIAS)},
    BU_BIAS: u32 = {fixed_point_lib::binary_uexponent(BE_BIAS)}
    >
    (x: sN[NB_IN][IN_HEIGHT][IN_WIDTH][IN_CHANNELS],
    W: sN[NB_IN][KERN_HEIGHT][KERN_WIDTH][IN_CHANNELS][OUT_FILTERS],
    b: sN[NB_IN][OUT_FILTERS])
    -> sN[NB_OUT][OUT_HEIGHT][OUT_WIDTH][OUT_FILTERS] {

    for (filter_idx, image): (u32, sN[NB_OUT][OUT_HEIGHT][OUT_WIDTH][OUT_FILTERS]) in u32:0..OUT_FILTERS {

        let computer_plane = for (i, plane): (u32, sN[NB_OUT][OUT_HEIGHT][OUT_WIDTH]) in u32:0..OUT_WIDTH {
            let computed_row = for (j, plane_row): (u32, sN[NB_OUT][OUT_HEIGHT]) in u32:0..OUT_HEIGHT {

                // Compute convolution across channels
                let conv_pixel = for (ch_idx, pixel): (u32, sN[NB_CONV]) in u32:0..IN_CHANNELS {
                    // Compute convolution for 1 channel
                    for (ii, ch_pixel): (u32, sN[NB_CONV]) in u32:0..KERN_WIDTH {
                        for (jj, acc): (u32, sN[NB_CONV]) in u32:0..KERN_HEIGHT {
                            fixed_point_fix::fmadd_already_widened
                                <NB_IN, EN_IN, BU_IN, 
                                NB_IN, EN_IN, BU_IN, 
                                NB_CONV, EN_CONV, BU_CONV>(x[ch_idx][i+ii][j+jj], W[filter_idx][ch_idx][ii][jj], acc)
                        }(ch_pixel)
                    }(pixel)
                }(sN[NB_CONV]:0);  

                // Add bias & truncate to output type
                let pixel_with_bias = fixed_point_fix::add<NB_CONV, EN_CONV, BU_CONV, NB_IN, EN_IN, BU_IN>(conv_pixel, b[filter_idx]);
                let common_pixel = fixed_point_fix::to_common_type<NB_OUT, BU_OUT, NB_BIAS, EN_BIAS, BU_BIAS>(pixel_with_bias);
                update(plane_row, j, common_pixel)

            }(sN[NB_OUT][OUT_HEIGHT]:[sN[NB_OUT]:0, ...]);
            update(plane, i, computed_row)

        }(sN[NB_OUT][OUT_HEIGHT][OUT_WIDTH]:[sN[NB_OUT][OUT_HEIGHT]:[sN[NB_OUT]:0, ...], ...]);
        update(image, filter_idx, computer_plane)

    // Whole image initialization
    }(sN[NB_OUT][OUT_HEIGHT][OUT_WIDTH][OUT_FILTERS]:[
        sN[NB_OUT][OUT_HEIGHT][OUT_WIDTH]:[
            sN[NB_OUT][OUT_HEIGHT]:[sN[NB_OUT]:0, 
            ...], ...], ...])
}

pub fn conv_relu_latency
    <NB_IN: u32, EN_IN: u32, BU_IN: u32, 
    NB_OUT: u32, EN_OUT: u32, BU_OUT: u32,
    // Input Image
    IN_HEIGHT: u32, IN_WIDTH: u32, IN_CHANNELS: u32,
    // Kernel Dims
    KERN_HEIGHT: u32, KERN_WIDTH: u32, OUT_FILTERS: u32,
    // Output Image
    OUT_HEIGHT: u32 = {IN_HEIGHT - KERN_HEIGHT + u32:1}, OUT_WIDTH: u32 = {IN_HEIGHT - KERN_HEIGHT + u32:1},
    BE_OUT: s32 = {fixed_point_lib::binary_exponent(EN_OUT, BU_OUT)}, 
    BE_IN: s32 = {fixed_point_lib::binary_exponent(EN_IN, BU_IN)},      // binary exp X
    // Precision inference MUL
    BE_MUL: s32 = {BE_IN + BE_IN},                               // binary exp MUL
    NB_MUL: u32 = {NB_IN + NB_IN},                               // number bits MUL
    EN_MUL: u32 = {fixed_point_lib::is_negative(BE_MUL)},        // exp is negative MUL
    BU_MUL: u32 = {fixed_point_lib::binary_uexponent(BE_MUL)},   // unsigned exp MUL
    // Precision Inference CONV
    NB_CONV: u32 = {NB_MUL + std::clog2(KERN_HEIGHT*KERN_WIDTH*IN_CHANNELS)},    // number bits CONV
    BE_CONV: s32 = {BE_MUL},                                                     // binary exp CONV
    EN_CONV: u32 = {fixed_point_lib::is_negative(BE_CONV)},                      // exp is negative CONV
    BU_CONV: u32 = {fixed_point_lib::binary_uexponent(BE_CONV)},
    // Precision Inference BIAS
    NB_BIAS: u32 = {
        fixed_point_lib::aligned_width(NB_CONV, BE_CONV, NB_IN, BE_IN) +
        if fixed_point_lib::num_bits_overlapping(NB_CONV, BE_CONV, NB_IN, BE_IN) == u32:0 { u32:0 } else { u32:1 }},
    BE_BIAS: s32 = {std::min(BE_CONV, BE_IN)},         
    EN_BIAS: u32 = {fixed_point_lib::is_negative(BE_BIAS)},
    BU_BIAS: u32 = {fixed_point_lib::binary_uexponent(BE_BIAS)}
    >
    (x: sN[NB_IN][IN_HEIGHT][IN_WIDTH][IN_CHANNELS],
    W: sN[NB_IN][KERN_HEIGHT][KERN_WIDTH][IN_CHANNELS][OUT_FILTERS],
    b: sN[NB_IN][OUT_FILTERS])
    -> sN[NB_OUT][OUT_HEIGHT][OUT_WIDTH][OUT_FILTERS] {

    for (filter_idx, image): (u32, sN[NB_OUT][OUT_HEIGHT][OUT_WIDTH][OUT_FILTERS]) in u32:0..OUT_FILTERS {

        let computer_plane = for (i, plane): (u32, sN[NB_OUT][OUT_HEIGHT][OUT_WIDTH]) in u32:0..OUT_WIDTH {
            let computed_row = for (j, plane_row): (u32, sN[NB_OUT][OUT_HEIGHT]) in u32:0..OUT_HEIGHT {

                // Compute convolution across channels
                let conv_pixel = for (ch_idx, pixel): (u32, sN[NB_CONV]) in u32:0..IN_CHANNELS {
                    // Compute convolution for 1 channel
                    for (ii, ch_pixel): (u32, sN[NB_CONV]) in u32:0..KERN_WIDTH {
                        for (jj, acc): (u32, sN[NB_CONV]) in u32:0..KERN_HEIGHT {
                            fixed_point_fix::fmadd_already_widened
                                <NB_IN, EN_IN, BU_IN, 
                                NB_IN, EN_IN, BU_IN, 
                                NB_CONV, EN_CONV, BU_CONV>(x[ch_idx][i+ii][j+jj], W[filter_idx][ch_idx][ii][jj], acc)
                        }(ch_pixel)
                    }(pixel)
                }(sN[NB_CONV]:0);  

                // Add bias & truncate to output type
                let pixel_with_bias = fixed_point_fix::add<NB_CONV, EN_CONV, BU_CONV, NB_IN, EN_IN, BU_IN>(conv_pixel, b[filter_idx]);
                let common_pixel = fixed_point_fix::to_common_type<NB_OUT, BU_OUT, NB_BIAS, EN_BIAS, BU_BIAS>(pixel_with_bias);
                let relu_pixel = activations::relu_1elem<NB_OUT>(common_pixel);
                update(plane_row, j, relu_pixel)

            }(sN[NB_OUT][OUT_HEIGHT]:[sN[NB_OUT]:0, ...]);
            update(plane, i, computed_row)

        }(sN[NB_OUT][OUT_HEIGHT][OUT_WIDTH]:[sN[NB_OUT][OUT_HEIGHT]:[sN[NB_OUT]:0, ...], ...]);
        update(image, filter_idx, computer_plane)

    // Whole image initialization
    }(sN[NB_OUT][OUT_HEIGHT][OUT_WIDTH][OUT_FILTERS]:[
        sN[NB_OUT][OUT_HEIGHT][OUT_WIDTH]:[
            sN[NB_OUT][OUT_HEIGHT]:[sN[NB_OUT]:0, 
            ...], ...], ...])
}



#[test]
fn conv2d_latency_test_uniform_io() { 
    // x = 
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    let x  = sN[16][5][5][1]:[sN[16][5][5]:[sN[16][5]:[sN[16]:1024, ...], ...], ...];

    // w = 
    //  | 1, 1, 1|
    //  | 2, 2, 2|
    //  | 3, 3, 3|
    let w = sN[16][3][3][1][1]:[[[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:2048, sN[16]:2048, sN[16]:2048],
        [sN[16]:3072, sN[16]:3072, sN[16]:3072],
    ]]];
    let b = sN[16][1]:[sN[16]:0];

    // expected = 
    //  | 18, 18, 18|
    //  | 18, 18, 18|
    //  | 18, 18, 18|
    let expected = sN[16][3][3][1]:[[
        [sN[16]:18432, sN[16]:18432, sN[16]:18432],
        [sN[16]:18432, sN[16]:18432, sN[16]:18432],
        [sN[16]:18432, sN[16]:18432, sN[16]:18432],
    ]];
    assert_eq(expected, conv2d_latency<u32:16, u32:1, u32:10, u32:16, u32:1, u32:10>(x, w, b));
}

#[test]
fn conv2d_latency_test_bias() { 
    // x = 
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    let x  = sN[16][5][5][1]:[sN[16][5][5]:[sN[16][5]:[sN[16]:1024, ...], ...], ...];

    // w = 
    //  | 1, 1, 1|
    //  | 2, 2, 2|
    //  | 3, 3, 3|
    let w = sN[16][3][3][1][1]:[[[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:2048, sN[16]:2048, sN[16]:2048],
        [sN[16]:3072, sN[16]:3072, sN[16]:3072],
    ]]];
    // b = | 1 |
    let b = sN[16][1]:[sN[16]:1024];

    // expected = 
    //  | 19, 19, 19|
    //  | 19, 19, 19|
    //  | 19, 19, 19|
    let expected = sN[16][3][3][1]:[[
        [sN[16]:19456, sN[16]:19456, sN[16]:19456],
        [sN[16]:19456, sN[16]:19456, sN[16]:19456],
        [sN[16]:19456, sN[16]:19456, sN[16]:19456],
    ]];
    assert_eq(expected, conv2d_latency<u32:16, u32:1, u32:10, u32:16, u32:1, u32:10>(x, w, b));
}

#[test]
fn conv2d_latency_test_pattern() { 
    // x = 
    //  | 1, 1, 1, 1, 1|
    //  | 0, 0, 0, 0, 0|
    //  | 2, 2, 2, 2, 2|
    //  | 0, 0, 0, 0, 0|
    //  | 1, 1, 1, 1, 1|
    let x = sN[16][5][5][1]:[[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:2048, sN[16]:2048, sN[16]:2048, sN[16]:2048, sN[16]:2048],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ]];

    // w = 
    //  | 1, 1, 1|
    //  | 2, 2, 2|
    //  | 3, 3, 3|
    let w = sN[16][3][3][1][1]:[[[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:2048, sN[16]:2048, sN[16]:2048],
        [sN[16]:3072, sN[16]:3072, sN[16]:3072],
    ]]];
    // b = | 0 |
    let b = sN[16][1]:[sN[16]:0];

    // expected = 
    //  | 21, 21, 21|
    //  | 12, 12, 12|
    //  | 15, 15, 15|
    let expected = sN[16][3][3][1]:[[
        [sN[16]:21504, sN[16]:21504, sN[16]:21504],
        [sN[16]:12288, sN[16]:12288, sN[16]:12288],
        [sN[16]:15360, sN[16]:15360, sN[16]:15360],
    ]];
    assert_eq(expected, conv2d_latency<u32:16, u32:1, u32:10, u32:16, u32:1, u32:10>(x, w, b));
}

#[test]
fn conv2d_latency_test_mutiple_filters() { 
    // x = 
    //  | 1, 1, 1, 1, 1|
    //  | 0, 0, 0, 0, 0|
    //  | 2, 2, 2, 2, 2|
    //  | 0, 0, 0, 0, 0|
    //  | 1, 1, 1, 1, 1|
    let x = sN[16][5][5][1]:[[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:2048, sN[16]:2048, sN[16]:2048, sN[16]:2048, sN[16]:2048],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ]];

    // w = 
    //  | 1, 1, 1|  | 1, 1, 1|  | 0, 0, 0|
    //  | 2, 2, 2|  | 1, 1, 1|  | 0, 0, 0|
    //  | 3, 3, 3|  | 1, 1, 1|  | 0, 0, 0|
    let w = sN[16][3][3][1][3]:[[[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:2048, sN[16]:2048, sN[16]:2048],
        [sN[16]:3072, sN[16]:3072, sN[16]:3072],
    ]],[[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ]],[[
        [sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0],
    ]]];
    // b = | 0, 0 ,-2|
    let b = sN[16][3]:[sN[16]:0, sN[16]:0, sN[16]:-2048];

    // expected = 
    //  | 21, 21, 21|  | 6, 6, 6|  | 0, 0, 0|
    //  | 12, 12, 12|  | 9, 9, 9|  | 0, 0, 0|
    //  | 15, 15, 15|  | 6, 6, 6|  | 0, 0, 0|
    let expected = sN[16][3][3][3]:[[
        [sN[16]:21504, sN[16]:21504, sN[16]:21504],
        [sN[16]:12288, sN[16]:12288, sN[16]:12288],
        [sN[16]:15360, sN[16]:15360, sN[16]:15360]
    ],[
        [sN[16]:9216, sN[16]:9216, sN[16]:9216],
        [sN[16]:6144, sN[16]:6144, sN[16]:6144],
        [sN[16]:9216, sN[16]:9216, sN[16]:9216]
    ],[
        [sN[16]:-2048, sN[16]:-2048, sN[16]:-2048],
        [sN[16]:-2048, sN[16]:-2048, sN[16]:-2048],
        [sN[16]:-2048, sN[16]:-2048, sN[16]:-2048],
    ]];
    assert_eq(expected, conv2d_latency<u32:16, u32:1, u32:10, u32:16, u32:1, u32:10>(x, w, b));
}

#[test]
fn conv2d_latency_test_mutiple_channels() { 
    // x = 
    //  | 1, 1, 1, 1, 1|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 0, 0, 0, 0, 0|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 2, 2, 2, 2, 2|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 0, 0, 0, 0, 0|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 1, 1, 1, 1, 1|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    let x = sN[16][5][5][3]:[[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:2048, sN[16]:2048, sN[16]:2048, sN[16]:2048, sN[16]:2048],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
    ],];

    // w = 
    //  | 1, 1, 1|  | 1, 1, 1|  | 0, 0, 0|
    //  | 2, 2, 2|  | 1, 1, 1|  | 0, 0, 0|
    //  | 1, 1, 1|  | 1, 1, 1|  | 0, 0, 0|
    let w = sN[16][3][3][3][1]:[[[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:2048, sN[16]:2048, sN[16]:2048],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0],
    ]]];
    // b = | 1 |
    let b = sN[16][1]:[sN[16]:0];

    // expected = 
    //  | 18, 18, 18|
    //  | 21, 21, 21|
    //  | 18, 18, 18|
    let expected = sN[16][3][3][1]:[[
        [sN[16]:18432, sN[16]:18432, sN[16]:18432],
        [sN[16]:21504, sN[16]:21504, sN[16]:21504],
        [sN[16]:18432, sN[16]:18432, sN[16]:18432]
    ]];
    assert_eq(expected, conv2d_latency<u32:16, u32:1, u32:10, u32:16, u32:1, u32:10>(x, w, b));
}

#[test]
fn conv2d_latency_test_mutiple_channels_and_filters() { 
    // x = 
    //  | 1, 1, 1, 1, 1|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 0, 0, 0, 0, 0|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 2, 2, 2, 2, 2|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 0, 0, 0, 0, 0|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 1, 1, 1, 1, 1|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    let x = sN[16][5][5][3]:[[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:2048, sN[16]:2048, sN[16]:2048, sN[16]:2048, sN[16]:2048],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
    ],];

    // w = 
    //  | 1, 1, 1|  | 1, 1, 1|  | 0, 0, 0|
    //  | 2, 2, 2|  | 1, 1, 1|  | 0, 0, 0|
    //  | 1, 1, 1|  | 1, 1, 1|  | 0, 0, 0|

    //  | 1, 1, 1|  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|  | 1, 1, 1|

    //  | 0, 0, 0|  | 0, 0, 0|  | 0, 0, 0|
    //  | 0, 0, 0|  | 0, 0, 0|  | 0, 0, 0|
    //  | 0, 0, 0|  | 0, 0, 0|  | 0, 0, 0|
    let w = sN[16][3][3][3][3]:[
    [[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:2048, sN[16]:2048, sN[16]:2048],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0],
    ]],

    [[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ]],

    [[
        [sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0],
    ],[
        [sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0],
    ],[
        [sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0],
    ]],];
    // b = | 0, 0, 0|
    let b = sN[16][3]:[sN[16]:0, sN[16]:0, sN[16]:0];

    // expected = 
    //  | 18, 18, 18|  | 18, 18, 18|  | 0, 0, 0|
    //  | 21, 21, 21|  | 15, 15, 15|  | 0, 0, 0|
    //  | 18, 18, 18|  | 18, 18, 18|  | 0, 0, 0|
    let expected = sN[16][3][3][3]:[[
        [sN[16]:18432, sN[16]:18432, sN[16]:18432],
        [sN[16]:21504, sN[16]:21504, sN[16]:21504],
        [sN[16]:18432, sN[16]:18432, sN[16]:18432]
    ],[
        [sN[16]:18432, sN[16]:18432, sN[16]:18432],
        [sN[16]:15360, sN[16]:15360, sN[16]:15360],
        [sN[16]:18432, sN[16]:18432, sN[16]:18432]
    ],[
        [sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0],
    ]];

    assert_eq(expected, conv2d_latency<u32:16, u32:1, u32:10, u32:16, u32:1, u32:10>(x, w, b));
}

#[test]
fn conv2d_latency_test_two_layers() { 
    // x = 
    //  | 1, 1, 1, 1, 1|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 0, 0, 0, 0, 0|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 2, 2, 2, 2, 2|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 0, 0, 0, 0, 0|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 1, 1, 1, 1, 1|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    let x = sN[16][5][5][3]:[[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:2048, sN[16]:2048, sN[16]:2048, sN[16]:2048, sN[16]:2048],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
    ],];

    // w = 
    //  | 1, 1, 1|  | 1, 1, 1|  | 0, 0, 0|
    //  | 2, 2, 2|  | 1, 1, 1|  | 0, 0, 0|
    //  | 1, 1, 1|  | 1, 1, 1|  | 0, 0, 0|

    //  | 1, 1, 1|  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|  | 1, 1, 1|
    let w0 = sN[16][3][3][3][2]:[
    [[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:2048, sN[16]:2048, sN[16]:2048],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0],
    ]],

    [[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ]]];
    // b = | -17, -17|
    let b0 = sN[16][2]:[sN[16]:-17408, sN[16]:-17408];

    // w1 = 
    //  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|
    let w1 = sN[16][3][3][2][1]:[
    [[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ]]];
    // b = | 0 |
    let b1 = sN[16][1]:[sN[16]:0];

    // expected = | 18 |
    let expected = sN[16][1][1][1]:[[
        [sN[16]:18432],
    ]];

    let z0 = conv2d_latency<u32:16, u32:1, u32:10, u32:16, u32:1, u32:10>(x, w0, b0);
    let z1 = conv2d_latency<u32:16, u32:1, u32:10, u32:16, u32:1, u32:10>(z0, w1, b1);
    assert_eq(expected, z1);
}

#[test]
fn conv_relu_latency_test_two_layers() { 
    // x = 
    //  | 1, 1, 1, 1, 1|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 0, 0, 0, 0, 0|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 2, 2, 2, 2, 2|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 0, 0, 0, 0, 0|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 1, 1, 1, 1, 1|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    let x = sN[16][5][5][3]:[[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:2048, sN[16]:2048, sN[16]:2048, sN[16]:2048, sN[16]:2048],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0, sN[16]:0],
    ],];

    // w = 
    //  | 1, 1, 1|  | 1, 1, 1|  | 0, 0, 0|
    //  | 2, 2, 2|  | 1, 1, 1|  | 0, 0, 0|
    //  | 1, 1, 1|  | 1, 1, 1|  | 0, 0, 0|

    //  | 1, 1, 1|  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|  | 1, 1, 1|
    let w0 = sN[16][3][3][3][2]:[
    [[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:2048, sN[16]:2048, sN[16]:2048],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0],
        [sN[16]:0, sN[16]:0, sN[16]:0],
    ]],

    [[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ]]];
    // b = | -17, -17|
    let b0 = sN[16][2]:[sN[16]:-17408, sN[16]:-17408];

    // w1 = 
    //  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|
    let w1 = sN[16][3][3][2][1]:[
    [[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ],[
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
        [sN[16]:1024, sN[16]:1024, sN[16]:1024],
    ]]];
    // b = | 0 |
    let b1 = sN[16][1]:[sN[16]:0];

    // expected = | 18 |
    let expected = sN[16][1][1][1]:[[
        [sN[16]:24576],
    ]];

    let z0 = conv_relu_latency<u32:16, u32:1, u32:10, u32:16, u32:1, u32:10>(x, w0, b0);
    let z1 = conv_relu_latency<u32:16, u32:1, u32:10, u32:16, u32:1, u32:10>(z0, w1, b1);
    assert_eq(expected, z1);
}