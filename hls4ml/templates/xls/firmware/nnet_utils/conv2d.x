import std;
import fixed_point;

import ap_types.fixed_point_util;
import nnet_utils.activations;

type FixedPoint = fixed_point::FixedPoint;
type RoundingMode = fixed_point_util::RoundingMode;
type OverflowMode = fixed_point_util::OverflowMode;

pub fn conv2d_latency
    <NB_OUT: u32, BE_OUT: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    NB_IN: u32, BE_IN: s32, 
    // Input Image
    IN_HEIGHT: u32, IN_WIDTH: u32, IN_CHANNELS: u32,
    // Kernel Dims
    KERN_HEIGHT: u32, KERN_WIDTH: u32, OUT_FILTERS: u32,
    // Output Image
    OUT_HEIGHT: u32 = {IN_HEIGHT - KERN_HEIGHT + u32:1}, OUT_WIDTH: u32 = {IN_HEIGHT - KERN_HEIGHT + u32:1},
    // Precision inference MUL
    BE_MUL: s32 = {BE_IN + BE_IN},                               // binary exp MUL
    NB_MUL: u32 = {NB_IN + NB_IN},                               // number bits MUL
    // Precision Inference CONV
    NB_CONV: u32 = {NB_MUL + std::clog2(KERN_HEIGHT*KERN_WIDTH*IN_CHANNELS)},    // number bits CONV
    BE_CONV: s32 = {BE_MUL},                                                     // binary exp CONV
    // Precision Inference BIAS
    NB_BIAS: u32 = {
        fixed_point_util::aligned_width(NB_CONV, BE_CONV, NB_IN, BE_IN) +
        if fixed_point_util::num_bits_overlapping(NB_CONV, BE_CONV, NB_IN, BE_IN) == u32:0 { u32:0 } else { u32:1 }},
    BE_BIAS: s32 = {std::min(BE_CONV, BE_IN)},
    >
    (x: FixedPoint<NB_IN, BE_IN>[IN_HEIGHT][IN_WIDTH][IN_CHANNELS],
    W: FixedPoint<NB_IN, BE_IN>[KERN_HEIGHT][KERN_WIDTH][IN_CHANNELS][OUT_FILTERS],
    b: FixedPoint<NB_IN, BE_IN>[OUT_FILTERS])
    -> FixedPoint<NB_OUT, BE_OUT>[OUT_HEIGHT][OUT_WIDTH][OUT_FILTERS] {

    for (filter_idx, image) in 0..OUT_FILTERS {

        let computer_plane = for (i, plane) in 0..OUT_WIDTH {
            let computed_row = for (j, plane_row) in 0..OUT_HEIGHT {

                // Compute convolution across channels
                let conv_pixel = for (ch_idx, pixel) in 0..IN_CHANNELS {
                    // Compute convolution for 1 channel
                    for (ii, ch_pixel) in 0..KERN_WIDTH {
                        for (jj, acc) in 0..KERN_HEIGHT {
                            fixed_point_util::fmadd_already_widened
                                <NB_IN, BE_IN, 
                                NB_IN, BE_IN, 
                                NB_CONV, BE_CONV>(x[ch_idx][i+ii][j+jj], W[filter_idx][ch_idx][ii][jj], acc)
                        }(ch_pixel)
                    }(pixel)
                }(zero!<FixedPoint<NB_CONV, BE_CONV>>());  

                // Add bias & truncate to output type
                let pixel_with_bias = fixed_point::add<NB_CONV, BE_CONV, NB_IN, BE_IN>(conv_pixel, b[filter_idx]);
                let out_pixel = fixed_point_util::resize<NB_OUT, BE_OUT, ROUNDING, OVERFLOW>(pixel_with_bias);
                update(plane_row, j, out_pixel)

            }(zero!<FixedPoint<NB_OUT,BE_OUT>[OUT_HEIGHT]>());
            update(plane, i, computed_row)

        }(zero!<FixedPoint<NB_OUT,BE_OUT>[OUT_HEIGHT][OUT_WIDTH]>());
        update(image, filter_idx, computer_plane)

    // Whole image initialization
    }(zero!<FixedPoint<NB_OUT,BE_OUT>[OUT_HEIGHT][OUT_WIDTH][OUT_FILTERS]>())
}

// TODO: used only for tests
pub fn conv_relu_latency
    <NB_OUT: u32, BE_OUT: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    NB_IN: u32, BE_IN: s32, 
    // Input Image
    IN_HEIGHT: u32, IN_WIDTH: u32, IN_CHANNELS: u32,
    // Kernel Dims
    KERN_HEIGHT: u32, KERN_WIDTH: u32, OUT_FILTERS: u32,
    // Output Image
    OUT_HEIGHT: u32 = {IN_HEIGHT - KERN_HEIGHT + u32:1}, OUT_WIDTH: u32 = {IN_HEIGHT - KERN_HEIGHT + u32:1}
    >
    (x: FixedPoint<NB_IN, BE_IN>[IN_HEIGHT][IN_WIDTH][IN_CHANNELS],
    W: FixedPoint<NB_IN, BE_IN>[KERN_HEIGHT][KERN_WIDTH][IN_CHANNELS][OUT_FILTERS],
    b: FixedPoint<NB_IN, BE_IN>[OUT_FILTERS])
    -> FixedPoint<NB_OUT, BE_OUT>[OUT_HEIGHT][OUT_WIDTH][OUT_FILTERS] {

    let y: FixedPoint<NB_OUT, BE_OUT>[OUT_HEIGHT][OUT_WIDTH][OUT_FILTERS] = conv2d_latency<NB_OUT, BE_OUT, ROUNDING, OVERFLOW>(x, W, b);
    let relu_y = for (filter_idx, image) in 0..OUT_FILTERS {
        let relu_plane = for (i, plane) in 0..OUT_WIDTH {
            let relu_row = for (j, plane_row) in 0..OUT_HEIGHT {
                let relu_pixel = activations::relu_1elem<NB_OUT>(y[filter_idx][i][j]);
                update(plane_row, j, relu_pixel)
            }(zero!<FixedPoint<NB_OUT,BE_OUT>[OUT_HEIGHT]>());
            update(plane, i, relu_row)
        }(zero!<FixedPoint<NB_OUT,BE_OUT>[OUT_HEIGHT][OUT_WIDTH]>());
        update(image, filter_idx, relu_plane)
    }(zero!<FixedPoint<NB_OUT,BE_OUT>[OUT_HEIGHT][OUT_WIDTH][OUT_FILTERS]>());
    relu_y
}



#[test]
fn conv2d_latency_test_uniform_io() { 
    // x = 
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    let x = fixed_point_util::make_fixed_points_3d<-10>(s16[5][5][1]:[s16[5][5]:[s16[5]:[s16:1024, ...], ...], ...]);

    // w = 
    //  | 1, 1, 1|
    //  | 2, 2, 2|
    //  | 3, 3, 3|
    let w = fixed_point_util::make_fixed_points_4d<-10>(s16[3][3][1][1]:[[[
        [s16:1024, 1024, 1024],
        [s16:2048, 2048, 2048],
        [s16:3072, 3072, 3072],
    ]]]);
    let b = fixed_point_util::make_fixed_points_1d<-10>(s16[1]:[s16:0]);

    // expected = 
    //  | 18, 18, 18|
    //  | 18, 18, 18|
    //  | 18, 18, 18|
    // TODO: herefater we have to specify integer type inside each 1d array because of type inference bug in DSLX:
    // It loses types in make_fixed_points_2d, _3d etc.,
    // and assert_eq fails with a message like:
    // lhs and rhs were not equal: [ [ FixedPoint { 
    // < significand: s16:0
    // > significand: u0:0 
    // } ] ]
    let expected = fixed_point_util::make_fixed_points_3d<-10>(s16[3][3][1]:[[
        [s16:18432, 18432, 18432],
        [s16:18432, 18432, 18432],
        [s16:18432, 18432, 18432],
    ]]);
    assert_eq(expected, conv2d_latency<16, -10, RoundingMode::TRN, OverflowMode::WRAP>(x, w, b));
}

#[test]
fn conv2d_latency_test_bias() { 
    // x = 
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    let x = fixed_point_util::make_fixed_points_3d<-10>(s16[5][5][1]:[s16[5][5]:[s16[5]:[s16:1024, ...], ...], ...]);

    // w = 
    //  | 1, 1, 1|
    //  | 2, 2, 2|
    //  | 3, 3, 3|
    let w = fixed_point_util::make_fixed_points_4d<-10>(s16[3][3][1][1]:[[[
        [s16:1024, 1024, 1024],
        [s16:2048, 2048, 2048],
        [s16:3072, 3072, 3072],
    ]]]);
    // b = | 1 |
    let b = fixed_point_util::make_fixed_points_1d<-10>(s16[1]:[s16:1024]);

    // expected = 
    //  | 19, 19, 19|
    //  | 19, 19, 19|
    //  | 19, 19, 19|
    let expected = fixed_point_util::make_fixed_points_3d<-10>(s16[3][3][1]:[[
        [s16:19456, 19456, 19456],
        [s16:19456, 19456, 19456],
        [s16:19456, 19456, 19456],
    ]]);
    assert_eq(expected, conv2d_latency<16, -10, RoundingMode::TRN, OverflowMode::WRAP>(x, w, b));
}

#[test]
fn conv2d_latency_test_pattern() { 
    // x = 
    //  | 1, 1, 1, 1, 1|
    //  | 0, 0, 0, 0, 0|
    //  | 2, 2, 2, 2, 2|
    //  | 0, 0, 0, 0, 0|
    //  | 1, 1, 1, 1, 1|
    let x = fixed_point_util::make_fixed_points_3d<-10>(s16[5][5][1]:[[
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:0, 0, 0, 0, 0],
        [s16:2048, 2048, 2048, 2048, 2048],
        [s16:0, 0, 0, 0, 0],
        [s16:1024, 1024, 1024, 1024, 1024],
    ]]);

    // w = 
    //  | 1, 1, 1|
    //  | 2, 2, 2|
    //  | 3, 3, 3|
    let w = fixed_point_util::make_fixed_points_4d<-10>(s16[3][3][1][1]:[[[
        [s16:1024, 1024, 1024],
        [s16:2048, 2048, 2048],
        [s16:3072, 3072, 3072],
    ]]]);
    // b = | 0 |
    let b = fixed_point_util::make_fixed_points_1d<-10>(s16[1]:[s16:0]);

    // expected = 
    //  | 21, 21, 21|
    //  | 12, 12, 12|
    //  | 15, 15, 15|
    let expected = fixed_point_util::make_fixed_points_3d<-10>(s16[3][3][1]:[[
        [s16:21504, 21504, 21504],
        [s16:12288, 12288, 12288],
        [s16:15360, 15360, 15360],
    ]]);
    assert_eq(expected, conv2d_latency<16, -10, RoundingMode::TRN, OverflowMode::WRAP>(x, w, b));
}

#[test]
fn conv2d_latency_test_mutiple_filters() { 
    // x = 
    //  | 1, 1, 1, 1, 1|
    //  | 0, 0, 0, 0, 0|
    //  | 2, 2, 2, 2, 2|
    //  | 0, 0, 0, 0, 0|
    //  | 1, 1, 1, 1, 1|
    let x = fixed_point_util::make_fixed_points_3d<-10>(s16[5][5][1]:[[
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:0, 0, 0, 0, 0],
        [s16:2048, 2048, 2048, 2048, 2048],
        [s16:0, 0, 0, 0, 0],
        [s16:1024, 1024, 1024, 1024, 1024],
    ]]);

    // w = 
    //  | 1, 1, 1|  | 1, 1, 1|  | 0, 0, 0|
    //  | 2, 2, 2|  | 1, 1, 1|  | 0, 0, 0|
    //  | 3, 3, 3|  | 1, 1, 1|  | 0, 0, 0|
    let w = fixed_point_util::make_fixed_points_4d<-10>(s16[3][3][1][3]:[[[
        [s16:1024, 1024, 1024],
        [s16:2048, 2048, 2048],
        [s16:3072, 3072, 3072],
    ]],[[
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
    ]],[[
        [s16:0, 0, 0],
        [s16:0, 0, 0],
        [s16:0, 0, 0],
    ]]]);
    // b = | 0, 0 ,-2|
    let b = fixed_point_util::make_fixed_points_1d<-10>(s16[3]:[s16:0, 0, -2048]);

    // expected = 
    //  | 21, 21, 21|  | 6, 6, 6|  | 0, 0, 0|
    //  | 12, 12, 12|  | 9, 9, 9|  | 0, 0, 0|
    //  | 15, 15, 15|  | 6, 6, 6|  | 0, 0, 0|
    let expected = fixed_point_util::make_fixed_points_3d<-10>(s16[3][3][3]:[[
        [s16:21504, 21504, 21504],
        [s16:12288, 12288, 12288],
        [s16:15360, 15360, 15360]
    ],[
        [s16:9216, 9216, 9216],
        [s16:6144, 6144, 6144],
        [s16:9216, 9216, 9216]
    ],[
        [s16:-2048, -2048, -2048],
        [s16:-2048, -2048, -2048],
        [s16:-2048, -2048, -2048],
    ]]);
    assert_eq(expected, conv2d_latency<16, -10, RoundingMode::TRN, OverflowMode::WRAP>(x, w, b));
}

#[test]
fn conv2d_latency_test_mutiple_channels() { 
    // x = 
    //  | 1, 1, 1, 1, 1|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 0, 0, 0, 0, 0|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 2, 2, 2, 2, 2|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 0, 0, 0, 0, 0|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 1, 1, 1, 1, 1|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    let x = fixed_point_util::make_fixed_points_3d<-10>(s16[5][5][3]:[[
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:0, 0, 0, 0, 0],
        [s16:2048, 2048, 2048, 2048, 2048],
        [s16:0, 0, 0, 0, 0],
        [s16:1024, 1024, 1024, 1024, 1024],
    ],[
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:1024, 1024, 1024, 1024, 1024],
    ],[
        [s16:0, 0, 0, 0, 0],
        [s16:0, 0, 0, 0, 0],
        [s16:0, 0, 0, 0, 0],
        [s16:0, 0, 0, 0, 0],
        [s16:0, 0, 0, 0, 0],
    ],]);

    // w = 
    //  | 1, 1, 1|  | 1, 1, 1|  | 0, 0, 0|
    //  | 2, 2, 2|  | 1, 1, 1|  | 0, 0, 0|
    //  | 1, 1, 1|  | 1, 1, 1|  | 0, 0, 0|
    let w = fixed_point_util::make_fixed_points_4d<-10>(s16[3][3][3][1]:[[[
        [s16:1024, 1024, 1024],
        [s16:2048, 2048, 2048],
        [s16:1024, 1024, 1024],
    ],[
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
    ],[
        [s16:0, 0, 0],
        [s16:0, 0, 0],
        [s16:0, 0, 0],
    ]]]);
    // b = | 1 |
    let b = fixed_point_util::make_fixed_points_1d<-10>(s16[1]:[s16:0]);

    // expected = 
    //  | 18, 18, 18|
    //  | 21, 21, 21|
    //  | 18, 18, 18|
    let expected = fixed_point_util::make_fixed_points_3d<-10>(s16[3][3][1]:[[
        [s16:18432, 18432, 18432],
        [s16:21504, 21504, 21504],
        [s16:18432, 18432, 18432]
    ]]);
    assert_eq(expected, conv2d_latency<16, -10, RoundingMode::TRN, OverflowMode::WRAP>(x, w, b));
}

#[test]
fn conv2d_latency_test_mutiple_channels_and_filters() { 
    // x = 
    //  | 1, 1, 1, 1, 1|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 0, 0, 0, 0, 0|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 2, 2, 2, 2, 2|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 0, 0, 0, 0, 0|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 1, 1, 1, 1, 1|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    let x = fixed_point_util::make_fixed_points_3d<-10>(s16[5][5][3]:[[
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:0, 0, 0, 0, 0],
        [s16:2048, 2048, 2048, 2048, 2048],
        [s16:0, 0, 0, 0, 0],
        [s16:1024, 1024, 1024, 1024, 1024],
    ],[
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:1024, 1024, 1024, 1024, 1024],
    ],[
        [s16:0, 0, 0, 0, 0],
        [s16:0, 0, 0, 0, 0],
        [s16:0, 0, 0, 0, 0],
        [s16:0, 0, 0, 0, 0],
        [s16:0, 0, 0, 0, 0],
    ],]);

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
    let w = fixed_point_util::make_fixed_points_4d<-10>(s16[3][3][3][3]:[
    [[
        [s16:1024, 1024, 1024],
        [s16:2048, 2048, 2048],
        [s16:1024, 1024, 1024],
    ],[
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
    ],[
        [s16:0, 0, 0],
        [s16:0, 0, 0],
        [s16:0, 0, 0],
    ]],

    [[
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
    ],[
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
    ],[
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
    ]],

    [[
        [s16:0, 0, 0],
        [s16:0, 0, 0],
        [s16:0, 0, 0],
    ],[
        [s16:0, 0, 0],
        [s16:0, 0, 0],
        [s16:0, 0, 0],
    ],[
        [s16:0, 0, 0],
        [s16:0, 0, 0],
        [s16:0, 0, 0],
    ]],]);
    // b = | 0, 0, 0|
    let b = fixed_point_util::make_fixed_points_1d<-10>(s16[3]:[s16:0, 0, 0]);

    // expected = 
    //  | 18, 18, 18|  | 18, 18, 18|  | 0, 0, 0|
    //  | 21, 21, 21|  | 15, 15, 15|  | 0, 0, 0|
    //  | 18, 18, 18|  | 18, 18, 18|  | 0, 0, 0|
    let expected = fixed_point_util::make_fixed_points_3d<-10>(s16[3][3][3]:[[
        [s16:18432, 18432, 18432],
        [s16:21504, 21504, 21504],
        [s16:18432, 18432, 18432]
    ],[
        [s16:18432, 18432, 18432],
        [s16:15360, 15360, 15360],
        [s16:18432, 18432, 18432]
    ],[
        [s16:0, 0, 0],
        [s16:0, 0, 0],
        [s16:0, 0, 0],
    ]]);

    assert_eq(expected, conv2d_latency<16, -10, RoundingMode::TRN, OverflowMode::WRAP>(x, w, b));
}

#[test]
fn conv2d_latency_test_two_layers() { 
    // x = 
    //  | 1, 1, 1, 1, 1|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 0, 0, 0, 0, 0|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 2, 2, 2, 2, 2|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 0, 0, 0, 0, 0|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 1, 1, 1, 1, 1|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    let x = fixed_point_util::make_fixed_points_3d<-10>(s16[5][5][3]:[[
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:0, 0, 0, 0, 0],
        [s16:2048, 2048, 2048, 2048, 2048],
        [s16:0, 0, 0, 0, 0],
        [s16:1024, 1024, 1024, 1024, 1024],
    ],[
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:1024, 1024, 1024, 1024, 1024],
    ],[
        [s16:0, 0, 0, 0, 0],
        [s16:0, 0, 0, 0, 0],
        [s16:0, 0, 0, 0, 0],
        [s16:0, 0, 0, 0, 0],
        [s16:0, 0, 0, 0, 0],
    ],]);

    // w = 
    //  | 1, 1, 1|  | 1, 1, 1|  | 0, 0, 0|
    //  | 2, 2, 2|  | 1, 1, 1|  | 0, 0, 0|
    //  | 1, 1, 1|  | 1, 1, 1|  | 0, 0, 0|

    //  | 1, 1, 1|  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|  | 1, 1, 1|
    let w0 = fixed_point_util::make_fixed_points_4d<-10>(s16[3][3][3][2]:[
    [[
        [s16:1024, 1024, 1024],
        [s16:2048, 2048, 2048],
        [s16:1024, 1024, 1024],
    ],[
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
    ],[
        [s16:0, 0, 0],
        [s16:0, 0, 0],
        [s16:0, 0, 0],
    ]],

    [[
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
    ],[
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
    ],[
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
    ]]]);
    // b = | -17, -17|
    let b0 = fixed_point_util::make_fixed_points_1d<-10>(s16[2]:[-17408, -17408]);

    // w1 = 
    //  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|
    let w1 = fixed_point_util::make_fixed_points_4d<-10>(s16[3][3][2][1]:[
    [[
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
    ],[
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
    ]]]);
    // b = | 0 |
    let b1 = fixed_point_util::make_fixed_points_1d<-10>(s16[1]:[s16:0]);

    // expected = | 18 |
    let expected = fixed_point_util::make_fixed_points_3d<-10>(s16[1][1][1]:[[
        [s16:18432],
    ]]);

    let z0 = conv2d_latency<16, -10, RoundingMode::TRN, OverflowMode::WRAP>(x, w0, b0);
    let z1 = conv2d_latency<16, -10, RoundingMode::TRN, OverflowMode::WRAP>(z0, w1, b1);
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
    let x = fixed_point_util::make_fixed_points_3d<-10>(s16[5][5][3]:[[
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:0, 0, 0, 0, 0],
        [s16:2048, 2048, 2048, 2048, 2048],
        [s16:0, 0, 0, 0, 0],
        [s16:1024, 1024, 1024, 1024, 1024],
    ],[
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:1024, 1024, 1024, 1024, 1024],
        [s16:1024, 1024, 1024, 1024, 1024],
    ],[
        [s16:0, 0, 0, 0, 0],
        [s16:0, 0, 0, 0, 0],
        [s16:0, 0, 0, 0, 0],
        [s16:0, 0, 0, 0, 0],
        [s16:0, 0, 0, 0, 0],
    ],]);

    // w = 
    //  | 1, 1, 1|  | 1, 1, 1|  | 0, 0, 0|
    //  | 2, 2, 2|  | 1, 1, 1|  | 0, 0, 0|
    //  | 1, 1, 1|  | 1, 1, 1|  | 0, 0, 0|

    //  | 1, 1, 1|  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|  | 1, 1, 1|
    let w0 = fixed_point_util::make_fixed_points_4d<-10>(s16[3][3][3][2]:[
    [[
        [s16:1024, 1024, 1024],
        [s16:2048, 2048, 2048],
        [s16:1024, 1024, 1024],
    ],[
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
    ],[
        [s16:0, 0, 0],
        [s16:0, 0, 0],
        [s16:0, 0, 0],
    ]],

    [[
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
    ],[
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
    ],[
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
    ]]]);
    // b = | -17, -17|
    let b0 = fixed_point_util::make_fixed_points_1d<-10>(s16[2]:[-17408, -17408]);

    // w1 = 
    //  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|
    let w1 = fixed_point_util::make_fixed_points_4d<-10>(s16[3][3][2][1]:[
    [[
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
    ],[
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
        [s16:1024, 1024, 1024],
    ]]]);
    // b = | 0 |
    let b1 = fixed_point_util::make_fixed_points_1d<-10>(s16[1]:[s16:0]);

    // expected = | 18 |
    let expected = fixed_point_util::make_fixed_points_3d<-10>(s16[1][1][1]:[[
        [s16:24576],
    ]]);

    let z0 = conv_relu_latency<16, -10, RoundingMode::TRN, OverflowMode::WRAP>(x, w0, b0);
    let z1 = conv_relu_latency<16, -10, RoundingMode::TRN, OverflowMode::WRAP>(z0, w1, b1);
    assert_eq(expected, z1);
}
