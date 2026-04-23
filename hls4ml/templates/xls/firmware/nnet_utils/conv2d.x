import std;
import fixed_point;

import ap_types.fixed_point_util;
import nnet_utils.activations;

type FixedPoint = fixed_point::FixedPoint;
type RoundingMode = fixed_point_util::RoundingMode;
type OverflowMode = fixed_point_util::OverflowMode;

pub enum DataFormat: u1 {
    CHANNELS_LAST = 0,
    CHANNELS_FIRST = 1
}

fn to_height_width_chans(dim_0: u32, dim_1: u32, dim_2: u32, data_format: DataFormat) -> u32[3] {    
    match data_format {
        DataFormat::CHANNELS_LAST  => [dim_0, dim_1, dim_2],
        // CHW -> HWC
        DataFormat::CHANNELS_FIRST => [dim_1, dim_2, dim_0]
    }
}

fn from_height_width_chans(height: u32, width: u32, channels: u32, data_format: DataFormat) -> u32[3] {    
    match data_format {
        DataFormat::CHANNELS_LAST  => [height, width, channels],
        // HWC -> CHW
        DataFormat::CHANNELS_FIRST => [channels, height, width]
    }
}

#[test]
fn test_data_format() {
    let (h, w, c) = (1,2,3);
    for (data_format, _) in [DataFormat::CHANNELS_LAST, DataFormat::CHANNELS_FIRST] {
        let dims = from_height_width_chans(h, w, c, data_format);
        let hwc = to_height_width_chans(dims[0], dims[1], dims[2], data_format);
        assert_eq(hwc, [h, w, c]);
        assert_eq(
            dims == hwc,
            data_format == DataFormat::CHANNELS_LAST
        );
    }(());
}

pub fn conv2d_latency
    <OUT_NB: u32, OUT_BE: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    STRIDE_HEIGHT: u32, STRIDE_WIDTH: u32,
    PAD_TOP: u32, PAD_BOTTOM: u32,
    PAD_LEFT: u32, PAD_RIGHT: u32,
    DATA_FORMAT: DataFormat,
    // All parameters below can be deduced automatically
    IN_NB: u32, IN_BE: s32, 
    // Input Image
    // Dimensions: (IN_HEIGHT, IN_WIDTH, IN_CHANNELS) or (IN_CHANNELS, IN_HEIGHT, IN_WIDTH),
    // depending on DATA_FORMAT
    IN_DIM_0: u32, IN_DIM_1: u32, IN_DIM_2: u32,
    // Kernel
    KERN_NB: u32, KERN_BE: s32,
    KERN_HEIGHT: u32, KERN_WIDTH: u32, OUT_FILTERS: u32,
    // Bias
    BIAS_NB: u32, BIAS_BE: s32,
    // Input image
    IN_HEIGHT: u32 = {to_height_width_chans(IN_DIM_0, IN_DIM_1, IN_DIM_2, DATA_FORMAT)[0]},
    IN_WIDTH: u32 = {to_height_width_chans(IN_DIM_0, IN_DIM_1, IN_DIM_2, DATA_FORMAT)[1]},
    IN_CHANNELS: u32 = {to_height_width_chans(IN_DIM_0, IN_DIM_1, IN_DIM_2, DATA_FORMAT)[2]},
    // Output Image
    OUT_HEIGHT: u32 = {((IN_HEIGHT + PAD_TOP + PAD_BOTTOM - KERN_HEIGHT) / STRIDE_HEIGHT) + 1},
    OUT_WIDTH: u32 = {((IN_WIDTH  + PAD_LEFT + PAD_RIGHT  - KERN_WIDTH) / STRIDE_WIDTH) + 1},
    // Output dimension
    OUT_DIM_0: u32 = {from_height_width_chans(OUT_HEIGHT, OUT_WIDTH, OUT_FILTERS, DATA_FORMAT)[0]},
    OUT_DIM_1: u32 = {from_height_width_chans(OUT_HEIGHT, OUT_WIDTH, OUT_FILTERS, DATA_FORMAT)[1]},
    OUT_DIM_2: u32 = {from_height_width_chans(OUT_HEIGHT, OUT_WIDTH, OUT_FILTERS, DATA_FORMAT)[2]},
    // Precision inference MUL
    MUL_BE: s32 = {IN_BE + IN_BE},
    MUL_NB: u32 = {IN_NB + IN_NB},
    // Precision Inference CONV
    // TODO support custom accum_t precision
    CONV_NB: u32 = {MUL_NB + std::clog2(KERN_HEIGHT * KERN_WIDTH * IN_CHANNELS)},
    CONV_BE: s32 = {MUL_BE}
    >
    (x: FixedPoint<IN_NB, IN_BE>[IN_DIM_2][IN_DIM_1][IN_DIM_0],
    kernel: FixedPoint<KERN_NB, KERN_BE>[OUT_FILTERS][IN_CHANNELS][KERN_WIDTH][KERN_HEIGHT],
    bias: FixedPoint<BIAS_NB, BIAS_BE>[OUT_FILTERS])
    -> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_2][OUT_DIM_1][OUT_DIM_0] {

    for (out_i_0, out_3d) in 0..OUT_DIM_0 {
        let out_2d = for (out_i_1, out_2d) in 0..OUT_DIM_1 {
            let out_1d = for (out_i_2, out_1d) in 0..OUT_DIM_2 {
                let ijf = to_height_width_chans(out_i_0, out_i_1, out_i_2, DATA_FORMAT);
                let out_i = ijf[0];
                let out_j = ijf[1];
                let filter_idx = ijf[2];

                let in_i: s32 = ((out_i as s32) * (STRIDE_HEIGHT as s32)) - (PAD_TOP as s32);
                let in_j: s32 = ((out_j as s32) * (STRIDE_WIDTH as s32)) - (PAD_LEFT as s32);
                // Compute convolution across channels:
                // res[out_i, out_j, filt] = sum(x[in_i+di, in_j+dj, ch_idx] * w[di, dj, ch_idx, filt])
                let conv_pixel = for (ch_idx, pixel_chans) in 0..IN_CHANNELS {
                    // Compute convolution for a single channel:
                    // acc = sum(x[i+di, j+dj] * w[di, dj])
                    for (di, pixel_ch) in 0..KERN_HEIGHT {
                        for (dj, acc) in 0..KERN_WIDTH {
                            let ii = in_i + (di as s32);
                            let jj = in_j + (dj as s32);
                            // Pad with zeros
                            let val = if ii < s32:0
                                    || ii >= IN_HEIGHT as s32
                                    || jj < s32:0
                                    || jj >= IN_WIDTH as s32 {
                                zero!<FixedPoint<IN_NB, IN_BE>>()
                            } else {
                                let ii = ii as u32;
                                let jj = jj as u32;
                                match DATA_FORMAT{
                                    DataFormat::CHANNELS_LAST  => x[ii][jj][ch_idx],
                                    DataFormat::CHANNELS_FIRST => x[ch_idx][ii][jj]
                                }
                            };
                            let w = kernel[di][dj][ch_idx][filter_idx];
                            fixed_point_util::fmadd_already_widened(val, w, acc)
                        }(pixel_ch)
                    }(pixel_chans)
                }(zero!<FixedPoint<CONV_NB, CONV_BE>>());
                let conv_pixel_with_bias = fixed_point::add(conv_pixel, bias[filter_idx]);
                let conv_pixel_with_bias = fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(conv_pixel_with_bias);

                update(out_1d, out_i_2, conv_pixel_with_bias)

            }(zero!<FixedPoint<OUT_NB,OUT_BE>[OUT_DIM_2]>());

            update(out_2d, out_i_1, out_1d)

        }(zero!<FixedPoint<OUT_NB,OUT_BE>[OUT_DIM_2][OUT_DIM_1]>());

        update(out_3d, out_i_0, out_2d)
    
    }(zero!<FixedPoint<OUT_NB,OUT_BE>[OUT_DIM_2][OUT_DIM_1][OUT_DIM_0]>())
}

// Set some default parameters reused in all tests.
// TODO: test other parameters
fn conv2d_latency_default<
    IN_NB: u32, IN_BE: s32, 
    // Input Image
    IN_HEIGHT: u32, IN_WIDTH: u32, IN_CHANNELS: u32,
    // Kernel Dims
    KERN_HEIGHT: u32, KERN_WIDTH: u32, OUT_FILTERS: u32,
    // Output Image
    OUT_HEIGHT: u32 = {IN_HEIGHT + u32:1 - KERN_HEIGHT},
    OUT_WIDTH: u32 = {IN_WIDTH + u32:1 - KERN_WIDTH},
    // Default parameters:
    OUT_NB: u32 = {IN_NB},
    OUT_BE: s32 = {IN_BE},
    STRIDE_HEIGHT: u32 = {u32:1},
    STRIDE_WIDTH: u32 = {u32:1},
    PAD_TOP: u32 = {u32:0},
    PAD_BOTTOM: u32 = {u32:0},
    PAD_LEFT: u32 = {u32:0},
    PAD_RIGHT: u32 = {u32:0}
    >
    (x: FixedPoint<IN_NB, IN_BE>[IN_CHANNELS][IN_WIDTH][IN_HEIGHT],
    weights: FixedPoint<IN_NB, IN_BE>[OUT_FILTERS][IN_CHANNELS][KERN_WIDTH][KERN_HEIGHT],
    bias: FixedPoint<IN_NB, IN_BE>[OUT_FILTERS])
    -> FixedPoint<OUT_NB, OUT_BE>[OUT_FILTERS][OUT_WIDTH][OUT_HEIGHT] {

    conv2d_latency<
        OUT_NB, OUT_BE,
        RoundingMode::TRN, OverflowMode::WRAP,
        STRIDE_HEIGHT, STRIDE_WIDTH,
        PAD_TOP, PAD_BOTTOM,
        PAD_LEFT, PAD_RIGHT,
        DataFormat::CHANNELS_LAST
    >(x, weights, bias)
}

// Same but with CHANNELS_FIRST
fn conv2d_latency_default_first<
    IN_NB: u32, IN_BE: s32, 
    // Input Image
    IN_HEIGHT: u32, IN_WIDTH: u32, IN_CHANNELS: u32,
    // Kernel Dims
    KERN_HEIGHT: u32, KERN_WIDTH: u32, OUT_FILTERS: u32,
    // Output Image
    OUT_HEIGHT: u32 = {IN_HEIGHT + u32:1 - KERN_HEIGHT},
    OUT_WIDTH: u32 = {IN_WIDTH + u32:1 - KERN_WIDTH},
    // Default parameters:
    OUT_NB: u32 = {IN_NB},
    OUT_BE: s32 = {IN_BE},
    STRIDE_HEIGHT: u32 = {u32:1},
    STRIDE_WIDTH: u32 = {u32:1},
    PAD_TOP: u32 = {u32:0},
    PAD_BOTTOM: u32 = {u32:0},
    PAD_LEFT: u32 = {u32:0},
    PAD_RIGHT: u32 = {u32:0}
    >
    (x: FixedPoint<IN_NB, IN_BE>[IN_WIDTH][IN_HEIGHT][IN_CHANNELS],
    weights: FixedPoint<IN_NB, IN_BE>[OUT_FILTERS][IN_CHANNELS][KERN_WIDTH][KERN_HEIGHT],
    bias: FixedPoint<IN_NB, IN_BE>[OUT_FILTERS])
    -> FixedPoint<OUT_NB, OUT_BE>[OUT_WIDTH][OUT_HEIGHT][OUT_FILTERS] {

    conv2d_latency<
        OUT_NB, OUT_BE,
        RoundingMode::TRN, OverflowMode::WRAP,
        STRIDE_HEIGHT, STRIDE_WIDTH,
        PAD_TOP, PAD_BOTTOM,
        PAD_LEFT, PAD_RIGHT,
        DataFormat::CHANNELS_FIRST
    >(x, weights, bias)
}

// All inputs are zero => we test only dimensions
// TODO: test also padding and stride
fn test_zero<
    IN_HEIGHT: u32, IN_WIDTH: u32,
    IN_CHANNELS: u32,
    KERN_HEIGHT: u32, KERN_WIDTH: u32,
    OUT_FILTERS: u32,
    OUT_HEIGHT: u32 = {IN_HEIGHT + u32:1 - KERN_HEIGHT},
    OUT_WIDTH: u32 = {IN_WIDTH + u32:1 - KERN_WIDTH},
    >() {
    
    let x = zero!<FixedPoint<16, -10>[IN_CHANNELS][IN_WIDTH][IN_HEIGHT]>();
    
    let w = zero!<FixedPoint<16, -10>[OUT_FILTERS][IN_CHANNELS][KERN_WIDTH][KERN_HEIGHT]>();
    let b = zero!<FixedPoint<16, -10>[OUT_FILTERS]>();
    
    let expected = zero!<FixedPoint<16, -10>[OUT_FILTERS][OUT_WIDTH][OUT_HEIGHT]>();

    assert_eq(expected, conv2d_latency_default(x, w, b));
    
    // CHANNELS_FIRST
    let x_first = zero!<FixedPoint<16, -10>[IN_WIDTH][IN_HEIGHT][IN_CHANNELS]>();
    let expected_first = zero!<FixedPoint<16, -10>[OUT_WIDTH][OUT_HEIGHT][OUT_FILTERS]>();
    assert_eq(expected_first, conv2d_latency_default_first(x_first, w, b));
}

#[test]
fn test_zero_1() {
    let IN_HEIGHT = u32:1;
    let IN_WIDTH = u32:1;
    let IN_CHANNELS = u32:1;
    let KERN_HEIGHT = u32:1;
    let KERN_WIDTH = u32:1;
    let OUT_FILTERS = u32:1;
    test_zero<IN_HEIGHT, IN_WIDTH, IN_CHANNELS, KERN_HEIGHT, KERN_WIDTH, OUT_FILTERS>();
}

#[test]
fn test_zero_2() {
    let IN_HEIGHT = u32:2;
    let IN_WIDTH = u32:2;
    let IN_CHANNELS = u32:1;
    let KERN_HEIGHT = u32:1;
    let KERN_WIDTH = u32:1;
    let OUT_FILTERS = u32:1;
    test_zero<IN_HEIGHT, IN_WIDTH, IN_CHANNELS, KERN_HEIGHT, KERN_WIDTH, OUT_FILTERS>();
}

#[test]
fn test_zero_multi() {
    let IN_HEIGHT = u32:9;
    let IN_WIDTH = u32:10;
    let IN_CHANNELS = u32:4;
    let KERN_HEIGHT = u32:3;
    let KERN_WIDTH = u32:2;
    let OUT_FILTERS = u32:5;
    test_zero<IN_HEIGHT, IN_WIDTH, IN_CHANNELS, KERN_HEIGHT, KERN_WIDTH, OUT_FILTERS>();
}


#[test]
fn conv2d_latency_test_uniform_io() { 
    // x = 
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    let x = fixed_point_util::make_fixed_points_3d<-10>(s16[1][5][5]:[s16[1][5]:[s16[1]:[s16:1024], ...], ...]);

    // w = 
    //  | 1, 1, 1|
    //  | 2, 2, 2|
    //  | 3, 3, 3|
    let w = fixed_point_util::make_fixed_points_4d<-10>(s16[1][1][3][3]:[
        s16[1][1][3]:[[[s16:1024]], ...],
        s16[1][1][3]:[[[s16:2048]], ...],
        s16[1][1][3]:[[[s16:3072]], ...]
    ]);
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
    let expected = fixed_point_util::make_fixed_points_3d<-10>(s16[1][3][3]:[
        s16[1][3]:[[s16:18432], ...], ...]);
    assert_eq(expected, conv2d_latency_default(x, w, b));

    // CHANNELS_FIRST
    let x_first = fixed_point_util::make_fixed_points_3d<-10>(s16[5][5][1]:[
        s16[5][5]:[s16[5]:[s16:1024, ...], ...]]);
    let expected_first = fixed_point_util::make_fixed_points_3d<-10>(s16[3][3][1]:[
            s16[3][3]:[s16[3]:[s16:18432, ...], ...]]);
    assert_eq(expected_first, conv2d_latency_default_first(x_first, w, b));
}

#[test]
fn conv2d_latency_test_bias() { 
    // x = 
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    //  | 1, 1, 1, 1, 1|
    let x = fixed_point_util::make_fixed_points_3d<-10>(s16[1][5][5]:[s16[1][5]:[s16[1]:[s16:1024], ...], ...]);

    // w = 
    //  | 1, 1, 1|
    //  | 2, 2, 2|
    //  | 3, 3, 3|
    let w = fixed_point_util::make_fixed_points_4d<-10>(s16[1][1][3][3]:[
        s16[1][1][3]:[[[s16:1024]], ...],
        s16[1][1][3]:[[[s16:2048]], ...],
        s16[1][1][3]:[[[s16:3072]], ...]
    ]);
    // b = | 1 |
    let b = fixed_point_util::make_fixed_points_1d<-10>(s16[1]:[s16:1024]);

    // expected = 
    //  | 19, 19, 19|
    //  | 19, 19, 19|
    //  | 19, 19, 19|
    let expected = fixed_point_util::make_fixed_points_3d<-10>(s16[1][3][3]:[
        s16[1][3]:[[s16:19456], ...], ...]);
    assert_eq(expected, conv2d_latency_default(x, w, b));
    
    // CHANNELS_FIRST
    let x_first = fixed_point_util::make_fixed_points_3d<-10>(s16[5][5][1]:[s16[5][5]:[s16[5]:[s16:1024, ...], ...]]);
    let expected_first = fixed_point_util::make_fixed_points_3d<-10>(s16[3][3][1]:[
            s16[3][3]:[s16[3]:[s16:19456, ...], ...]]);
    assert_eq(expected_first, conv2d_latency_default_first(x_first, w, b));
}

#[test]
fn conv2d_latency_test_pattern() { 
    // x = 
    //  | 1, 1, 1, 1, 1|
    //  | 0, 0, 0, 0, 0|
    //  | 2, 2, 2, 2, 2|
    //  | 0, 0, 0, 0, 0|
    //  | 1, 1, 1, 1, 1|
    let x = fixed_point_util::make_fixed_points_3d<-10>(s16[1][5][5]:[
        s16[1][5]:[[s16:1024], ...],
        s16[1][5]:[[s16:0], ...],
        s16[1][5]:[[s16:2048], ...],
        s16[1][5]:[[s16:0], ...],
        s16[1][5]:[[s16:1024], ...]
    ]);

    // w = 
    //  | 1, 1, 1|
    //  | 2, 2, 2|
    //  | 3, 3, 3|
    let w = fixed_point_util::make_fixed_points_4d<-10>(s16[1][1][3][3]:[
        s16[1][1][3]:[[[s16:1024]], ...],
        s16[1][1][3]:[[[s16:2048]], ...],
        s16[1][1][3]:[[[s16:3072]], ...]
    ]);
    // b = | 0 |
    let b = fixed_point_util::make_fixed_points_1d<-10>(s16[1]:[s16:0]);

    // expected = 
    //  | 21, 21, 21|
    //  | 12, 12, 12|
    //  | 15, 15, 15|
    let expected = fixed_point_util::make_fixed_points_3d<-10>(s16[1][3][3]:[
        s16[1][3]:[[s16:21504], ...],
        s16[1][3]:[[s16:12288], ...],
        s16[1][3]:[[s16:15360], ...]
    ]);
    assert_eq(expected, conv2d_latency_default(x, w, b));
}

#[test]
fn conv2d_latency_test_mutiple_filters() { 
    // x = 
    //  | 1, 1, 1, 1, 1|
    //  | 0, 0, 0, 0, 0|
    //  | 2, 2, 2, 2, 2|
    //  | 0, 0, 0, 0, 0|
    //  | 1, 1, 1, 1, 1|
    let x = fixed_point_util::make_fixed_points_3d<-10>(s16[1][5][5]:[
        s16[1][5]:[[s16:1024], ...],
        s16[1][5]:[[s16:0], ...],
        s16[1][5]:[[s16:2048], ...],
        s16[1][5]:[[s16:0], ...],
        s16[1][5]:[[s16:1024], ...]
    ]);

    // w = 
    //  | 1, 1, 1|  | 1, 1, 1|  | 0, 0, 0|
    //  | 2, 2, 2|  | 1, 1, 1|  | 0, 0, 0|
    //  | 3, 3, 3|  | 1, 1, 1|  | 0, 0, 0|
    let w = fixed_point_util::make_fixed_points_4d<-10>(s16[3][1][3][3]:[
        s16[3][1][3]:[[[s16:1024, 1024, 0]], ...],
        s16[3][1][3]:[[[s16:2048, 1024, 0]], ...],
        s16[3][1][3]:[[[s16:3072, 1024, 0]], ...],
    ]);

    // b = | 0, 0 ,-2|
    let b = fixed_point_util::make_fixed_points_1d<-10>(s16[3]:[s16:0, 0, -2048]);

    // expected = 
    //  | 21, 21, 21|  | 9, 9, 9|  | -2, -2, -2|
    //  | 12, 12, 12|  | 6, 6, 6|  | -2, -2, -2|
    //  | 15, 15, 15|  | 9, 9, 9|  | -2, -2, -2|
    let expected = fixed_point_util::make_fixed_points_3d<-10>(s16[3][3][3]:[
        s16[3][3]:[[s16:21504, 9216, -2048], ...],
        s16[3][3]:[[s16:12288, 6144, -2048], ...],
        s16[3][3]:[[s16:15360, 9216, -2048], ...]
    ]);
    assert_eq(expected, conv2d_latency_default(x, w, b));
}

#[test]
fn conv2d_latency_test_mutiple_channels() { 
    // x = 
    //  | 1, 1, 1, 1, 1|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 0, 0, 0, 0, 0|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 2, 2, 2, 2, 2|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 0, 0, 0, 0, 0|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 1, 1, 1, 1, 1|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    let x = fixed_point_util::make_fixed_points_3d<-10>(s16[3][5][5]:[
        s16[3][5]:[[s16:1024, 1024, 0], ...],
        s16[3][5]:[[s16:0, 1024, 0], ...],
        s16[3][5]:[[s16:2048, 1024, 0], ...],
        s16[3][5]:[[s16:0, 1024, 0], ...],
        s16[3][5]:[[s16:1024, 1024, 0], ...]
    ]);

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
    let w = fixed_point_util::make_fixed_points_4d<-10>(s16[1][3][3][3]:[
        s16[1][3][3]:[[[s16:1024], [s16:1024], [s16:0]], ...],
        s16[1][3][3]:[[[s16:2048], [s16:1024], [s16:0]], ...],
        s16[1][3][3]:[[[s16:1024], [s16:1024], [s16:0]], ...]
    ]);
    // b = | 0 |
    let b = fixed_point_util::make_fixed_points_1d<-10>(s16[1]:[s16:0]);

    // expected = 
    //  | 18, 18, 18|
    //  | 21, 21, 21|
    //  | 18, 18, 18|
    let expected = fixed_point_util::make_fixed_points_3d<-10>(s16[1][3][3]:[
        s16[1][3]:[[s16:18432], ...],
        s16[1][3]:[[s16:21504], ...],
        s16[1][3]:[[s16:18432], ...]
    ]);
    assert_eq(expected, conv2d_latency_default(x, w, b));
}

#[test]
fn conv2d_latency_test_mutiple_channels_and_filters() { 
    // x = 
    //  | 1, 1, 1, 1, 1|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 0, 0, 0, 0, 0|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 2, 2, 2, 2, 2|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 0, 0, 0, 0, 0|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 1, 1, 1, 1, 1|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    let x = fixed_point_util::make_fixed_points_3d<-10>(s16[3][5][5]:[
        s16[3][5]:[[s16:1024, 1024, 0], ...],
        s16[3][5]:[[s16:0, 1024, 0], ...],
        s16[3][5]:[[s16:2048, 1024, 0], ...],
        s16[3][5]:[[s16:0, 1024, 0], ...],
        s16[3][5]:[[s16:1024, 1024, 0], ...]
    ]);

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
        s16[3][3][3]:[[
            [s16:1024, 1024, 0],
            [s16:1024, 1024, 0],
            [s16:0, 1024, 0]
        ], ...],
        s16[3][3][3]:[[
            [s16:2048, 1024, 0],
            [s16:1024, 1024, 0],
            [s16:0, 1024, 0]
        ], ...],
        s16[3][3][3]:[[
            [s16:1024, 1024, 0],
            [s16:1024, 1024, 0],
            [s16:0, 1024, 0]
        ], ...]
    ]);
    // b = | 0, 0, 0|
    let b = fixed_point_util::make_fixed_points_1d<-10>(s16[3]:[s16:0, 0, 0]);

    // expected = 
    //  | 18, 18, 18|  | 18, 18, 18|  | 0, 0, 0|
    //  | 21, 21, 21|  | 15, 15, 15|  | 0, 0, 0|
    //  | 18, 18, 18|  | 18, 18, 18|  | 0, 0, 0|
    let expected = fixed_point_util::make_fixed_points_3d<-10>(s16[3][3][3]:[
        s16[3][3]:[[s16:18432, 18432, 0], ...],
        s16[3][3]:[[s16:21504, 15360, 0], ...],
        s16[3][3]:[[s16:18432, 18432, 0], ...]
    ]);
    assert_eq(expected, conv2d_latency_default(x, w, b));
}

#[test]
fn conv2d_latency_test_two_layers() { 
    // x = 
    //  | 1, 1, 1, 1, 1|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 0, 0, 0, 0, 0|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 2, 2, 2, 2, 2|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 0, 0, 0, 0, 0|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    //  | 1, 1, 1, 1, 1|  | 1, 1, 1, 1, 1|  | 0, 0, 0, 0, 0|
    let x = fixed_point_util::make_fixed_points_3d<-10>(s16[3][5][5]:[
        s16[3][5]:[[s16:1024, 1024, 0], ...],
        s16[3][5]:[[s16:0, 1024, 0], ...],
        s16[3][5]:[[s16:2048, 1024, 0], ...],
        s16[3][5]:[[s16:0, 1024, 0], ...],
        s16[3][5]:[[s16:1024, 1024, 0], ...]
    ]);

    // w = 
    //  | 1, 1, 1|  | 1, 1, 1|  | 0, 0, 0|
    //  | 2, 2, 2|  | 1, 1, 1|  | 0, 0, 0|
    //  | 1, 1, 1|  | 1, 1, 1|  | 0, 0, 0|

    //  | 1, 1, 1|  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|  | 1, 1, 1|
    let w0 = fixed_point_util::make_fixed_points_4d<-10>(s16[2][3][3][3]:[
        s16[2][3][3]:[[
            [s16:1024, 1024],
            [s16:1024, 1024],
            [s16:0, 1024]
        ], ...],
        s16[2][3][3]:[[
            [s16:2048, 1024],
            [s16:1024, 1024],
            [s16:0, 1024]
        ], ...],
        s16[2][3][3]:[[
            [s16:1024, 1024],
            [s16:1024, 1024],
            [s16:0, 1024]
        ], ...]
    ]);
    // b = | -17, -17|
    let b0 = fixed_point_util::make_fixed_points_1d<-10>(s16[2]:[-17408, -17408]);

    // w1 = 
    //  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|
    //  | 1, 1, 1|  | 1, 1, 1|
    let w1 = fixed_point_util::make_fixed_points_4d<-10>(s16[1][2][3][3]:[
        s16[1][2][3]:[s16[1][2]:[[s16:1024], ...], ...], ...
    ]);
    // b = | 0 |
    let b1 = fixed_point_util::make_fixed_points_1d<-10>(s16[1]:[s16:0]);

    // expected = | 18 |
    let expected = fixed_point_util::make_fixed_points_3d<-10>(s16[1][1][1]:[[
        [s16:18432],
    ]]);

    let z0 = conv2d_latency_default(x, w0, b0);
    let z1 = conv2d_latency_default(z0, w1, b1);
    assert_eq(expected, z1);
}
