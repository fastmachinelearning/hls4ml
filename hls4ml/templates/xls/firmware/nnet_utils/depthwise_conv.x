import std;
import fixed_point;

import ap_types.fixed_point_util;
import nnet_utils.data_format;

type FixedPoint = fixed_point::FixedPoint;
type RoundingMode = fixed_point_util::RoundingMode;
type OverflowMode = fixed_point_util::OverflowMode;
pub type DataFormat = data_format::DataFormat;

pub fn depthwise_conv_1d
    <OUT_NB: u32, OUT_BE: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    STRIDE: u32,
    PAD_LEFT: u32, PAD_RIGHT: u32,
    DATA_FORMAT: DataFormat,
    // Input
    IN_NB: u32, IN_BE: s32,
    IN_DIM_0: u32, IN_DIM_1: u32,
    // Kernel
    KERN_NB: u32, KERN_BE: s32,
    KERN_SIZE: u32, OUT_FILTERS: u32,
    // Bias
    BIAS_NB: u32, BIAS_BE: s32,
    // Derived input dims
    IN_SIZE: u32 = {data_format::to_size_chans(IN_DIM_0, IN_DIM_1, DATA_FORMAT)[0]},
    IN_CHANNELS: u32 = {data_format::to_size_chans(IN_DIM_0, IN_DIM_1, DATA_FORMAT)[1]},
    DEPTH_MULTIPLIER: u32 = {OUT_FILTERS / IN_CHANNELS},
    // Output size
    OUT_SIZE: u32 = {((IN_SIZE + PAD_LEFT + PAD_RIGHT - KERN_SIZE) / STRIDE) + 1},
    // Output dims
    OUT_DIM_0: u32 = {data_format::from_size_chans(OUT_SIZE, OUT_FILTERS, DATA_FORMAT)[0]},
    OUT_DIM_1: u32 = {data_format::from_size_chans(OUT_SIZE, OUT_FILTERS, DATA_FORMAT)[1]},
    // Precision
    MUL_BE: s32 = {IN_BE + KERN_BE},
    MUL_NB: u32 = {IN_NB + KERN_NB},
    CONV_NB: u32 = {MUL_NB + std::clog2(KERN_SIZE)},
    CONV_BE: s32 = {MUL_BE}
    >
(
    x: FixedPoint<IN_NB, IN_BE>[IN_DIM_1][IN_DIM_0],
    kernel: FixedPoint<KERN_NB, KERN_BE>[DEPTH_MULTIPLIER][IN_CHANNELS][KERN_SIZE],
    bias: FixedPoint<BIAS_NB, BIAS_BE>[OUT_FILTERS]
)
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_1][OUT_DIM_0] {

    for (out_i_0, out_2d) in 0..OUT_DIM_0 {
        let out_1d = for (out_i_1, out_1d) in 0..OUT_DIM_1 {

            let ij = data_format::to_size_chans(out_i_0, out_i_1, DATA_FORMAT);
            let out_pos = ij[0];
            let filter_idx = ij[1];
            let ch_idx = filter_idx / DEPTH_MULTIPLIER;
            let depth_idx = filter_idx % DEPTH_MULTIPLIER;

            let in_pos: s32 = ((out_pos as s32) * (STRIDE as s32)) - (PAD_LEFT as s32);

            let conv_pixel = for (k, acc) in 0..KERN_SIZE {
                let ii = in_pos + (k as s32);

                let val = if ii < s32:0 || ii >= IN_SIZE as s32 {
                    zero!<FixedPoint<IN_NB, IN_BE>>()
                } else {
                    let ii = ii as u32;
                    match DATA_FORMAT {
                        DataFormat::CHANNELS_LAST  => x[ii][ch_idx],
                        DataFormat::CHANNELS_FIRST => x[ch_idx][ii]
                    }
                };

                let w = kernel[k][ch_idx][depth_idx];
                fixed_point_util::fmadd_already_widened(val, w, acc)
            }(zero!<FixedPoint<CONV_NB, CONV_BE>>());

            let conv_pixel_with_bias = fixed_point::add(conv_pixel, bias[filter_idx]);
            let conv_pixel_with_bias =
                fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(conv_pixel_with_bias);

            update(out_1d, out_i_1, conv_pixel_with_bias)

        }(zero!<FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_1]>());

        update(out_2d, out_i_0, out_1d)

    }(zero!<FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_1][OUT_DIM_0]>())
}

pub fn depthwise_conv_2d
    <OUT_NB: u32, OUT_BE: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    STRIDE_HEIGHT: u32, STRIDE_WIDTH: u32,
    PAD_TOP: u32, PAD_BOTTOM: u32,
    PAD_LEFT: u32, PAD_RIGHT: u32,
    DATA_FORMAT: DataFormat,
    // Input
    IN_NB: u32, IN_BE: s32,
    IN_DIM_0: u32, IN_DIM_1: u32, IN_DIM_2: u32,
    // Kernel
    KERN_NB: u32, KERN_BE: s32,
    KERN_HEIGHT: u32, KERN_WIDTH: u32, OUT_FILTERS: u32,
    // Bias
    BIAS_NB: u32, BIAS_BE: s32,
    // Derived input dims
    IN_HEIGHT: u32 = {data_format::to_height_width_chans(IN_DIM_0, IN_DIM_1, IN_DIM_2, DATA_FORMAT)[0]},
    IN_WIDTH: u32 = {data_format::to_height_width_chans(IN_DIM_0, IN_DIM_1, IN_DIM_2, DATA_FORMAT)[1]},
    IN_CHANNELS: u32 = {data_format::to_height_width_chans(IN_DIM_0, IN_DIM_1, IN_DIM_2, DATA_FORMAT)[2]},
    DEPTH_MULTIPLIER: u32 = {OUT_FILTERS / IN_CHANNELS},
    // Output dims
    OUT_HEIGHT: u32 = {((IN_HEIGHT + PAD_TOP + PAD_BOTTOM - KERN_HEIGHT) / STRIDE_HEIGHT) + 1},
    OUT_WIDTH: u32 = {((IN_WIDTH + PAD_LEFT + PAD_RIGHT - KERN_WIDTH) / STRIDE_WIDTH) + 1},
    OUT_DIM_0: u32 = {data_format::from_height_width_chans(OUT_HEIGHT, OUT_WIDTH, OUT_FILTERS, DATA_FORMAT)[0]},
    OUT_DIM_1: u32 = {data_format::from_height_width_chans(OUT_HEIGHT, OUT_WIDTH, OUT_FILTERS, DATA_FORMAT)[1]},
    OUT_DIM_2: u32 = {data_format::from_height_width_chans(OUT_HEIGHT, OUT_WIDTH, OUT_FILTERS, DATA_FORMAT)[2]},
    // Precision
    MUL_BE: s32 = {IN_BE + KERN_BE},
    MUL_NB: u32 = {IN_NB + KERN_NB},
    CONV_NB: u32 = {MUL_NB + std::clog2(KERN_HEIGHT * KERN_WIDTH)},
    CONV_BE: s32 = {MUL_BE}
    >
(
    x: FixedPoint<IN_NB, IN_BE>[IN_DIM_2][IN_DIM_1][IN_DIM_0],
    kernel: FixedPoint<KERN_NB, KERN_BE>[DEPTH_MULTIPLIER][IN_CHANNELS][KERN_WIDTH][KERN_HEIGHT],
    bias: FixedPoint<BIAS_NB, BIAS_BE>[OUT_FILTERS]
)
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_2][OUT_DIM_1][OUT_DIM_0] {

    for (out_i_0, out_3d) in 0..OUT_DIM_0 {
        let out_2d = for (out_i_1, out_2d) in 0..OUT_DIM_1 {
            let out_1d = for (out_i_2, out_1d) in 0..OUT_DIM_2 {
                let ijf = data_format::to_height_width_chans(out_i_0, out_i_1, out_i_2, DATA_FORMAT);
                let out_i = ijf[0];
                let out_j = ijf[1];
                let filter_idx = ijf[2];
                let ch_idx = filter_idx / DEPTH_MULTIPLIER;
                let depth_idx = filter_idx % DEPTH_MULTIPLIER;

                let in_i: s32 = ((out_i as s32) * (STRIDE_HEIGHT as s32)) - (PAD_TOP as s32);
                let in_j: s32 = ((out_j as s32) * (STRIDE_WIDTH as s32)) - (PAD_LEFT as s32);

                let conv_pixel = for (di, pixel_ch) in 0..KERN_HEIGHT {
                    for (dj, acc) in 0..KERN_WIDTH {
                        let ii = in_i + (di as s32);
                        let jj = in_j + (dj as s32);

                        let val = if ii < s32:0
                                || ii >= IN_HEIGHT as s32
                                || jj < s32:0
                                || jj >= IN_WIDTH as s32 {
                            zero!<FixedPoint<IN_NB, IN_BE>>()
                        } else {
                            let ii = ii as u32;
                            let jj = jj as u32;
                            match DATA_FORMAT {
                                DataFormat::CHANNELS_LAST  => x[ii][jj][ch_idx],
                                DataFormat::CHANNELS_FIRST => x[ch_idx][ii][jj]
                            }
                        };

                        let w = kernel[di][dj][ch_idx][depth_idx];
                        fixed_point_util::fmadd_already_widened(val, w, acc)
                    }(pixel_ch)
                }(zero!<FixedPoint<CONV_NB, CONV_BE>>());

                let conv_pixel_with_bias = fixed_point::add(conv_pixel, bias[filter_idx]);
                let conv_pixel_with_bias =
                    fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(conv_pixel_with_bias);

                update(out_1d, out_i_2, conv_pixel_with_bias)

            }(zero!<FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_2]>());

            update(out_2d, out_i_1, out_1d)

        }(zero!<FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_2][OUT_DIM_1]>());

        update(out_3d, out_i_0, out_2d)

    }(zero!<FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_2][OUT_DIM_1][OUT_DIM_0]>())
}

// Testing

fn depthwise_conv_1d_default<
    IN_NB: u32, IN_BE: s32,
    IN_SIZE: u32, IN_CHANNELS: u32,
    KERN_SIZE: u32, DEPTH_MULTIPLIER: u32,
    OUT_FILTERS: u32 = {IN_CHANNELS * DEPTH_MULTIPLIER},
    OUT_SIZE: u32 = {IN_SIZE + u32:1 - KERN_SIZE},
    OUT_NB: u32 = {IN_NB},
    OUT_BE: s32 = {IN_BE},
    STRIDE: u32 = {u32:1},
    PAD_LEFT: u32 = {u32:0},
    PAD_RIGHT: u32 = {u32:0}
>
(
    x: FixedPoint<IN_NB, IN_BE>[IN_CHANNELS][IN_SIZE],
    weights: FixedPoint<IN_NB, IN_BE>[DEPTH_MULTIPLIER][IN_CHANNELS][KERN_SIZE],
    bias: FixedPoint<IN_NB, IN_BE>[OUT_FILTERS]
)
-> FixedPoint<OUT_NB, OUT_BE>[OUT_FILTERS][OUT_SIZE] {

    depthwise_conv_1d<
        OUT_NB, OUT_BE,
        RoundingMode::TRN, OverflowMode::WRAP,
        STRIDE,
        PAD_LEFT, PAD_RIGHT,
        DataFormat::CHANNELS_LAST
    >(x, weights, bias)
}

fn depthwise_conv_1d_default_first<
    IN_NB: u32, IN_BE: s32,
    IN_SIZE: u32, IN_CHANNELS: u32,
    KERN_SIZE: u32, DEPTH_MULTIPLIER: u32,
    OUT_FILTERS: u32 = {IN_CHANNELS * DEPTH_MULTIPLIER},
    OUT_SIZE: u32 = {IN_SIZE + u32:1 - KERN_SIZE},
    OUT_NB: u32 = {IN_NB},
    OUT_BE: s32 = {IN_BE},
    STRIDE: u32 = {u32:1},
    PAD_LEFT: u32 = {u32:0},
    PAD_RIGHT: u32 = {u32:0}
>
(
    x: FixedPoint<IN_NB, IN_BE>[IN_SIZE][IN_CHANNELS],
    weights: FixedPoint<IN_NB, IN_BE>[DEPTH_MULTIPLIER][IN_CHANNELS][KERN_SIZE],
    bias: FixedPoint<IN_NB, IN_BE>[OUT_FILTERS]
)
-> FixedPoint<OUT_NB, OUT_BE>[OUT_SIZE][OUT_FILTERS] {

    depthwise_conv_1d<
        OUT_NB, OUT_BE,
        RoundingMode::TRN, OverflowMode::WRAP,
        STRIDE,
        PAD_LEFT, PAD_RIGHT,
        DataFormat::CHANNELS_FIRST
    >(x, weights, bias)
}

fn depthwise_conv_2d_default<
    IN_NB: u32, IN_BE: s32,
    IN_HEIGHT: u32, IN_WIDTH: u32, IN_CHANNELS: u32,
    KERN_HEIGHT: u32, KERN_WIDTH: u32, DEPTH_MULTIPLIER: u32,
    OUT_FILTERS: u32 = {IN_CHANNELS * DEPTH_MULTIPLIER},
    OUT_HEIGHT: u32 = {IN_HEIGHT + u32:1 - KERN_HEIGHT},
    OUT_WIDTH: u32 = {IN_WIDTH + u32:1 - KERN_WIDTH},
    OUT_NB: u32 = {IN_NB},
    OUT_BE: s32 = {IN_BE},
    STRIDE_HEIGHT: u32 = {u32:1},
    STRIDE_WIDTH: u32 = {u32:1},
    PAD_TOP: u32 = {u32:0},
    PAD_BOTTOM: u32 = {u32:0},
    PAD_LEFT: u32 = {u32:0},
    PAD_RIGHT: u32 = {u32:0}
>
(
    x: FixedPoint<IN_NB, IN_BE>[IN_CHANNELS][IN_WIDTH][IN_HEIGHT],
    weights: FixedPoint<IN_NB, IN_BE>[DEPTH_MULTIPLIER][IN_CHANNELS][KERN_WIDTH][KERN_HEIGHT],
    bias: FixedPoint<IN_NB, IN_BE>[OUT_FILTERS]
)
-> FixedPoint<OUT_NB, OUT_BE>[OUT_FILTERS][OUT_WIDTH][OUT_HEIGHT] {

    depthwise_conv_2d<
        OUT_NB, OUT_BE,
        RoundingMode::TRN, OverflowMode::WRAP,
        STRIDE_HEIGHT, STRIDE_WIDTH,
        PAD_TOP, PAD_BOTTOM,
        PAD_LEFT, PAD_RIGHT,
        DataFormat::CHANNELS_LAST
    >(x, weights, bias)
}

fn depthwise_conv_2d_default_first<
    IN_NB: u32, IN_BE: s32,
    IN_HEIGHT: u32, IN_WIDTH: u32, IN_CHANNELS: u32,
    KERN_HEIGHT: u32, KERN_WIDTH: u32, DEPTH_MULTIPLIER: u32,
    OUT_FILTERS: u32 = {IN_CHANNELS * DEPTH_MULTIPLIER},
    OUT_HEIGHT: u32 = {IN_HEIGHT + u32:1 - KERN_HEIGHT},
    OUT_WIDTH: u32 = {IN_WIDTH + u32:1 - KERN_WIDTH},
    OUT_NB: u32 = {IN_NB},
    OUT_BE: s32 = {IN_BE},
    STRIDE_HEIGHT: u32 = {u32:1},
    STRIDE_WIDTH: u32 = {u32:1},
    PAD_TOP: u32 = {u32:0},
    PAD_BOTTOM: u32 = {u32:0},
    PAD_LEFT: u32 = {u32:0},
    PAD_RIGHT: u32 = {u32:0}
>
(
    x: FixedPoint<IN_NB, IN_BE>[IN_WIDTH][IN_HEIGHT][IN_CHANNELS],
    weights: FixedPoint<IN_NB, IN_BE>[DEPTH_MULTIPLIER][IN_CHANNELS][KERN_WIDTH][KERN_HEIGHT],
    bias: FixedPoint<IN_NB, IN_BE>[OUT_FILTERS]
)
-> FixedPoint<OUT_NB, OUT_BE>[OUT_WIDTH][OUT_HEIGHT][OUT_FILTERS] {

    depthwise_conv_2d<
        OUT_NB, OUT_BE,
        RoundingMode::TRN, OverflowMode::WRAP,
        STRIDE_HEIGHT, STRIDE_WIDTH,
        PAD_TOP, PAD_BOTTOM,
        PAD_LEFT, PAD_RIGHT,
        DataFormat::CHANNELS_FIRST
    >(x, weights, bias)
}

#[test]
fn test_zero_1d() {
    let x = zero!<FixedPoint<16, -10>[2][5]>();
    let w = zero!<FixedPoint<16, -10>[2][2][3]>();
    let b = zero!<FixedPoint<16, -10>[4]>();

    let expected = zero!<FixedPoint<16, -10>[4][3]>();
    assert_eq(expected, depthwise_conv_1d_default(x, w, b));

    let x_first = zero!<FixedPoint<16, -10>[5][2]>();
    let expected_first = zero!<FixedPoint<16, -10>[3][4]>();
    assert_eq(expected_first, depthwise_conv_1d_default_first(x_first, w, b));
}

#[test]
fn test_depthwise_conv_1d_uniform_io() {
    let x = fixed_point_util::make_fixed_points_2d<-10>(s16[2][4]:[
        s16[2]:[s16:1024, s16:2048],
        s16[2]:[s16:1024, s16:2048],
        s16[2]:[s16:1024, s16:2048],
        s16[2]:[s16:1024, s16:2048]
    ]);
    let w = fixed_point_util::make_fixed_points_3d<-10>(s16[2][2][2]:[
        s16[2][2]:[[s16:1024, s16:1024], [s16:1024, s16:1024]],
        s16[2][2]:[[s16:1024, s16:2048], [s16:1024, s16:2048]]
    ]);
    let b = fixed_point_util::make_fixed_points_1d<-10>(s16[4]:[s16:0, s16:1024, s16:0, s16:1024]);

    let expected = fixed_point_util::make_fixed_points_2d<-10>(s16[4][3]:[
        s16[4]:[s16:2048, s16:4096, s16:4096, s16:7168],
        s16[4]:[s16:2048, s16:4096, s16:4096, s16:7168],
        s16[4]:[s16:2048, s16:4096, s16:4096, s16:7168]
    ]);
    assert_eq(expected, depthwise_conv_1d_default(x, w, b));

    let x_first = fixed_point_util::make_fixed_points_2d<-10>(s16[4][2]:[
        s16[4]:[s16:1024, s16:1024, s16:1024, s16:1024],
        s16[4]:[s16:2048, s16:2048, s16:2048, s16:2048]
    ]);
    let expected_first = fixed_point_util::make_fixed_points_2d<-10>(s16[3][4]:[
        s16[3]:[s16:2048, s16:2048, s16:2048],
        s16[3]:[s16:4096, s16:4096, s16:4096],
        s16[3]:[s16:4096, s16:4096, s16:4096],
        s16[3]:[s16:7168, s16:7168, s16:7168]
    ]);
    assert_eq(expected_first, depthwise_conv_1d_default_first(x_first, w, b));
}

#[test]
fn test_zero_2d() {
    let x = zero!<FixedPoint<16, -10>[2][4][4]>();
    let w = zero!<FixedPoint<16, -10>[2][2][2][2]>();
    let b = zero!<FixedPoint<16, -10>[4]>();

    let expected = zero!<FixedPoint<16, -10>[4][3][3]>();
    assert_eq(expected, depthwise_conv_2d_default(x, w, b));

    let x_first = zero!<FixedPoint<16, -10>[4][4][2]>();
    let expected_first = zero!<FixedPoint<16, -10>[3][3][4]>();
    assert_eq(expected_first, depthwise_conv_2d_default_first(x_first, w, b));
}

#[test]
fn test_depthwise_conv_2d_uniform_io() {
    let x = fixed_point_util::make_fixed_points_3d<-10>(s16[2][3][3]:[
        s16[2][3]:[[s16:1024, s16:2048], [s16:1024, s16:2048], [s16:1024, s16:2048]],
        s16[2][3]:[[s16:1024, s16:2048], [s16:1024, s16:2048], [s16:1024, s16:2048]],
        s16[2][3]:[[s16:1024, s16:2048], [s16:1024, s16:2048], [s16:1024, s16:2048]]
    ]);
    let w = fixed_point_util::make_fixed_points_4d<-10>(s16[2][2][2][2]:[
        s16[2][2][2]:[
            [[s16:1024, s16:1024], [s16:1024, s16:1024]],
            [[s16:1024, s16:0], [s16:1024, s16:0]]
        ],
        s16[2][2][2]:[
            [[s16:1024, s16:0], [s16:1024, s16:0]],
            [[s16:1024, s16:1024], [s16:1024, s16:1024]]
        ]
    ]);
    let b = fixed_point_util::make_fixed_points_1d<-10>(s16[4]:[s16:0, s16:1024, s16:0, s16:1024]);

    let expected = fixed_point_util::make_fixed_points_3d<-10>(s16[4][2][2]:[
        s16[4][2]:[[s16:4096, s16:3072, s16:8192, s16:5120], [s16:4096, s16:3072, s16:8192, s16:5120]],
        s16[4][2]:[[s16:4096, s16:3072, s16:8192, s16:5120], [s16:4096, s16:3072, s16:8192, s16:5120]]
    ]);
    assert_eq(expected, depthwise_conv_2d_default(x, w, b));

    let x_first = fixed_point_util::make_fixed_points_3d<-10>(s16[3][3][2]:[
        s16[3][3]:[s16[3]:[s16:1024, s16:1024, s16:1024], s16[3]:[s16:1024, s16:1024, s16:1024], s16[3]:[s16:1024, s16:1024, s16:1024]],
        s16[3][3]:[s16[3]:[s16:2048, s16:2048, s16:2048], s16[3]:[s16:2048, s16:2048, s16:2048], s16[3]:[s16:2048, s16:2048, s16:2048]]
    ]);
    let expected_first = fixed_point_util::make_fixed_points_3d<-10>(s16[2][2][4]:[
        s16[2][2]:[s16[2]:[s16:4096, s16:4096], s16[2]:[s16:4096, s16:4096]],
        s16[2][2]:[s16[2]:[s16:3072, s16:3072], s16[2]:[s16:3072, s16:3072]],
        s16[2][2]:[s16[2]:[s16:8192, s16:8192], s16[2]:[s16:8192, s16:8192]],
        s16[2][2]:[s16[2]:[s16:5120, s16:5120], s16[2]:[s16:5120, s16:5120]]
    ]);
    assert_eq(expected_first, depthwise_conv_2d_default_first(x_first, w, b));
}
