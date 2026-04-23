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

fn to_size_chans(dim_0: u32, dim_1: u32, data_format: DataFormat) -> u32[2] {    
    match data_format {
        DataFormat::CHANNELS_LAST  => [dim_0, dim_1],
        DataFormat::CHANNELS_FIRST => [dim_1, dim_0]
    }
}

fn from_size_chans(size: u32, channels: u32, data_format: DataFormat) -> u32[2] {    
    match data_format {
        DataFormat::CHANNELS_LAST  => [size, channels],
        DataFormat::CHANNELS_FIRST => [channels, size]
    }
}

pub fn conv1d_latency
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
    IN_SIZE: u32 = {to_size_chans(IN_DIM_0, IN_DIM_1, DATA_FORMAT)[0]},
    IN_CHANNELS: u32 = {to_size_chans(IN_DIM_0, IN_DIM_1, DATA_FORMAT)[1]},
    // Output size
    OUT_SIZE: u32 = {((IN_SIZE + PAD_LEFT + PAD_RIGHT - KERN_SIZE) / STRIDE) + 1},
    // Output dims
    OUT_DIM_0: u32 = {from_size_chans(OUT_SIZE, OUT_FILTERS, DATA_FORMAT)[0]},
    OUT_DIM_1: u32 = {from_size_chans(OUT_SIZE, OUT_FILTERS, DATA_FORMAT)[1]},
    // Precision
    MUL_BE: s32 = {IN_BE + IN_BE},
    MUL_NB: u32 = {IN_NB + IN_NB},
    CONV_NB: u32 = {MUL_NB + std::clog2(KERN_SIZE * IN_CHANNELS)},
    CONV_BE: s32 = {MUL_BE}
    >
(
    x: FixedPoint<IN_NB, IN_BE>[IN_DIM_1][IN_DIM_0],
    kernel: FixedPoint<KERN_NB, KERN_BE>[OUT_FILTERS][IN_CHANNELS][KERN_SIZE],
    bias: FixedPoint<BIAS_NB, BIAS_BE>[OUT_FILTERS]
)
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_1][OUT_DIM_0] {

    for (out_i_0, out_2d) in 0..OUT_DIM_0 {
        let out_1d = for (out_i_1, out_1d) in 0..OUT_DIM_1 {

            let ij = to_size_chans(out_i_0, out_i_1, DATA_FORMAT);
            let out_pos = ij[0];
            let filter_idx = ij[1];

            let in_pos: s32 = ((out_pos as s32) * (STRIDE as s32)) - (PAD_LEFT as s32);

            let conv_pixel = for (ch_idx, pixel_chans) in 0..IN_CHANNELS {

                for (k, acc) in 0..KERN_SIZE {
                    let ii = in_pos + (k as s32);

                    let val = if ii < s32:0
                            || ii >= IN_SIZE as s32 {
                        zero!<FixedPoint<IN_NB, IN_BE>>()
                    } else {
                        let ii = ii as u32;
                        match DATA_FORMAT {
                            DataFormat::CHANNELS_LAST  => x[ii][ch_idx],
                            DataFormat::CHANNELS_FIRST => x[ch_idx][ii]
                        }
                    };

                    let w = kernel[k][ch_idx][filter_idx];
                    fixed_point_util::fmadd_already_widened(val, w, acc)
                }(pixel_chans)

            }(zero!<FixedPoint<CONV_NB, CONV_BE>>());

            let conv_pixel_with_bias = fixed_point::add(conv_pixel, bias[filter_idx]);
            let conv_pixel_with_bias = fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(conv_pixel_with_bias);

            update(out_1d, out_i_1, conv_pixel_with_bias)

        }(zero!<FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_1]>());

        update(out_2d, out_i_0, out_1d)

    }(zero!<FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_1][OUT_DIM_0]>())
}

// Testing

fn conv1d_latency_default<
    IN_NB: u32, IN_BE: s32,
    // Input
    IN_SIZE: u32, IN_CHANNELS: u32,
    // Kernel
    KERN_SIZE: u32, OUT_FILTERS: u32,
    // Output
    OUT_SIZE: u32 = {IN_SIZE + u32:1 - KERN_SIZE},
    // Defaults
    OUT_NB: u32 = {IN_NB},
    OUT_BE: s32 = {IN_BE},
    STRIDE: u32 = {u32:1},
    PAD_LEFT: u32 = {u32:0},
    PAD_RIGHT: u32 = {u32:0}
>
(
    x: FixedPoint<IN_NB, IN_BE>[IN_CHANNELS][IN_SIZE],
    weights: FixedPoint<IN_NB, IN_BE>[OUT_FILTERS][IN_CHANNELS][KERN_SIZE],
    bias: FixedPoint<IN_NB, IN_BE>[OUT_FILTERS]
)
-> FixedPoint<OUT_NB, OUT_BE>[OUT_FILTERS][OUT_SIZE] {

    conv1d_latency<
        OUT_NB, OUT_BE,
        RoundingMode::TRN, OverflowMode::WRAP,
        STRIDE,
        PAD_LEFT, PAD_RIGHT,
        DataFormat::CHANNELS_LAST
    >(x, weights, bias)
}

fn conv1d_latency_default_first<
    IN_NB: u32, IN_BE: s32,

    // Input
    IN_SIZE: u32, IN_CHANNELS: u32,

    // Kernel
    KERN_SIZE: u32, OUT_FILTERS: u32,

    // Output
    OUT_SIZE: u32 = {IN_SIZE + u32:1 - KERN_SIZE},

    // Defaults
    OUT_NB: u32 = {IN_NB},
    OUT_BE: s32 = {IN_BE},
    STRIDE: u32 = {u32:1},
    PAD_LEFT: u32 = {u32:0},
    PAD_RIGHT: u32 = {u32:0}
>
(
    x: FixedPoint<IN_NB, IN_BE>[IN_SIZE][IN_CHANNELS],
    weights: FixedPoint<IN_NB, IN_BE>[OUT_FILTERS][IN_CHANNELS][KERN_SIZE],
    bias: FixedPoint<IN_NB, IN_BE>[OUT_FILTERS]
)
-> FixedPoint<OUT_NB, OUT_BE>[OUT_SIZE][OUT_FILTERS] {

    conv1d_latency<
        OUT_NB, OUT_BE,
        RoundingMode::TRN, OverflowMode::WRAP,
        STRIDE,
        PAD_LEFT, PAD_RIGHT,
        DataFormat::CHANNELS_FIRST
    >(x, weights, bias)
}

fn test_zero_1d<
    IN_SIZE: u32,
    IN_CHANNELS: u32,
    KERN_SIZE: u32,
    OUT_FILTERS: u32,
    OUT_SIZE: u32 = {IN_SIZE + u32:1 - KERN_SIZE}
>() {

    let x = zero!<FixedPoint<16, -10>[IN_CHANNELS][IN_SIZE]>();
    let w = zero!<FixedPoint<16, -10>[OUT_FILTERS][IN_CHANNELS][KERN_SIZE]>();
    let b = zero!<FixedPoint<16, -10>[OUT_FILTERS]>();

    let expected = zero!<FixedPoint<16, -10>[OUT_FILTERS][OUT_SIZE]>();
    assert_eq(expected, conv1d_latency_default(x, w, b));

    // CHANNELS_FIRST
    let x_first = zero!<FixedPoint<16, -10>[IN_SIZE][IN_CHANNELS]>();
    let expected_first = zero!<FixedPoint<16, -10>[OUT_SIZE][OUT_FILTERS]>();
    assert_eq(expected_first, conv1d_latency_default_first(x_first, w, b));
}

#[test]
fn test_zero_1d_1() {
    let IN_SIZE = u32:1;
    let IN_CHANNELS = u32:1;
    let KERN_SIZE = u32:1;
    let OUT_FILTERS = u32:1;
    test_zero_1d<IN_SIZE, IN_CHANNELS, KERN_SIZE, OUT_FILTERS>();
}

#[test]
fn test_zero_1d_2() {
    let IN_SIZE = u32:5;
    let IN_CHANNELS = u32:2;
    let KERN_SIZE = u32:3;
    let OUT_FILTERS = u32:4;
    test_zero_1d<IN_SIZE, IN_CHANNELS, KERN_SIZE, OUT_FILTERS>();
}

#[test]
fn conv1d_latency_test_uniform_io() {

    // x = [1,1,1,1,1]
    let x = fixed_point_util::make_fixed_points_2d<-10>(
        s16[1][5]:[[s16:1024], ...]
    );

    // w = [1,2,3]
    let w = fixed_point_util::make_fixed_points_3d<-10>(
        s16[1][1][3]:[
            s16[1][1]:[[s16:1024]],
            s16[1][1]:[[s16:2048]],
            s16[1][1]:[[s16:3072]]
        ]
    );

    let b = fixed_point_util::make_fixed_points_1d<-10>(s16[1]:[s16:0]);

    // expected = [6,6,6] (scaled: 6 * 1024 = 6144)
    let expected = fixed_point_util::make_fixed_points_2d<-10>(
        s16[1][3]:[[s16:6144], ...]
    );

    assert_eq(expected, conv1d_latency_default(x, w, b));

    // CHANNELS_FIRST
    let x_first = fixed_point_util::make_fixed_points_2d<-10>(
        s16[5][1]:[s16[5]:[s16:1024, ...]]
    );

    let expected_first = fixed_point_util::make_fixed_points_2d<-10>(
        s16[3][1]:[s16[3]:[s16:6144, ...]]
    );

    assert_eq(expected_first, conv1d_latency_default_first(x_first, w, b));
}
