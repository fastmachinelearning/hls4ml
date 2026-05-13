import std;
import fixed_point;

import ap_types.fixed_point_util;
import nnet_utils.activations;
import nnet_utils.data_format;

type FixedPoint = fixed_point::FixedPoint;
type RoundingMode = fixed_point_util::RoundingMode;
type OverflowMode = fixed_point_util::OverflowMode;
type DataFormat = data_format::DataFormat;

pub enum PoolingOperation: u1 {
    MAX = 0,
    AVERAGE = 1
}


pub fn pooling_1d
    <OUT_NB: u32, OUT_BE: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    POOLING_OP: PoolingOperation,
    // Pool
    POOL_SIZE: u32,
    STRIDE: u32,
    PAD_LEFT: u32, PAD_RIGHT: u32,
    COUNT_PAD: bool,
    DATA_FORMAT: DataFormat,
    // Input
    IN_NB: u32, IN_BE: s32,
    IN_DIM_0: u32, IN_DIM_1: u32,
    // Derived input dims
    IN_SIZE: u32 = {data_format::to_size_chans(IN_DIM_0, IN_DIM_1, DATA_FORMAT)[0]},
    IN_CHANNELS: u32 = {data_format::to_size_chans(IN_DIM_0, IN_DIM_1, DATA_FORMAT)[1]},
    // Output size
    OUT_SIZE: u32 = {((IN_SIZE + PAD_LEFT + PAD_RIGHT - POOL_SIZE) / STRIDE) + 1},
    // Output dims
    OUT_DIM_0: u32 = {data_format::from_size_chans(OUT_SIZE, IN_CHANNELS, DATA_FORMAT)[0]},
    OUT_DIM_1: u32 = {data_format::from_size_chans(OUT_SIZE, IN_CHANNELS, DATA_FORMAT)[1]},
    // Precision for max_or_sum accumulator
    ACC_NB: u32 = {match POOLING_OP {
        PoolingOperation::MAX => IN_NB,
        PoolingOperation::AVERAGE => IN_NB + std::clog2(POOL_SIZE)
    }},
    ACC_BE: s32 = {IN_BE},
    >
(
    x: FixedPoint<IN_NB, IN_BE>[IN_DIM_1][IN_DIM_0]
)
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_1][OUT_DIM_0] {

    for (out_i_0, out_2d) in 0..OUT_DIM_0 {
        let out_1d = for (out_i_1, out_1d) in 0..OUT_DIM_1 {

            let ij = data_format::to_size_chans(out_i_0, out_i_1, DATA_FORMAT);
            let out_pos = ij[0];
            let ch_idx = ij[1];

            let in_pos: s32 = ((out_pos as s32) * (STRIDE as s32)) - (PAD_LEFT as s32);

            // Initial value
            let max_or_sum: FixedPoint<ACC_NB, ACC_BE> = match POOLING_OP {
                PoolingOperation::MAX => fixed_point_util::min_value<ACC_NB, ACC_BE>(),
                PoolingOperation::AVERAGE => zero!<FixedPoint<ACC_NB, ACC_BE>>()
            };
            let (max_or_sum, num_elements) = for (k, (max_or_sum, num_elements)) in 0..POOL_SIZE {
                let ii = in_pos + (k as s32);

                if ii < s32:0 || ii >= IN_SIZE as s32 {
                    if COUNT_PAD {
                        (max_or_sum, num_elements + u32:1)
                    } else {
                        // Padding elements are ignored
                        (max_or_sum, num_elements)
                    }
                } else {
                    let ii = ii as u32;
                    let val = match DATA_FORMAT {
                        DataFormat::CHANNELS_LAST  => x[ii][ch_idx],
                        DataFormat::CHANNELS_FIRST => x[ch_idx][ii]
                    };
                    let max_or_sum = match POOLING_OP {
                        PoolingOperation::MAX => {
                            // val and acc have the same precision in this case,
                            // widening is needed only to prevent compilation error.
                            assert_fmt!(ACC_NB == IN_NB, "max_pooling_op_width");
                            const_assert!(ACC_BE == IN_BE);
                            let val_widened = fixed_point::make_fixed_point<ACC_BE>(val.significand as sN[ACC_NB]);
                            fixed_point_util::max(max_or_sum, val_widened)
                        },
                        PoolingOperation::AVERAGE => fixed_point_util::add_already_widened(val, max_or_sum)
                    };
                    (max_or_sum, num_elements + u32:1)
                }
            }((max_or_sum, u32:0));

            // TODO is it valid case?
            // assert_fmt!(num_elements > 0, "pooling_1d_zero_elements");

            let pool_result = match POOLING_OP {
                PoolingOperation::MAX => fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(
                    max_or_sum
                ),
                PoolingOperation::AVERAGE =>{
                    let avg_significand = max_or_sum.significand / (num_elements as sN[ACC_NB]);
                    let avg = fixed_point::make_fixed_point<ACC_BE>(avg_significand);
                    fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(
                        avg
                    )
                }
            };
            update(out_1d, out_i_1, pool_result)
        }(zero!<FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_1]>());

        update(out_2d, out_i_0, out_1d)

    }(zero!<FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_1][OUT_DIM_0]>())
}

pub fn pooling_2d
    <OUT_NB: u32, OUT_BE: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    POOLING_OP: PoolingOperation,
    // Pool
    POOL_HEIGHT: u32, POOL_WIDTH: u32,
    STRIDE_HEIGHT: u32, STRIDE_WIDTH: u32,
    PAD_TOP: u32, PAD_BOTTOM: u32,
    PAD_LEFT: u32, PAD_RIGHT: u32,
    COUNT_PAD: bool,
    DATA_FORMAT: DataFormat,
    // Input
    IN_NB: u32, IN_BE: s32,
    IN_DIM_0: u32, IN_DIM_1: u32, IN_DIM_2: u32,
    // Derived input dims
    IN_HEIGHT: u32 = {data_format::to_height_width_chans(IN_DIM_0, IN_DIM_1, IN_DIM_2, DATA_FORMAT)[0]},
    IN_WIDTH: u32 = {data_format::to_height_width_chans(IN_DIM_0, IN_DIM_1, IN_DIM_2, DATA_FORMAT)[1]},
    IN_CHANNELS: u32 = {data_format::to_height_width_chans(IN_DIM_0, IN_DIM_1, IN_DIM_2, DATA_FORMAT)[2]},
    // Output size
    OUT_HEIGHT: u32 = {((IN_HEIGHT + PAD_TOP + PAD_BOTTOM - POOL_HEIGHT) / STRIDE_HEIGHT) + 1},
    OUT_WIDTH: u32 = {((IN_WIDTH + PAD_LEFT + PAD_RIGHT - POOL_WIDTH) / STRIDE_WIDTH) + 1},
    // Output dims
    OUT_DIM_0: u32 = {data_format::from_height_width_chans(OUT_HEIGHT, OUT_WIDTH, IN_CHANNELS, DATA_FORMAT)[0]},
    OUT_DIM_1: u32 = {data_format::from_height_width_chans(OUT_HEIGHT, OUT_WIDTH, IN_CHANNELS, DATA_FORMAT)[1]},
    OUT_DIM_2: u32 = {data_format::from_height_width_chans(OUT_HEIGHT, OUT_WIDTH, IN_CHANNELS, DATA_FORMAT)[2]},
    // Precision for max_or_sum accumulator
    ACC_NB: u32 = {match POOLING_OP {
        PoolingOperation::MAX => IN_NB,
        PoolingOperation::AVERAGE => IN_NB + std::clog2(POOL_HEIGHT * POOL_WIDTH)
    }},
    ACC_BE: s32 = {IN_BE},
    >
(
    x: FixedPoint<IN_NB, IN_BE>[IN_DIM_2][IN_DIM_1][IN_DIM_0]
)
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_2][OUT_DIM_1][OUT_DIM_0] {

    for (out_i_0, out_3d) in 0..OUT_DIM_0 {
        let out_2d = for (out_i_1, out_2d) in 0..OUT_DIM_1 {
            let out_1d = for (out_i_2, out_1d) in 0..OUT_DIM_2 {

                let ijc = data_format::to_height_width_chans(out_i_0, out_i_1, out_i_2, DATA_FORMAT);
                let out_i = ijc[0];
                let out_j = ijc[1];
                let ch_idx = ijc[2];

                let in_i: s32 = ((out_i as s32) * (STRIDE_HEIGHT as s32)) - (PAD_TOP as s32) ;
                let in_j: s32 = ((out_j as s32) * (STRIDE_WIDTH as s32)) - (PAD_LEFT as s32);

                // Initial value
                let max_or_sum: FixedPoint<ACC_NB, ACC_BE> = match POOLING_OP {
                    PoolingOperation::MAX => fixed_point_util::min_value<ACC_NB, ACC_BE>(),
                    PoolingOperation::AVERAGE => zero!<FixedPoint<ACC_NB, ACC_BE>>()
                };
                let (max_or_sum, num_elements) = for (di, (max_or_sum, num_elements)) in 0..POOL_HEIGHT {
                    for (dj, (max_or_sum, num_elements)) in 0..POOL_WIDTH {
                        let ii = in_i + (di as s32);
                        let jj = in_j + (dj as s32);

                        if ii < s32:0 || ii >= IN_HEIGHT as s32 || jj < s32:0 || jj >= IN_WIDTH as s32 {
                            if COUNT_PAD {
                                (max_or_sum, num_elements + u32:1)
                            } else {
                                // Padding elements are ignored
                                (max_or_sum, num_elements)
                            }
                        } else {
                            let ii = ii as u32;
                            let jj = jj as u32;
                            let val = match DATA_FORMAT {
                                DataFormat::CHANNELS_LAST  => x[ii][jj][ch_idx],
                                DataFormat::CHANNELS_FIRST => x[ch_idx][ii][jj]
                            };
                            let max_or_sum = match POOLING_OP {
                                PoolingOperation::MAX => {
                                    // val and acc have the same precision in this case,
                                    // widening is needed only to prevent compilation error.
                                    assert_fmt!(ACC_NB == IN_NB, "max_pooling_op_width");
                                    const_assert!(ACC_BE == IN_BE);
                                    let val_widened = fixed_point::make_fixed_point<ACC_BE>(val.significand as sN[ACC_NB]);
                                    fixed_point_util::max(max_or_sum, val_widened)
                                },
                                PoolingOperation::AVERAGE => fixed_point_util::add_already_widened(val, max_or_sum)
                            };
                            (max_or_sum, num_elements + u32:1)
                        }
                    }((max_or_sum, num_elements))
                }((max_or_sum, u32:0));

                // TODO is it valid case?
                // assert_fmt!(num_elements > 0, "pooling2d_zero_elements");
                let pool_result = match POOLING_OP {
                    PoolingOperation::MAX => fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(
                        max_or_sum
                    ),
                    PoolingOperation::AVERAGE =>{
                        let avg_significand = max_or_sum.significand / (num_elements as sN[ACC_NB]);
                        let avg = fixed_point::make_fixed_point<ACC_BE>(avg_significand);
                        fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(
                            avg
                        )
                    }
                };
                update(out_1d, out_i_2, pool_result)
            }(zero!<FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_2]>());

            update(out_2d, out_i_1, out_1d)
        }(zero!<FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_2][OUT_DIM_1]>());

        update(out_3d, out_i_0, out_2d)
    }(zero!<FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_2][OUT_DIM_1][OUT_DIM_0]>())
}

pub fn global_pooling_1d<
    OUT_NB: u32, OUT_BE: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    POOLING_OP: PoolingOperation,
    DATA_FORMAT: DataFormat,
    // Input
    IN_NB: u32, IN_BE: s32,
    IN_DIM_0: u32, IN_DIM_1: u32,
    // Derived input dims
    IN_SIZE: u32 = {data_format::to_size_chans(IN_DIM_0, IN_DIM_1, DATA_FORMAT)[0]},
    IN_CHANNELS: u32 = {data_format::to_size_chans(IN_DIM_0, IN_DIM_1, DATA_FORMAT)[1]},
    // For global pooling, pool size is equal to input size
    POOL_SIZE: u32 = {IN_SIZE},
    STRIDE: u32 = {1},
    PAD_LEFT: u32 = {0}, PAD_RIGHT: u32 = {0},
    COUNT_PAD: bool = {false},
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_1][IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[IN_CHANNELS] {
    let res_2d = pooling_1d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW, POOLING_OP, POOL_SIZE, STRIDE, PAD_LEFT, PAD_RIGHT, COUNT_PAD, DATA_FORMAT>(x);
    fixed_point_util::flatten_2d(res_2d)
}

pub fn global_pooling_2d<
    OUT_NB: u32, OUT_BE: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    POOLING_OP: PoolingOperation,
    DATA_FORMAT: DataFormat,
    // Input
    IN_NB: u32, IN_BE: s32,
    IN_DIM_0: u32, IN_DIM_1: u32, IN_DIM_2: u32,
    // Derived input dims
    IN_HEIGHT: u32 = {data_format::to_height_width_chans(IN_DIM_0, IN_DIM_1, IN_DIM_2, DATA_FORMAT)[0]},
    IN_WIDTH: u32 = {data_format::to_height_width_chans(IN_DIM_0, IN_DIM_1, IN_DIM_2, DATA_FORMAT)[1]},
    IN_CHANNELS: u32 = {data_format::to_height_width_chans(IN_DIM_0, IN_DIM_1, IN_DIM_2, DATA_FORMAT)[2]},
    // For global pooling, pool size is equal to input size
    POOL_HEIGHT: u32 = {IN_HEIGHT},
    POOL_WIDTH: u32 = {IN_WIDTH},
    STRIDE_HEIGHT: u32 = {1},
    STRIDE_WIDTH: u32 = {1},
    PAD_TOP: u32 = {0}, PAD_BOTTOM: u32 = {0},
    PAD_LEFT: u32 = {0}, PAD_RIGHT: u32 = {0},
    COUNT_PAD: bool = {false},
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_2][IN_DIM_1][IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[IN_CHANNELS] {
    let res_3d = pooling_2d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW, POOLING_OP, POOL_HEIGHT, POOL_WIDTH, STRIDE_HEIGHT, STRIDE_WIDTH, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT, COUNT_PAD, DATA_FORMAT>(x);
    fixed_point_util::flatten_3d(res_3d)
}

// Testing

// Test constant input for 1D and 2D

fn test_pooling_const_case<
    POOLING_OP: PoolingOperation,
    DATA_FORMAT: DataFormat,
    COUNT_PAD: bool,
    IN_HEIGHT: u32,
    IN_WIDTH: u32,
    IN_CHANNELS: u32,
    POOL_HEIGHT: u32,
    POOL_WIDTH: u32,
    STRIDE_HEIGHT: u32,
    STRIDE_WIDTH: u32,
    PAD_TOP: u32, PAD_BOTTOM: u32,
    PAD_LEFT: u32, PAD_RIGHT: u32,
    // Input
    NB: u32, BE: s32,
    // 2d
    IN_2D_DIM_0: u32 = {data_format::from_height_width_chans(IN_HEIGHT, IN_WIDTH, IN_CHANNELS, DATA_FORMAT)[0]},
    IN_2D_DIM_1: u32 = {data_format::from_height_width_chans(IN_HEIGHT, IN_WIDTH, IN_CHANNELS, DATA_FORMAT)[1]},
    IN_2D_DIM_2: u32 = {data_format::from_height_width_chans(IN_HEIGHT, IN_WIDTH, IN_CHANNELS, DATA_FORMAT)[2]},
    // 1d
    IN_SIZE: u32 = {IN_HEIGHT},
    IN_1D_DIM_0: u32 = {data_format::from_size_chans(IN_SIZE, IN_CHANNELS, DATA_FORMAT)[0]},
    IN_1D_DIM_1: u32 = {data_format::from_size_chans(IN_SIZE, IN_CHANNELS, DATA_FORMAT)[1]},
    // Output 2d
    OUT_HEIGHT: u32 = {((IN_HEIGHT + PAD_TOP + PAD_BOTTOM - POOL_HEIGHT) / STRIDE_HEIGHT) + 1},
    OUT_WIDTH: u32 = {((IN_WIDTH + PAD_LEFT + PAD_RIGHT - POOL_WIDTH) / STRIDE_WIDTH) + 1},
    OUT_2D_DIM_0: u32 = {data_format::from_height_width_chans(OUT_HEIGHT, OUT_WIDTH, IN_CHANNELS, DATA_FORMAT)[0]},
    OUT_2D_DIM_1: u32 = {data_format::from_height_width_chans(OUT_HEIGHT, OUT_WIDTH, IN_CHANNELS, DATA_FORMAT)[1]},
    OUT_2D_DIM_2: u32 = {data_format::from_height_width_chans(OUT_HEIGHT, OUT_WIDTH, IN_CHANNELS, DATA_FORMAT)[2]},
    // Output 1d
    OUT_SIZE: u32 = {OUT_HEIGHT},
    OUT_1D_DIM_0: u32 = {data_format::from_size_chans(OUT_SIZE, IN_CHANNELS, DATA_FORMAT)[0]},
    OUT_1D_DIM_1: u32 = {data_format::from_size_chans(OUT_SIZE, IN_CHANNELS, DATA_FORMAT)[1]},
>(value: FixedPoint<NB, BE>) {
    let R = RoundingMode::TRN;
    let O = OverflowMode::WRAP;

    let input_for_1d = fixed_point_util::const_array_2d<IN_1D_DIM_0, IN_1D_DIM_1>(value);
    let expected_1d = fixed_point_util::const_array_2d<OUT_1D_DIM_0, OUT_1D_DIM_1>(value);

    let input_for_2d = fixed_point_util::const_array_3d<IN_2D_DIM_0, IN_2D_DIM_1, IN_2D_DIM_2>(value);
    let expected_2d = fixed_point_util::const_array_3d<OUT_2D_DIM_0, OUT_2D_DIM_1, OUT_2D_DIM_2>(value);
    let expected_global = fixed_point_util::const_array_1d<IN_CHANNELS>(value);

    let pooling_1d_result = pooling_1d<
        NB, BE, R, O,
        POOLING_OP,
        POOL_HEIGHT,
        STRIDE_HEIGHT, PAD_TOP, PAD_BOTTOM,
        COUNT_PAD,
        DATA_FORMAT
        >(input_for_1d);
    let pooling_2d_result = pooling_2d<
        NB, BE, R, O,
        POOLING_OP,
        POOL_HEIGHT, POOL_WIDTH,
        STRIDE_HEIGHT, STRIDE_WIDTH,
        PAD_TOP, PAD_BOTTOM,
        PAD_LEFT, PAD_RIGHT,
        COUNT_PAD,
        DATA_FORMAT
        >(input_for_2d);

    if (COUNT_PAD == false ||
        POOLING_OP == PoolingOperation::MAX ||
        (PAD_TOP + PAD_BOTTOM + PAD_LEFT + PAD_RIGHT == 0) ||
        value == zero!<FixedPoint<NB, BE>>()
    ){
        assert_eq(expected_1d, pooling_1d_result);
        assert_eq(expected_2d, pooling_2d_result);
    }
    else{
        // TODO check element values instead of skipping the test
        trace_fmt!("test_pooling_const_case: skip because output array will not be constant: COUNT_PAD={}, POOLING_OP={}, PAD_TOP={}, PAD_BOTTOM={}, PAD_LEFT={}, PAD_RIGHT={}, value={}",
            COUNT_PAD, POOLING_OP, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT, value);
    };

    let global_pooling_1d_result = global_pooling_1d<NB, BE, R, O, POOLING_OP, DATA_FORMAT>(input_for_1d);
    let global_pooling_2d_result = global_pooling_2d<NB, BE, R, O, POOLING_OP, DATA_FORMAT>(input_for_2d);
    assert_eq(expected_global, global_pooling_1d_result);
    assert_eq(expected_global, global_pooling_2d_result);
}

fn test_pooling_const_cases<
    IN_HEIGHT: u32,
    IN_WIDTH: u32,
    IN_CHANNELS: u32,
    POOL_HEIGHT: u32,
    POOL_WIDTH: u32,
    STRIDE_HEIGHT: u32,
    STRIDE_WIDTH: u32,
    PAD_TOP: u32,
    PAD_BOTTOM: u32,
    PAD_LEFT: u32,
    PAD_RIGHT: u32,
    // Input
    NB: u32, BE: s32,
>(value: FixedPoint<NB, BE>) {
    let POOLING_OP = PoolingOperation::MAX;
    let DATA_FORMAT = DataFormat::CHANNELS_LAST;
    let COUNT_PAD = false;
    test_pooling_const_case<POOLING_OP, DATA_FORMAT, COUNT_PAD, IN_HEIGHT, IN_WIDTH, IN_CHANNELS, POOL_HEIGHT, POOL_WIDTH, STRIDE_HEIGHT, STRIDE_WIDTH, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT>(value);
    let COUNT_PAD = true;
    test_pooling_const_case<POOLING_OP, DATA_FORMAT, COUNT_PAD, IN_HEIGHT, IN_WIDTH, IN_CHANNELS, POOL_HEIGHT, POOL_WIDTH, STRIDE_HEIGHT, STRIDE_WIDTH, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT>(value);

    let DATA_FORMAT = DataFormat::CHANNELS_FIRST;
    let COUNT_PAD = false;
    test_pooling_const_case<POOLING_OP, DATA_FORMAT, COUNT_PAD, IN_HEIGHT, IN_WIDTH, IN_CHANNELS, POOL_HEIGHT, POOL_WIDTH, STRIDE_HEIGHT, STRIDE_WIDTH, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT>(value);
    let COUNT_PAD = true;
    test_pooling_const_case<POOLING_OP, DATA_FORMAT, COUNT_PAD, IN_HEIGHT, IN_WIDTH, IN_CHANNELS, POOL_HEIGHT, POOL_WIDTH, STRIDE_HEIGHT, STRIDE_WIDTH, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT>(value);

    let POOLING_OP = PoolingOperation::AVERAGE;
    let DATA_FORMAT = DataFormat::CHANNELS_LAST;
    let COUNT_PAD = false;
    test_pooling_const_case<POOLING_OP, DATA_FORMAT, COUNT_PAD, IN_HEIGHT, IN_WIDTH, IN_CHANNELS, POOL_HEIGHT, POOL_WIDTH, STRIDE_HEIGHT, STRIDE_WIDTH, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT>(value);
    let COUNT_PAD = true;
    test_pooling_const_case<POOLING_OP, DATA_FORMAT, COUNT_PAD, IN_HEIGHT, IN_WIDTH, IN_CHANNELS, POOL_HEIGHT, POOL_WIDTH, STRIDE_HEIGHT, STRIDE_WIDTH, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT>(value);

    let DATA_FORMAT = DataFormat::CHANNELS_FIRST;
    let COUNT_PAD = false;
    test_pooling_const_case<POOLING_OP, DATA_FORMAT, COUNT_PAD, IN_HEIGHT, IN_WIDTH, IN_CHANNELS, POOL_HEIGHT, POOL_WIDTH, STRIDE_HEIGHT, STRIDE_WIDTH, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT>(value);
    let COUNT_PAD = true;
    test_pooling_const_case<POOLING_OP, DATA_FORMAT, COUNT_PAD, IN_HEIGHT, IN_WIDTH, IN_CHANNELS, POOL_HEIGHT, POOL_WIDTH, STRIDE_HEIGHT, STRIDE_WIDTH, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT>(value);
}

#[test]
fn test_pooling_const() {
    let IN_HEIGHT = u32:5;
    let IN_WIDTH = u32:6;
    let IN_CHANNELS = u32:2;
    let POOL_HEIGHT = u32:3;
    let POOL_WIDTH = u32:2;
    let STRIDE_HEIGHT = u32:2;
    let STRIDE_WIDTH = u32:2;
    let PAD_TOP = u32:1;
    let PAD_BOTTOM = u32:1;
    let PAD_LEFT = u32:1;
    let PAD_RIGHT = u32:1;
    let zero = fixed_point::make_fixed_point<-10>(s16:0);
    let one = fixed_point::make_fixed_point<-10>(s16:1024);
    let min_value = fixed_point_util::min_value<16, -10>();
    let max_value = fixed_point_util::max_value<16, -10>();
    map(
        [zero, one, min_value, max_value],
        test_pooling_const_cases<IN_HEIGHT, IN_WIDTH, IN_CHANNELS, POOL_HEIGHT, POOL_WIDTH, STRIDE_HEIGHT, STRIDE_WIDTH, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT>
    );
}

// Test pooling_1d with non-constant input and simple parameters

// TODO inline and remove this function, use pooling_1d with explicit named parameters instead
pub fn pooling_1d_default
    <POOLING_OP: PoolingOperation,
    DATA_FORMAT: DataFormat,
    // Input
    IN_NB: u32, IN_BE: s32,
    IN_DIM_0: u32, IN_DIM_1: u32,
    // Defaults
    OUT_NB: u32 = {IN_NB}, OUT_BE: s32 = {IN_BE},
    ROUNDING: RoundingMode = {RoundingMode::TRN},
    OVERFLOW: OverflowMode = {OverflowMode::WRAP},
    POOL_SIZE: u32 = {u32:3},
    STRIDE: u32 = {u32:1},
    PAD_LEFT: u32 = {u32:0},
    PAD_RIGHT: u32 = {u32:0},
    COUNT_PAD: bool = {false},
    // Derived input dims
    IN_SIZE: u32 = {data_format::to_size_chans(IN_DIM_0, IN_DIM_1, DATA_FORMAT)[0]},
    IN_CHANNELS: u32 = {data_format::to_size_chans(IN_DIM_0, IN_DIM_1, DATA_FORMAT)[1]},
    // Output size
    OUT_SIZE: u32 = {((IN_SIZE + PAD_LEFT + PAD_RIGHT - POOL_SIZE) / STRIDE) + 1},
    // Output dims
    OUT_DIM_0: u32 = {data_format::from_size_chans(OUT_SIZE, IN_CHANNELS, DATA_FORMAT)[0]},
    OUT_DIM_1: u32 = {data_format::from_size_chans(OUT_SIZE, IN_CHANNELS, DATA_FORMAT)[1]},
    >
(
    x: FixedPoint<IN_NB, IN_BE>[IN_DIM_1][IN_DIM_0]
)
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_1][OUT_DIM_0] {
    pooling_1d<
        OUT_NB, OUT_BE,
        ROUNDING, OVERFLOW,
        POOLING_OP, POOL_SIZE,
        STRIDE, PAD_LEFT, PAD_RIGHT, COUNT_PAD,
        DATA_FORMAT
    >(x)
}

#[test]
fn test_pooling_1d() {
    let NB = u32:16;
    let BE = s32:0;
    let R = RoundingMode::TRN;
    let O = OverflowMode::WRAP;

    let IN_SIZE = u32:5;
    let CHANNELS = u32:1;
    let OUT_SIZE = u32:3;
    let x_flat = fixed_point_util::make_fixed_points_1d<0>([s16:1,2,3,4,5]);
    let expected_max_flat = fixed_point_util::make_fixed_points_1d<0>([s16:3,4,5]);
    let expected_avg_flat = fixed_point_util::make_fixed_points_1d<0>([s16:2,3,4]);
    let expected_global_max_flat = fixed_point_util::make_fixed_points_1d<0>([s16:5]);
    let expected_global_avg_flat = fixed_point_util::make_fixed_points_1d<0>([s16:3]);

    // CHANNELS_LAST
    let x_last = fixed_point_util::reshape_to_2d<IN_SIZE, CHANNELS>(x_flat);
    let expected_max_last = fixed_point_util::reshape_to_2d<OUT_SIZE, CHANNELS>(expected_max_flat);
    let expected_avg_last = fixed_point_util::reshape_to_2d<OUT_SIZE, CHANNELS>(expected_avg_flat);
    assert_eq(
        expected_max_last,
        pooling_1d_default<PoolingOperation::MAX, DataFormat::CHANNELS_LAST>(x_last)
    );
    assert_eq(
        expected_avg_last,
        pooling_1d_default<PoolingOperation::AVERAGE, DataFormat::CHANNELS_LAST>(x_last)
    );
    assert_eq(
        expected_global_max_flat,
        global_pooling_1d<NB, BE, R, O, PoolingOperation::MAX, DataFormat::CHANNELS_LAST>(x_last)
    );
    assert_eq(
        expected_global_avg_flat,
        global_pooling_1d<NB, BE, R, O, PoolingOperation::AVERAGE, DataFormat::CHANNELS_LAST>(x_last)
    );

    // CHANNELS_FIRST
    let x_first = fixed_point_util::reshape_to_2d<CHANNELS, IN_SIZE>(x_flat);
    let expected_max_first = fixed_point_util::reshape_to_2d<CHANNELS, OUT_SIZE>(expected_max_flat);
    let expected_avg_first = fixed_point_util::reshape_to_2d<CHANNELS, OUT_SIZE>(expected_avg_flat);
    assert_eq(
        expected_max_first,
        pooling_1d_default<PoolingOperation::MAX, DataFormat::CHANNELS_FIRST>(x_first)
    );
    assert_eq(
        expected_avg_first,
        pooling_1d_default<PoolingOperation::AVERAGE, DataFormat::CHANNELS_FIRST>(x_first)
    );
    assert_eq(
        expected_global_max_flat,
        global_pooling_1d<NB, BE, R, O, PoolingOperation::MAX, DataFormat::CHANNELS_FIRST>(x_first)
    );
    assert_eq(
        expected_global_avg_flat,
        global_pooling_1d<NB, BE, R, O, PoolingOperation::AVERAGE, DataFormat::CHANNELS_FIRST>(x_first)
    );
}

// Test pooling_2d with non-constant input and simple parameters

#[test]
fn test_pooling_2d() {
    let NB = u32:16;
    let BE = s32:0;
    let R = RoundingMode::TRN;
    let O = OverflowMode::WRAP;

    let IN_HEIGHT = u32:4;
    let IN_WIDTH = u32:4;
    let CHANNELS = u32:1;
    let OUT_HEIGHT = u32:2;
    let OUT_WIDTH = u32:2;
    let POOL_HEIGHT = u32:2;
    let POOL_WIDTH = u32:2;
    let STRIDE_HEIGHT = u32:2;
    let STRIDE_WIDTH = u32:2;
    let PAD_TOP = u32:0;
    let PAD_BOTTOM = u32:0;
    let PAD_LEFT = u32:0;
    let PAD_RIGHT = u32:0;
    let x_flat = fixed_point_util::make_fixed_points_1d<0>([
        s16:1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    ]);
    let expected_max_flat = fixed_point_util::make_fixed_points_1d<0>([s16:6, 8, 14, 16]);
    let expected_avg_flat = fixed_point_util::make_fixed_points_1d<0>([s16:3, 5, 11, 13]);
    let expected_global_max_flat = fixed_point_util::make_fixed_points_1d<0>([s16:16]);
    let expected_global_avg_flat = fixed_point_util::make_fixed_points_1d<0>([s16:8]);

    // CHANNELS_LAST
    let x_last = fixed_point_util::reshape_to_3d<IN_HEIGHT, IN_WIDTH, CHANNELS>(x_flat);
    let expected_max_last = fixed_point_util::reshape_to_3d<OUT_HEIGHT, OUT_WIDTH, CHANNELS>(expected_max_flat);
    let expected_avg_last = fixed_point_util::reshape_to_3d<OUT_HEIGHT, OUT_WIDTH, CHANNELS>(expected_avg_flat);
    assert_eq(
        expected_max_last,
        pooling_2d<NB, BE, R, O, PoolingOperation::MAX, POOL_HEIGHT, POOL_WIDTH, STRIDE_HEIGHT, STRIDE_WIDTH, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT, false, DataFormat::CHANNELS_LAST>(x_last)
    );
    assert_eq(
        expected_avg_last,
        pooling_2d<NB, BE, R, O, PoolingOperation::AVERAGE, POOL_HEIGHT, POOL_WIDTH, STRIDE_HEIGHT, STRIDE_WIDTH, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT, false, DataFormat::CHANNELS_LAST>(x_last)
    );
    assert_eq(
        expected_global_max_flat,
        global_pooling_2d<NB, BE, R, O, PoolingOperation::MAX, DataFormat::CHANNELS_LAST>(x_last)
    );
    assert_eq(
        expected_global_avg_flat,
        global_pooling_2d<NB, BE, R, O, PoolingOperation::AVERAGE, DataFormat::CHANNELS_LAST>(x_last)
    );

    // CHANNELS_FIRST
    let x_first = fixed_point_util::reshape_to_3d<CHANNELS, IN_HEIGHT, IN_WIDTH>(x_flat);
    let expected_max_first = fixed_point_util::reshape_to_3d<CHANNELS, OUT_HEIGHT, OUT_WIDTH>(expected_max_flat);
    let expected_avg_first = fixed_point_util::reshape_to_3d<CHANNELS, OUT_HEIGHT, OUT_WIDTH>(expected_avg_flat);
    assert_eq(
        expected_max_first,
        pooling_2d<NB, BE, R, O, PoolingOperation::MAX, POOL_HEIGHT, POOL_WIDTH, STRIDE_HEIGHT, STRIDE_WIDTH, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT, false, DataFormat::CHANNELS_FIRST>(x_first)
    );
    assert_eq(
        expected_avg_first,
        pooling_2d<NB, BE, R, O, PoolingOperation::AVERAGE, POOL_HEIGHT, POOL_WIDTH, STRIDE_HEIGHT, STRIDE_WIDTH, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT, false, DataFormat::CHANNELS_FIRST>(x_first)
    );
    assert_eq(
        expected_global_max_flat,
        global_pooling_2d<NB, BE, R, O, PoolingOperation::MAX, DataFormat::CHANNELS_FIRST>(x_first)
    );
    assert_eq(
        expected_global_avg_flat,
        global_pooling_2d<NB, BE, R, O, PoolingOperation::AVERAGE, DataFormat::CHANNELS_FIRST>(x_first)
    );
}
