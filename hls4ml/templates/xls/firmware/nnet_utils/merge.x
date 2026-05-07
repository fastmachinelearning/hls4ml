import std;
import fixed_point;

import ap_types.fixed_point_util;

type FixedPoint = fixed_point::FixedPoint;
type RoundingMode = fixed_point_util::RoundingMode;
type OverflowMode = fixed_point_util::OverflowMode;

pub fn add
    <OUT_NB: u32, OUT_BE: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    X_NB: u32, X_BE: s32,
    Y_NB: u32, Y_BE: s32,
    DIM: u32
>
(
    x: FixedPoint<X_NB, X_BE>[DIM],
    y: FixedPoint<Y_NB, Y_BE>[DIM]
)
-> FixedPoint<OUT_NB, OUT_BE>[DIM] {
    for (i, res) in 0..DIM {
        update(res, i,
            fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(
                fixed_point::add(x[i], y[i])
        ))
    }(zero!<FixedPoint<OUT_NB, OUT_BE>[DIM]>())
}

#[test]
fn test_add() {
    let x = fixed_point_util::make_fixed_points_1d<0>([s8:1, 2, 3]);
    let y = fixed_point_util::make_fixed_points_1d<-1>([s16:2, 4, 6]);
    let result = add<8, 0, RoundingMode::TRN, OverflowMode::WRAP>(x, y);
    let expected = fixed_point_util::make_fixed_points_1d<0>([s8:2, 4, 6]);
    assert_eq(result, expected);
}

pub fn subtract
    <OUT_NB: u32, OUT_BE: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    X_NB: u32, X_BE: s32,
    Y_NB: u32, Y_BE: s32,
    DIM: u32
>
(
    x: FixedPoint<X_NB, X_BE>[DIM],
    y: FixedPoint<Y_NB, Y_BE>[DIM]
)
-> FixedPoint<OUT_NB, OUT_BE>[DIM] {
    for (i, res) in 0..DIM {
        update(res, i,
            fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(
                fixed_point::sub(x[i], y[i])
        ))
    }(zero!<FixedPoint<OUT_NB, OUT_BE>[DIM]>())
}

#[test]
fn test_subtract() {
    let x = fixed_point_util::make_fixed_points_1d<0>([s8:10, 20, 30]);
    let y = fixed_point_util::make_fixed_points_1d<-1>([s7:2, 4, 6]);
    let result = subtract<8, 0, RoundingMode::TRN, OverflowMode::WRAP>(x, y);
    let expected = fixed_point_util::make_fixed_points_1d<0>([s8:9, 18, 27]);
    assert_eq(result, expected);
}

pub fn multiply
    <OUT_NB: u32, OUT_BE: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    X_NB: u32, X_BE: s32,
    Y_NB: u32, Y_BE: s32,
    DIM: u32
>
(
    x: FixedPoint<X_NB, X_BE>[DIM],
    y: FixedPoint<Y_NB, Y_BE>[DIM]
)
-> FixedPoint<OUT_NB, OUT_BE>[DIM] {
    for (i, res) in 0..DIM {
        update(res, i,
            fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(
                fixed_point::mul(x[i], y[i])
        ))
    }(zero!<FixedPoint<OUT_NB, OUT_BE>[DIM]>())
}

#[test]
fn test_multiply() {
    let x = fixed_point_util::make_fixed_points_1d<0>([s8:2, 3, 4]);
    let y = fixed_point_util::make_fixed_points_1d<-1>([s7:4, 4, 4]);
    let result = multiply<8, 0, RoundingMode::TRN, OverflowMode::WRAP>(x, y);
    let expected = fixed_point_util::make_fixed_points_1d<0>([s8:4, 6, 8]);
    assert_eq(result, expected);
}

pub fn maximum
    <OUT_NB: u32, OUT_BE: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    X_NB: u32, X_BE: s32,
    Y_NB: u32, Y_BE: s32,
    DIM: u32
>
(
    x: FixedPoint<X_NB, X_BE>[DIM],
    y: FixedPoint<Y_NB, Y_BE>[DIM]
)
-> FixedPoint<OUT_NB, OUT_BE>[DIM] {
    for (i, res) in 0..DIM {
        // NB: cannot compare significants directly if BINARY_EXPONENT's are different.
        let diff = fixed_point::sub(x[i], y[i]);
        let max_value = if(std::msb(diff.significand) == u1:0) {
            fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(x[i])
        } else {
            fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(y[i])
        };
        update(res, i, max_value)
    }(zero!<FixedPoint<OUT_NB, OUT_BE>[DIM]>())
}

#[test]
fn test_maximum() {
    let x = fixed_point_util::make_fixed_points_1d<0>([s8:5, 10, 3]);
    let y = fixed_point_util::make_fixed_points_1d<-1>([s7:10, 7, 18]);
    let result = maximum<8, 0, RoundingMode::TRN, OverflowMode::WRAP>(x, y);
    let expected = fixed_point_util::make_fixed_points_1d<0>([s8:5, 10, 9]);
    assert_eq(result, expected);
}

pub fn minimum
    <OUT_NB: u32, OUT_BE: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    X_NB: u32, X_BE: s32,
    Y_NB: u32, Y_BE: s32,
    DIM: u32
>
(
    x: FixedPoint<X_NB, X_BE>[DIM],
    y: FixedPoint<Y_NB, Y_BE>[DIM]
)
-> FixedPoint<OUT_NB, OUT_BE>[DIM] {
    for (i, res) in 0..DIM {
        // NB: cannot compare significants directly if BINARY_EXPONENT's are different.
        let diff = fixed_point::sub(x[i], y[i]);
        let min_value = if(std::msb(diff.significand) == u1:1) {
            fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(x[i])
        } else {
            fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(y[i])
        };
        update(res, i, min_value)
    }(zero!<FixedPoint<OUT_NB, OUT_BE>[DIM]>())
}

#[test]
fn test_minimum() {
    let x = fixed_point_util::make_fixed_points_1d<0>([s8:5, 10, 3]);
    let y = fixed_point_util::make_fixed_points_1d<-1>([s7:10, 7, 18]);
    let result = minimum<8, 0, RoundingMode::TRN, OverflowMode::WRAP>(x, y);
    let expected = fixed_point_util::make_fixed_points_1d<0>([s8:5, 3, 3]);
    assert_eq(result, expected);
}

pub fn average
    <OUT_NB: u32, OUT_BE: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    X_NB: u32, X_BE: s32,
    Y_NB: u32, Y_BE: s32,
    DIM: u32
>
(
    x: FixedPoint<X_NB, X_BE>[DIM],
    y: FixedPoint<Y_NB, Y_BE>[DIM]
)
-> FixedPoint<OUT_NB, OUT_BE>[DIM] {
    let ONE_HALF = fixed_point::make_fixed_point<-1>(s2:1);
    for (i, res) in 0..DIM {
        let sum = fixed_point::add(x[i], y[i]);
        let avg = fixed_point::mul(sum, ONE_HALF);
        update(res, i,
            fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(avg)
        )
    }(zero!<FixedPoint<OUT_NB, OUT_BE>[DIM]>())
}

#[test]
fn test_average() {
    let x = fixed_point_util::make_fixed_points_1d<0>([s8:1, 2, 3]);
    let y = fixed_point_util::make_fixed_points_1d<-1>([s16:2, 4, 10]);
    let result = average<8, 0, RoundingMode::TRN, OverflowMode::WRAP>(x, y);
    let expected = fixed_point_util::make_fixed_points_1d<0>([s8:1, 2, 4]);
    assert_eq(result, expected);
}

pub fn dot
    <OUT_NB: u32, OUT_BE: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    X_NB: u32, X_BE: s32,
    Y_NB: u32, Y_BE: s32,
    DIM: u32
>
(
    x: FixedPoint<X_NB, X_BE>[DIM],
    y: FixedPoint<Y_NB, Y_BE>[DIM]
)
-> FixedPoint<OUT_NB, OUT_BE>[1] {
    [fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(
        fixed_point_util::dot_prod(x,y)
    )]
}

#[test]
fn test_dot() {
    let x = fixed_point_util::make_fixed_points_1d<0>([s8:1, 2, 3]);
    let y = fixed_point_util::make_fixed_points_1d<-1>([s16:2, 4, 10]);
    let result = dot<8, 0, RoundingMode::TRN, OverflowMode::WRAP>(x, y);
    let expected = fixed_point_util::make_fixed_points_1d<0>([s8:20]);
    assert_eq(result, expected);
}

pub fn concatenate1d
    <OUT_NB: u32, OUT_BE: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    X_NB: u32, X_BE: s32,
    Y_NB: u32, Y_BE: s32,
    X_DIM: u32, 
    Y_DIM: u32, 
    OUT_DIM: u32 = {X_DIM + Y_DIM}
>
(
    x: FixedPoint<X_NB, X_BE>[X_DIM],
    y: FixedPoint<Y_NB, Y_BE>[Y_DIM]
)
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM] {
    let x_out = fixed_point_util::resize_1d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(x);
    let y_out = fixed_point_util::resize_1d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(y);
    x_out ++ y_out
}

#[test]
fn test_concatenate1d() {
    let x = fixed_point_util::make_fixed_points_1d<0>([s8:1, 2, 3]);
    let y = fixed_point_util::make_fixed_points_1d<-1>([s16:2, 4, 10]);
    let result = concatenate1d<8, 0, RoundingMode::TRN, OverflowMode::WRAP>(x, y);
    let expected = fixed_point_util::make_fixed_points_1d<0>([s8:1,2,3,1,2,5]);
    assert_eq(result, expected);
}


// Shape of concatenate(x,y)
fn concatenate_out_shape<N: u32>(
    axis: u32,
    x_shape: u32[N], y_shape: u32[N]
) -> u32[N] {
    // assert!(axis < N, "concatenate_illegal_axis");
    for (i, out_shape) in 0..N {
        let x = x_shape[i];
        let y = y_shape[i];
        let out_dim = if (i == axis) {
            x + y
        } else {
            // assert!(x == y, "concatenate_shape_mismatch");
            x
        };
        update(out_shape, i, out_dim)
    }(x_shape)
}

pub fn concatenate2d
    <OUT_NB: u32, OUT_BE: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    AXIS: u32,
    X_NB: u32, X_BE: s32,
    Y_NB: u32, Y_BE: s32,
    X_DIM_0: u32, X_DIM_1: u32, 
    Y_DIM_0: u32, Y_DIM_1: u32, 
    OUT_DIM_0: u32 = {concatenate_out_shape(AXIS, [X_DIM_0, X_DIM_1], [Y_DIM_0, Y_DIM_1])[0]},
    OUT_DIM_1: u32 = {concatenate_out_shape(AXIS, [X_DIM_0, X_DIM_1], [Y_DIM_0, Y_DIM_1])[1]},
>
(
    x: FixedPoint<X_NB, X_BE>[X_DIM_1][X_DIM_0],
    y: FixedPoint<Y_NB, Y_BE>[Y_DIM_1][Y_DIM_0]
)
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_1][OUT_DIM_0] {
    const_assert!(AXIS < 2);
    let x_out = fixed_point_util::resize_2d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(x);
    let y_out = fixed_point_util::resize_2d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(y);
    let res = zero!<FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_1][OUT_DIM_0]>();
    for (i, res) in 0..OUT_DIM_0 {
        for (j, res) in 0..OUT_DIM_1 {
            let value = match AXIS {
                u32:0 => {
                    if (i < X_DIM_0) {
                        x_out[i][j] 
                    } else {
                        y_out[i - X_DIM_0][j] 
                    }
                },
                u32:1 => {
                    if (j < X_DIM_1){
                        x_out[i][j] 
                    } 
                    else {
                        y_out[i][j - X_DIM_1] 
                    }
                },
                _ => fail!("concatenate2d_axis", res[0][0])
            };
            update(res, (i, j), value)
        }(res)
    }(res)
}

#[test]
fn test_concatenate2d() {
    let x = fixed_point_util::make_fixed_points_2d<0>([[s8:1, 2, 3]]);
    let y = fixed_point_util::make_fixed_points_2d<-1>([[s16:2, 4, 10]]);
    
    let expected_0 = fixed_point_util::make_fixed_points_2d<0>([[s8:1,2,3],[s8:1,2,5]]);
    let result_0 = concatenate2d<8, 0, RoundingMode::TRN, OverflowMode::WRAP, 0>(x, y);
    assert_eq(result_0, expected_0);

    let expected_1 = fixed_point_util::make_fixed_points_2d<0>([[s8:1,2,3,1,2,5]]);
    let result_1 = concatenate2d<8, 0, RoundingMode::TRN, OverflowMode::WRAP, 1>(x, y);
    assert_eq(result_1, expected_1);
}

pub fn concatenate3d
    <OUT_NB: u32, OUT_BE: s32,
    ROUNDING: RoundingMode,
    OVERFLOW: OverflowMode,
    AXIS: u32,
    X_NB: u32, X_BE: s32,
    Y_NB: u32, Y_BE: s32,
    X_DIM_0: u32, X_DIM_1: u32, X_DIM_2: u32,
    Y_DIM_0: u32, Y_DIM_1: u32, Y_DIM_2: u32,
    OUT_DIM_0: u32 = {concatenate_out_shape(AXIS, [X_DIM_0, X_DIM_1, X_DIM_2], [Y_DIM_0, Y_DIM_1, Y_DIM_2])[0]},
    OUT_DIM_1: u32 = {concatenate_out_shape(AXIS, [X_DIM_0, X_DIM_1, X_DIM_2], [Y_DIM_0, Y_DIM_1, Y_DIM_2])[1]},
    OUT_DIM_2: u32 = {concatenate_out_shape(AXIS, [X_DIM_0, X_DIM_1, X_DIM_2], [Y_DIM_0, Y_DIM_1, Y_DIM_2])[2]},
>
(
    x: FixedPoint<X_NB, X_BE>[X_DIM_2][X_DIM_1][X_DIM_0],
    y: FixedPoint<Y_NB, Y_BE>[Y_DIM_2][Y_DIM_1][Y_DIM_0]
)
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_2][OUT_DIM_1][OUT_DIM_0] {
    const_assert!(AXIS < 3);
    let x_out = fixed_point_util::resize_3d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(x);
    let y_out = fixed_point_util::resize_3d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(y);
    let res = zero!<FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_2][OUT_DIM_1][OUT_DIM_0]>();
    for (i, res) in 0..OUT_DIM_0 {
        for (j, res) in 0..OUT_DIM_1 {
            for (k, res) in 0..OUT_DIM_2 {
                let value = match AXIS {
                    u32:0 => {
                        if (i < X_DIM_0) {
                            x_out[i][j][k]
                        } else {
                            y_out[i - X_DIM_0][j][k]
                        }
                    },
                    u32:1 => {
                        if (j < X_DIM_1){
                            x_out[i][j][k]
                        } 
                        else {
                            y_out[i][j - X_DIM_1][k]
                        }
                    },
                    u32:2 => {
                        if (k < X_DIM_2){
                            x_out[i][j][k]
                        } 
                        else {
                            y_out[i][j][k - X_DIM_2]
                        }
                    },
                    _ => fail!("concatenate3d_axis", res[0][0][0])
                };
                update(res, (i, j, k), value)
            }(res)
        }(res)
    }(res)
}

#[test]
fn test_concatenate3d() {
    let x = fixed_point_util::reshape_to_3d<1,2,3>(
        fixed_point_util::make_fixed_points_1d<0>([
            s8:1, 2, 3, 4, 5, 6
        ])
    );
    let y = fixed_point_util::reshape_to_3d<1,2,3>(
        fixed_point_util::make_fixed_points_1d<-1>([
            s8:20, 40, 60, 80, 100, 120
        ])
    );
    
    let expected_0 = fixed_point_util::reshape_to_3d<2,2,3>(
        fixed_point_util::make_fixed_points_1d<0>([
            s8:1,2,3,4,5,6,10,20,30,40,50,60
        ])
    );
    let result_0 = concatenate3d<8, 0, RoundingMode::TRN, OverflowMode::WRAP, 0>(x, y);
    assert_eq(result_0, expected_0);

    let expected_1 = fixed_point_util::reshape_to_3d<1,4,3>(
        fixed_point_util::make_fixed_points_1d<0>([
            s8:1,2,3,4,5,6,10,20,30,40,50,60
        ])
    );
    let result_1 = concatenate3d<8, 0, RoundingMode::TRN, OverflowMode::WRAP, 1>(x, y);
    assert_eq(result_1, expected_1);

    let expected_2 = fixed_point_util::reshape_to_3d<1,2,6>(
        fixed_point_util::make_fixed_points_1d<0>([
            s8:1,2,3,10,20,30,4,5,6,40,50,60
        ])
    );
    let result_2 = concatenate3d<8, 0, RoundingMode::TRN, OverflowMode::WRAP, 2>(x, y);
    assert_eq(result_2, expected_2);
}