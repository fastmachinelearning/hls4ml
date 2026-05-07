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