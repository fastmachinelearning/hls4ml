import fixed_point;

import ap_types.fixed_point_util;

type FixedPoint = fixed_point::FixedPoint;
type RoundingMode = fixed_point_util::RoundingMode;
type OverflowMode = fixed_point_util::OverflowMode;

pub fn reshape_1d_to_1d<
    OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    OUT_DIM_0: u32,
    IN_NB: u32, IN_BE: s32, IN_DIM_0: u32,
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_0] {
    const_assert!(IN_DIM_0 == OUT_DIM_0);
    fixed_point_util::resize_1d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(x)
}

pub fn reshape_1d_to_2d<
    OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    OUT_DIM_0: u32, OUT_DIM_1: u32,
    IN_NB: u32, IN_BE: s32, IN_DIM_0: u32,
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_1][OUT_DIM_0] {
    const_assert!(IN_DIM_0 == OUT_DIM_0 * OUT_DIM_1);
    let x_flat = fixed_point_util::resize_1d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(x);
    fixed_point_util::reshape_to_2d<OUT_DIM_0, OUT_DIM_1>(x_flat)
}

pub fn reshape_1d_to_3d<
    OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    OUT_DIM_0: u32, OUT_DIM_1: u32, OUT_DIM_2: u32,
    IN_NB: u32, IN_BE: s32, IN_DIM_0: u32,
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_2][OUT_DIM_1][OUT_DIM_0] {
    const_assert!(IN_DIM_0 == OUT_DIM_0 * OUT_DIM_1 * OUT_DIM_2);
    let x_flat = fixed_point_util::resize_1d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(x);
    fixed_point_util::reshape_to_3d<OUT_DIM_0, OUT_DIM_1, OUT_DIM_2>(x_flat)
}

pub fn reshape_1d_to_4d<
    OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    OUT_DIM_0: u32, OUT_DIM_1: u32, OUT_DIM_2: u32, OUT_DIM_3: u32,
    IN_NB: u32, IN_BE: s32, IN_DIM_0: u32,
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_3][OUT_DIM_2][OUT_DIM_1][OUT_DIM_0] {
    const_assert!(IN_DIM_0 == OUT_DIM_0 * OUT_DIM_1 * OUT_DIM_2 * OUT_DIM_3);
    let x_flat = fixed_point_util::resize_1d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(x);
    fixed_point_util::reshape_to_4d<OUT_DIM_0, OUT_DIM_1, OUT_DIM_2, OUT_DIM_3>(x_flat)
}

pub fn reshape_2d_to_1d<
    OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    OUT_DIM_0: u32,
    IN_NB: u32, IN_BE: s32, IN_DIM_0: u32, IN_DIM_1: u32,
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_1][IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_0] {
    const_assert!(IN_DIM_0 * IN_DIM_1 == OUT_DIM_0);
    let x_flat = fixed_point_util::flatten_2d(x);
    fixed_point_util::resize_1d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(x_flat)
}

pub fn reshape_2d_to_2d<
    OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    OUT_DIM_0: u32, OUT_DIM_1: u32,
    IN_NB: u32, IN_BE: s32, IN_DIM_0: u32, IN_DIM_1: u32,
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_1][IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_1][OUT_DIM_0] {
    const_assert!(IN_DIM_0 * IN_DIM_1 == OUT_DIM_0 * OUT_DIM_1);
    let x_flat = fixed_point_util::resize_1d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(fixed_point_util::flatten_2d(x));
    fixed_point_util::reshape_to_2d<OUT_DIM_0, OUT_DIM_1>(x_flat)
}

pub fn reshape_2d_to_3d<
    OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    OUT_DIM_0: u32, OUT_DIM_1: u32, OUT_DIM_2: u32,
    IN_NB: u32, IN_BE: s32, IN_DIM_0: u32, IN_DIM_1: u32,
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_1][IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_2][OUT_DIM_1][OUT_DIM_0] {
    const_assert!(IN_DIM_0 * IN_DIM_1 == OUT_DIM_0 * OUT_DIM_1 * OUT_DIM_2);
    let x_flat = fixed_point_util::resize_1d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(fixed_point_util::flatten_2d(x));
    fixed_point_util::reshape_to_3d<OUT_DIM_0, OUT_DIM_1, OUT_DIM_2>(x_flat)
}

pub fn reshape_2d_to_4d<
    OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    OUT_DIM_0: u32, OUT_DIM_1: u32, OUT_DIM_2: u32, OUT_DIM_3: u32,
    IN_NB: u32, IN_BE: s32, IN_DIM_0: u32, IN_DIM_1: u32,
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_1][IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_3][OUT_DIM_2][OUT_DIM_1][OUT_DIM_0] {
    const_assert!(IN_DIM_0 * IN_DIM_1 == OUT_DIM_0 * OUT_DIM_1 * OUT_DIM_2 * OUT_DIM_3);
    let x_flat = fixed_point_util::resize_1d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(fixed_point_util::flatten_2d(x));
    fixed_point_util::reshape_to_4d<OUT_DIM_0, OUT_DIM_1, OUT_DIM_2, OUT_DIM_3>(x_flat)
}

pub fn reshape_3d_to_1d<
    OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    OUT_DIM_0: u32,
    IN_NB: u32, IN_BE: s32, IN_DIM_0: u32, IN_DIM_1: u32, IN_DIM_2: u32,
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_2][IN_DIM_1][IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_0] {
    const_assert!(IN_DIM_0 * IN_DIM_1 * IN_DIM_2 == OUT_DIM_0);
    let x_flat = fixed_point_util::flatten_3d(x);
    fixed_point_util::resize_1d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(x_flat)
}

pub fn reshape_3d_to_2d<
    OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    OUT_DIM_0: u32, OUT_DIM_1: u32,
    IN_NB: u32, IN_BE: s32, IN_DIM_0: u32, IN_DIM_1: u32, IN_DIM_2: u32,
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_2][IN_DIM_1][IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_1][OUT_DIM_0] {
    const_assert!(IN_DIM_0 * IN_DIM_1 * IN_DIM_2 == OUT_DIM_0 * OUT_DIM_1);
    let x_flat = fixed_point_util::resize_1d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(fixed_point_util::flatten_3d(x));
    fixed_point_util::reshape_to_2d<OUT_DIM_0, OUT_DIM_1>(x_flat)
}

pub fn reshape_3d_to_3d<
    OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    OUT_DIM_0: u32, OUT_DIM_1: u32, OUT_DIM_2: u32,
    IN_NB: u32, IN_BE: s32, IN_DIM_0: u32, IN_DIM_1: u32, IN_DIM_2: u32,
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_2][IN_DIM_1][IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_2][OUT_DIM_1][OUT_DIM_0] {
    const_assert!(IN_DIM_0 * IN_DIM_1 * IN_DIM_2 == OUT_DIM_0 * OUT_DIM_1 * OUT_DIM_2);
    let x_flat = fixed_point_util::resize_1d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(fixed_point_util::flatten_3d(x));
    fixed_point_util::reshape_to_3d<OUT_DIM_0, OUT_DIM_1, OUT_DIM_2>(x_flat)
}

pub fn reshape_3d_to_4d<
    OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    OUT_DIM_0: u32, OUT_DIM_1: u32, OUT_DIM_2: u32, OUT_DIM_3: u32,
    IN_NB: u32, IN_BE: s32, IN_DIM_0: u32, IN_DIM_1: u32, IN_DIM_2: u32,
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_2][IN_DIM_1][IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_3][OUT_DIM_2][OUT_DIM_1][OUT_DIM_0] {
    const_assert!(IN_DIM_0 * IN_DIM_1 * IN_DIM_2 == OUT_DIM_0 * OUT_DIM_1 * OUT_DIM_2 * OUT_DIM_3);
    let x_flat = fixed_point_util::resize_1d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(fixed_point_util::flatten_3d(x));
    fixed_point_util::reshape_to_4d<OUT_DIM_0, OUT_DIM_1, OUT_DIM_2, OUT_DIM_3>(x_flat)
}

pub fn reshape_4d_to_1d<
    OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    OUT_DIM_0: u32,
    IN_NB: u32, IN_BE: s32, IN_DIM_0: u32, IN_DIM_1: u32, IN_DIM_2: u32, IN_DIM_3: u32,
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_3][IN_DIM_2][IN_DIM_1][IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_0] {
    const_assert!(IN_DIM_0 * IN_DIM_1 * IN_DIM_2 * IN_DIM_3 == OUT_DIM_0);
    let x_flat = fixed_point_util::flatten_4d(x);
    fixed_point_util::resize_1d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(x_flat)
}

pub fn reshape_4d_to_2d<
    OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    OUT_DIM_0: u32, OUT_DIM_1: u32,
    IN_NB: u32, IN_BE: s32, IN_DIM_0: u32, IN_DIM_1: u32, IN_DIM_2: u32, IN_DIM_3: u32,
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_3][IN_DIM_2][IN_DIM_1][IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_1][OUT_DIM_0] {
    const_assert!(IN_DIM_0 * IN_DIM_1 * IN_DIM_2 * IN_DIM_3 == OUT_DIM_0 * OUT_DIM_1);
    let x_flat = fixed_point_util::resize_1d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(fixed_point_util::flatten_4d(x));
    fixed_point_util::reshape_to_2d<OUT_DIM_0, OUT_DIM_1>(x_flat)
}

pub fn reshape_4d_to_3d<
    OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    OUT_DIM_0: u32, OUT_DIM_1: u32, OUT_DIM_2: u32,
    IN_NB: u32, IN_BE: s32, IN_DIM_0: u32, IN_DIM_1: u32, IN_DIM_2: u32, IN_DIM_3: u32,
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_3][IN_DIM_2][IN_DIM_1][IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_2][OUT_DIM_1][OUT_DIM_0] {
    const_assert!(IN_DIM_0 * IN_DIM_1 * IN_DIM_2 * IN_DIM_3 == OUT_DIM_0 * OUT_DIM_1 * OUT_DIM_2);
    let x_flat = fixed_point_util::resize_1d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(fixed_point_util::flatten_4d(x));
    fixed_point_util::reshape_to_3d<OUT_DIM_0, OUT_DIM_1, OUT_DIM_2>(x_flat)
}

pub fn reshape_4d_to_4d<
    OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    OUT_DIM_0: u32, OUT_DIM_1: u32, OUT_DIM_2: u32, OUT_DIM_3: u32,
    IN_NB: u32, IN_BE: s32, IN_DIM_0: u32, IN_DIM_1: u32, IN_DIM_2: u32, IN_DIM_3: u32,
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_3][IN_DIM_2][IN_DIM_1][IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_3][OUT_DIM_2][OUT_DIM_1][OUT_DIM_0] {
    const_assert!(IN_DIM_0 * IN_DIM_1 * IN_DIM_2 * IN_DIM_3 == OUT_DIM_0 * OUT_DIM_1 * OUT_DIM_2 * OUT_DIM_3);
    let x_flat = fixed_point_util::resize_1d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(fixed_point_util::flatten_4d(x));
    fixed_point_util::reshape_to_4d<OUT_DIM_0, OUT_DIM_1, OUT_DIM_2, OUT_DIM_3>(x_flat)
}

#[test]
fn test_reshape_1d_to_4d() {
    let R = RoundingMode::TRN;
    let O = OverflowMode::WRAP;
    let x = fixed_point_util::make_fixed_points_1d<0>([s16:1, 2, 3, 4, 5, 6, 7, 8]);
    let expected = fixed_point_util::make_fixed_points_4d<0>([
        [[[s16:1, 2], [s16:3, 4]]],
        [[[s16:5, 6], [s16:7, 8]]],
    ]);
    assert_eq(expected, reshape_1d_to_4d<16, 0, R, O, 2, 1, 2, 2>(x));
}

#[test]
fn test_reshape_2d_to_4d() {
    let R = RoundingMode::TRN;
    let O = OverflowMode::WRAP;
    let x = fixed_point_util::make_fixed_points_2d<0>([[s16:1, 2, 3, 4], [s16:5, 6, 7, 8]]);
    let expected = fixed_point_util::make_fixed_points_4d<0>([
        [[[s16:1, 2], [s16:3, 4]]],
        [[[s16:5, 6], [s16:7, 8]]],
    ]);
    assert_eq(expected, reshape_2d_to_4d<16, 0, R, O, 2, 1, 2, 2>(x));
}

#[test]
fn test_reshape_4d_to_1d() {
    let R = RoundingMode::TRN;
    let O = OverflowMode::WRAP;
    let x = fixed_point_util::make_fixed_points_4d<0>([
        [[[s16:1, 2], [s16:3, 4]]],
        [[[s16:5, 6], [s16:7, 8]]],
    ]);
    let expected = fixed_point_util::make_fixed_points_1d<0>([s16:1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq(expected, reshape_4d_to_1d<16, 0, R, O, 8>(x));
}

#[test]
fn test_reshape_3d_to_2d_resize() {
    let R = RoundingMode::TRN;
    let O = OverflowMode::WRAP;
    let x = fixed_point_util::make_fixed_points_3d<0>([
        [[s8:1, 2], [s8:3, 4]],
        [[s8:5, 6], [s8:7, 8]],
    ]);
    let expected = fixed_point_util::make_fixed_points_2d<-1>([[s16:2, 4], [s16:6, 8], [s16:10, 12], [s16:14, 16]]);
    assert_eq(expected, reshape_3d_to_2d<16, -1, R, O, 4, 2>(x));
}
