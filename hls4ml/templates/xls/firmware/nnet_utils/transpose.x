import std;
import fixed_point;

import ap_types.fixed_point_util;

type FixedPoint = fixed_point::FixedPoint;
type RoundingMode = fixed_point_util::RoundingMode;
type OverflowMode = fixed_point_util::OverflowMode;

// Simple bubble sort used for checking permutation indices.
fn sort<S: bool, N: u32, DIM: u32>(x: xN[S][N][DIM])-> xN[S][N][DIM] {
    let res = x;
    for (i, res) in 0..DIM {
        for (j, res) in 0..DIM {
            if j > i && res[j] < res[i] {
                update(update(res, i, res[j]), j, res[i])
            } else {
                res
            }
        }(res)
    }(res)
}

#[test]
fn test_sort() {
    assert_eq(sort([u32:0, 1]), [u32:0, 1]);
    assert_eq(sort([u32:1, 0]), [u32:0, 1]);
    assert_eq(sort([u32:3, 1, 2, 0]), [u32:0, 1, 2, 3]);
    assert_eq(sort([u32:2, 1, 2, 0]), [u32:0, 1, 2, 2]);
}

fn permute<S: bool, N: u32, DIM: u32>(x: xN[S][N][DIM], perm: u32[DIM])-> xN[S][N][DIM] {
    let range: u32[DIM] = 0..DIM;
    assert_fmt!(sort(perm) == range, "invalid_perm");
    for (i, res) in 0..DIM {
       update(res, i, x[perm[i]])
    }(x)
}

#[test]
fn test_permute() {
    assert_eq(permute([0,1,2], [1,2,0]), [1,2,0]);
    assert_eq(permute([3,4,5], [1,0,2]), [4,3,5]);
}

pub fn transpose_1d<
    OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    PERM_0: u32,
    IN_NB: u32, IN_BE: s32, DIM: u32,
>
(x: FixedPoint<IN_NB, IN_BE>[DIM])
-> FixedPoint<OUT_NB, OUT_BE>[DIM] {
    const_assert!(PERM_0 == u32:0);
    fixed_point_util::resize_1d<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(x)
}

pub fn transpose_2d<
    OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    // Permutation: (0,1) or (1,0)
    PERM_0: u32, PERM_1: u32,
    IN_NB: u32, IN_BE: s32, IN_DIM_0: u32, IN_DIM_1: u32,
    OUT_DIM_0: u32 = {[IN_DIM_0, IN_DIM_1][PERM_0]},
    OUT_DIM_1: u32 = {[IN_DIM_0, IN_DIM_1][PERM_1]},
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_1][IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_1][OUT_DIM_0] {
    const_assert!(sort([PERM_0, PERM_1]) == u32[2]:[0, 1]);
    let res = zero!<FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_1][OUT_DIM_0]>();
    for (i_0, res) in 0..IN_DIM_0 {
        for (i_1, res) in 0..IN_DIM_1 {
            let out_idx = permute([i_0, i_1], [PERM_0, PERM_1]);
            update(
                res,
                (out_idx[0], out_idx[1]),
                fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(x[i_0][i_1])
            )
        }(res)
    }(res)
}

pub fn transpose_3d<
    OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    PERM_0: u32, PERM_1: u32, PERM_2: u32,
    IN_NB: u32, IN_BE: s32, IN_DIM_0: u32, IN_DIM_1: u32, IN_DIM_2: u32,
    OUT_DIM_0: u32 = {[IN_DIM_0, IN_DIM_1, IN_DIM_2][PERM_0]},
    OUT_DIM_1: u32 = {[IN_DIM_0, IN_DIM_1, IN_DIM_2][PERM_1]},
    OUT_DIM_2: u32 = {[IN_DIM_0, IN_DIM_1, IN_DIM_2][PERM_2]},
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_2][IN_DIM_1][IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_2][OUT_DIM_1][OUT_DIM_0] {
    const_assert!(sort([PERM_0, PERM_1, PERM_2]) == u32[3]:[0, 1, 2]);
    let res = zero!<FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_2][OUT_DIM_1][OUT_DIM_0]>();
    for (i_0, res) in 0..IN_DIM_0 {
        for (i_1, res) in 0..IN_DIM_1 {
            for (i_2, res) in 0..IN_DIM_2 {
                let out_idx = permute([i_0, i_1, i_2], [PERM_0, PERM_1, PERM_2]);
                update(
                    res,
                    (out_idx[0], out_idx[1], out_idx[2]),
                    fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(x[i_0][i_1][i_2])
                )
            }(res)
        }(res)
    }(res)
}


pub fn transpose_4d<
OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
PERM_0: u32, PERM_1: u32, PERM_2: u32, PERM_3: u32,
IN_NB: u32, IN_BE: s32, IN_DIM_0: u32, IN_DIM_1: u32, IN_DIM_2: u32, IN_DIM_3: u32,
OUT_DIM_0: u32 = {[IN_DIM_0, IN_DIM_1, IN_DIM_2, IN_DIM_3][PERM_0]},
OUT_DIM_1: u32 = {[IN_DIM_0, IN_DIM_1, IN_DIM_2, IN_DIM_3][PERM_1]},
OUT_DIM_2: u32 = {[IN_DIM_0, IN_DIM_1, IN_DIM_2, IN_DIM_3][PERM_2]},
OUT_DIM_3: u32 = {[IN_DIM_0, IN_DIM_1, IN_DIM_2, IN_DIM_3][PERM_3]},
>
(x: FixedPoint<IN_NB, IN_BE>[IN_DIM_3][IN_DIM_2][IN_DIM_1][IN_DIM_0])
-> FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_3][OUT_DIM_2][OUT_DIM_1][OUT_DIM_0] {
    const_assert!(sort([PERM_0, PERM_1, PERM_2, PERM_3]) == u32[4]:[0, 1, 2, 3]);
    let res = zero!<FixedPoint<OUT_NB, OUT_BE>[OUT_DIM_3][OUT_DIM_2][OUT_DIM_1][OUT_DIM_0]>();
    for (i_0, res) in 0..IN_DIM_0 {
        for (i_1, res) in 0..IN_DIM_1 {
            for (i_2, res) in 0..IN_DIM_2 {
                for (i_3, res) in 0..IN_DIM_3 {
                    let out_idx = permute([i_0, i_1, i_2, i_3], [PERM_0, PERM_1, PERM_2, PERM_3]);
                    update(
                        res,
                        (out_idx[0], out_idx[1], out_idx[2], out_idx[3]),
                        fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(x[i_0][i_1][i_2][i_3])
                    )
                }(res)
            }(res)
        }(res)
    }(res)
}

// Testing

#[test]
fn test_transpose_2d() {
    let R = RoundingMode::TRN;
    let O = OverflowMode::WRAP;
    let x = fixed_point_util::make_fixed_points_2d<0>([[s16:1, 2, 3], [s16:4, 5, 6]]);
    let x_t = fixed_point_util::make_fixed_points_2d<0>([[s16:1, 4], [s16:2, 5], [s16:3, 6]]);

    assert_eq(x, transpose_2d<16, 0, R, O, 0, 1>(x));
    assert_eq(x_t, transpose_2d<16, 0, R, O, 1, 0>(x));
}

#[test]
fn test_transpose_3d() {
    let R = RoundingMode::TRN;
    let O = OverflowMode::WRAP;
    let x = fixed_point_util::make_fixed_points_3d<0>([
        [[s16:1, 2], [s16:3, 4], [s16:5, 6]],
        [[s16:7, 8], [s16:9, 10], [s16:11, 12]],
    ]);
    let x_102 = fixed_point_util::make_fixed_points_3d<0>([
        [[s16:1, 2], [s16:7, 8]],
        [[s16:3, 4], [s16:9, 10]],
        [[s16:5, 6], [s16:11, 12]],
    ]);
    let x_120 = fixed_point_util::make_fixed_points_3d<0>([
        [[s16:1, 7], [s16:2, 8]],
        [[s16:3, 9], [s16:4, 10]],
        [[s16:5, 11], [s16:6, 12]],
    ]);
    let x_210 = fixed_point_util::make_fixed_points_3d<0>([
        [[s16:1, 7], [s16:3, 9], [s16:5, 11]],
        [[s16:2, 8], [s16:4, 10], [s16:6, 12]],
    ]);

    assert_eq(x, transpose_3d<16, 0, R, O, 0, 1, 2>(x));
    assert_eq(x_102, transpose_3d<16, 0, R, O, 1, 0, 2>(x));
    assert_eq(x_120, transpose_3d<16, 0, R, O, 1, 2, 0>(x));
    assert_eq(x_210, transpose_3d<16, 0, R, O, 2, 1, 0>(x));
}

#[test]
fn test_transpose_4d() {
    let R = RoundingMode::TRN;
    let O = OverflowMode::WRAP;
    let x = fixed_point_util::make_fixed_points_4d<0>([
        [[[s16:1, 2], [s16:3, 4]], [[s16:5, 6], [s16:7, 8]]],
        [[[s16:9, 10], [s16:11, 12]], [[s16:13, 14], [s16:15, 16]]],
    ]);
    let x_3210 = fixed_point_util::make_fixed_points_4d<0>([
        [[[s16:1, 9], [s16:5, 13]], [[s16:3, 11], [s16:7, 15]]],
        [[[s16:2, 10], [s16:6, 14]], [[s16:4, 12], [s16:8, 16]]],
    ]);

    assert_eq(x, transpose_4d<16, 0, R, O, 0, 1, 2, 3>(x));
    assert_eq(x_3210, transpose_4d<16, 0, R, O, 3, 2, 1, 0>(x));
}
