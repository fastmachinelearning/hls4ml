import std;
import fixed_point;

import ap_types.fixed_point_util;

type FixedPoint = fixed_point::FixedPoint;
type RoundingMode = fixed_point_util::RoundingMode;
type OverflowMode = fixed_point_util::OverflowMode;

pub fn normalize<
    OUT_NB: u32, OUT_BE: s32, ROUNDING: RoundingMode, OVERFLOW: OverflowMode,
    IN_NB: u32, IN_BE: s32, DIM: u32,
    SCALE_NB: u32, SCALE_BE: s32, SCALE_DIM: u32,
    BIAS_NB: u32, BIAS_BE: s32, BIAS_DIM: u32 = {SCALE_DIM}
>
(
    x: FixedPoint<IN_NB, IN_BE>[DIM],
    scale: FixedPoint<SCALE_NB, SCALE_BE>[SCALE_DIM],
    bias: FixedPoint<BIAS_NB, BIAS_BE>[BIAS_DIM],
)
-> FixedPoint<OUT_NB, OUT_BE>[DIM] {
    for (i, acc) in 0..DIM {
        update(acc, i, 
            fixed_point_util::resize<OUT_NB, OUT_BE, ROUNDING, OVERFLOW>(
                fixed_point::add(bias[i % BIAS_DIM],
                    fixed_point::mul(scale[i % SCALE_DIM], x[i])
                )
            )
        )
    }(zero!<FixedPoint<OUT_NB, OUT_BE>[DIM]>())
}

#[test]
fn normalize_mixed_precision_test() {
    let x = fixed_point_util::make_fixed_points_1d<0>(s4[4]:[
        1, 2, 3, 4
    ]);
    let scale = fixed_point_util::make_fixed_points_1d<-1>(s3[2]:[
        2, 3
    ]);
    let bias = fixed_point_util::make_fixed_points_1d<-2>(s3[2]:[
        1, 2
    ]);

    let expected = fixed_point_util::make_fixed_points_1d<-2>(s6[4]:[
        5, 14, 13, 26
    ]);

    assert_eq(expected,
        normalize<6, -2, RoundingMode::TRN, OverflowMode::WRAP>(x, scale, bias)
    );
}
