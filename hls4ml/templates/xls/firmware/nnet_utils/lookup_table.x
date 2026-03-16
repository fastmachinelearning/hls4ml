import std;
import fixed_point;
import ap_types.fixed_point_util;

type FixedPoint = fixed_point::FixedPoint;

// Table of values f(x[i]), i=0..SIZE
// where x[i] = x_min + i * dx,
// dx = 2^(LOG2_STEP)
pub struct LookupTable<
    NB_IN: u32, BE_IN: s32,
    NB_OUT: u32, BE_OUT: s32,
    SIZE: u32,
    LOG2_STEP: s32>{
    
    x_min: FixedPoint<NB_IN, BE_IN>,
    values: FixedPoint<NB_OUT, BE_OUT>[SIZE]
}

fn const_validate_lookup_table_params<
    NB_IN: u32, BE_IN: s32,
    NB_OUT: u32, BE_OUT: s32,
    SIZE: u32,
    LOG2_STEP: s32>(){

    const_assert!(SIZE >= 1);

    // Step should not be smaller than allowed by FixedPoint, i.e. 2^(BE_IN)
    const_assert!(LOG2_STEP >= BE_IN);
    let SHIFT = (LOG2_STEP - BE_IN) as u32;
    
    // Check that DELTA = (x_max - x_min) does not overflow
    let DELTA = ((SIZE - 1) as uN[32 + SHIFT]) << SHIFT;
    let MAX_DELTA = std::unsigned_max_value<NB_IN>();
    
    let NB_MAX = std::max(32 + SHIFT, NB_IN);
    const_assert!(DELTA as uN[NB_MAX] <= MAX_DELTA as uN[NB_MAX]);
}

// Check for overflows
fn validate_lookup_table<
    NB_IN: u32, BE_IN: s32,
    NB_OUT: u32, BE_OUT: s32,
    SIZE: u32,
    LOG2_STEP: s32>(
        lut: LookupTable<NB_IN, BE_IN, NB_OUT, BE_OUT, SIZE, LOG2_STEP>
    ){
    // Check statically everything that is possible
    const_validate_lookup_table_params<NB_IN, BE_IN, NB_OUT, BE_OUT, SIZE, LOG2_STEP>();

    // Now check for x_max overflow
    let SHIFT = (LOG2_STEP - BE_IN) as u32;
    // DELTA = x_max - x_min = step * (SIZE - 1)
    let DELTA = ((SIZE - 1) as sN[NB_IN + 1]) << SHIFT;
    let x_min = lut.x_min.significand as sN[NB_IN + 1];
    let x_max = x_min + DELTA;
    assert_fmt!(x_max <= std::signed_max_value<NB_IN>() as sN[NB_IN + 1], "lookup_table_x_max_overflow");
}

// Check arguments and create LUT
pub fn create<
    LOG2_STEP: s32,
    // Other parametes are deduced automatically, so we put them after LOG2_STEP
    NB_IN: u32, BE_IN: s32,
    NB_OUT: u32, BE_OUT: s32,
    SIZE: u32,
    >(
        x_min: FixedPoint<NB_IN, BE_IN>,
        values: FixedPoint<NB_OUT, BE_OUT>[SIZE]
    ) -> LookupTable<NB_IN, BE_IN, NB_OUT, BE_OUT, SIZE, LOG2_STEP> {

    let lut = LookupTable<NB_IN, BE_IN, NB_OUT, BE_OUT, SIZE, LOG2_STEP>{
        x_min: x_min,
        values: values
    };
    validate_lookup_table(lut);
    lut
}

pub fn eval<
    NB_IN: u32, BE_IN: s32,
    NB_OUT: u32, BE_OUT: s32,
    SIZE: u32, LOG2_STEP: s32>(
        lut: LookupTable<NB_IN, BE_IN, NB_OUT, BE_OUT, SIZE, LOG2_STEP>,
        fxp_x: FixedPoint<NB_IN, BE_IN>
    ) -> FixedPoint<NB_OUT, BE_OUT> {
    
    const_validate_lookup_table_params<NB_IN, BE_IN, NB_OUT, BE_OUT, SIZE, LOG2_STEP>();
    
    let SHIFT = (LOG2_STEP - BE_IN) as u32;   

    // add extra bit to avoid overflow
    let x = fxp_x.significand as sN[NB_IN + 1];
    let x_min = lut.x_min.significand as sN[NB_IN +1];
    let delta = x - x_min;
    
    let idx = delta >> SHIFT;
    // clamp
    let idx = std::max(0, idx) as u32;
    let idx = std::min(idx, SIZE - 1);

    lut.values[idx]
}

pub fn eval_1d<
    NB_IN: u32, BE_IN: s32,
    NB_OUT: u32, BE_OUT: s32,
    SIZE: u32, LOG2_STEP: s32,
    DIM: u32>(
        lut: LookupTable<NB_IN, BE_IN, NB_OUT, BE_OUT, SIZE, LOG2_STEP>,
        x: FixedPoint<NB_IN, BE_IN>[DIM]
    ) -> FixedPoint<NB_OUT, BE_OUT>[DIM] {
    
    for (i, res) in 0..DIM{
        update(res, i, eval(lut, x[i]))
    }(zero!<FixedPoint<NB_OUT, BE_OUT>[DIM]>())
}

// =========================================================================
// --------------------------------- Tests ----------------------------------

fn from_integer<NB: u32, BE: s32, NB_IN: u32>(x:sN[NB_IN]) -> FixedPoint<NB, BE> {
    let res:FixedPoint<NB_IN,0> = fixed_point::from_integer(x);
    let res:FixedPoint<NB, BE> = fixed_point::to_common_type<NB, BE, NB_IN, 0>(res);
    res
}

fn x_values<
    NB_IN: u32, BE_IN: s32,
    NB_OUT: u32, BE_OUT: s32,
    SIZE: u32, LOG2_STEP: s32>(
        lut: LookupTable<NB_IN, BE_IN, NB_OUT, BE_OUT, SIZE, LOG2_STEP>
    ) -> FixedPoint<NB_IN, BE_IN>[SIZE] {

    let SHIFT = (LOG2_STEP - BE_IN) as u32;
    let step = fixed_point::make_fixed_point<BE_IN>(sN[NB_IN]:1 << SHIFT);
    let (_, res) = for (i, (x, xs)) in 0..SIZE{
        let x_next = fixed_point_util::add_already_widened(x, step);
        (x_next, update(xs, i, x))
    }((lut.x_min, zero!<FixedPoint<NB_IN, BE_IN>[SIZE]>()));
    res
}

fn plus_one<
    NB_OUT: u32, BE_OUT: s32,
    NB_IN: u32, BE_IN: s32>(
        x: FixedPoint<NB_IN, BE_IN>
    ) -> FixedPoint<NB_OUT, BE_OUT>{
    
    fixed_point::to_common_type<NB_OUT, BE_OUT>(
        fixed_point::add(x, fixed_point::from_integer(s2:1))
    )
}

#[test]
fn test_lookup_table(){
    
    let NB_IN = u32:8;
    let BE_IN = s32:-3;
    let NB_OUT = NB_IN + 1;
    let BE_OUT = BE_IN - 1;

    // xs = [-3,-2,..6]
    let LOG2_STEP = s32:0;
    let SIZE = u32:10;
    
    let x_min = s32:-3;
    let xs = x_min..(x_min + (SIZE as s32));
    let ys = (x_min + 1)..(x_min + (SIZE as s32) + 1);
    
    
    let xs_lut = map(xs, from_integer<NB_IN, BE_IN>);    
    let ys_lut = map(ys, from_integer<NB_OUT, BE_OUT>);

    let lut = create<LOG2_STEP>(
        from_integer<NB_IN, BE_IN>(x_min),
        map(ys, from_integer<NB_OUT, BE_OUT>)
    );

    
    let lut_keys = x_values(lut);
    let lut_values = lut.values;
    
    // Check consistency
    assert_eq(lut_keys, xs_lut);
    assert_eq(lut_values, eval_1d(lut, lut_keys));
    
    // TODO check intermediate values
    // TODO check input outside of lut_keys
    // Check overflow
    {
        let x = fixed_point_util::make_fixed_points_1d<BE_IN>([
            std::signed_min_value<NB_IN>(),
            std::signed_max_value<NB_IN>()
        ]);
        let expected = [
            lut_values[0],
            lut_values[SIZE - 1]
        ];
        assert_eq(expected, eval_1d(lut, x));
    }
}
