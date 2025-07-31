
import std;


pub fn idx_from_real_val
    <TABLE_SZ: u32, NB: u32,
    N: u32 = {std::clog2(TABLE_SZ)},
    LOW_END: u32 = {NB - (NB - N)}>
    (x: sN[NB]) -> uN[N] {

    let unsgined_x = x as uN[NB];
    let idx = (unsgined_x >> LOW_END) & ((uN[NB]:1 << N) - uN[NB]:1);
    idx as uN[N]
}

#[test]
fn idx_from_real_val_test() {
    let x = sN[16]:1024; 
    let expected = uN[10]:1;  
    assert_eq(expected, idx_from_real_val<u32:1024, u32:16, u32:10>(x));

    let x = sN[16]:4096; 
    let expected = uN[10]:4;  
    assert_eq(expected, idx_from_real_val<u32:1024, u32:16, u32:10>(x));

    let x = sN[18]:4096; 
    let expected = uN[10]:4;  
    assert_eq(expected, idx_from_real_val<u32:1024, u32:18, u32:10>(x));
}

// hls-fpga-machine-learning insert exponent table


// hls-fpga-machine-learning insert inversion table

