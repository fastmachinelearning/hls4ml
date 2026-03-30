import fixed_point;
import ap_types.fixed_point_util;

// hls-fpga-machine-learning insert imports


// hls-fpga-machine-learning debugging


// Input and output types: arrays of FixedPoint
pub fn myproject_fixed_point(x: InputType) -> OutputType {
    // hls-fpga-machine-learning insert layers
}

// Input and output types: arrays of sN[N]
pub fn myproject_bits(x: InputTypeBits) -> OutputTypeBits {
    // hls-fpga-machine-learning convert from bits
}

// Top-level function
pub fn myproject(x: InputTypeBits) -> OutputTypeBits {
    myproject_bits(x)
}