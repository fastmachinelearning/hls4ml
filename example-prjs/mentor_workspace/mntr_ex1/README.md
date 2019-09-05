# Mapping Xilinx on SystemC Arbitrary-Precision Datatypes

This is a first attempt to use SystemC (Mentor Graphics Catapult HLS) arbitrary
precision datatypes as a replacement for the Xilinx ones. The long-term goal
is to reduce the source code modifications when porting *hls4ml* from Xilinx
Vivado HLS to Mentor Graphics Catapult HLS. 

## Implementation Notes

Anoop Saha from Mentor Graphics has provided us with a first version of the
source code. The remapping leverage a feature of C++11 call [template
aliasing](https://en.cppreference.com/w/cpp/language/type_alias):

```
template<int W>
using ap_int = sc_bigint<W>;

template<int W>
using ap_uint = sc_biguint<W>;

template<int W, int I, sc_q_mode Q, sc_o_mode O>
using ap_fixed = sc_fixed<W,I,Q,O>;

template<int W, int I, sc_q_mode Q, sc_o_mode O>
using ap_ufixed = sc_ufixed<W,I,Q,O>;
```

Anoop's source code:
- `inc/ap_int.h`: the code above plus remap (using #define) of rounding and saturation modes
- `inc/ap_fixed.h`: just includes `ap_int.h`
- `src/t.cxx`: small testcase

## Compile and Run

You need Catapult HLS installed on your local machine, but for this example you
do not need any license. Update the `envsetup.sh` script to match the
installation paths on your local machine. Finally:
```
cd sim
source envsetup.sh
```

At this point you can compile and run the test case:
```
make run
```

**Please, note that we are using the the SystemC implementation of `sc_fixed`
types and the `g++` compiler in the Catapult HLS installation.**

## Portability

This example can be also compiled and linked against a vanilla SystemC library
from [Accellera](https://accellera.org/downloads/standards/systemc) and the
algorithmic datatypes from [HLS LIBS](https://hlslibs.org). Both these libraries
are released under an open source license (please, check those details).
