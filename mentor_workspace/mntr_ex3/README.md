# Synthesis of a HLS4ML Project with Catapult HLS

This **hls4ml** project supports `ac_fixed` and `ac_int` arbitrary-precision
datatypes provided with Catapult HLS. There are few differences between
`sc_fixed`, `sc_int` and `ac_fixed`, `ac_int`.

- The project **runs Catapult HLS**.
- The project has been generated with [keras-config.yml](../../keras-to-hls/keras-config.yml) and manually modified.
- The generators have not been modified.
- The [nnet_utils](../../nnet_utils) have been modified. See the following notes.

## Implementation Notes

- Remapping of `ap_fixed<>` from the Arbitrary Precision datatype library to `ac_fixed<>` from the Algorithmic C datatype library (which has differencies w.r.t. SystemC `sc_fixed<>` and `sc_int<>`). See [inc](./inc) folder.
  ```
  #include <ac_int.h>
  #include <ac_fixed.h>

  template<int W>
  using ap_int = ac_int<W, true>;

  template<int W>
  using ap_uint = ac_int<W, false>;

  template<int W, int I>
  using ap_fixed = ac_fixed<W,I,true>;

  template<int W, int I>
  using ap_ufixed = ac_fixed<W,I,false>;
  ```

- Vivado HLS has a wider support for fixed/floating point arithmetic. Catapult HLS may
  - not support the synthesis of some operators (e.g. division `/`, it is necessary to use various functions that implement those operators)
  - have problems with implicit casts (e.g. it is necessary to make everything explicit)
  - have more *arithmetic functions* for the `ac_fixed` and `ac_int` w.r.t. `sc_fixed` and `sc_int` (e.g. `mgc_ac_exp` and `mgc_ac_log` (CORDIC) are not provided for `sc_fixed`)
  - have limited support for floating point arithmetic, thus it is necessary to convert the functions from float to fixed point. For example:
    ```
    // *************************************************
    //       Softplus Activation
    // *************************************************
    #ifdef MNTR_CATAPULT_HLS
    inline ac_fixed<18,8,true> softplus_fcn_float(ac_fixed<18,8,true> input) {
       ac_fixed<18,8,true> _exp; mgc_ac_exp(input, _exp);
       ac_fixed<18,8,true> _log; mgc_ac_log(_exp, _log);
       return _log + ac_fixed<18,8,true>(1.);
     }
     #else
     inline float softplus_fcn_float(float input) {
       return std::log(std::exp(input) + 1.);
     }
     #endif
     ```
   This porting requires some error propagation analysis to avoid overflow and lost of precision.

- Change the Streaming data interface.
  ```
  #include "hls_stream.h"
  hls::stream< data_T > &data
  ```
  to
  ```
  #include "ac_channel.h"
  ac_channel< data_T > &data
  ```

- Some of these changes are under conditional macros.
  ```
  #ifdef MNTR_CATAPULT_HLS
    // Catapult HLS code
  #else
    // Vivado HLS code
  #endif
  ```

- Currently I am getting an error (in Catapult HLS libraries). This may require some support from Mentor.
  ```
  # Error: $MGC_HOME/shared/include/math/mgc_ac_hcordic.h(98): class "MgcAcHtrigAssert<false>" has no member "test" (CRD-135) 
  # Error: $MGC_HOME/shared/include/math/mgc_ac_hcordic.h(98):           detected during: (CRD-135) 
  # Error: $MGC_HOME/shared/include/math/mgc_ac_hcordic.h(98):             instantiation of "ac_fixed<W, 0, false, AC_TRN, AC_WRAP> MgcAcHtrig::ln2<W>() [with W=196]" at line 409 (CRD-135)
  # Error: $MGC_HOME/shared/include/math/mgc_ac_hcordic.h(98):             instantiation of "void mgc_ac_exp(const ac_fixed<AW, AI, AS, AQ, AV> &, ac_fixed<ZW, ZI, ZS, ZQ, ZV> &) [with AW=18, AI=8, AS=true, AQ=AC_TRN, AV=AC_W
  RAP, ZW=18, ZI=8, ZS=true, ZQ=AC_TRN, ZV=AC_WRAP]" at line 190 of "/extras/giuseppe/research/projects/hls4ml/hls4ml.git/nnet_utils/nnet_activation.h" (CRD-135) 
  ```

## Synthesis

You need Catapult HLS installed on your local machine. Update the
`syn/envsetup.sh` script to match the installation paths on your local machine.
Finally:

```
cd syn
source envsetup.sh
make hls
```
