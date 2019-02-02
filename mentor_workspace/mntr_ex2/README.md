# Porting of an HLS Project to SystemC Arbitrary-Precision Datatypes

This **hls4ml** project supports SystemC arbitrary-precision datatypes provided
with Catapult HLS.

- The project **does not run HLS** and it is meant for **simulation only**.
- The project has been generated with [keras-config.yml](../../keras-to-hls/keras-config.yml) and manually modified.
- The generators have not been modified.
- The [nnet_utils](../../nnet_utils) have been modified. See the following notes. 

## Implementation Notes

- Remapping of `ap_fixed<>` from the Arbitrary Precision datatype library to `sc_fixed<>` from the Algorithmic C datatype library.
  ```
  #include < ap_int.h > // From Vivado HLS
  ap_fixed < a, b >
  ```
  to
  ```
  #include < ap_int.h > // This includes < ac_int.h >
  ap_fixed < a, b, AP_TRN_ZERO, AP_WRAP >
  ```

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

## Compile and Run

You need Catapult HLS installed on your local machine, but for this example you
do not need any license. Update the `sim/envsetup-mntr.sh` and
`sim/envsetup-xlnx.sh` scripts to match the installation paths on your local
machine. Finally:

### Catapult HLS

Use a new console.
```
cd sim
source envsetup-mntr.sh
make clean
make run
```

We are using the Algorithmic Datatypes (`ac_fixed`), the SystemC implementation
and the `g++` compiler in the Catapult HLS installation.

### Vivado HLS

Use a new console.
```
cd sim
source envsetup-xlnx.sh
make -f Makefile.xlnx.mak clean
make -f Makefile.xlnx.mak run
```

We are using the Xilinx HLS libraries.

## Portability

This example can be also compiled and linked against a vanilla SystemC library
from [Accellera](https://accellera.org/downloads/standards/systemc) and the
algorithmic datatypes from [HLS LIBS](https://hlslibs.org). Both these libraries
are released under an open source license (please, check those details).
