# HLSConfig

In the hls4ml configuration file, it is possible to specify the model *Precision* and *ReuseFactor* with fine granularity.
Under the `HLSConfig` heading, these can be set for the `Model`, per `LayerType`, per `LayerName`, and for named variables within the layer (for precision only).
The most basic configuration may look like this:
```
HLSConfig:
  Model:
    Precision: ap_fixed<16,6>
    ReuseFactor: 1
```
This configuration use `ap_fixed<16,6>` for every variable and a ReuseFactor of 1 throughout.

Specify all `Dense` layers to use a different precision like this:
```
HLSConfig:
  Model:
    Precision: ap_fixed<16,6>
    ReuseFactor: 1
  LayerType:
    Dense:
      Precision: ap_fixed<14,5>
```
In this case, all variables in any `Dense` layers will be represented with `ap_fixed<14,5>` while any other layer types will use `ap_fixed<16,6>`.

A specific layer can be targeted like this:
```
HLSConfig:
  Model:
    Precision: ap_fixed<16,6>
    ReuseFactor: 16
  LayerName:
    dense1:
      Precision: 
        weight: ap_fixed<14,2>
        bias: ap_fixed<14,4>
        result: ap_fixed<16,6>
      ReuseFactor: 12
      Strategy: Resource
```
In this case, the default model configuration will use `ap_fixed<16,6>` and a `ReuseFactor` of 16.
The layer named `dense1` (defined in the user provided model architecture file) will instead use different precision for the `weight`, `bias`, and `result` (output) variables, a `ReuseFactor` of 12, and the `Resource` strategy (while the model default is `Latency` strategy.

More than one layer can have a configuration specified, e.g.:
```
HLSConfig:
  Model:
   ...
  LayerName:
    dense1:
      ...
    batchnormalization1:
      ...
    dense2:
      ...
```
