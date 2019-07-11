# Debug

Vladimir Loncar observed an issue with the current implementions in `nvt/large_mlp_wDep_v4` and `nvt/large_mlp_wDep_v6`.

```
Here is my C program with original (function "dense"), v4 (function "dense_v4")
and v6 (function "dense_v6") implementations, tested on some random integers.
The results for original and v4 match, v6 is different. Changing the weights to
a constant (change the line 230-231) makes the results the same
```

To fix the issue for the attached C example we must transpose the weights. The [HLS writer](../../hls-writer/hls_writer.py) in `nvt/large_mlp_wDep_v6` does that (approx. lines 1039-1040).

## Quick Run

Run with all ones as input weights:
```
make clean
make all-ones
make run
```

Run with _different-value_ weights:
```
make clean
make all
make run
```

