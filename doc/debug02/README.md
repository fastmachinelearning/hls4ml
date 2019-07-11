# Debug

Vladimir Loncar's implementation for `nvt/large_mlp_wDep_v4` and `nvt/large_mlp_wDep_v6` where modulus operators are removed.

This works `(RF > n_in) => (RF % n_in == 0)` which is `(RF % n_in == 0) or (RF >= n_in)`.

# Quick Run

```
make clean
make run
```

