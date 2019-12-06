# Tcl do file for compile of mnist_mlp interface



quietly set cmd [format "vlog -timescale 1ps/1ps +incdir+%s/environment_packages/mnist_mlp_env_pkg" $env(UVMF_VIP_LIBRARY_HOME)]
quietly set cmd [format "%s %s/environment_packages/mnist_mlp_env_pkg/mnist_mlp_env_pkg.sv" $cmd $env(UVMF_VIP_LIBRARY_HOME)]
eval $cmd


