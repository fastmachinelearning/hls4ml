
// UVM
+incdir+${UVM_HOME}/src
${UVM_HOME}/src/uvm_pkg.sv

// Common UVMF files
-f ${UVMF_HOME}/common/common_sve.f

// BFM Files
-F ${UVMF_VIP_LIBRARY_HOME}/interface_packages/ccs_pkg/ccs_pkg_sve.F

// Environment Files
-F ${UVMF_VIP_LIBRARY_HOME}/environment_packages/mnist_mlp_env_pkg/mnist_mlp_env_pkg_sve.F

// Bench Files
+incdir+./tb/tests
./tb/tests/mnist_mlp_bench_tests_pkg.sv

+incdir+./tb/sequences
./tb/sequences/mnist_mlp_bench_sequences_pkg.sv

+incdir+./tb/parameters
./tb/parameters/mnist_mlp_bench_parameters_pkg.sv

./tb/testbench/hdl_top.sv
./tb/testbench/hvl_top.sv

