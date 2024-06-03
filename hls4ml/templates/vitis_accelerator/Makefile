# Absolute path to top directory of accelerator project
PWD = $(shell readlink -f .)

# Checks for XILINX_VITIS
ifndef XILINX_VITIS
$(error XILINX_VITIS variable is not set, please set correctly and rerun)
endif

# Checks for XILINX_XRT
ifndef XILINX_XRT
$(error XILINX_XRT variable is not set, please set correctly and rerun)
endif

# Checks for XILINX_VIVADO
ifndef XILINX_VIVADO
$(error XILINX_VIVADO variable is not set, please set correctly and rerun)
endif

# Checks for g++
ifneq ($(shell expr $(shell g++ -dumpversion) \>= 5), 1)
CXX := $(XILINX_VIVADO)/tps/lnx64/gcc-6.2.0/bin/g++
$(warning [WARNING]: g++ version older. Using g++ provided by the tool : $(CXX))
endif

KERN_LIBRARIES += -I./ -I./firmware/ -I./firmware/weights -I./firmware/nnet_utils/

.PHONY: all
all: hls xclbin

# Building kernel
./build/myproject_kernel.xo: kernel_wrapper.cpp
	mkdir -p ./build && mkdir -p ./build/xo
	v++ -c -t hw --config ./accelerator_card.cfg --temp_dir build/xo kernel_wrapper.cpp firmware/myproject.cpp -o ./build/myproject_kernel.xo $(KERN_LIBRARIES)

# hls-fpga-machine-learning packaging 

# Building Host 
INCLUDES += -I$(XILINX_XRT)/include/ -I$(XILINX_VIVADO)/include/ -I$(XILINX_HLS)/include/ \
			-I$(PWD)/libs/ -I$(PWD)/firmware/ -I$(PWD)/firmware/nnet_utils/
CXXFLAGS += -Wall -std=c++11 -Wno-unknown-pragmas -g -O0 
LDFLAGS = -L$(XILINX_XRT)/lib/ -lstdc++ -lpthread -lrt -lOpenCL

host: myproject_host_cl.cpp libs/xcl2.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(INCLUDES) $(LDFLAGS) 

.PHONY: hls
hls: ./build/myproject_kernel.xo

.PHONY: xclbin
xclbin: ./build/kernel_wrapper.xclbin host

# Cleaning stuff
.PHONY: cleanxclbin
	-rm -rf host tb_data/hw_results.dat
	-rm -rf *kernel_wrapper*.log 
	-rm -rf build/kernel_wrapper.xclbin* build/kernel_wrapper.xsa* build/kernel_wrapper.ltx build/kernel_wrapper.mdb
	-rm -rf build/xclbin

.PHONY: cleanhls
	-rm -rf build/myproject_kernel.xo*
	-rm -rf build/xo
	-rm -rf *myproject_kernel*.log