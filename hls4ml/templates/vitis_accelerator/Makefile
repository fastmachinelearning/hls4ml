.PHONY: clean

PLATFORM=myplatform

app.exe: ./myproject_host.cpp
	g++ -g -std=c++14 ./myproject_host.cpp -o app.exe \
		-I firmware/ap_types -I/opt/xilinx/xrt/include \
		-L${XILINX_XRT}/lib/ -lxrt_coreutil -lOpenCL -pthread -lrt -lstdc++

myproject_kernel.xo: myproject_kernel.cpp
	v++ --target hw --compile -I"./" --config accelerator_card.cfg -o"build/myproject_kernel.xo" "myproject_kernel.cpp"

ifneq (,$(findstring vck5000,$(PLATFORM)))
myproject_kernel.xsa: ./build/myproject_kernel.xo
	v++ -l -t hw --config ./accelerator_card.cfg ./build/myproject_kernel.xo -o myproject_kernel.xsa

myproject_kernel.xclbin: ./myproject_kernel.xsa
	v++ --package -t hw --config ./accelerator_card.cfg ./myproject_kernel.xsa -o myproject_kernel.xclbin
else
myproject_kernel.xclbin: ./myproject_kernel.xo
	v++ -l -t hw --config ./accelerator_card.cfg ./build/myproject_kernel.xo -o myproject_kernel.xclbin
endif

emconfig.json:
	emconfigutil --platform  xilinx_vck5000_gen4x8_qdma_2_202220_1 --nd 1

clean:
	rm -f KERAS_3layer.json KERAS_3layer_weights.h5  prj_tuto_vitis_stream.tar.gz
	rm -rf myproject_kernel* myproject_kernel* .exe *json *csv *log *summary _x xilinx* .run .Xil .ipcache *.jou

# Unless specified, use the current directory name as the v++ build target
TARGET ?= $(notdir $(CURDIR))
