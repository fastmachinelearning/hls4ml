v++ -l -t hw --platform {PLATFORM_XPFM} {KERNEL_XO} --config buildConfig.cfg -o {PROJECT_NAME}.xclbin --save-temps
[ -f ../../export/system.bit ] && rm -f ../../export/system.bit
[ -f ../../export/system.hwh ] && rm -f ../../export/system.hwh

xclbinutil --dump-section BITSTREAM:RAW:../../export/system.bit --input {PROJECT_NAME}.xclbin
cp _x/link/vivado/vpl/prj/prj.gen/sources_1/bd/vitis_design/hw_handoff/vitis_design.hwh ../../export/system.hwh