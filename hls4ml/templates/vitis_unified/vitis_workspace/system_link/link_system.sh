# Generate XSA from TCL if platform source is a generator script
{XSA_GENERATOR_BLOCK}

v++ -l -t hw --platform {PLATFORM_PATH} {KERNEL_XO} --config link_system.cfg -o {PROJECT_NAME}.xclbin --save-temps
[ -f ../../export/system.bit ] && rm -f ../../export/system.bit
[ -f ../../export/system.hwh ] && rm -f ../../export/system.hwh

xclbinutil --dump-section BITSTREAM:RAW:../../export/system.bit --input {PROJECT_NAME}.xclbin
cp _x/link/vivado/vpl/prj/prj.gen/sources_1/bd/vitis_design/hw_handoff/vitis_design.hwh ../../export/system.hwh

# Copy final reports (timing, clock, resource usage) to final_reports
mkdir -p ../../final_reports
[ -d _x/reports/link/imp ] && cp -p _x/reports/link/imp/*.rpt ../../final_reports/ 2>/dev/null || true
[ -f {PROJECT_NAME}.xclbin.link_summary ] && cp -p {PROJECT_NAME}.xclbin.link_summary ../../final_reports/ 2>/dev/null || true
