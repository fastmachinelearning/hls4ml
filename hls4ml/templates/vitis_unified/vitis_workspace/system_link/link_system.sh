#!/bin/bash
set -e

# Generate XSA from TCL if platform source is a generator script
{XSA_GENERATOR_BLOCK}

v++ -l -t hw --platform {PLATFORM_PATH} {KERNEL_XO} --config link_system.cfg -o {PROJECT_NAME}.xclbin --save-temps
[ -f ../../export/system.bit ] && rm -f ../../export/system.bit
[ -f ../../export/system.hwh ] && rm -f ../../export/system.hwh

xclbinutil --dump-section BITSTREAM:RAW:../../export/system.bit --input {PROJECT_NAME}.xclbin
cp _x/link/vivado/vpl/prj/prj.gen/sources_1/bd/vitis_design/hw_handoff/vitis_design.hwh ../../export/system.hwh

# Generate routed vectorless power estimate
POWER_REPORT_TCL=report_power.tcl
printf '%s\n' \
'set dcp_candidates [lsort [glob -nocomplain "_x/link/vivado/vpl/prj/prj.runs/impl_1/*_routed.dcp"]]' \
'if {[llength $dcp_candidates] == 0} {' \
'  puts "WARNING: No routed DCP found for power report."' \
'  exit 0' \
'}' \
'set routed_dcp [lindex $dcp_candidates 0]' \
'file mkdir ../../final_reports' \
'open_checkpoint $routed_dcp' \
'report_power -file ../../final_reports/impl_1_power_routed.rpt' \
'exit' > "$POWER_REPORT_TCL"

vivado -mode batch -source "$POWER_REPORT_TCL" \
  -log ../../final_reports/power_report.log \
  -journal ../../final_reports/power_report.jou \
  || echo "WARNING: Power report generation failed"

# Copy final reports (timing, clock, resource usage) to final_reports
mkdir -p ../../final_reports
[ -d _x/reports/link ] && cp -p _x/reports/link/*power* ../../final_reports/ 2>/dev/null || true
[ -d _x/reports/link/imp ] && cp -p _x/reports/link/imp/*.rpt ../../final_reports/ 2>/dev/null || true
[ -f {PROJECT_NAME}.xclbin.link_summary ] && cp -p {PROJECT_NAME}.xclbin.link_summary ../../final_reports/ 2>/dev/null || true

# Copy HLS reports (compile, cosim)
HLS_REPORTS_DIR="../{PROJECT_NAME}/vitis_unified_project/reports"
[ -f "$HLS_REPORTS_DIR/hls_compile.rpt" ] && cp -p "$HLS_REPORTS_DIR/hls_compile.rpt" ../../final_reports/ || true
[ -f "$HLS_REPORTS_DIR/hls_cosim.rpt" ] && cp -p "$HLS_REPORTS_DIR/hls_cosim.rpt" ../../final_reports/ || true
