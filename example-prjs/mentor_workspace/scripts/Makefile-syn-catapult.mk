help:
	@echo "make <TAB> for usage"
.PHONY: help

CCS := $(shell ls -t Catapult*ccs 2> /dev/null | head -n1)

gui:
	catapult $(CCS) &
.PHONY: gui

hls-fpga-gui:
	catapult -file build_prj_fpga.tcl -logfile ./catapult-fpga.log  &
.PHONY: hls-fpga-gui

hls-fpga-sh:
	catapult -shell -file build_prj_fpga.tcl -logfile ./catapult-fpga.log 
.PHONY: hls-fpga-sh

hls-asic-gui:
	catapult -file build_prj_asic.tcl -logfile ./catapult-asic.log  &
.PHONY: hls-asic-gui

hls-asic-sh:
	catapult -shell -file build_prj_asic.tcl -logfile ./catapult-asic.log 
.PHONY: hls-asic-sh

kill-all:
	pkill -f catapult || pkill -f vsim
.PHONY: kill-all

report:
	@../../report-catapult.sh $(PROJECT) | tee report.log
.PHONY: report

validate-c:
	@python ../../scripts/validate.py -r ./tb_data/tb_output_predictions.dat -i ./tb_data/csim_results.log | tee validate-c.log
.PHONY: validate-c

validate-rtl:
	@python ../../script/validate.py -r ./tb_data/tb_output_predictions.dat -i ./tb_data/rtl_cosim_results.log | tee validate-rtl.log
.PHONY: validate-rtl


clean:
	@echo "INFO: make ultraclean"
.PHONY: clean

ultraclean: clean
	rm -rf Catapult* transcript vivado.jou vivado_*.str design_checker_constraints.tcl  design_checker_pre_build.tcl *.csv *.log *.pinfo slec_* tb_data/*log
.PHONY: ultraclean

