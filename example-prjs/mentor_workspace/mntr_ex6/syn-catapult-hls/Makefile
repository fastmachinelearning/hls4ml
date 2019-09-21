help:
	@echo "make <TAB> for usage"
.PHONY: help

CCS := $(shell ls -t Catapult*ccs 2> /dev/null | head -n1)

gui:
	catapult $(CCS) &
.PHONY: gui

hls-gui:
	catapult -file build_prj.tcl &
.PHONY: hls-gui

hls-sh:
	catapult -shell -file build_prj.tcl
.PHONY: hls-sh

kill-all:
	pkill -f catapult || pkill -f vsim
.PHONY: kill-all

report:
	./report.sh keras1layer | tee report.log
.PHONY: report

validate:
	@python ../script/validate.py -r ./tb_data/tb_output_predictions.dat -i ./tb_data/csim_results.log | tee validate.log
.PHONY: validate

clean:
	@echo "INFO: make ultraclean"
.PHONY: clean

ultraclean: clean
	rm -rf Catapult* transcript vivado.jou vivado_*.str design_checker_constraints.tcl  design_checker_pre_build.tcl *.csv *log slec_*
.PHONY: ultraclean

