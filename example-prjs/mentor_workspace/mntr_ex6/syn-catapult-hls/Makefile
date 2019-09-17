help:
	@echo "make <TAB> for usage"
.PHONY: help

CCS := $(shell ls -t Catapult*ccs 2> /dev/null | head -n1)

gui:
	catapult $(CCS) &
.PHONY: gui

hls-gui:
	catapult -file project.tcl &
.PHONY: hls-gui

hls-sh:
	catapult -shell -file project.tcl
.PHONY: hls-sh

kill-all:
	pkill -f catapult || pkill -f vsim
.PHONY: kill-all

report:
	./report.sh keras1layer
.PHONY: report

clean:
	@echo "INFO: make ultraclean"
.PHONY: clean

ultraclean: clean
	rm -rf Catapult* transcript vivado.jou vivado_*.str design_checker_constraints.tcl  design_checker_pre_build.tcl *.csv *log slec_*
.PHONY: ultraclean

