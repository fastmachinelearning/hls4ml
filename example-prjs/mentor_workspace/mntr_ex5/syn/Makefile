help:
	@echo "make <TAB> for usage"
.PHONY: help

CCS := $(shell ls -t Catapult*ccs | head -n1)

gui:
	catapult $(CCS) &
.PHONY: gui

hls-solution1-gui:
	catapult -file project_solution1.tcl &
.PHONY: hls-solution1-gui

#hls-solution1-sh:
#	catapult -shell -file project_solution1.tcl
#.PHONY: hls-solution1-sh

#hls-solution2-gui:
#	catapult -file project_solution1.tcl &
#.PHONY: hls-solution2-gui

#hls-solution2-sh:
#	catapult -shell -file project_solution1.tcl
#.PHONY: hls-solution2-sh

kill-all:
	pkill -f catapult || pkill -f vsim
.PHONY: kill-all

clean:
	rm -rf *log slec_*
.PHONY: clean

ultraclean: clean
	rm -rf Catapult* transcript vivado.jou vivado_*.str design_checker_constraints.tcl  design_checker_pre_build.tcl 
.PHONY: ultraclean

