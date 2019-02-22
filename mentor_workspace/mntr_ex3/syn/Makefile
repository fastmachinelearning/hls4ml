help:
	@echo "make <TAB> for usage"
.PHONY: help

CCS := $(shell ls -t Catapult*ccs | head -n1)

gui:
	catapult $(CCS) &
.PHONY: gui

hls-solution1-gui:
	catapult -file project_solution1.tcl &
.PHONY: hls-gui

hls-solution1-sh:
	catapult -shell -file project_solution1.tcl
.PHONY: hls-sh

hls-solution2-gui:
	catapult -file project_solution1.tcl &
.PHONY: hls-gui

hls-solution2-sh:
	catapult -shell -file project_solution1.tcl
.PHONY: hls-sh

kill-all:
	pkill -f catapult || pkill -f vsim
.PHONY: kill-all

clean:
	rm -rf *log slec_*
.PHONY: clean

distclean: clean
	rm -rf Catapult*
.PHONY: distclean

