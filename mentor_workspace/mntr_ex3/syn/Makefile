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
	pkill -f vsim
	pkill -f catapult
.PHONY: kill-all

clean:
	rm -rf *log slec_*
.PHONY: clean

distclean: clean
	rm -rf Catapult*
.PHONY: distclean

