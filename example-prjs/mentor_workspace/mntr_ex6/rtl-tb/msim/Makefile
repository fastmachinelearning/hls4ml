TOP_MODULE = keras1layer_tb

WORK_LIB = work

VLOG = vlog
VLIB = vlib
VSIM = vsim

VPATH :=
VPATH += ../src
VPATH += ../tb

V_SOURCES :=
V_SOURCES += $(subst ../tb/,,$(wildcard ../tb/*.v))
V_SOURCES += $(subst ../src/,,$(wildcard ../src/*.v))

.SUFFIXES: .v

setup: $(WORK_LIB)

compile: $(V_SOURCES)
	$(VLOG) $^
.PHONY: compile

run: setup compile
	$(VSIM) -voptargs=+acc $(TOP_MODULE)
.PHONY: run

$(WORK_LIB):
	$(VLIB) $(WORK_LIB)

clean:
	rm -rf work modelsim.ini vsim.wlf transcript *.cr.mti *.mpf *.vcd
.PHONY: clean
