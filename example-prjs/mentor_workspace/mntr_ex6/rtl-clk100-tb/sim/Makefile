TARGET = keras1layer

EXE = $(TARGET).x

VCD_FILE = $(TARGET).vcd
CONFIG_FILE = show.gtkw

VPATH :=
VPATH += ../src
VPATH += ../tb

V_SOURCES :=
V_SOURCES += $(subst ../tb/,,$(wildcard ../tb/*.v))
V_SOURCES += $(subst ../src/,,$(wildcard ../src/*.v))

.SUFFIXES: .v .x

all:
	@echo "INFO: make <TAB> for targets"
.PHONY: all

compile: $(EXE)

$(EXE): $(V_SOURCES)
	iverilog $^ -o $@

run: compile
	./$(EXE)
.PHONY: run

view: run
	gtkwave $(VCD_FILE) $(CONFIG_FILE) &
.PHONY: view

clean:
	rm -rf $(EXE) $(VCD_FILE)
.PHONY: clean

