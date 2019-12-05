#### BE CAREFUL: You may not need to edit this Makefile. ####

ifndef V
	QUIET_AR            = @echo 'MAKE:' AR $@;
	QUIET_BUILD         = @echo 'MAKE:' BUILD $@;
	QUIET_C             = @echo 'MAKE:' CC $@;
	QUIET_CXX           = @echo 'MAKE:' CXX $@;
	QUIET_CHECKPATCH    = @echo 'MAKE:' CHECKPATCH $(subst .o,.cpp,$@);
	QUIET_CHECK         = @echo 'MAKE:' CHECK $(subst .o,.cpp,$@);
	QUIET_LINK          = @echo 'MAKE:' LINK $@;
	QUIET_CP            = @echo 'MAKE:' CP $@;
	QUIET_MKDIR         = @echo 'MAKE:' MKDIR $@;
	QUIET_MAKE          = @echo 'MAKE:' MAKE $@;
	QUIET_INFO          = @echo -n 'MAKE:' INFO '';
	QUIET_RUN           = @echo 'MAKE:' RUN '';
	QUIET_CLEAN         = @echo 'MAKE:' CLEAN ${PWD};
endif

all:
	@echo "INFO: make <TAB> for targets"
.PHONY: all

CXX          = g++
TARGET_ARCH = linux64

INCDIR :=
INCDIR += -I../$(MODEL_DIR)/firmware/nnet_utils/
INCDIR += -I../$(MODEL_DIR)/firmware
INCDIR += -I../$(MODEL_DIR)/firmware/weights

CXX_FLAGS :=
CXX_FLAGS += -MMD
CXX_FLAGS += -Wall
CXX_FLAGS += -Wno-uninitialized
CXX_FLAGS += -Wno-unknown-pragmas
CXX_FLAGS += -Wno-unused-label
CXX_FLAGS += -Wno-sign-compare
CXX_FLAGS += -Wno-unused-variable
CXX_FLAGS += -Wno-narrowing
CXX_FLAGS += -std=c++11
CXX_FLAGS += -D__WEIGHTS_DIR__="../mnist_mlp/firmware/weights"

LD_FLAGS :=
LD_FLAGS += -lm

LD_LIBS :=

VPATH :=
VPATH += ../inc
VPATH += ../$(MODEL_DIR)/
VPATH += ../$(MODEL_DIR)/firmware
VPATH += ../$(MODEL_DIR)/firmware/weights
VPATH += ../$(MODEL_DIR)/nnet_utils

CXX_SOURCES :=
CXX_SOURCES += $(subst ../$(MODEL_DIR)/,,$(wildcard ../$(MODEL_DIR)/*.cpp))
CXX_SOURCES += $(subst ../$(MODEL_DIR)/firmware/,,$(wildcard ../$(MODEL_DIR)/firmware/*.cpp))

.SUFFIXES: .cpp .h .o

# Vivado HLS
vivado: CXX_FLAGS += -O3
vivado: INCDIR += -I$(XILINX_VIVADO)/include
vivado: $(MODEL)
.PHONY: vivado

#debug-vivado: CXX_FLAGS += -O0
#debug-vivado: CXX_FLAGS += -g
#debug-vivado: INCDIR += -I$(XILINX_VIVADO)/include
#debug-vivado: $(MODEL)
#	$(QUIET_INFO)echo "Compiled with debugging flags!"
#.PHONY: debug-vivado

# Catapult HLS
catapult: INCDIR += -I../inc
catapult: INCDIR += -I$(SYSTEMC)/include
catapult: CXX_FLAGS += -O3
catapult: CXX_FLAGS += -DMNTR_CATAPULT_HLS
catapult: LD_LIBS += -L$(SYSTEMC)/lib
catapult: LD_FLAGS += -lsystemc
catapult: LD_FLAGS += -lpthread
catapult: $(MODEL)
.PHONY: catapult

#debug-catapult: INCDIR += -I../inc
#debug-catapult: INCDIR += -I$(SYSTEMC)/include
#debug-catapult: CXX_FLAGS += -g
#debug-catapult: CXX_FLAGS += -O0
#debug-catapult: CXX_FLAGS += -DMNTR_CATAPULT_HLS
#debug-catapult: LD_LIBS += -L$(SYSTEMC)/lib
#debug-catapult: LD_FLAGS += -lsystemc
#debug-catapult: LD_FLAGS += -lpthread
#debug-catapult: $(MODEL)
#	$(QUIET_INFO)echo "Compiled with debugging flags!"
#.PHONY: debug-catapult

CXX_OBJECTS := $(CXX_SOURCES:.cpp=.o)
-include $(CXX_OBJECTS:.o=.d)

$(MODEL): $(CXX_OBJECTS)
	$(QUIET_LINK)$(CXX) -o $@ $(CXX_OBJECTS) ${LD_LIBS} ${LD_FLAGS}

.cpp.o:
	$(QUIET_CXX)$(CXX) $(CXX_FLAGS) ${INCDIR} -c $<

run-vivado: vivado
	$(QUIET_RUN)set -o pipefail; ./$(MODEL) | tee run-vivado.log
.PHONY: run-vivado

run-catapult: catapult
	$(QUIET_RUN)set -o pipefail; ./$(MODEL) | tee run-catapult.log
.PHONY: run-catapult

validate-catapult:
	@set -o pipefail; python ../../scripts/validate.py \
		-r ./tb_data/tb_output_predictions.dat \
		-i ./tb_data/catapult_fpga_csim_results.log \
		-t catapult \
		| tee validate-catapult.log
.PHONY: validate-catapult

validate-vivado:
	@set -o pipefail; python ../../scripts/validate.py \
		-r ./tb_data/tb_output_predictions.dat \
		-i ./tb_data/vivado_csim_results.log \
		-t vivado \
		| tee validate-vivado.log
.PHONY: validate-vivado

#valgrind:
#	$(QUIET_RUN)valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./$(MODEL)
#.PHONY: valgrind

#gdb:
#	$(QUIET_RUN)gdb ./$(MODEL)
#.PHONY: gdb

clean:
	$(QUIET_CLEAN)rm -rf $(MODEL) *.o *.d
.PHONY: clean

ultraclean: clean
	$(QUIET_CLEAN)rm -rf ./tb_data/*.log *.log *.png
.PHONY: ultraclean
