#V=1
MODEL = keras1layer
MODEL_DIR = keras_1layer

#### You may not need to edit the remaining of this Makefile. ####

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

all: release
.PHONY: all

CXX          = g++
TARGET_ARCH = linux64

INCDIR :=
INCDIR += -I../inc
INCDIR += -I../$(MODEL_DIR)/firmware/nnet_utils/
INCDIR += -I../$(MODEL_DIR)/firmware
INCDIR += -I../$(MODEL_DIR)/firmware/weights
INCDIR += -I$(SYSTEMC)/include

CXX_FLAGS :=
CXX_FLAGS += -MMD
CXX_FLAGS += -Wall
CXX_FLAGS += -Wno-unknown-pragmas
CXX_FLAGS += -Wno-unused-label
CXX_FLAGS += -Wno-sign-compare
CXX_FLAGS += -Wno-unused-variable
CXX_FLAGS += -Wno-narrowing
CXX_FLAGS += -std=c++11
CXX_FLAGS += -DMNTR_CATAPULT_HLS
#CXX_FLAGS += -D__SYNTHESIS__

release: CXX_FLAGS += -O3
release: $(MODEL)
.PHONY: realease

debug: CXX_FLAGS += -O0
debug: CXX_FLAGS += -g
debug: $(MODEL)
	$(QUIET_INFO)echo "Compiled with debugging flags!"
.PHONY: debug

LD_FLAGS :=
LD_FLAGS += -lm
LD_FLAGS += -lsystemc
LD_FLAGS += -lpthread

LD_LIBS :=
LD_LIBS += -L$(SYSTEMC)/lib

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

CXX_OBJECTS := $(CXX_SOURCES:.cpp=.o)
-include $(CXX_OBJECTS:.o=.d)

$(MODEL): $(CXX_OBJECTS)
	$(QUIET_LINK)$(CXX) -o $@ $(CXX_OBJECTS) ${LD_LIBS} ${LD_FLAGS}

.cpp.o:
	$(QUIET_CXX)$(CXX) $(CXX_FLAGS) ${INCDIR} -c $<

run: release
	$(QUIET_RUN)./$(MODEL)
.PHONY: run

validate:
	@python ../script/validate.py -r ./tb_data/tb_output_predictions.dat -i ./tb_data/csim_results.log | tee validate.log
.PHONY: validate

valgrind: debug
	$(QUIET_RUN)valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./$(MODEL)
.PHONY: valgrind

gdb:
	$(QUIET_RUN)gdb ./$(MODEL)
.PHONY: gdb

clean:
	$(QUIET_CLEAN)rm -rf *.o $(MODEL)
.PHONY: clean

