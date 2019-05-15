# V=1 for a more verbose compilation

ifndef V
	QUIET_AR            = @echo '   ' AR $@;
	QUIET_BUILD         = @echo '   ' BUILD $@;
	QUIET_C             = @echo '   ' CC $@;
	QUIET_CXX           = @echo '   ' CXX $@;
	QUIET_CHECKPATCH    = @echo '   ' CHECKPATCH $(subst .o,.cpp,$@);
	QUIET_CHECK         = @echo '   ' CHECK $(subst .o,.cpp,$@);
	QUIET_LINK          = @echo '   ' LINK $@;
	QUIET_CP            = @echo '   ' CP $@;
	QUIET_MKDIR         = @echo '   ' MKDIR $@;
	QUIET_MAKE          = @echo '   ' MAKE $@;
	QUIET_INFO          = @echo -n '   ' INFO '';
	QUIET_RUN           = @echo '   ' RUN '';
	QUIET_CLEAN         = @echo '   ' CLEAN ${PWD};
endif

all: run-release

TARGET = myproject_test

CXX = clang++

INCDIR :=
INCDIR += -I..
INCDIR += -I../py
INCDIR += -I../firmware
INCDIR += -I../firmware/weights
INCDIR += -I../../../nnet_utils
INCDIR += -I$(XILINX_VIVADO)/include

# For Windows machines: provide the path to your Vivado installation
#INCDIR += -IC:/Xilinx/Vivado/2018.3/include

CXX_FLAGS :=
CXX_FLAGS += -MMD
CXX_FLAGS += -Wall
CXX_FLAGS += -Wno-unknown-pragmas
CXX_FLAGS += -Wno-unused-label
CXX_FLAGS += -Wno-sign-compare
CXX_FLAGS += -Wno-unused-variable
CXX_FLAGS += -Wno-mismatched-tags
#CXX_FLAGS += -Wno-int-in-bool-context # GCC only
CXX_FLAGS += -DWEIGHTS_DIR=../firmware/weights

release: CXX_FLAGS += -O0
release: $(TARGET)

debug: CXX_FLAGS += -DDEBUG
debug: CXX_FLAGS += -O0
debug: CXX_FLAGS += -g
debug: $(TARGET)

LD_FLAGS :=

LD_LIBS :=

VPATH :=
VPATH += ../
VPATH += ../py
VPATH += ../firmware
VPATH += ../firmware/weights
VPATH += ../../../nnet_utils

CXX_SOURCES :=
CXX_SOURCES += $(subst ../,,$(wildcard ../*.cpp))
CXX_SOURCES += $(subst ../firmware/,,$(wildcard ../firmware/*.cpp))

#CXX_HEADERS :=
#CXX_HEADERS += $(subst ../,,$(wildcard ../*.h))
#CXX_HEADERS += $(subst ../firmware/,,$(wildcard ../firmware/*.h))
#CXX_HEADERS += $(subst ../firmware/weights/,,$(wildcard ../firmware/weights/*.h))
#CXX_HEADERS += $(subst ../../../nnet_utils/,,$(wildcard ../../../nnet_utils/*.h))

.SUFFIXES: .cpp .h .o

CXX_OBJECTS := $(CXX_SOURCES:.cpp=.o)
-include $(CXX_OBJECTS:.o=.d)


#$(CXX_OBJECTS): $(CXX_HEADERS)

$(TARGET): $(CXX_OBJECTS)
	$(QUIET_LINK)$(CXX) -o $@ $(CXX_OBJECTS) ${LD_LIBS} ${LD_FLAGS}

.cpp.o:
	$(QUIET_CXX)$(CXX) $(CXX_FLAGS) ${INCDIR} -c $<

run-debug: debug
	$(QUIET_RUN)./$(TARGET) | tee results.log
.PHONY: run-debug

run-release: release
	$(QUIET_RUN)./$(TARGET) | tee results.log
.PHONY: run-release

valgrind: debug
	$(QUIET_RUN)valgrind --leak-check=full --max-stackframe=16777216 --show-leak-kinds=all --track-origins=yes ./$(TARGET)
.PHONY: valgrind

gdb:
	$(QUIET_RUN)gdb ./$(TARGET)
.PHONY: gdb

clean:
	$(QUIET_CLEAN)rm -f *.o *.d $(TARGET) *.bin *.log *.out
.PHONY: clean
