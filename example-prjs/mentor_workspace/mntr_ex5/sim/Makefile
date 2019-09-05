#V=1

# Define V=1 for a more verbose compilation
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

INPUT =

# Define the main target

TARGET = mnist-mlp
all: release
.PHONY: all

# We'll use g++ for C++ compilation.

CXX          = g++
TARGET_ARCH = linux64

# Let's leave a place holder for additional include directories

INCDIR :=
INCDIR += -I../inc
INCDIR += -I../mnist_mlp/nnet_utils
INCDIR += -I../mnist_mlp/firmware
INCDIR += -I../mnist_mlp/firmware/weights
INCDIR += -I$(SYSTEMC)/include

# Compilation options
CXX_FLAGS :=
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
release: $(TARGET)
.PHONY: realease

debug: CXX_FLAGS += -O0
debug: CXX_FLAGS += -g
debug: $(TARGET)
	$(QUIET_INFO)echo "Compiled with debugging flags!"
.PHONY: debug

# Linking options:
# For example, use "-lm" for the math library.

LD_FLAGS :=
LD_FLAGS += -lm
LD_FLAGS += -lsystemc
LD_FLAGS += -lpthread

# List the libraries you need to link with in LD_LIBS.
LD_LIBS :=
LD_LIBS += -L$(SYSTEMC)/lib

# The VPATH is a list of directories to be searched for missing source and
# headers files.
VPATH :=
VPATH += ../inc
VPATH += ../mnist_mlp
VPATH += ../mnist_mlp/firmware
VPATH += ../mnist_mlp/firmware/weights
VPATH += ../mnist_mlp/nnet_utils

# List of the source and header files. Note that they will be searched first in
# the current directory and then in the directories specified in the VPATH
# variable.
CXX_SOURCES :=
CXX_SOURCES += sc_main.cpp
CXX_SOURCES += mnist_mlp.cpp

CXX_HEADERS :=
CXX_HEADERS += ap_fixed.h
CXX_HEADERS += ap_int.h
CXX_HEADERS += b2.h
CXX_HEADERS += b4.h
CXX_HEADERS += w2.h
CXX_HEADERS += w4.h
CXX_HEADERS += parameters.h

.SUFFIXES: .cpp .h .o

CXX_OBJECTS := $(CXX_SOURCES:.cpp=.o)

$(CXX_OBJECTS): $(CXX_HEADERS)

$(TARGET): $(CXX_OBJECTS)
	$(QUIET_LINK)$(CXX) -o $@ $(CXX_OBJECTS) ${LD_LIBS} ${LD_FLAGS}

.cpp.o:
	$(QUIET_CXX)$(CXX) $(CXX_FLAGS) ${INCDIR} -c $<

logs:
	$(QUIET_RUN)mkdir -p logs
.PHONY: logs

run: release logs
	$(QUIET_RUN)./$(TARGET) $(INPUT)
.PHONY: run

valgrind: debug logs
	$(QUIET_RUN)valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./$(TARGET) $(INPUT)
.PHONY: valgrind

gdb: logs
	$(QUIET_RUN)gdb ./$(TARGET)
.PHONY: gdb

clean:
	$(QUIET_CLEAN)rm -rf *.o $(TARGET) logs
.PHONY: clean

