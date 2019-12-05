# Copyright 2018 Columbia University, SLD Group

# list subdirectories that contain memory libraries
# (must be in subdirectories of the current directory)
MEMLIBS = ./memlib

FPGA_MEMTECH_PATH = ../../memgen/tech/virtexup

memlib-fpga: memlist.txt
	@rm -rf memlib/*
	@../../memgen/memgen.py $(FPGA_MEMTECH_PATH) memlist.txt ./memlib | tee memgen.log
	@echo "Memories generated successfully"
