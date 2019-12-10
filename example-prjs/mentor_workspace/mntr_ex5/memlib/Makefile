# Copyright 2018 Columbia University, SLD Group

# list subdirectories that contain memory libraries
# (must be in subdirectories of the current directory)
#MEMLIBS = ./memlib

FPGA_MEMTECH_PATH = ../../memgen/tech/virtexup
ASIC_MEMTECH_PATH = ../../memgen/tech/cmos32soi

all:
	@echo "INFO: make <TAB>"
.PHONY: all

memlib-fpga: memlist.txt
	@rm -rf memlib
	@../../memgen/memgen.py $(FPGA_MEMTECH_PATH) memlist.txt ./memlib | tee memgen-fpga.log
	@rm -rf fpga && mv memlib fpga
	@echo "INFO: Memories generated successfully: $(PWD)/fpga/"
.PHONY: memlib-fpga

memlib-asic: memlist.txt
	@rm -rf memlib
	@../../memgen/memgen.py $(ASIC_MEMTECH_PATH) memlist.txt ./memlib | tee memgen-asic.log
	@rm -rf asic && mv memlib asic
	@echo "INFO: Memories generated successfully: $(PWD)/asic/"
.PHONY: memlib-asic

clean:
	@echo "INFO: make clean"
.PHONY: clean

ultraclean:
	rm -rf fpga asic memlib *.log
.PHONY: ultraclean
