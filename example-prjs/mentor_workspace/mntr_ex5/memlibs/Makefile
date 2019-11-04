# Copyright 2018 Columbia University, SLD Group

# list subdirectories that contain memory libraries
# (must be in subdirectories of the current directory)
MEMLIBS = ./memlib

MEMTECH_PATH = ./scripts/memgen/virtex7

memlib: memlist.txt
	@rm -rf memlib/*
	@./scripts/memgen/memgen.py $(MEMTECH_PATH) memlist.txt ./memlib | tee memgen.log
	@echo "Memories generated successfully"
.PHONY: memblib
