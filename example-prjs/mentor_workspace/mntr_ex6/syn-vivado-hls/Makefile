help:
	@echo "make <TAB> for usage"
.PHONY: help

hls:
	vivado_hls -f ./build_prj.tcl
.PHONY: hls

clean:
	rm -rf *log
.PHONY: clean

ultraclean: clean
	rm -rf ./*_prj
.PHONY: ultraclean
