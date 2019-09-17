PROJECT = keras1layer

help:
	@echo "INFO: make <TAB> for usage"
.PHONY: help

hls:
	vivado_hls -f ./build_prj.tcl
.PHONY: hls

report:
	@./report.sh $(PROJECT)
.PHONY: report

view-csv: report
	./csv-viewer.sh $(PROJECT).csv
.PHONY: view-csv

clean:
	@echo "INFO: make ultraclean"
.PHONY: clean

ultraclean: clean
	rm -rf *.log *.csv ./*_prj
.PHONY: ultraclean
