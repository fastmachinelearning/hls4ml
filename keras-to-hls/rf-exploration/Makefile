SANDBOX_DIR=$(HOME)/sandbox1

export SANDBOX_DIR

dse:
	bash ./rf-exploration.sh
.PHONY: dse

gnuplot:
	gnuplot rf-reporting.gnuplot
.PHONY: gnuplot

clean:
	@echo "INFO: please manually clean"
	@echo "INFO:   $(PWD)/rf-exploration-results"
	@echo "INFO:   $(SANDBOX_DIR)"
.PHONY: clean

sandbox-info:
	@echo "INFO: $(SANDBOX_DIR)"
.PHONY: sandbox-info
