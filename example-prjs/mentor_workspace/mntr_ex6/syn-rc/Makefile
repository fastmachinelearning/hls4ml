all:
	@echo "INFO: make <TAB> for targets"
.PHONY: all

run-rc-fermi-lab:
	time rc -cmdfile ./output/fermi_lab/rc.cmd -logfile ./output/fermi_lab/rc.log -f ./scripts/fermi_lab/rc.tcl < ./scripts/terminator.rc
.PHONY: run-rc-fermi-lab

run-rc-catapult-hls:
	time rc -cmdfile ./output/catapult_hls/rc.cmd -logfile ./output/catapult_hls/rc.log -f ./scripts/catapult_hls/rc.tcl < ./scripts/terminator.rc
.PHONY: run-rc-catapult-hls

clean:
	@echo "INFO: make <TAB> for targets"
.PHONY: clean

clean-fermi-lab:
	rm -rf libscore_work fv output/fermi_lab/* ./report/fermi_lab/*
.PHONY: clean-fermi-lab

clean-catapult-hls:
	rm -rf libscore_work fv output/catapult_hls/* ./report/catapult_hls/*
.PHONY: clean-catapult-hls

ultraclean: clean-fermi-lab clean-catapult-hls
.PHONY: ultraclean
