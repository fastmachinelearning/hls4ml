SCRIPT_PY := test_backends.py

help:
	@echo "INFO: make<TAB> to show targets"
.PHONY: help

run:
	python $(SCRIPT_PY)
.PHONY: run

run-profile:
	python $(SCRIPT_PY) profile
.PHONY: run-profile

vivado-gui:
	vivado ./wrapped_qresource64/myproject_vivado_accelerator/myproject.xpr
.PHONY: vivado-gui

clean:
	rm -rf training_dir
	rm -rf __pycache__
	rm -rf *axi_m_backend
	rm -rf *qresource*
	rm -f *axi_m_backend.tar.gz
	rm -f *.npy
	rm -f *.log
	rm -f *.jou
	rm -f *.str
	rm -rf *.idea
	rm -rf NA
.PHONY: clean

ultraclean: clean
	rm -f model/*.h5
	rm -f npy/*.npy
.PHONY: clean
