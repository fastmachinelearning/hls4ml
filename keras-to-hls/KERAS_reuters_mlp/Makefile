csim:
	make -C csim
.PHONY: csim

hls:
	vivado_hls -f build_prj.tcl
.PHONY: hls

#less myproject_prj/solution1/syn/report/myproject_csynth.rpt

hls-gui: hls
	vivado_hls -p myproject_prj
.PHONY: hls-gui

gui:
	vivado_hls -p myproject_prj
.PHONY: gui

results:
	less myproject_prj/solution1/syn/report/myproject_csynth.rpt
.PHONY: results

clean:
	rm -rf myproject_prj *log
	make -C csim clean
.PHONY: clean
