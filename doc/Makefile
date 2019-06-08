PY_FILE=nnet_large_layer.py
DIA_FILE=nnet_large_layer.dia
PDF_FILE=nnet_large_layer.pdf

dia2pdf: $(PDF_FILE)
.PHONY: dia2pdf

$(PDF_FILE): $(DIA_FILE)
	dia -e $@ $<

view-dia: $(DIA_FILE)
	dia $< &
.PHONY: view-dia

view-pdf: $(PDF_FILE)
	evince $< &
.PHONY: view-pdf

run:
	python ./$(PY_FILE)
.PHONY: run
