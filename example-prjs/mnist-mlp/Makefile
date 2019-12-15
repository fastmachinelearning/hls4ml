all: generate

# Usage
# $(call hls4ml_buikd_run,IMAGE_SIZE,WORD_SIZE,INTEGER_SIZE)
define hls4ml_build_run
	python generate.py --image_size $(1) --word_size $(2) --integer_size $(3)
	hls4ml convert -c keras-config-$(1)x$(1)-fxd-$(2)-$(3).yml
	hls4ml build -p mnist_$(1)x$(1)_fxd_$(2)_$(3)_mlp -c
endef

generate:
	$(call hls4ml_build_run,28,18,8)
	$(call hls4ml_build_run,16,18,8)
	$(call hls4ml_build_run,14,18,8)
	$(call hls4ml_build_run,12,18,8)
.PHONY: generate

clean:
	rm -rf mnist_*_mlp* keras-config-*.yml
.PHONY: clean	
