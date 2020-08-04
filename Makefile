
GPU_DIR=pyrknn/kernels/gpu/impl/
CPU_DIR=pyrknn/kernels/cpu/impl/

ifeq ($(PYRKNN_USE_CUDA), 1)
DEPFILE=$(GPU_DIR)/dense/denknngpu.so $(GPU_DIR)/sparse/spknngpu.so
endif

DEPFILES=$(CPU_DIR)/exact/libexact.so $(CPU_DIR)/sparse/libspknncpu.so $(DEPFILE)

.PHONY: prknn

target: prknn $(DEPFILES)
	python setup.py build_ext --inplace

prknn: guard
	rsync -rupE --delete src/pyrknn/. pyrknn

%.so: prknn
	make -C $(dir $@) -j8

guard:
	@test -n "$(PYRKNN_USE_CUDA)" || (echo 'PYRKNN_USE_CUDA must be set' && exit 1)

clean:
	rm -rf pyrknn
	rm -rf build

