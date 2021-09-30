
METHOD = FIKNN
FILE = src/$(METHOD)_Interface
BUILD_DIR = build
RES_DIR = results
INCLUDES = -I${TACC_CUDA_DIR}/samples/common/inc/ -Iinclude
BUILD_FILE = GPU_$(METHOD)
CPP_FILE = test/* src/FIKNN_dense.cu src/FIKNN_sparse.cu src/rnd_gen.cpp 

sparse:
	nvcc ${INCLUDES} -lineinfo $(CPP_FILE) $(FILE)_sparse.cpp -o $(BUILD_DIR)/$(BUILD_FILE)_sparse
	./$(BUILD_DIR)/$(BUILD_FILE)_sparse
dense:
	nvcc ${INCLUDES} -lineinfo $(CPP_FILE) $(FILE)_dense.cpp -o $(BUILD_DIR)/$(BUILD_FILE)_dense
	./$(BUILD_DIR)/$(BUILD_FILE)_dense

