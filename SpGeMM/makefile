
METHOD = TSKNN
FILE = src/$(METHOD)_Interface.cpp
BUILD_DIR = build
RES_DIR = results
INCLUDES = -I${CUDA_HOME}/samples/common/inc/ -Iinclude
BUILD_FILE = GPU_$(METHOD)
CPP_FILE = test/knn_seq.cpp src/*

all:
	./$(BUILD_DIR)/$(BUILD_FILE)

compile:
	nvcc ${INCLUDES} -lineinfo $(CPP_FILE) $(FILE) -o $(BUILD_DIR)/$(BUILD_FILE)


cr:
	nvcc ${INCLUDES} -lineinfo $(CPP_FILE) $(FILE) -o $(BUILD_DIR)/$(BUILD_FILE)
	./$(BUILD_DIR)/$(BUILD_FILE)

clean:
	rm -rf build/*
