#!/bin/bash

export LD_LIBRARY_PATH=$(pwd)/dense:$(pwd)/gemm:$(pwd)/merge:$(pwd)/orthogonal:$(pwd)/readSVM:$(pwd)/reorder:$(pwd)/singleton:$(pwd)/singleton:$(pwd)/sort:$(pwd)/sparse:$(pwd)/transpose:$(pwd)/util:$LD_LIBRARY_PATH
