.PHONE: clean build run

NVCC_FLAGS = -std=c++17 -lcublas

clean:
	rm -f matmul

build: 
	nvcc $(NVCC_FLAGS) -O2 -o matmul matmul.cu

run: build
	./matmul

build-debug:
	nvcc $(NVCC_FLAGS) -g -G -o matmul-debug matmul.cu

build-profile:
	nvcc $(NVCC_FLAGS) -DPROFILE -O2 -o matmul-profile matmul.cu

gdb: build-debug
	cuda-gdb ./matmul
