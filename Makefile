.PHONE: clean build run

NVCC_FLAGS = -std=c++17 -lcublas -ccbin /usr/bin/g++-12

clean:
	rm -f matmul

build: 
	nvcc $(NVCC_FLAGS) -O2 -o matmul matmul.cu

run: build
	./matmul

build-debug:
	nvcc $(NVCC_FLAGS) -DPROFILE -O0 -g -G -o matmul-debug matmul.cu

build-profile:
	nvcc $(NVCC_FLAGS) -DPROFILE -O2 -o matmul-profile matmul.cu

gdb: build-debug
	cuda-gdb ./matmul
