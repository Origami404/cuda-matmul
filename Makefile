.PHONE: clean build run

clean:
	rm -f matmul

build: 
	nvcc -std=c++17 -O2 -o matmul matmul.cu

run: build
	./matmul

build-debug:
	nvcc -std=c++17 -g -G -o matmul matmul.cu

gdb: build-debug
	cuda-gdb ./matmul
