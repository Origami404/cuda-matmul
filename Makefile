matmul: matmul.cu
	nvcc -std=c++17 -O2 -o matmul $<

.PHONE: clean build run

clean:
	rm -f matmul

build: matmul

run: build
	./matmul

gdb: build
	nvcc -std=c++17 -g -G -o matmul $<
	cuda-gdb ./matmul
