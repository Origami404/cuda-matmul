matmul: matmul.cu
	nvcc -std=c++17 -O2 -G -o matmul $<

.PHONE: clean build run

clean:
	rm -f matmul

build: matmul

run: build
	./matmul

gdb: build
	cuda-gdb ./matmul
