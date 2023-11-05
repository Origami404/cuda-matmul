.PHONE: clean build run

clean:
	rm -f matmul

build: 
	nvcc -std=c++17 -O2 -o matmul matmul.cu

run: build
	./matmul

build-debug:
	nvcc -std=c++17 -g -G -o matmul-debug matmul.cu

build-profile:
	nvcc -std=c++17 -DPROFILE -O2 -o matmul-profile matmul.cu

gdb: build-debug
	cuda-gdb ./matmul
