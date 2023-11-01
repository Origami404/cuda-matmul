matmul: matmul.cu
	nvcc -std=c++17 -o matmul $<

.PHONE: clean build run

clean:
	rm -f matmul

build: matmul

run: build
	./matmul