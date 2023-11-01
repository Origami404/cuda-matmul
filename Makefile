matmul: matmul.cu
	nvcc -o matmul $<

.PHONE: clean build run

clean:
	rm -f matmul

build: matmul

run: build
	./matmul