# Learn CUDA from Matmul

This is a simple project to learn CUDA from matrix multiplication (8192 * 8192 * 8192).

## Hardware info

- RTX 3060 Max-Q 6GB on laptop, max power 115W
- CUDA core: 3840
- Max frenquency: 2100MHz
    - when running, the actual frenquency is roughly 1920MHz
- Memory bits: 192bit
- Memory frenquency: 6000MHz

Max memory bandwidth:

$$
    2 * 192 \ \textrm{bits} * 6000 \ \textrm{MHz} 
    = 2 * \frac{192 * 6000}{8000} \frac{\textrm{GiB}}{\textrm{s}} 
    = 288 \ \textrm{GiB/s}
$$

Max mul-add throughput:

$$
    3840 * 2 * 1920 \textrm{MHz} 
    = \frac{3840 * 2 * 1920}{10^6} \textrm{TFLOPS} 
    = 14.7456 \ \textrm{TiFLOPS}
$$

> 1 FMA is counted as 2 FLOP

## Time record (first try)

### First version

```
matmul time: 17463ms
Throughput: 0.06 TFLOPS (0.43%)
```

### Adjust block size, `16 -> 8`

```
matmul time: 7625ms
Throughput: 0.144 TFLOPS (0.978%)
```

### Open `-O2`

```
matmul time: 6674ms
Throughput: 0.165 TFLOPS (1.117%)
```

### Make each thread calculate 4x4 elements

```
matmul time: 3611ms
Throughput: 0.304 TFLOPS (2.065%)
```

### Adjust loop order

Pre-load part of `A`/`B` to local memory. 

```
matmul time: 736ms
Throughput: 1.494 TFLOPS (10.131%)
```

### Adjust thread block size, `4x4 -> 8x8`

```
matmul time: 492ms
Throughput: 2.235 TFLOPS (15.156%)
```

### Add normal matrix transpose

```
matmul time: 335ms
Throughput: 3.282 TFLOPS (22.258%)
```

### Tiled matrix transpose

```
matmul time: 249ms
Throughput: 4.416 TFLOPS (29.946%)
```

### In-place matrix transpose

```
matmul time: 225ms
Throughput: 4.887 TFLOPS (33.140%)
```

## Time record (second try)

### Init, shared memory & block tile

```
Throughput: 0.874 TFLOPS 
    (5.925% Max)
    (10.619% cuBLAS)
```

### Preload sA/sB to register

No improvement.

### Make each thread calculate 1x32 elements (1D tiling)

> it looks like we make 32 thread a 32xf32 vector,
> and they do 32 times works for each row in the block.

```
matmul time: 381.333 ms
Throughput: 2.883 TFLOPS 
    (19.554% Max)
    (35.043% cuBLAS)
```

After adjusted the load loop, the performance has a little improvement.

```
matmul time: 347.733 ms
Throughput: 3.162 TFLOPS 
    (21.443% Max)
    (38.429% cuBLAS)
```

### Make each thread calculate 8x8 elements (2D tiling)

TODO: has bug

Let each thread directly write to global memory, can save 1/3 shared memory. No decrease in performance.

### Vectorize gmem load/store

```
matmul time: 308.433 ms
Throughput: 3.565 TFLOPS 
    (24.176% Max)
    (43.326% cuBLAS)
```

## Time record (3rd try)

### 3-layer tiling

Block-Warp-Thread, with vectorized gmem load/store.

```
matmul time: 365.567 ms
Throughput: 3.008 TFLOPS 
    (20.397% Max)
    (36.554% cuBLAS)
```

## Reference

- [通用图形处理器设计](https://book.douban.com/subject/35998320/)
- [如何加速矩阵乘法——优化GEMM (CPU单线程篇)](https://renzibei.com/2021/06/30/optimize-gemm/)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
