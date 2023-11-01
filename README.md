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

## Time record

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
