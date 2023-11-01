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

> ed4b46bb773e479eeb6758c8d4d24bd930218a63

```
bash -c "time ./matmul"

real    0m18.354s
user    0m18.182s
sys     0m0.153s
```