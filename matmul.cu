#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>

constexpr size_t N = 8192;
auto constexpr BLK_N = 32;

static inline bool feq(float a, float b) { return abs(a - b) <= 1e-5f; }
// use macro for both host and device
#define at(arr, i, j) ((arr)[(i)*N + (j)])
#define L(a, b, n) ((a) * (n) + (b))

// Kernel function to add the elements of two arrays
__global__ void matmul(float *C, float *A, float *B) {
  __shared__ float sC[BLK_N][BLK_N];
  __shared__ float sA[BLK_N][BLK_N];
  __shared__ float sB[BLK_N][BLK_N];

  // all the indices are named as ijk, where:
  //    i: means who use, t for thread, b for block
  //    j: means related to which matrix, m for all matrix, b for block matrix
  //    k: means related to which axis, x for x axis, y for y axis, c for count

  auto const ty = threadIdx.y;
  auto const tx = threadIdx.x;
  auto const by = blockIdx.y;
  auto const bx = blockIdx.x;

  sC[ty][tx] = 0.0f;

  for (auto bk = 0; bk < N / BLK_N; bk++) {
    sA[ty][tx] = at(A, by * BLK_N + ty, bk * BLK_N + tx);
    sB[ty][tx] = at(B, bk * BLK_N + ty, bx * BLK_N + tx);

    __syncthreads();

    float pA[BLK_N], pB[BLK_N];
    for (auto tk = 0; tk < BLK_N; tk++) {
      pA[tk] = sA[ty][tk];
    }
    for (auto tk = 0; tk < BLK_N; tk++) {
      pB[tk] = sB[tk][tx];
    }

    __syncthreads();

    float sum = 0.0f;
    for (auto tk = 0; tk < BLK_N; tk++) {
      sum += pA[tk] * pB[tk];
    }
    sC[ty][tx] += sum;
  }

  at(C, by * BLK_N + ty, bx * BLK_N + tx) = sC[ty][tx];
}

// host copies of A, B, C
float hA[N * N], hB[N * N], hC[N * N];

auto test() {
  // init two matrixes
  for (auto y = 0; y < N; y++) {
    for (auto x = 0; x < N; x++) {
      at(hA, y, x) = y == x ? 1.0f : 0.0f;
    }
  }

  for (auto y = 0; y < N; y++) {
    for (auto x = 0; x < N; x++) {
      at(hB, y, x) = y == x ? 1.0f : 0.0f;
    }
  }

  // device copies of A, B, C
  float *A, *B, *C;

  // Allocate Unified Memory – accessible from CPU or GPU
  auto constexpr MAT_SIZE = N * N * sizeof(float);
  cudaMallocManaged(&A, MAT_SIZE);
  cudaMallocManaged(&B, MAT_SIZE);
  cudaMallocManaged(&C, MAT_SIZE);

  cudaMemcpy(A, hA, MAT_SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(B, hB, MAT_SIZE, cudaMemcpyHostToDevice);
  cudaMemset(C, 1, MAT_SIZE);

  dim3 constexpr blockDim{BLK_N, BLK_N, 1};
  dim3 constexpr gridDim{N / BLK_N, N / BLK_N, 1};

  auto const matmul_begin = std::chrono::steady_clock::now();

  matmul<<<gridDim, blockDim>>>(C, A, B);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  auto const matmul_end = std::chrono::steady_clock::now();

  cudaMemcpy(hC, C, MAT_SIZE, cudaMemcpyDeviceToHost);

  // Check for errors
  for (auto i = 0; i < N; i++) {
    for (auto j = 0; j < N; j++) {
      auto const v = at(C, i, j);
      if ((i == j && !feq(v, 1.0f)) || (i != j && !feq(v, 0.0f))) {
        std::cout << "Error at (" << i << ", " << j << "): " << v << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }

  // Free memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  auto const matmul_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(matmul_end -
                                                            matmul_begin)
          .count();

  return matmul_time;
}

int main(void) {
#ifndef PROFILE
  auto constexpr WARMUP_N = 5;
  auto constexpr TEST_N = 30;

  for (auto i = 0; i < WARMUP_N; i++) {
    test();
  }

  auto matmul_time = 0.0;
  for (auto i = 0; i < TEST_N; i++) {
    matmul_time += test();
  }
  matmul_time /= TEST_N;
#else
  auto const matmul_time = test();
#endif

  // each cell needs N fma, and there are N * N cells
  auto const matmul_tflops = 2.0 * N * N * N / matmul_time / 1e9;
  auto constexpr theory_max_tflops = 14.7456;
  auto constexpr cublas_tflops = 8.228;

  std::cout << std::setprecision(3) << std::fixed;
  std::cout << "matmul time: " << matmul_time << " ms" << std::endl;
  std::cout << "Throughput: " << matmul_tflops << " TFLOPS \n"
            << "    (" << matmul_tflops / theory_max_tflops * 100 << "% Max)\n"
            << "    (" << matmul_tflops / cublas_tflops * 100 << "% cuBLAS)\n"
            << std::endl;

  return 0;
}
